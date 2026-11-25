"""Temporal leakage fixtures for the production feature/label path."""
from datetime import date, timedelta
from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest

from bft65b.features.contract import FEATURE_NAMES, compute_features
from bft65b.features.label import MembershipEvidence, OutcomeEvent, classify_label
from bft65b.features.leakage_audit import (
    _offer_sample, _sample_category, _synthetic_controls,
)
from bft65b.features import leakage_audit
from bft65b.features.preprocessing import make_preprocessing_pipeline
from bft65b.features.table import ReportHistoryState, build_quarter_tables
from bft65b.panel.asof import (AmbiguousRevisionError, FinancialReport,
                               PREDICTOR_FIELDS, PredictionRequest)


Q = date(2023, 6, 30)
R = date(2022, 12, 31)


def test_portable_leakage_runner_rejects_implicit_historical_runs(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "source_vintage_bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="explicit upstream run selection"):
        leakage_audit.run(tmp_path, "newaudit")
    assert not (tmp_path / "runs" / leakage_audit.NAMESPACE).exists()


def row(cert: int, repdte: date, asset: str, roa: float = 1.0) -> dict:
    return {"CERT": cert, "REPDTE": repdte, "ID": f"{cert}:{repdte}",
            "SOURCE_MISSING_FIELDS": [], "ASSET": Decimal(asset),
            "DEP": Decimal("80.0000"), "EQ": Decimal("10.0000"),
            "ROA": roa, "RBC1AAJ": 8.0}


def report(cert: int, repdte: date, asset: str,
           published: date | None = None) -> FinancialReport:
    values = dict.fromkeys(PREDICTOR_FIELDS)
    values.update(ASSET=Decimal(asset), DEP=Decimal("80.0000"),
                  EQ=Decimal("10.0000"), ROA=1.0, RBC1AAJ=8.0)
    return FinancialReport(cert, repdte, values, source_id=f"{cert}:{repdte}:{published}",
                           revision_published_at=published,
                           revision_time_verified=published is not None)


def candidate(anchor, q: date) -> dict:
    return {"CERT": anchor.cert, "Q": q, "SELECTED_REPDTE": anchor.repdte,
            "AVAILABILITY_DATE": anchor.available,
            "REPORT_AGE_DAYS": (q - anchor.repdte).days,
            "AVAILABILITY_BASIS": "provisional_120_day",
            "SOURCE_ID_DIAGNOSTIC": anchor.source_id,
            "SOURCE_MISSING_FIELDS_DIAGNOSTIC": list(anchor.source_missing),
            **anchor.source_values}


def label(cert: int, q: date, disposition="negative", membership="eligible",
          history=True, y=None) -> dict:
    return {"CERT": cert, "Q": q, "MEMBERSHIP": membership,
            "FEATURE_HISTORY_ELIGIBLE": history,
            "LABEL_DISPOSITION": disposition,
            "Y": (1 if disposition == "positive" else 0 if disposition == "negative" else None)
                 if y is None else y}


def test_production_cache_advances_only_on_available_quarters_and_preserves_other_bank():
    sources = [{"quarter": "20220930", "rows": 2},
               {"quarter": "20221231", "rows": 2},
               {"quarter": "20230630", "rows": 1}]
    payload = {
        "20220930": [row(22, date(2022, 9, 30), "70"),
                     row(11, date(2022, 9, 30), "50")],
        "20221231": [row(11, R, "100"), row(22, R, "90")],
        "20230630": [row(11, date(2023, 6, 30), "120")],
    }
    state = ReportHistoryState(sources)
    load = lambda source: payload[source["quarter"]]
    state.advance(date(2023, 3, 31), load)
    assert state.pointer == 1
    assert state.anchors[11].repdte == date(2022, 9, 30)
    state.advance(Q, load)
    assert state.pointer == 2 and state.scanned_reports == 4
    first = state.anchors[11]
    other = state.anchors[22]
    assert first.repdte == R
    assert first.features["asset_growth_qoq_pct"] == Decimal("100.000000")
    assert first.features["roa_four_quarter_mean_pct"] is None
    table, model, _ = build_quarter_tables(Q, [candidate(first, Q)],
                                         [label(11, Q)], state.anchors)
    assert model.to_pylist()[0]["asset_growth_qoq_pct"] == Decimal("100.000000")
    state.advance(date(2023, 9, 30), load)
    assert state.pointer == 2 and state.anchors[11] is first
    assert state.anchors[22] is other
    state.advance(date(2023, 12, 31), load)
    assert state.pointer == 3 and state.anchors[11].repdte == date(2023, 6, 30)
    assert state.anchors[11].features["asset_growth_qoq_pct"] is None
    assert state.anchors[11].features["roa_four_quarter_mean_pct"] is None
    assert state.anchors[22] is other
    assert "missing_or_unavailable_calendar_quarter" in state.anchors[11].feature_lineage_json
    assert table.num_rows == model.num_rows == 1


def test_source_row_order_canonicality_and_invalid_quarter_order():
    sources = [{"quarter": "20221231", "rows": 2}]
    left = [row(11, R, "100"), row(22, R, "90")]
    right = list(reversed(left))
    a, b = ReportHistoryState(sources), ReportHistoryState(sources)
    a.advance(Q, lambda source: left)
    b.advance(Q, lambda source: right)
    assert {cert: value.features for cert, value in a.anchors.items()} == {
        cert: value.features for cert, value in b.anchors.items()}
    with pytest.raises(ValueError, match="calendar order"):
        ReportHistoryState([sources[0], {"quarter": "20220930", "rows": 0}])
    with pytest.raises(ValueError, match="calendar order"):
        ReportHistoryState([sources[0], sources[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        ReportHistoryState(sources).advance(Q, lambda source: [left[0], left[0]])
    with pytest.raises(ValueError, match="wrong-quarter"):
        ReportHistoryState([{"quarter": "20221231", "rows": 1}]).advance(
            Q, lambda source: [row(11, date(2022, 9, 30), "100")])
    with pytest.raises(ValueError, match="calendar order"):
        a.advance(date(2023, 3, 31), lambda source: left)


def test_future_reports_and_revisions_do_not_change_earlier_values():
    request = PredictionRequest(11, Q, "verified_registry", Q, "synthetic")
    original = report(11, R, "100")
    earlier = compute_features(request, [original])
    future_quarter = report(11, date(2023, 3, 31), "999")
    future_revision = report(11, R, "888", date(2023, 7, 1))
    altered = compute_features(request, [future_quarter, future_revision, original])
    assert earlier.values == altered.values
    assert earlier.diagnostics == altered.diagnostics
    assert altered.source_repdte == R
    at_q = report(11, R, "101", Q)
    selected = compute_features(request, [at_q])
    assert selected.values["asset_size"] == Decimal("101")
    assert selected.source_repdte == R
    with pytest.raises(AmbiguousRevisionError):
        compute_features(request, [original, at_q])
    with pytest.raises(AmbiguousRevisionError):
        compute_features(request, [at_q, report(11, R, "102", Q)])


def test_join_rejects_wrong_keys_history_flags_and_disposition_mismatch():
    state = ReportHistoryState([{"quarter": "20221231", "rows": 1}])
    state.advance(Q, lambda source: [row(11, R, "100")])
    anchor = state.anchors[11]
    correct = candidate(anchor, Q)
    with pytest.raises(ValueError, match="Duplicate"):
        build_quarter_tables(Q, [correct, correct], [label(11, Q)], state.anchors)
    with pytest.raises(ValueError, match="wrong-quarter"):
        build_quarter_tables(Q, [dict(correct, Q=date(2023, 3, 31))],
                             [label(11, Q)], state.anchors)
    with pytest.raises(ValueError, match="one-to-one"):
        build_quarter_tables(Q, [correct], [], state.anchors)
    with pytest.raises(ValueError, match="feature-history"):
        build_quarter_tables(Q, [correct], [label(11, Q, history=False)], state.anchors)
    with pytest.raises(ValueError, match="feature-history"):
        build_quarter_tables(Q, [correct], [label(11, Q), label(22, Q)], state.anchors)
    with pytest.raises(ValueError, match="label disposition"):
        build_quarter_tables(Q, [correct], [label(11, Q, membership="ineligible")], state.anchors)
    with pytest.raises(ValueError, match="label disposition"):
        build_quarter_tables(Q, [correct], [label(11, Q, disposition="positive", y=0)], state.anchors)
    valid, model, stats = build_quarter_tables(
        Q, [correct], [label(11, Q), label(22, Q, disposition="excluded",
                                        membership="ineligible", history=False)], state.anchors)
    assert stats["label_only"] == 1 and stats["model_negative"] == 1
    assert valid.num_rows == model.num_rows == 1


def test_outcome_boundaries_assistance_merger_and_identity_uncertainty():
    member = MembershipEvidence(11, Q, "eligible", "retrospective_operational_v1",
                                None, "fixture", True)
    right = date(2024, 6, 30)
    event = lambda day, transaction="verified_failure", cert=11: OutcomeEvent(
        f"{transaction}:{day}:{cert}", cert, day, transaction,
        "chronological_candidate")
    classify = lambda events, coverage=right, m=member: classify_label(
        m, events, outcome_complete_through=coverage)
    assert classify([event(Q)]).disposition == "excluded"
    assert classify([event(Q + timedelta(days=1))]).y == 1
    assert classify([event(right)]).y == 1
    assert classify([event(right + timedelta(days=1))]).y == 0
    assert classify([], right - timedelta(days=1)).disposition == "unresolved"
    assert classify([event(right, "assistance_transaction")]).y == 0
    assert classify([event(right, "documented_merger")]).y == 0
    uncertain = MembershipEvidence(11, Q, "uncertain", "retrospective_operational_v1",
                                   None, "fixture", False)
    assert classify([], m=uncertain).disposition == "unresolved"
    assert classify([event(right, cert=None)]).disposition == "unresolved"


def test_outcome_and_future_metadata_perturbation_changes_label_only():
    state = ReportHistoryState([{"quarter": "20221231", "rows": 1}])
    source = row(11, R, "100")
    state.advance(Q, lambda source_item: [source])
    item = candidate(state.anchors[11], Q)
    first, first_model, _ = build_quarter_tables(Q, [item], [label(11, Q)], state.anchors)
    second, second_model, _ = build_quarter_tables(Q, [dict(item, FAILDATE="future",
        MEMBERSHIP_REASON="changed", DQ_FLAGS=["future"])],
        [label(11, Q, disposition="positive")], state.anchors)
    for feature in FEATURE_NAMES:
        assert first[feature].to_pylist() == second[feature].to_pylist()
    assert first_model["Y"].to_pylist() == [0]
    assert second_model["Y"].to_pylist() == [1]


def _fold(assets: list[str], growth: list[object], *, q: date, y: int) -> pd.DataFrame:
    rows = []
    for asset, value in zip(assets, growth):
        row = {name: 1.0 for name in FEATURE_NAMES}
        row.update(asset_size=Decimal(asset), asset_growth_qoq_pct=value,
                   roa_reported_pct=None, CERT=11, Q=q, Y=y,
                   FAILDATE="audit-only", MEMBERSHIP="eligible")
        rows.append(row)
    return pd.DataFrame(rows)


def test_two_forward_folds_and_heldout_perturbations_do_not_refit():
    early = _fold(["100", "200"], [None, None], q=date(2022, 12, 31), y=0)
    later = _fold(["300", "400"], [Decimal("2"), Decimal("4")], q=Q, y=0)
    heldout = _fold(["1000000"], [None], q=date(2023, 9, 30), y=1)
    for train in (early, later):
        pipeline = make_preprocessing_pipeline().fit(train)
        train_values = pipeline.transform(train).copy()
        stats = pipeline.named_steps["imputer"].statistics_.copy()
        means = pipeline.named_steps["scaler"].mean_.copy()
        variance = pipeline.named_steps["scaler"].var_.copy()
        scale = pipeline.named_steps["scaler"].scale_.copy()
        assert train_values.shape == (2, 7)
        assert list(pipeline.get_feature_names_out()) == list(FEATURE_NAMES)
        assert scale[3] == 1.0  # constant or entirely missing ROA
        pipeline.transform(heldout)
        extreme = heldout.copy()
        extreme["asset_size"] = Decimal("999999999")
        extreme["Y"] = 0
        extreme["FAILDATE"] = "different"
        pipeline.transform(extreme)
        np.testing.assert_array_equal(train_values, pipeline.transform(train))
        np.testing.assert_array_equal(stats, pipeline.named_steps["imputer"].statistics_)
        np.testing.assert_array_equal(means, pipeline.named_steps["scaler"].mean_)
        np.testing.assert_array_equal(variance, pipeline.named_steps["scaler"].var_)
        np.testing.assert_array_equal(scale, pipeline.named_steps["scaler"].scale_)
    assert make_preprocessing_pipeline().fit(early).named_steps["scaler"].mean_[0] == 150.0
    assert make_preprocessing_pipeline().fit(later).named_steps["scaler"].mean_[0] == 350.0
    combined = make_preprocessing_pipeline().fit(pd.concat([early, heldout], ignore_index=True))
    assert combined.named_steps["scaler"].mean_[0] != 150.0
    assert _synthetic_controls()["combined_fit_negative_control_detected"]


def test_deterministic_sample_category_and_order():
    ordinary = {name: Decimal("1") for name in FEATURE_NAMES}
    ordinary.update(CERT=11, FEATURE_MISSING_REASONS=[])
    gap = dict(ordinary, asset_growth_qoq_pct=None,
               FEATURE_MISSING_REASONS=["asset_growth_qoq_pct:missing_or_unavailable_calendar_quarter"])
    missing = dict(ordinary, equity_to_assets_pct=None,
                   FEATURE_MISSING_REASONS=["equity_to_assets_pct:explicit_null"])
    assert [_sample_category(row) for row in (ordinary, gap, missing)] == [
        "ordinary", "calendar_gap", "other_missing"]
    rows = [dict(ordinary, CERT=cert) for cert in range(11, 31)]
    a, b = {"ordinary": []}, {"ordinary": []}
    for item in rows:
        _offer_sample(a, item, "20230630", "fixture")
    for item in reversed(rows):
        _offer_sample(b, item, "20230630", "fixture")
    assert sorted(a["ordinary"]) == sorted(b["ordinary"])
    assert len(a["ordinary"]) == 8
