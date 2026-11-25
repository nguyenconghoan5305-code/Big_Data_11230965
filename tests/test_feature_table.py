"""Synthetic production-path feature and one-to-one label join gates."""
from collections import deque
from datetime import date, timedelta
from decimal import Decimal
import json

import pytest
from bft65b.features import table as feature_table

from bft65b.features.contract import FEATURE_NAMES, compute_features
from bft65b.features.table import (
    CANDIDATE_SCHEMA, MODEL_SCHEMA, _anchor_from_history, build_quarter_tables,
)
from bft65b.panel.asof import (FinancialReport, PREDICTOR_FIELDS,
                               PredictionRequest)


Q = date(2023, 6, 30)
R = date(2022, 12, 31)


def test_portable_table_requires_all_selected_upstreams(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "source_vintage_bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="explicit upstream run selection"):
        feature_table._verified_inputs(tmp_path)


def report(day, *, cert=11, asset="100", dep="80", eq="10", roa=1.0,
           leverage=8.0, absent=(), published=None):
    values = dict.fromkeys(PREDICTOR_FIELDS)
    values.update(ASSET=Decimal(asset) if asset is not None else None,
                  DEP=Decimal(dep) if dep is not None else None,
                  EQ=Decimal(eq) if eq is not None else None,
                  ROA=roa, RBC1AAJ=leverage)
    return FinancialReport(cert, day, values, source_id=f"source:{cert}:{day}",
                           source_missing_fields=tuple(absent),
                           revision_published_at=published,
                           revision_time_verified=published is not None)


def candidate(anchor, q=Q):
    return {"CERT": anchor.cert, "Q": q, "SELECTED_REPDTE": anchor.repdte,
            "AVAILABILITY_DATE": anchor.available,
            "REPORT_AGE_DAYS": (q - anchor.repdte).days,
            "AVAILABILITY_BASIS": "provisional_120_day",
            "SOURCE_ID_DIAGNOSTIC": anchor.source_id,
            "SOURCE_MISSING_FIELDS_DIAGNOSTIC": list(anchor.source_missing),
            **anchor.source_values}


def label(cert=11, q=Q, *, membership="eligible", disposition="positive",
          history=True):
    return {"CERT": cert, "Q": q, "MEMBERSHIP": membership,
            "FEATURE_HISTORY_ELIGIBLE": history,
            "LABEL_DISPOSITION": disposition,
            "Y": 1 if disposition == "positive" else 0 if disposition == "negative" else None}


def test_complete_calendar_history_and_join_lineage():
    history = deque([
        report(date(2022, 3, 31), asset="50", roa=4.0),
        report(date(2022, 6, 30), asset="60", roa=3.0),
        report(date(2022, 9, 30), asset="80", roa=2.0),
        report(R, asset="100", roa=1.0),
    ], maxlen=4)
    anchor = _anchor_from_history(history, Q)
    features, model, stats = build_quarter_tables(
        Q, [candidate(anchor)], [label(), label(12, history=False)], {11: anchor})
    row = features.to_pylist()[0]
    assert features.schema == CANDIDATE_SCHEMA and model.schema == MODEL_SCHEMA
    assert row["asset_growth_qoq_pct"] == Decimal("25.000000")
    assert row["roa_four_quarter_mean_pct"] == 2.5
    assert row["equity_to_assets_pct"] == Decimal("10.000000")
    assert row["deposits_to_assets_pct"] == Decimal("80.000000")
    assert row["SOURCE_ID_DIAGNOSTIC"].startswith("source:")
    lineage = json.loads(row["FEATURE_LINEAGE_JSON"])
    assert lineage["roa_four_quarter_mean_pct"]["required_repdte"] == [
        "2022-12-31", "2022-09-30", "2022-06-30", "2022-03-31"]
    assert model.to_pylist()[0]["Y"] == 1
    assert stats["candidate_keys"] == 1 and stats["label_keys"] == 2
    assert stats["matched"] == 1 and stats["label_only"] == 1
    assert stats["model_positive"] == 1 and stats["exclusion_reasons"]["no_feature_history"] == 1


def test_cross_year_gap_and_source_absent_are_not_imputed():
    q = date(2023, 12, 31)
    anchor = _anchor_from_history(deque([
        report(R, asset="80", roa=2.0),
        report(date(2023, 6, 30), asset="100", eq=None, absent=("EQ",)),
    ], maxlen=4), q)
    features, model, stats = build_quarter_tables(
        q, [candidate(anchor, q)], [label(q=q, disposition="negative")],
        {11: anchor})
    row = features.to_pylist()[0]
    assert row["asset_growth_qoq_pct"] is None
    assert row["roa_four_quarter_mean_pct"] is None
    assert row["equity_to_assets_pct"] is None
    assert "equity_to_assets_pct:source_absent" in row["FEATURE_MISSING_REASONS"]
    assert "asset_growth_qoq_pct:missing_or_unavailable_calendar_quarter" in row["FEATURE_MISSING_REASONS"]
    assert stats["model_nulls"]["roa_four_quarter_mean_pct"] == 1
    assert model.to_pylist()[0]["Y"] == 0


def test_exact_availability_boundary_and_future_revision():
    original = report(R)
    future = report(R, asset="999", published=Q + timedelta(days=1))
    future_quarter = report(date(2023, 3, 31), asset="500")
    request = PredictionRequest(11, Q, "verified_registry", Q, "fixture")
    before = compute_features(request, [original])
    after = compute_features(request, [original, future, future_quarter])
    assert before.values == after.values
    assert before.source_repdte == after.source_repdte == R
    exactly_q = report(R, asset="101", published=Q)
    # Two eligible versions with an undated original cannot be ordered.
    from bft65b.panel.asof import AmbiguousRevisionError
    with pytest.raises(AmbiguousRevisionError):
        compute_features(request, [original, exactly_q])
    dated_only = _anchor_from_history(deque([exactly_q], maxlen=4), Q)
    assert dated_only.available == Q
    assert dated_only.available > Q - timedelta(days=1)
    with pytest.raises(ValueError, match="Newly available"):
        _anchor_from_history(deque([exactly_q], maxlen=4), date(2023, 3, 31))


def test_duplicate_keys_missing_label_and_future_availability_fail_closed():
    anchor = _anchor_from_history(deque([report(R)], maxlen=4), Q)
    item = candidate(anchor)
    with pytest.raises(ValueError, match="Duplicate"):
        build_quarter_tables(Q, [item, item], [label()], {11: anchor})
    with pytest.raises(ValueError, match="Duplicate"):
        build_quarter_tables(Q, [item], [label(), label()], {11: anchor})
    with pytest.raises(ValueError, match="one-to-one"):
        build_quarter_tables(Q, [item], [], {11: anchor})
    changed = {**item, "AVAILABILITY_DATE": Q + timedelta(days=1)}
    with pytest.raises(ValueError, match="availability"):
        build_quarter_tables(Q, [changed], [label()], {11: anchor})


def test_forbidden_columns_are_absent_from_model_schema():
    assert MODEL_SCHEMA.names == ["CERT", "Q", "Y", *FEATURE_NAMES]
    assert not set(MODEL_SCHEMA.names) & {"FAILDATE", "QUALIFYING_SOURCE_IDS",
        "MEMBERSHIP", "MEMBERSHIP_REASON", "END_DATE", "SELECTED_REPDTE",
        "SOURCE_ID_DIAGNOSTIC", "DQ_FLAGS"}


def test_contradictory_label_evidence_is_not_an_ordinary_exclusion():
    anchor = _anchor_from_history(deque([report(R)], maxlen=4), Q)
    item = candidate(anchor)
    # A resolved positive cannot be assigned to an ineligible population row.
    with pytest.raises(ValueError, match="label disposition"):
        build_quarter_tables(Q, [item],
            [label(membership="ineligible", disposition="positive")], {11: anchor})
    # A label asserting feature history must have a corresponding PIT key.
    with pytest.raises(ValueError, match="feature-history"):
        build_quarter_tables(Q, [item], [label(), label(12, history=True)], {11: anchor})
