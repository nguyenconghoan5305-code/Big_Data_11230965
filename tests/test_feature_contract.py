"""Synthetic gates for the bounded Part 4 feature contract."""

from datetime import date
from decimal import Decimal

import pyarrow as pa
import pytest
from bft65b.features import contract

from bft65b.features.contract import (
    FEATURE_INPUTS, FEATURE_NAMES, FORBIDDEN_MODEL_INPUTS,
    _previous_quarter, compute_features, validate_schema,
)
from bft65b.panel.asof import (
    AmbiguousRevisionError, FinancialReport, PREDICTOR_FIELDS,
    PredictionRequest, population_from_available_reports,
)


def report(day, *, cert=11, asset="100", dep="80", eq="10", roa=1.0,
           leverage=8.0, absent=(), published=None):
    values = dict.fromkeys(PREDICTOR_FIELDS)
    values.update(ASSET=Decimal(asset) if asset is not None else None,
                  DEP=Decimal(dep) if dep is not None else None,
                  EQ=Decimal(eq) if eq is not None else None,
                  ROA=roa, RBC1AAJ=leverage)
    return FinancialReport(cert, day, values, source_missing_fields=tuple(absent),
                           revision_published_at=published,
                           revision_time_verified=published is not None)


def test_portable_contract_requires_explicit_upstreams(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "source_vintage_bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="explicit panel and PIT"):
        contract.validate_accepted(tmp_path, "newcontract")


def request(q, cert=11):
    return PredictionRequest(cert, q, "verified_registry", q, "synthetic-known-population")


def test_dictionary_inputs_and_schema_are_strict():
    assert len(FEATURE_NAMES) == 7
    assert not FEATURE_INPUTS & FORBIDDEN_MODEL_INPUTS
    schema = pa.schema([
        pa.field("CERT", pa.int64()), pa.field("REPDTE", pa.date32()),
        pa.field("ASSET", pa.decimal128(20, 4)),
        pa.field("DEP", pa.decimal128(20, 4)),
        pa.field("EQ", pa.decimal128(20, 4)),
        pa.field("ROA", pa.float64()), pa.field("RBC1AAJ", pa.float64()),
        pa.field("SOURCE_MISSING_FIELDS", pa.list_(pa.string())),
    ], metadata={b"bft65b.schema_version": b"2"})
    validate_schema(schema)
    with pytest.raises(ValueError, match="ASSET"):
        validate_schema(schema.set(2, pa.field("ASSET", pa.float64())))
    with pytest.raises(ValueError, match="v2"):
        validate_schema(schema.with_metadata({b"bft65b.schema_version": b"1"}))


def test_formulas_units_nulls_and_nonpositive_denominator():
    r = report(date(2022, 12, 31), asset="100", dep="80", eq="-5", roa=1.25)
    out = compute_features(request(date(2023, 6, 30)), [r])
    assert out.source_repdte == r.repdte
    assert out.values["asset_size"] == Decimal("100")
    assert out.values["equity_to_assets_pct"] == Decimal("-5.000000")
    assert out.values["deposits_to_assets_pct"] == Decimal("80.000000")
    assert out.values["roa_reported_pct"] == 1.25
    assert out.values["pca_leverage_reported_pct"] == 8.0
    assert out.values["asset_growth_qoq_pct"] is None
    assert out.diagnostics["asset_growth_qoq_pct"]["reason"] == "missing_or_unavailable_calendar_quarter"
    zero = compute_features(request(date(2023, 6, 30)), [report(r.repdte, asset="0")])
    assert zero.values["asset_size"] is None
    assert zero.values["equity_to_assets_pct"] is None
    assert zero.diagnostics["equity_to_assets_pct"]["reason"] == "nonpositive_denominator"
    absent = compute_features(request(date(2023, 6, 30)), [report(r.repdte, absent=("EQ",))])
    null = compute_features(request(date(2023, 6, 30)), [report(r.repdte, eq=None)])
    assert absent.values["equity_to_assets_pct"] is None
    assert null.values["equity_to_assets_pct"] is None
    assert absent.diagnostics["equity_to_assets_pct"]["reason"] == "source_absent"
    assert null.diagnostics["equity_to_assets_pct"]["reason"] == "explicit_null"


def test_cross_year_gap_invalidates_growth_and_rolling():
    assert _previous_quarter(date(2023, 3, 31), 1) == date(2022, 12, 31)
    reports = [report(date(2022, 9, 30), asset="50"),
               report(date(2022, 12, 31), asset="60"),
               report(date(2023, 6, 30), asset="90")]
    out = compute_features(request(date(2023, 12, 31)), reports)
    assert out.source_repdte == date(2023, 6, 30)
    assert out.values["asset_growth_qoq_pct"] is None
    assert out.values["roa_four_quarter_mean_pct"] is None
    assert out.diagnostics["asset_growth_qoq_pct"]["required_repdte"] == [
        "2023-06-30", "2023-03-31"]
    assert out.diagnostics["roa_four_quarter_mean_pct"]["reason"] == "missing_or_unavailable_calendar_quarter"


def test_consecutive_history_uses_exact_calendar_quarters():
    reports = [report(date(2022, 3, 31), asset="50", roa=1.0),
               report(date(2022, 6, 30), asset="60", roa=2.0),
               report(date(2022, 9, 30), asset="80", roa=3.0),
               report(date(2022, 12, 31), asset="100", roa=4.0)]
    out = compute_features(request(date(2023, 6, 30)), reports)
    assert out.values["asset_growth_qoq_pct"] == Decimal("25.000000")
    assert out.values["roa_four_quarter_mean_pct"] == 2.5
    assert out.diagnostics["roa_four_quarter_mean_pct"]["earliest_input_availability"] == "2023-04-30"


def test_future_publication_cannot_change_earlier_membership_or_values():
    base = report(date(2022, 12, 31), asset="100")
    future_revision = report(date(2022, 12, 31), asset="999", published=date(2023, 7, 1))
    future_report = report(date(2023, 3, 31), asset="777", published=date(2023, 8, 1))
    future_other_bank = report(date(2022, 12, 31), cert=22, asset="888",
                               published=date(2023, 7, 1))
    q = date(2023, 6, 30)
    before_roster = population_from_available_reports(q, [base])
    after_roster = population_from_available_reports(q, [base, future_revision,
                                                        future_report, future_other_bank])
    assert before_roster == after_roster
    before = compute_features(before_roster[0], [base])
    after = compute_features(after_roster[0], [base, future_revision, future_report,
                                              future_other_bank])
    assert before == after
    assert after.values["asset_size"] == Decimal("100")


def test_future_exit_and_snapshot_diagnostics_cannot_enter_values():
    base = report(date(2022, 12, 31))
    changed = report(base.repdte)
    changed.values.update(DQ_FLAGS=["FUTURE_EXIT_DERIVED"],
                          EXIT_CLASSIFICATION="confirmed_failure",
                          INSTITUTION_ACTIVE_SNAPSHOT=0,
                          FAILDATE=date(2024, 1, 1))
    q = date(2023, 6, 30)
    assert compute_features(request(q), [base]).values == compute_features(
        request(q), [changed]).values


def test_inclusive_boundary_and_eligible_revision_ambiguity():
    original = report(date(2022, 12, 31))
    at_q = report(original.repdte, asset="101", published=date(2023, 6, 30))
    with pytest.raises(AmbiguousRevisionError):
        compute_features(request(date(2023, 6, 30)), [original, at_q])
    with pytest.raises(AmbiguousRevisionError):
        compute_features(request(date(2023, 6, 30)), [at_q,
            report(original.repdte, asset="102", published=date(2023, 6, 30))])
    assert compute_features(request(date(2023, 3, 31)), [original]).status == "no_eligible_history"
    assert compute_features(request(date(2023, 6, 30)), [original]).status == "selected"


def test_invalid_types_and_prediction_limit():
    r = report(date(2022, 12, 31))
    with pytest.raises(ValueError, match="2025Q2"):
        compute_features(request(date(2025, 9, 30)), [r])
    bad = report(r.repdte)
    bad.values["ROA"] = float("nan")
    with pytest.raises(TypeError, match="finite"):
        compute_features(request(date(2023, 6, 30)), [bad])
