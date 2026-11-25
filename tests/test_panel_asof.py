"""Synthetic point-in-time report-selection and population leakage contracts."""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from bft65b.config import EXTENDED_FINANCIAL_FIELDS
from bft65b.panel.asof import (
    AVAILABILITY_LAG_DAYS, PREDICTOR_FIELDS, AmbiguousRevisionError,
    FinancialReport, PredictionRequest, availability_date,
    population_from_available_reports, select_asof,
)


Q1 = date(2023, 3, 31)
Q2 = date(2023, 6, 30)
Q3 = date(2023, 9, 30)
OLD = date(2022, 12, 31)
OLDER = date(2022, 9, 30)


def report(cert=1, repdte=OLD, asset="10.0000", *, published=None,
           verified=False, extras=None, source_id="synthetic-report"):
    values = {field: None for field in PREDICTOR_FIELDS}
    values["ASSET"] = Decimal(asset)
    values.update(extras or {})
    return FinancialReport(cert, repdte, values, source_id,
                           (), published, verified)


def request(cert=1, q=Q2, known_at=date(2023, 1, 1)):
    return PredictionRequest(cert, q, "verified_registry", known_at,
                             f"synthetic-registry:{cert}")


def test_predictor_whitelist_is_exact_selected_numeric_financials():
    assert len(PREDICTOR_FIELDS) == 23
    assert set(PREDICTOR_FIELDS) == set(EXTENDED_FINANCIAL_FIELDS) - {
        "CERT", "REPDTE", "STNAME"}
    assert not set(PREDICTOR_FIELDS) & {"FAILDATE", "RESTYPE", "RESTYPE1",
        "EXIT_CATEGORY", "INSTITUTION_ACTIVE_SNAPSHOT", "DQ_FLAGS",
        "SOURCE_MISSING_FIELDS", "ID"}


def test_provisional_lag_uses_calendar_days_and_inclusive_cutoff():
    base = report()
    cutoff = OLD + timedelta(days=AVAILABILITY_LAG_DAYS)
    assert cutoff == date(2023, 4, 30)
    assert availability_date(base) == cutoff
    assert availability_date(base) > cutoff - timedelta(days=1)
    assert availability_date(base) <= cutoff
    assert availability_date(base) < cutoff + timedelta(days=1)
    early_verified_release = report(published=date(2023, 1, 20), verified=True)
    assert availability_date(early_verified_release) == cutoff


@pytest.mark.parametrize("release,selected", [
    (Q2 - timedelta(days=1), True),
    (Q2, True),
    (Q2 + timedelta(days=1), False),
])
def test_verified_revision_one_day_before_on_and_after_prediction_q(release, selected):
    item = report(published=release, verified=True)
    result = select_asof([request()], [item])[0]
    assert (result.status == "selected") is selected
    assert result.source_availability_date == (release if selected else None)


def test_latest_eligible_report_wins_and_age_is_recorded():
    items = [report(repdte=OLDER, asset="2.0000"),
             report(repdte=OLD, asset="3.0000")]
    chosen = select_asof([request()], items)[0]
    assert chosen.status == "selected"
    assert chosen.source_repdte == OLD
    assert chosen.source_availability_date == date(2023, 4, 30)
    assert chosen.report_age_days == (Q2 - OLD).days
    assert chosen.predictors["ASSET"] == Decimal("3.0000")
    assert chosen.availability_basis == "provisional_120_day"


def test_no_eligible_history_is_not_zero_or_imputed():
    item = report(repdte=OLD)
    result = select_asof([request(q=Q1)], [item])[0]
    assert result.status == "no_eligible_history"
    assert result.predictors is None
    assert result.source_repdte is None and result.report_age_days is None
    assert population_from_available_reports(Q1, [item]) == []


def test_bank_isolation_and_future_q_report_cannot_create_earlier_population():
    old_bank = report(cert=1)
    future_bank = report(cert=2, repdte=Q2)
    roster = population_from_available_reports(Q2, [old_bank, future_bank])
    assert [(row.cert, row.q) for row in roster] == [(1, Q2)]
    assert roster[0].population_known_at == date(2023, 4, 30)
    assert select_asof(roster, [old_bank, future_bank])[0].predictors["ASSET"] == Decimal("10.0000")
    second = select_asof([request(cert=2)], [old_bank, future_bank])[0]
    assert second.status == "no_eligible_history"


def test_population_evidence_must_precede_q_and_cannot_use_future_exit_source():
    with pytest.raises(ValueError, match="evidenced no later"):
        PredictionRequest(1, Q2, "verified_registry", Q2 + timedelta(days=1), "later")
    with pytest.raises(ValueError, match="evidenced no later"):
        PredictionRequest(1, Q2, "future_exit", Q1, "later-exit")
    with pytest.raises(ValueError, match="reconcile"):
        select_asof([PredictionRequest(1, Q2, "available_report_roster", Q1,
                                      "fabricated-old-roster")], [report()])
    with pytest.raises(ValueError, match="Duplicate prediction"):
        select_asof([request(), request()], [report()])


def test_conflicting_versions_without_verified_order_or_with_tied_release_fail():
    undated = [report(asset="10.0000"),
               report(asset="99.0000", published=date(2023, 5, 1), verified=True)]
    with pytest.raises(AmbiguousRevisionError, match="undated or tied"):
        select_asof([request()], undated)
    with pytest.raises(AmbiguousRevisionError, match="undated or tied"):
        population_from_available_reports(Q2, undated)
    tied = [report(asset="10.0000", published=date(2023, 5, 1), verified=True),
            report(asset="99.0000", published=date(2023, 5, 1), verified=True)]
    with pytest.raises(AmbiguousRevisionError, match="undated or tied"):
        select_asof([request()], tied)
    with pytest.raises(AmbiguousRevisionError, match="undated or tied"):
        population_from_available_reports(Q2, tied)


def test_revision_published_exactly_at_q_is_eligible_but_undated_tie_is_ambiguous():
    revision = report(asset="99.0000", published=Q2, verified=True,
                      source_id="revision-at-q")
    roster = population_from_available_reports(Q2, [revision])
    selected = select_asof(roster, [revision])[0]
    assert len(roster) == 1 and roster[0].population_known_at == Q2
    assert selected.source_availability_date == Q2
    assert selected.predictors["ASSET"] == Decimal("99.0000")
    original = report(asset="10.0000")
    with pytest.raises(AmbiguousRevisionError, match="undated or tied"):
        population_from_available_reports(Q2, [original, revision])
    with pytest.raises(AmbiguousRevisionError, match="undated or tied"):
        select_asof([request()], [original, revision])


def test_verified_later_revision_changes_only_later_prediction():
    base = report(asset="10.0000", published=date(2023, 5, 1), verified=True)
    revised = report(asset="20.0000", published=date(2023, 7, 1), verified=True,
                     source_id="synthetic-revision")
    early, later = select_asof([request(q=Q2), request(q=Q3)], [base, revised])
    assert early.predictors["ASSET"] == Decimal("10.0000")
    assert later.predictors["ASSET"] == Decimal("20.0000")
    assert early.source_repdte == later.source_repdte == OLD
    assert early.source_availability_date == date(2023, 5, 1)
    assert later.source_availability_date == date(2023, 7, 1)
    assert population_from_available_reports(Q2, [base, revised]) == \
        population_from_available_reports(Q2, [base])
    assert population_from_available_reports(Q3, [base, revised])[0].population_known_at == \
        date(2023, 5, 1)


def test_undated_panel_original_is_unchanged_by_future_verified_revision():
    row = {"CERT": 1, "REPDTE": OLD, "ID": "accepted-original",
           "SOURCE_MISSING_FIELDS": [],
           **{field: None for field in PREDICTOR_FIELDS}}
    row["ASSET"] = Decimal("10.0000")
    original = FinancialReport.from_panel_row(row)
    future_revision = report(asset="99.0000", published=date(2023, 7, 1),
                             verified=True, source_id="later-revision")
    baseline_population = population_from_available_reports(Q2, [original])
    baseline_selected = select_asof(baseline_population, [original])
    revised_population = population_from_available_reports(Q2,
                                                            [original, future_revision])
    revised_selected = select_asof(revised_population, [original, future_revision])
    assert revised_population == baseline_population
    assert revised_selected == baseline_selected
    assert revised_selected[0].predictors["ASSET"] == Decimal("10.0000")


def test_future_report_exit_and_snapshot_perturbation_cannot_change_earlier_selection():
    base = report(asset="10.0000", published=date(2023, 5, 1), verified=True,
        extras={"DQ_FLAGS": ["future_derived_flag"],
                "EXIT_CATEGORY": "confirmed_failure",
                "INSTITUTION_ACTIVE_SNAPSHOT": 0,
                "RESTYPE": "FAILURE"})
    before_population = population_from_available_reports(Q2, [base])
    before = select_asof(before_population, [base])
    altered = report(asset="10.0000", published=date(2023, 5, 1), verified=True,
        extras={"DQ_FLAGS": ["changed_after_q"],
                "EXIT_CATEGORY": "ambiguous_exit",
                "INSTITUTION_ACTIVE_SNAPSHOT": 1,
                "RESTYPE": "ASSISTANCE"})
    future_report = report(cert=2, repdte=Q2, asset="999999.0000")
    later_revision = report(asset="999.0000", published=date(2023, 7, 1), verified=True)
    after_population = population_from_available_reports(Q2, [altered, future_report, later_revision])
    after = select_asof(after_population, [altered, future_report, later_revision])
    assert after_population == before_population
    assert after == before
    assert len(after[0].predictors) == 23
    assert not set(after[0].predictors) & {"DQ_FLAGS", "EXIT_CATEGORY",
        "INSTITUTION_ACTIVE_SNAPSHOT", "RESTYPE"}


def test_panel_adapter_keeps_source_absence_diagnostic_but_excludes_quality_flags():
    row = {"CERT": 1, "REPDTE": OLD, "STNAME": "Example Bank",
           "ID": "source-1", "SOURCE_MISSING_FIELDS": ["P3ASSET"],
           "DQ_FLAGS": ["future_derived_flag"], "EXIT_CATEGORY": "confirmed_failure"}
    row.update({field: None for field in PREDICTOR_FIELDS})
    row["ASSET"] = Decimal("1.0000")
    selected = select_asof([request()], [FinancialReport.from_panel_row(row)])[0]
    assert selected.predictors["ASSET"] == Decimal("1.0000")
    assert selected.predictors["P3ASSET"] is None
    assert selected.diagnostics["source_missing_fields"] == ("P3ASSET",)
    assert selected.diagnostics["source_id"] == "source-1"
    assert "DQ_FLAGS" not in selected.predictors and "EXIT_CATEGORY" not in selected.diagnostics


def test_report_contract_rejects_unverified_revision_timestamps_and_missing_columns():
    with pytest.raises(ValueError, match="explicitly verified"):
        report(published=Q2, verified=False)
    with pytest.raises(ValueError, match="lacks predictor"):
        FinancialReport(1, OLD, {"ASSET": Decimal("1")})
