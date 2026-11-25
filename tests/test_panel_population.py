"""Synthetic, outcome-free checks of the P2 population membership policy."""

from datetime import date, timedelta

import pytest

from bft65b.panel.asof import FinancialReport, PREDICTOR_FIELDS, population_from_available_reports, select_asof
from bft65b.panel.population import DatedEvent, _quarter_stats, classify, quarter_distance


Q = date(2023, 6, 30)


def row(repdte=date(2022, 12, 31), cert=1, baseline="eligible"):
    return {"CERT": cert, "Q": Q, "SELECTED_REPDTE": repdte,
            "AVAILABILITY_DATE": repdte + timedelta(days=120),
            "REPORT_AGE_DAYS": (Q - repdte).days,
            "ELIGIBILITY_DISPOSITION": baseline}


def event(kind, effective, known, cert=1, source="official:1", verified=True):
    return DatedEvent(cert, kind, effective, known, source, verified)


def test_four_quarter_window_is_inclusive_and_preserves_gap_uncertainty():
    assert quarter_distance(date(2022, 6, 30), Q) == 4
    assert classify(row(date(2022, 6, 30)))["MEMBERSHIP_DISPOSITION"] == "recent_reporter"
    assert classify(row(date(2022, 3, 31)))["MEMBERSHIP_DISPOSITION"] == "activity_uncertain_stale"
    # Once a later report is available, a temporarily missing reporter re-enters.
    resumed = {**row(date(2023, 3, 31)), "Q": date(2023, 9, 30),
               "REPORT_AGE_DAYS": 183}
    assert classify(resumed)["MEMBERSHIP_DISPOSITION"] == "recent_reporter"


@pytest.mark.parametrize("kind,expected", [
    ("failure", "known_exit"), ("merger", "known_exit"),
    ("other_exit", "known_exit"), ("assistance", "recent_reporter"),
])
def test_verified_event_kind_and_same_day_boundary(kind, expected):
    result = classify(row(), events=[event(kind, Q, Q)])
    assert result["MEMBERSHIP_DISPOSITION"] == expected
    assert result["MEMBERSHIP_REASON"] == (f"known_{kind}_by_q" if expected == "known_exit"
                                                else "selected_report_within_window")


def test_event_before_q_but_learned_after_q_is_not_exclusion():
    past = event("merger", Q - timedelta(days=1), Q + timedelta(days=1))
    future = event("failure", Q + timedelta(days=1), Q + timedelta(days=1))
    unknown = event("failure", Q - timedelta(days=1), None)
    assert classify(row(), events=[past, future, unknown])["MEMBERSHIP_DISPOSITION"] == "recent_reporter"
    assert classify(row(), events=[event("failure", Q - timedelta(days=1), Q)])[
        "MEMBERSHIP_DISPOSITION"] == "known_exit"


def test_conflicting_identity_and_multiple_events_do_not_duplicate():
    result = classify(row(), events=[event("failure", Q, Q), event("merger", Q, Q,
                                                                    source="official:2")])
    assert result["MEMBERSHIP_DISPOSITION"] == "unresolved_identity"
    assert classify(row(), events=[event("identity_conflict", Q, Q)])[
        "MEMBERSHIP_DISPOSITION"] == "unresolved_identity"


def test_missing_history_has_no_candidate_and_no_invented_membership():
    assert _quarter_stats([], [], set())["candidate_bank_quarters"] == 0
    with pytest.raises(ValueError, match="not as-of consistent"):
        classify({**row(), "AVAILABILITY_DATE": Q + timedelta(days=1)})


def test_retrospective_candidate_disposition_and_future_fields_do_not_change_membership():
    original = row(baseline="eligible")
    changed = {**original, "ELIGIBILITY_DISPOSITION": "confirmed_failure_before_or_on_q",
               "ASSET": 999, "DQ_FLAGS": ["future"], "INSTITUTION_ACTIVE_SNAPSHOT": 0}
    past_result = classify(original)
    changed_result = classify(changed, events=[event("failure", Q + timedelta(days=1),
                                                     Q + timedelta(days=1))])
    assert past_result["MEMBERSHIP_DISPOSITION"] == changed_result["MEMBERSHIP_DISPOSITION"]
    assert past_result["SELECTED_REPDTE"] == changed_result["SELECTED_REPDTE"]
    assert past_result["AVAILABILITY_DATE"] == changed_result["AVAILABILITY_DATE"]
    assert "ASSET" not in changed_result and "DQ_FLAGS" not in changed_result
    assert changed_result["CANDIDATE_DISPOSITION"] == "confirmed_failure_before_or_on_q"


def test_future_report_revision_event_and_snapshot_perturbation_keeps_earlier_predictor():
    def report(repdte, asset, published=None, snapshot=1):
        values = {field: None for field in PREDICTOR_FIELDS}
        values["ASSET"] = asset
        values["INSTITUTION_ACTIVE_SNAPSHOT"] = snapshot
        return FinancialReport(1, repdte, values, revision_published_at=published,
                               revision_time_verified=published is not None)

    original = report(date(2022, 12, 31), 10)
    modified_original = report(date(2022, 12, 31), 10, snapshot=0)
    future = report(date(2023, 6, 30), 90)
    future_revision = report(date(2022, 12, 31), 99, date(2023, 7, 1))
    before = select_asof(population_from_available_reports(Q, [original]), [original])[0]
    after = select_asof(population_from_available_reports(Q, [modified_original, future,
                                                               future_revision]),
                        [modified_original, future, future_revision])[0]
    assert before.source_repdte == after.source_repdte == date(2022, 12, 31)
    assert before.predictors == after.predictors
    assert before.predictors["ASSET"] == 10
    assert classify(row(), events=[event("failure", date(2023, 7, 1),
                                        date(2023, 7, 1))])["MEMBERSHIP_DISPOSITION"] == "recent_reporter"


def test_reconciliation_counts_and_sensitivity_are_explicit():
    rows = [row(cert=1), row(date(2022, 3, 31), cert=2),
            row(date(2022, 6, 30), cert=3, baseline="unresolved_identity")]
    decisions = [classify(r) for r in rows]
    result = _quarter_stats(rows, decisions, {3})
    assert result["candidate_bank_quarters"] == result["candidate_distinct_banks"] == 3
    assert result["recent_reporter"] == 2
    assert result["activity_uncertain_stale"] == 1
    assert result["recent_entrants_vs_previous_quarter"] == 1
    assert result["baseline_eligible"] == 2
    assert result["baseline_unresolved_identity"] == 1
    assert result["sensitivity_recent_by_window_quarters"] == {"2": 1, "4": 2, "8": 3}
