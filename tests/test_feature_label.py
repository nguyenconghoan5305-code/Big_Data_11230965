"""Synthetic four-quarter outcome and population-separation gates."""

from datetime import date, timedelta

import pytest

from bft65b.features.label import (
    MembershipEvidence, OutcomeEvent, classify_label, label_batch,
)


Q = date(2023, 6, 30)
RIGHT = date(2024, 6, 30)


def member(q=Q, *, cert=11, status="eligible", identity=True):
    return MembershipEvidence(cert, q, status, "verified_active_v1", q,
                              "synthetic-register-row", identity)


def event(day, *, source="failure-1", cert=11, transaction="verified_failure",
          link="chronological_candidate", match="matched"):
    return OutcomeEvent(source, cert, day, transaction, link, match)


def decide(events=(), *, case=None, through=RIGHT):
    return classify_label(case or member(), events, outcome_complete_through=through)


def test_exact_left_and_right_boundaries():
    on_q = decide([event(Q)])
    assert (on_q.disposition, on_q.y, on_q.reason) == (
        "excluded", None, "verified_failure_before_or_on_q")
    after_q = decide([event(Q + timedelta(days=1))])
    assert (after_q.disposition, after_q.y, after_q.qualifying_source_ids) == (
        "positive", 1, ("failure-1",))
    at_right = decide([event(RIGHT)])
    assert (at_right.disposition, at_right.y, at_right.right_endpoint) == (
        "positive", 1, RIGHT)
    after_right = decide([event(RIGHT + timedelta(days=1))])
    assert (after_right.disposition, after_right.y) == ("negative", 0)


def test_followup_and_nonfailure_exit():
    assert decide([], through=RIGHT - timedelta(days=1)).reason == "incomplete_four_quarter_followup"
    assert decide([event(Q + timedelta(days=1), transaction="assistance_transaction")]).disposition == "negative"
    merger = event(Q + timedelta(days=1), transaction="documented_merger", link="documented")
    assert (decide([merger]).disposition, decide([merger]).reason) == (
        "negative", "complete_followup_with_nonfailure_exit")
    assert decide([]).disposition == "negative"


def test_identity_and_population_are_separate_from_future_outcome():
    future_failure = event(Q + timedelta(days=1))
    assert decide([future_failure], case=member(identity=False)).reason == "identity_unresolved_at_q"
    assert decide([future_failure], case=member(status="uncertain")).reason == "population_membership_uncertain_at_q"
    assert decide([future_failure], case=member(status="ineligible")).reason == "not_in_approved_population_at_q"
    with pytest.raises(ValueError, match="approved policy"):
        MembershipEvidence(11, Q, "eligible", "recent_reporter_4_quarters_v1",
                           Q, "unapproved-proxy", True)
    with pytest.raises(ValueError, match="known by q"):
        MembershipEvidence(11, Q, "eligible", "verified_active_v1",
                           Q + timedelta(days=1), "future-registry", True)


def test_unresolved_link_and_null_cert_cannot_make_negative():
    bad_link = event(Q + timedelta(days=1), link="unmatched_cert")
    assert decide([bad_link]).disposition == "unresolved"
    null_cert = event(Q + timedelta(days=1), cert=None)
    assert decide([null_cert]).reason == "unassignable_failure_source_record_in_window"
    old_null_cert = event(date(1976, 1, 1), cert=None)
    assert decide([old_null_cert]).disposition == "negative"


def test_unassigned_event_does_not_override_independent_positive():
    linked = event(Q + timedelta(days=1), source="linked")
    unassigned = event(Q + timedelta(days=2), source="unassigned", cert=None)
    result = decide([linked, unassigned])
    assert (result.disposition, result.y, result.qualifying_source_ids) == (
        "positive", 1, ("linked",))
    assert decide([unassigned]).disposition == "unresolved"
    same_bank_conflict = event(Q + timedelta(days=2), source="conflict",
                               transaction="failure_charter_uncertain")
    assert decide([linked, same_bank_conflict]).disposition == "unresolved"


def test_multiple_source_records_preserve_event_identity_without_row_multiplication():
    first = event(Q + timedelta(days=1), source="id-a")
    second = event(Q + timedelta(days=2), source="id-b")
    decision = decide([first, second])
    assert decision.disposition == "unresolved"
    assert decision.qualifying_source_ids == ("id-a", "id-b")
    assert isinstance(decision.y, type(None))
    with pytest.raises(ValueError, match="Duplicate outcome source IDs"):
        decide([first, first])
    assistance = event(Q + timedelta(days=2), source="assist-b",
                       transaction="assistance_transaction")
    assert decide([first, assistance]).disposition == "positive"


def test_preq_failure_and_late_modelling_q_are_excluded():
    pre = event(Q - timedelta(days=1))
    assert decide([pre]).reason == "verified_failure_before_or_on_q"
    later = member(date(2025, 9, 30))
    decision = decide([], case=later, through=date(2026, 9, 30))
    assert (decision.disposition, decision.reason, decision.y) == (
        "excluded", "prediction_after_2025q2", None)
    last = member(date(2025, 6, 30))
    assert decide([], case=last, through=date(2026, 6, 30)).right_endpoint == date(2026, 6, 30)


def test_positive_does_not_require_complete_negative_followup():
    decision = decide([event(Q + timedelta(days=1))], through=Q + timedelta(days=1))
    assert (decision.disposition, decision.y) == ("positive", 1)


def test_unknown_transaction_and_conflicting_source_match_remain_unresolved():
    ambiguous = event(Q + timedelta(days=1), transaction="failure_charter_uncertain")
    assert decide([ambiguous]).disposition == "unresolved"
    conflict = event(Q + timedelta(days=1), match="corroboration_conflict")
    assert decide([conflict]).disposition == "unresolved"


def test_one_source_event_can_label_several_earlier_quarters_without_duplicate_keys():
    one_failure = event(date(2024, 6, 30), source="one-source-id")
    observations = [member(day) for day in (date(2023, 6, 30),
        date(2023, 9, 30), date(2023, 12, 31), date(2024, 3, 31))]
    decisions = label_batch(observations, [one_failure],
                            outcome_complete_through=date(2024, 6, 30))
    assert len(decisions) == len({(item.cert, item.q) for item in decisions}) == 4
    assert sum(item.y == 1 for item in decisions) == 4
    assert {source for item in decisions for source in item.qualifying_source_ids} == {
        "one-source-id"}
    with pytest.raises(ValueError, match="Duplicate bank-quarter"):
        label_batch([observations[0], observations[0]], [one_failure],
                    outcome_complete_through=date(2024, 6, 30))
