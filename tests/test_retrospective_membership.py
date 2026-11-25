"""Synthetic gates for the historical cohort and separate label outcome."""
from datetime import date, timedelta
from decimal import Decimal

import pytest

from bft65b.features.label import MembershipEvidence, OutcomeEvent, classify_label
from bft65b.panel.asof import (FinancialReport, PREDICTOR_FIELDS,
                               population_from_available_reports, select_asof)
from bft65b.panel.membership_rule import (
    ActiveInterval, BankEvidence, derive_bank_evidence, retrospective_membership,
)
from bft65b.panel import retrospective
from bft65b.panel import retrospective_review
from bft65b.manifest import RunManifest


Q = date(2023, 6, 30)


def test_portable_cohort_rejects_implicit_accepted_upstreams(tmp_path, monkeypatch):
    (tmp_path / "final_report_bft65b.pdf").write_bytes(b"synthetic assignment")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "membership_source_bundle_v1.json").write_text("{}")
    monkeypatch.setattr(retrospective, "load_config", lambda _: object())
    monkeypatch.setattr(retrospective, "full_plan", lambda _: type("Plan", (), {
        "quarters": ("19840331",) * 166})())
    with pytest.raises(ValueError, match="explicit upstream run selection"):
        retrospective.run(tmp_path, "newrun")


def test_portable_cohort_rejects_mismatched_audit_lineage(tmp_path, monkeypatch):
    (tmp_path / "final_report_bft65b.pdf").write_bytes(b"synthetic assignment")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "membership_source_bundle_v1.json").write_text("{}")
    monkeypatch.setattr(retrospective, "load_config", lambda _: object())
    monkeypatch.setattr(retrospective, "full_plan", lambda _: type("Plan", (), {
        "quarters": ("19840331",) * 166})())
    monkeypatch.setattr(retrospective.RunManifest, "read", lambda _: type("Manifest", (), {
        "config": {"source_panel_run": "wrong", "source_pit_run": "selectedpit"}})())
    with pytest.raises(ValueError, match="lineage differs"):
        retrospective.run(tmp_path, "newrun", audit_run="selectedaudit",
                          panel_run="selectedpanel", candidate_run="selectedpit")


def test_portable_review_rejects_nonmatching_candidate_run(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "source_vintage_bundle.json").write_text("{}")
    monkeypatch.setattr(retrospective_review, "verify_membership", lambda *_: None)
    monkeypatch.setattr(retrospective_review, "verify_labels", lambda *_: None)
    monkeypatch.setattr(retrospective_review, "_read_json", lambda path: {
        "membership_run": "selectedmembership"} if "production_labels" in str(path)
        else {})
    monkeypatch.setattr(RunManifest, "read", lambda _: type(
        "Manifest", (), {"config": {"source_candidate_run": "selectedpit"}})())
    with pytest.raises(ValueError, match="selected PIT run"):
        retrospective_review.review(tmp_path, "selectedmembership", "selectedlabels",
                                    "newreview", candidate_run="wrongpit")


def institution(**updates):
    row = {"CERT": 11, "ESTYMD": "01/01/2020", "INSDATE": "12/31/9999",
           "ENDEFYMD": "12/31/9999", "ACTIVE": 1}
    row.update(updates)
    return row


def history(code, day, source="h1"):
    return {"ID": source, "CERT": 11, "CHANGECODE": code,
            "EFFDATE": day.isoformat() + "T00:00:00"}


def test_establishment_and_insurance_are_distinct():
    bank = derive_bank_evidence(11, institution(), [], [date(2020, 3, 31)])
    assert bank.insured_date is None
    assert bank.insurance_date_status == "unavailable_9999"
    assert retrospective_membership(bank, Q).status == "eligible"
    assert retrospective_membership(bank, date(2019, 12, 31)).status == "ineligible"


def test_exact_entry_and_exit_boundaries_and_assistance():
    bank = derive_bank_evidence(11, institution(ESTYMD="06/30/2023",
        ENDEFYMD="09/30/2023", ACTIVE=0),
        [history(830, date(2023, 7, 1))], [Q])
    assert retrospective_membership(bank, Q).status == "eligible"
    assert retrospective_membership(bank, date(2023, 9, 29)).status == "eligible"
    assert retrospective_membership(bank, date(2023, 9, 30)).status == "ineligible"
    same_day = derive_bank_evidence(11, institution(ESTYMD="06/30/2023",
        ENDEFYMD="06/30/2023", ACTIVE=0), [], [Q])
    assert retrospective_membership(same_day, Q).status == "ineligible"


def test_missing_institution_needs_entry_and_whole_bank_exit():
    rows = [history(110, date(2020, 1, 1), "entry"),
            history(223, date(2024, 1, 1), "merger")]
    bank = derive_bank_evidence(11, None, rows, [date(2020, 3, 31), Q])
    assert bank.disposition == "supported" and bank.source == "history_interval"
    assert retrospective_membership(bank, Q).status == "eligible"
    assert retrospective_membership(bank, date(2024, 1, 1)).status == "ineligible"
    for terminal in (225, 213, 830):  # partial transfer, assisted failure, assistance
        bad = derive_bank_evidence(11, None,
            [rows[0], history(terminal, date(2024, 1, 1), "not_ordinary_exit")],
            [date(2020, 3, 31), Q])
        assert retrospective_membership(bad, Q).status == "uncertain"


def test_identifier_change_and_inactive_missing_end_remain_uncertain():
    changed = derive_bank_evidence(11, institution(),
        [history(150, date(2023, 7, 1))], [date(2020, 3, 31)])
    assert retrospective_membership(changed, Q).status == "eligible"
    assert retrospective_membership(changed, date(2023, 9, 30)).status == "uncertain"
    unknown_exit = derive_bank_evidence(11, institution(ACTIVE=0), [], [Q])
    assert retrospective_membership(unknown_exit, Q).status == "uncertain"


def test_future_reports_and_predictor_values_do_not_change_earlier_membership():
    base = derive_bank_evidence(11, institution(), [], [date(2020, 3, 31)])
    later = derive_bank_evidence(11, institution(),
        [history(223, date(2024, 1, 1))],
        [date(2020, 3, 31), date(2024, 3, 31)])
    assert retrospective_membership(base, Q).status == retrospective_membership(later, Q).status
    # Predictor values are absent from BankEvidence by construction.
    assert not hasattr(base, "asset")


def test_reentry_and_temporary_reporting_gap():
    bank = BankEvidence(11, "supported", (
        ActiveInterval(date(2020, 1, 1), date(2022, 1, 1), "synthetic"),
        ActiveInterval(date(2023, 1, 1), None, "synthetic")),
        "synthetic", None, "missing", (date(2020, 3, 31), date(2023, 3, 31)), ())
    assert retrospective_membership(bank, date(2022, 6, 30)).status == "ineligible"
    assert retrospective_membership(bank, Q).status == "eligible"
    gap = derive_bank_evidence(11, institution(), [],
        [date(2020, 3, 31), date(2023, 3, 31)])
    assert retrospective_membership(gap, date(2021, 12, 31)).status == "eligible"


def test_retrospective_label_timing_and_nonfailure_exit():
    member = MembershipEvidence(11, Q, "eligible", "retrospective_operational_v1",
                                None, "institution-11", True)
    failure = OutcomeEvent("failure", 11, date(2024, 6, 30),
                           "verified_failure", "chronological_candidate")
    unrelated = OutcomeEvent("unassigned", None, date(2024, 1, 1),
                             "verified_failure", "unmatched")
    result = classify_label(member, [failure, unrelated], outcome_complete_through=date(2026, 6, 30))
    assert (result.disposition, result.y, result.qualifying_source_ids) == (
        "positive", 1, ("failure",))
    merger = OutcomeEvent("merger", 11, date(2024, 1, 1),
                          "documented_merger", "history_cert")
    assert classify_label(member, [merger], outcome_complete_through=date(2024, 6, 30)).y == 0
    assert classify_label(member, [merger, failure],
                          outcome_complete_through=date(2024, 6, 30)).disposition == "unresolved"
    assert classify_label(member, [], outcome_complete_through=date(2024, 6, 29)).disposition == "unresolved"
    with pytest.raises(ValueError, match="timing"):
        MembershipEvidence(11, Q, "eligible", "retrospective_operational_v1",
                           Q, "incorrect-vintage-claim", True)


def test_future_revision_changes_neither_cohort_nor_earlier_predictor():
    bank = derive_bank_evidence(11, institution(), [], [date(2022, 12, 31)])
    before = retrospective_membership(bank, Q)
    values = {field: None for field in PREDICTOR_FIELDS}
    values["ASSET"] = Decimal("10.0000")
    original = FinancialReport(11, date(2022, 12, 31), values)
    revised_values = {**values, "ASSET": Decimal("999.0000")}
    future_revision = FinancialReport(11, date(2022, 12, 31), revised_values,
                                      revision_published_at=date(2023, 7, 1),
                                      revision_time_verified=True)
    roster = population_from_available_reports(Q, [original])
    chosen = select_asof(roster, [original])[0]
    later = derive_bank_evidence(11, institution(), [],
                                 [date(2022, 12, 31), date(2023, 9, 30)])
    after = retrospective_membership(later, Q)
    changed_roster = population_from_available_reports(Q, [original, future_revision])
    changed = select_asof(changed_roster, [original, future_revision])[0]
    assert (before.status, before.reason) == (after.status, after.reason)
    assert [(item.cert, item.q) for item in roster] == [
        (item.cert, item.q) for item in changed_roster]
    assert chosen.source_repdte == changed.source_repdte
    assert chosen.source_availability_date == changed.source_availability_date
    assert chosen.predictors == changed.predictors


def test_effective_date_is_not_first_publication_date():
    bank = derive_bank_evidence(11, institution(ENDEFYMD="09/30/2023", ACTIVE=0),
                                [], [date(2022, 12, 31)])
    assert retrospective_membership(bank, date(2023, 9, 30)).status == "ineligible"
    # This is a reconstructed cohort statement. It cannot certify that an
    # operational predictor had a 2023-09-30 public registry vintage.
    assert not hasattr(bank, "first_publication_date")


def test_retrospective_correction_to_past_effective_date_can_revise_cohort():
    original = derive_bank_evidence(11, institution(), [], [date(2020, 3, 31)])
    corrected = derive_bank_evidence(11, institution(ESTYMD="01/01/2024"), [],
                                     [date(2020, 3, 31)])
    assert retrospective_membership(original, Q).status == "eligible"
    # An earlier observed report conflicts with the corrected establishment;
    # the historical cohort becomes uncertain. The report's values are not
    # read into this activity decision.
    assert retrospective_membership(corrected, Q).status == "uncertain"
