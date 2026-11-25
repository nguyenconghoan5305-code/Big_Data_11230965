"""Production label table integration with retrospective membership decisions."""
from datetime import date

import pytest

from bft65b.features.label import OutcomeEvent
from bft65b.features.production import LABEL_SCHEMA, label_quarter
from bft65b.features import production


Q = date(2023, 6, 30)


def test_portable_labels_reject_default_repair_and_mismatched_audit(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "source_vintage_bundle.json").write_text("{}")
    monkeypatch.setattr(production, "_root", lambda _: tmp_path)
    monkeypatch.setattr(production, "verify_membership", lambda *_: {"quarters": 166})
    monkeypatch.setattr(production.RunManifest, "read", lambda path: type("Manifest", (), {
        "config": {"source_audit_run": "audit"} if "task_4_2_membership" in str(path)
        else {"source_repair_run": "expectedrepair"}})())
    with pytest.raises(ValueError, match="explicit selected repair"):
        production.run(tmp_path, "newlabels", "selectedmembership")
    with pytest.raises(ValueError, match="lineage differ"):
        production.run(tmp_path, "newlabels", "selectedmembership", "wrongrepair")


def member(cert, status="eligible", history=True):
    return {"CERT": cert, "Q": Q, "MEMBERSHIP": status,
            "MEMBERSHIP_REASON": "test_interval", "EVIDENCE_SOURCE": "fixture",
            "FEATURE_HISTORY_ELIGIBLE": history}


def failure(source_id, cert, day):
    return OutcomeEvent(source_id, cert, day, "verified_failure",
                        "chronological_candidate", "matched")


def test_positive_survives_unassigned_failure_and_right_boundary():
    rows = [member(1), member(2, history=False)]
    linked = failure("a", 1, date(2024, 6, 30))
    unassigned = failure("orphan", None, date(2024, 4, 1))
    table, summary = label_quarter(rows, {1: [linked]}, [unassigned], Q,
                                   outcome_complete_through=date(2024, 6, 30))
    result = table.to_pylist()
    assert table.schema == LABEL_SCHEMA
    assert [(row["LABEL_DISPOSITION"], row["Y"]) for row in result] == [
        ("positive", 1), ("unresolved", None)]
    assert result[0]["QUALIFYING_SOURCE_IDS"] == ["a"]
    assert summary["positive"] == 1
    assert summary["unresolved"] == 1
    assert summary["feature_history_eligible"] == 1
    assert summary["distinct_qualifying_source_events"] == 1


def test_assistance_nonfailure_exit_and_incomplete_followup():
    assistance = OutcomeEvent("assist", 1, date(2023, 8, 1),
                              "assistance_transaction", "chronological_candidate")
    merger = OutcomeEvent("merge", 2, date(2023, 9, 1),
                          "documented_merger", "chronological_candidate")
    table, summary = label_quarter([member(1), member(2), member(3)],
                                   {1: [assistance], 2: [merger]}, [], Q,
                                   outcome_complete_through=date(2024, 6, 30))
    assert [row["LABEL_DISPOSITION"] for row in table.to_pylist()] == [
        "negative", "negative", "negative"]
    assert summary["reason:complete_followup_with_nonfailure_exit"] == 1
    table, summary = label_quarter([member(1)], {1: [assistance]}, [], Q,
                                   outcome_complete_through=date(2024, 6, 29))
    assert summary["unresolved"] == 1


def test_prior_failure_exclusion_and_uncertain_identity():
    earlier = failure("earlier", 1, Q)
    table, summary = label_quarter([member(1), member(2, "uncertain")],
                                   {1: [earlier]}, [], Q,
                                   outcome_complete_through=date(2024, 6, 30))
    assert [row["LABEL_DISPOSITION"] for row in table.to_pylist()] == [
        "excluded", "unresolved"]
    assert summary["reason:verified_failure_before_or_on_q"] == 1


def test_duplicate_and_wrong_quarter_membership_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        label_quarter([member(1), member(1)], {}, [], Q,
                      outcome_complete_through=date(2024, 6, 30))
    bad = member(1)
    bad["Q"] = date(2023, 9, 30)
    with pytest.raises(ValueError, match="wrong-quarter"):
        label_quarter([bad], {}, [], Q,
                      outcome_complete_through=date(2024, 6, 30))
