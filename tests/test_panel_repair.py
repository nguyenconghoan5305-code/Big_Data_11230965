"""Synthetic P2.3 failure/assistance and boundary-precedence contracts."""

from datetime import date
import json

import pytest

from bft65b.panel.build import DocumentedExit, classify_exits, link_failure_events
from bft65b.panel.repair import _source_rows, reconcile_events, transaction_disposition


Q1 = date(2023, 3, 31)
Q4 = date(2023, 12, 31)


def event(source_id="1", cert=10, when=Q4):
    return {"ID": source_id, "CERT": cert, "FAILDATE": when, "COST": 1.5}


def supplemental(source_id="1", cert=10, when="12/31/2023", status="FAILURE", method="PA"):
    return {"ID": source_id, "CERT": cert, "FAILDATE": when, "COST": 1.5,
            "RESTYPE": status, "RESTYPE1": method}


def linked(status="FAILURE", method="PA", *, cert=10, when=Q4):
    accepted = link_failure_events([event(cert=cert, when=when)], {cert: [Q1]})
    rows, audit = reconcile_events(accepted, [supplemental(cert=cert,
        when=f"{when.month}/{when.day}/{when.year}", status=status, method=method)])
    assert audit["matched"] == 1 and audit["supplement_extra_ids"] == 0
    return rows


@pytest.mark.parametrize("status,method,disposition,category", [
    ("FAILURE", "PA", "verified_failure", "confirmed_failure"),
    ("ASSISTANCE", "A/A", "assistance_transaction", "ambiguous_exit"),
    ("ASSISTANCE", "OBAM", "open_bank_assistance_code_unverified", "ambiguous_exit"),
    ("FAILURE", "MGR", "verified_failure", "confirmed_failure"),
    ("FAILURE", "REP", "failure_charter_uncertain", "ambiguous_exit"),
    ("FAILURE", "UNLISTED", "unmapped_transaction", "ambiguous_exit"),
    (None, "PA", "unknown_transaction", "ambiguous_exit"),
])
def test_transaction_status_is_separate_from_bank_link(status, method, disposition, category):
    rows = linked(status, method)
    assert rows[0]["TRANSACTION_DISPOSITION"] == disposition
    assert rows[0]["BANK_LINK_DISPOSITION"] == "chronological_candidate"
    assert classify_exits({10: [Q1]}, rows, {}, Q4)[0]["EXIT_CATEGORY"] == category


@pytest.mark.parametrize("source,match", [
    ([], "missing_source_id"),
    ([supplemental(), supplemental()], "duplicate_source_id"),
    ([supplemental(cert=11)], "corroboration_conflict"),
    ([supplemental(when="12/30/2023")], "corroboration_conflict"),
    ([dict(supplemental(), COST=2.5)], "corroboration_conflict"),
    ([dict(supplemental(), COST=True)], "corroboration_conflict"),
    ([dict(supplemental(), CERT=True)], "corroboration_conflict"),
])
def test_unmatched_or_conflicting_supplement_preserves_original_event(source, match):
    original = link_failure_events([event()], {10: [Q1]})
    rows, audit = reconcile_events(original, source)
    assert len(rows) == 1 and rows[0]["ID"] == "1"
    assert rows[0]["CERT"] == 10 and rows[0]["FAILDATE"] == Q4
    assert rows[0]["TRANSACTION_DISPOSITION"] == "unresolved_source_match"
    assert rows[0]["SUPPLEMENT_MATCH"] == match and audit[match] == 1
    assert classify_exits({10: [Q1]}, rows, {}, Q4)[0]["EXIT_CATEGORY"] == "ambiguous_exit"


def test_extra_supplemental_id_is_audited_without_creating_event():
    original = link_failure_events([event()], {10: [Q1]})
    rows, audit = reconcile_events(original, [supplemental(), supplemental(source_id="extra")])
    assert len(rows) == 1 and audit["supplement_extra_ids"] == 1


def test_failure_without_valid_bank_link_is_not_confirmed():
    accepted = link_failure_events([event(cert=99)], {10: [Q1]})
    rows, _ = reconcile_events(accepted, [supplemental(cert=99)])
    assert rows[0]["TRANSACTION_DISPOSITION"] == "verified_failure"
    assert rows[0]["BANK_LINK_DISPOSITION"] == "unmatched_cert"
    assert classify_exits({10: [Q1]}, rows, {}, Q4)[0]["EXIT_CATEGORY"] == "ambiguous_exit"


def test_boundary_day_precedence_and_conflicts():
    merger = DocumentedExit(10, "merger", "synthetic history", "m10", Q4)
    no_event = classify_exits({10: [Q4]}, [], {}, Q4)[0]
    assert no_event["EXIT_CATEGORY"] == "boundary_observed"
    accepted = link_failure_events([event()], {10: [Q4]})
    failure, _ = reconcile_events(accepted, [supplemental()])
    assistance, _ = reconcile_events(accepted, [supplemental(status="ASSISTANCE", method="A/A")])
    assert failure[0]["TRANSACTION_DISPOSITION"] == "verified_failure"
    assert classify_exits({10: [Q4]}, failure, {}, Q4)[0]["EXIT_CATEGORY"] == "confirmed_failure"
    assert classify_exits({10: [Q4]}, [], {}, Q4, [merger])[0]["EXIT_CATEGORY"] == "documented_merger"
    assert assistance[0]["TRANSACTION_DISPOSITION"] == "assistance_transaction"
    assert classify_exits({10: [Q4]}, assistance, {}, Q4)[0]["EXIT_CATEGORY"] == "boundary_observed"
    assert classify_exits({10: [Q4]}, failure, {}, Q4, [merger])[0]["EXIT_CATEGORY"] == "ambiguous_exit"
    unknown, _ = reconcile_events(accepted, [supplemental(status=None)])
    assert classify_exits({10: [Q4]}, unknown, {}, Q4)[0]["EXIT_CATEGORY"] == "ambiguous_exit"
    assert classify_exits({10: [Q4]}, assistance, {}, Q4, [merger])[0]["EXIT_CATEGORY"] == "ambiguous_exit"


def test_failed_bank_acquisition_is_not_ordinary_merger():
    rows = linked("FAILURE", "P&A")
    assert rows[0]["TRANSACTION_DISPOSITION"] == "verified_failure"
    assert classify_exits({10: [Q1]}, rows, {}, Q4)[0]["EXIT_CATEGORY"] == "confirmed_failure"
    merger = DocumentedExit(10, "merger", "synthetic history", "m10", Q4)
    assert classify_exits({10: [Q1]}, [], {}, Q4, [merger])[0]["EXIT_CATEGORY"] == "documented_merger"


def test_transaction_policy_rejects_conflicting_or_unknown_codes():
    assert transaction_disposition("ASSISTANCE", "PA")[0] == "transaction_conflict"
    assert transaction_disposition("FAILURE", "A/A")[0] == "transaction_conflict"
    assert transaction_disposition("FAILURE", "XYZ")[0] == "unmapped_transaction"
    assert transaction_disposition("FAILURE", "DINB")[0] == "verified_failure"
    assert transaction_disposition("FAILURE", "ABT")[0] == "verified_failure"


def test_supplemental_response_total_and_index_are_fail_closed(tmp_path):
    payload = {"meta": {"total": 1, "parameters": {
        "fields": "CERT,FAILDATE,COST,RESTYPE,RESTYPE1", "limit": "10000",
        "offset": "0", "filters": ""},
        "index": {"name": "failures_1787667198788", "createTimestamp": "2026-08-25T14:13:18Z"}},
        "data": [{"data": supplemental()}]}
    path = tmp_path / "official_response.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _source_rows(path)[0][0]["ID"] == "1"
    payload["meta"]["total"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="total"):
        _source_rows(path)
    payload["meta"]["total"] = 1
    payload["meta"]["index"]["name"] = "changed_source"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="index"):
        _source_rows(path)
