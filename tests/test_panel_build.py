"""Synthetic panel and exit cases with explicit source-event counts."""

from datetime import date
from decimal import Decimal
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.acquisition.storage import arrow_schema
from bft65b.config import EXTENDED_FINANCIAL_FIELDS
from bft65b.panel.build import (
    DocumentedExit, _panel_reconciliation, _write_verified, classify_exits,
    link_failure_events, reporting_gaps,
)


Q1 = date(2023, 3, 31)
Q2 = date(2023, 6, 30)
Q3 = date(2023, 9, 30)
Q4 = date(2023, 12, 31)


def failure(source_id, cert, when):
    return {"ID": source_id, "CERT": cert, "FAILDATE": when,
            "COST": None, "SOURCE_MISSING_FIELDS": [], "DQ_FLAGS": [],
            "TRANSACTION_DISPOSITION": "verified_failure"}


def by_cert(rows):
    return {row["CERT"]: row for row in rows}


def test_confirmed_failure_keeps_one_event_with_multiple_prior_reports():
    reports = {1: [Q1, Q2]}
    events = link_failure_events([failure("event-1", 1, date(2023, 8, 15))], reports)
    assert len(events) == 1
    assert events[0]["BANK_LINK_DISPOSITION"] == "chronological_candidate"
    assert events[0]["PRIOR_REPORT_COUNT"] == 2
    assert events[0]["POST_EVENT_REPORT_COUNT"] == 0
    exits = classify_exits(reports, events, {}, Q4)
    assert len(exits) == 1
    assert exits[0]["EXIT_CATEGORY"] == "confirmed_failure"
    refs = json.loads(exits[0]["EVIDENCE_JSON"])
    assert {ref["identifier"] for ref in refs} == {"1", "event-1"}
    assert any(ref["date"] == "2023-08-15" for ref in refs)


@pytest.mark.parametrize("kind,category", [
    ("merger", "documented_merger"), ("other_exit", "documented_other_exit")])
def test_dated_documented_exit_requires_source_id_and_effective_date(kind, category):
    evidence = DocumentedExit(2, kind, "synthetic structure history", "history-2",
                              date(2023, 7, 1))
    result = classify_exits({2: [Q1]}, [], {}, Q4, [evidence])[0]
    assert result["EXIT_CATEGORY"] == category
    refs = json.loads(result["EVIDENCE_JSON"])
    assert refs[-1] == {"source": "synthetic structure history", "identifier": "history-2",
                        "date": "2023-07-01", "rule": kind}


def test_conflicting_failure_and_merger_remains_ambiguous():
    reports = {3: [Q1]}
    events = link_failure_events([failure("f3", 3, date(2023, 8, 1))], reports)
    merger = DocumentedExit(3, "merger", "synthetic structure history", "m3",
                            date(2023, 8, 2))
    result = classify_exits(reports, events, {}, Q4, [merger])[0]
    assert result["EXIT_CATEGORY"] == "ambiguous_exit"
    assert result["CLASSIFICATION_RULE"] == "Conflicting or nonqualifying event evidence"
    assert {r["identifier"] for r in json.loads(result["EVIDENCE_JSON"])} == {"3", "f3", "m3"}


def test_missing_unmatched_and_future_only_failure_links_keep_source_ids():
    events = [failure("null-cert", None, date(2023, 5, 1)),
              failure("unmatched", 4, date(2023, 5, 1)),
              failure("future", 5, date(2023, 5, 1))]
    linked = link_failure_events(events, {5: [Q2]})
    assert len(linked) == len({row["ID"] for row in linked}) == 3
    assert {row["ID"]: row["BANK_LINK_DISPOSITION"] for row in linked} == {
        "null-cert": "missing_cert", "unmatched": "unmatched_cert",
        "future": "future_reports_only"}
    assert by_cert(classify_exits({5: [Q2]}, linked, {}, Q4))[5]["EXIT_CATEGORY"] == "ambiguous_exit"


def test_post_event_reports_conflict_and_multiple_event_ids_are_not_merged():
    reports = {6: [Q1, Q3], 7: [Q1]}
    events = link_failure_events([failure("f6", 6, date(2023, 5, 1)),
                                  failure("f7a", 7, date(2023, 5, 1)),
                                  failure("f7b", 7, date(2023, 7, 1))], reports)
    assert len(events) == 3
    assert events[0]["BANK_LINK_DISPOSITION"] == "post_event_report_conflict"
    exits = by_cert(classify_exits(reports, events, {}, Q4))
    assert exits[6]["EXIT_CATEGORY"] == "ambiguous_exit"
    assert exits[7]["EXIT_CATEGORY"] == "ambiguous_exit"
    assert len(json.loads(exits[7]["EVIDENCE_JSON"])) == 3


def test_temporary_gap_then_resumption_never_creates_exit_or_zero_quarter():
    reports = {8: [Q1, Q3]}
    gaps = reporting_gaps(reports)
    assert gaps == [{"CERT": 8, "PREVIOUS_REPDTE": Q1, "NEXT_REPDTE": Q3,
                     "MISSING_INTERVENING_QUARTERS": 1}]
    exits = classify_exits(reports, [], {}, Q4)
    assert len(exits) == 1 and exits[0]["LAST_REPDTE"] == Q3
    assert exits[0]["EXIT_CATEGORY"] == "ambiguous_exit"
    assert Q2 not in reports[8]


def test_final_quarter_is_boundary_observation_not_exit():
    result = classify_exits({9: [Q4]}, [], {9: {"ACTIVE": 0, "CHANGEC1": "223",
        "SOURCE_MISSING_FIELDS": []}}, Q4)[0]
    assert result["EXIT_CATEGORY"] == "boundary_observed"
    assert result["INSTITUTION_ACTIVE_SNAPSHOT"] == 0
    assert result["INSTITUTION_CHANGEC1_SNAPSHOT"] == "223"
    assert result["SNAPSHOT_AUDIT_ONLY"] is True


def test_snapshot_change_code_does_not_prove_merger_or_exit():
    result = classify_exits({10: [Q1]}, [], {10: {"ACTIVE": 0, "CHANGEC1": "223",
        "SOURCE_MISSING_FIELDS": []}}, Q4)[0]
    assert result["EXIT_CATEGORY"] == "ambiguous_exit"
    assert result["INSTITUTION_CHANGEC1_SNAPSHOT"] == "223"


def test_panel_key_reconciliation_preserves_values_and_flags(tmp_path):
    schema = arrow_schema("financials", "20230331",
                          financial_fields=EXTENDED_FINANCIAL_FIELDS).append(
        pa.field("DQ_FLAGS", pa.list_(pa.field("element", pa.string())), nullable=False))
    source = pa.Table.from_pylist([{"CERT": 1, "REPDTE": Q1,
        "ASSET": Decimal("10.0000"), "SOURCE_MISSING_FIELDS": [],
        "DQ_FLAGS": ["source_explicit_null:P3ASSET"]}], schema=schema)
    target = tmp_path / "financials" / "REPDTE=20230331" / "part-000.parquet"
    digest = _write_verified(target, source)
    with pq.ParquetFile(target) as parquet:
        result = parquet.read()
    assert len(digest) == 64
    assert _panel_reconciliation("20230331", source, result) == {
        "quarter": "20230331", "input_rows": 1, "input_keys": 1,
        "output_rows": 1, "output_keys": 1, "input_duplicate_keys": 0,
        "output_duplicate_keys": 0, "missing_keys": 0, "added_keys": 0}
    assert result.to_pylist()[0]["DQ_FLAGS"] == ["source_explicit_null:P3ASSET"]


def test_panel_duplicate_or_added_key_fails_reconciliation():
    source = pa.Table.from_pylist([{"CERT": 1, "REPDTE": Q1}])
    duplicate = pa.Table.from_pylist([{"CERT": 1, "REPDTE": Q1},
                                       {"CERT": 1, "REPDTE": Q1}])
    with pytest.raises(ValueError, match="Panel differs"):
        _panel_reconciliation("20230331", source, duplicate)
    added = pa.Table.from_pylist([{"CERT": 1, "REPDTE": Q1},
                                   {"CERT": 2, "REPDTE": Q1}])
    with pytest.raises(ValueError, match="Panel differs"):
        _panel_reconciliation("20230331", source, added)


def test_duplicate_failure_source_id_is_rejected():
    with pytest.raises(ValueError, match="source ID"):
        link_failure_events([failure("f", 1, date(2023, 8, 1)),
                             failure("f", 1, date(2023, 8, 2))], {1: [Q1]})
