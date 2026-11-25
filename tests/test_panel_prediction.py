"""Synthetic P2.4 full-application contracts, without FDIC data or network calls."""

from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.config import load_config
from bft65b.panel.asof import FinancialReport, PREDICTOR_FIELDS
from bft65b.panel.prediction import (
    build_prediction_quarter, candidate_disposition, prediction_quarters,
    read_panel_file,
)


Q = date(2023, 6, 30)
OLD = date(2022, 12, 31)


def _schema():
    return pa.schema([
        pa.field("CERT", pa.int64()), pa.field("Q", pa.date32()),
        pa.field("POPULATION_KNOWN_AT", pa.date32()),
        pa.field("SELECTED_REPDTE", pa.date32()),
        pa.field("AVAILABILITY_DATE", pa.date32()),
        pa.field("REPORT_AGE_DAYS", pa.int32()),
        pa.field("ELIGIBILITY_DISPOSITION", pa.string()),
        pa.field("AVAILABILITY_BASIS", pa.string()),
        pa.field("SOURCE_ID_DIAGNOSTIC", pa.string()),
        pa.field("SOURCE_MISSING_FIELDS_DIAGNOSTIC",
                 pa.list_(pa.field("element", pa.string()))),
        *(pa.field(field, pa.float64()) for field in PREDICTOR_FIELDS),
    ])


def _report(cert=1, repdte=OLD, asset=10.0, *, published=None):
    values = {field: None for field in PREDICTOR_FIELDS}
    values["ASSET"] = asset
    values.update({"DQ_FLAGS": ["future_flag"], "EXIT_CATEGORY": "confirmed_failure",
                   "INSTITUTION_ACTIVE_SNAPSHOT": 0})
    return FinancialReport(cert, repdte, values, "synthetic", (), published,
                           published is not None)


def _event(day, status="verified_failure", link="chronological_candidate", cert=1):
    return {"CERT": cert, "FAILDATE": day,
            "TRANSACTION_DISPOSITION": status, "BANK_LINK_DISPOSITION": link}


def test_prediction_scope_exactly_166_quarters():
    config = load_config(Path("config/financials_full_v2.json"))
    quarters = prediction_quarters(config)
    assert len(quarters) == 166
    assert quarters[:2] == ("19840331", "19840630")
    assert quarters[-1] == "20250630"


def test_physical_partition_reader_preserves_repdte_date(tmp_path):
    path = tmp_path / "REPDTE=20221231" / "part-000.parquet"
    path.parent.mkdir()
    source = pa.table({"CERT": pa.array([1], type=pa.int64()),
                       "REPDTE": pa.array([OLD], type=pa.date32())})
    pq.write_table(source, path)
    assert read_panel_file(path).equals(source)
    assert read_panel_file(path, ["CERT"])["CERT"].to_pylist() == [1]


def test_prediction_output_list_child_name_survives_parquet_round_trip(tmp_path):
    source, _ = build_prediction_quarter(Q, [_report()], [], _schema())
    path = tmp_path / "Q=20230630" / "part-000.parquet"
    path.parent.mkdir()
    pq.write_table(source, path, compression="zstd", compression_level=3)
    with pq.ParquetFile(path) as parquet:
        assert parquet.schema_arrow.equals(source.schema, check_metadata=True)
        assert parquet.read().equals(source, check_metadata=False)


def test_candidate_lag_boundary_latest_eligible_and_no_future_predictor():
    reports = [_report(asset=10.0), _report(repdte=date(2023, 3, 31), asset=99.0)]
    table, summary = build_prediction_quarter(Q, reports, [], _schema())
    row = table.to_pylist()[0]
    assert summary["candidates"] == summary["eligible"] == 1
    assert summary["no_eligible_history"] == 0
    assert row["SELECTED_REPDTE"] == OLD
    assert row["AVAILABILITY_DATE"] == date(2023, 4, 30)
    assert row["AVAILABILITY_DATE"] <= Q
    assert row["ASSET"] == 10.0
    assert "DQ_FLAGS" not in table.column_names
    assert "EXIT_CATEGORY" not in table.column_names


@pytest.mark.parametrize("event_day,expected", [
    (Q - timedelta(days=1), "confirmed_failure_before_or_on_q"),
    (Q, "confirmed_failure_before_or_on_q"),
    (Q + timedelta(days=1), "eligible"),
])
def test_failure_before_or_on_prediction_is_excluded_without_changing_target(event_day, expected):
    table, summary = build_prediction_quarter(Q, [_report()], [_event(event_day)], _schema())
    assert table.to_pylist()[0]["ELIGIBILITY_DISPOSITION"] == expected
    assert summary["candidates"] == 1
    assert summary["eligible"] == (expected == "eligible")
    assert summary["excluded_confirmed_failure_before_or_on_q"] == (expected != "eligible")


def test_assistance_and_unresolved_identity_are_distinct():
    bank_events = defaultdict(list)
    bank_events[1] = [_event(Q, status="assistance_transaction")]
    known = {1: [_report()]}
    assert candidate_disposition(1, Q, bank_events, known) == "eligible"
    bank_events[1] = [_event(date(2023, 5, 1), status="verified_failure",
                             link="post_event_report_conflict")]
    # A retrospective link flag alone cannot expose a later report at q.
    assert candidate_disposition(1, Q, bank_events, known) == "confirmed_failure_before_or_on_q"
    future_report = _report(repdte=Q)
    assert candidate_disposition(1, Q, bank_events, {1: [known[1][0], future_report]}) == "confirmed_failure_before_or_on_q"
    known_post = _report(repdte=Q)
    later_q = date(2023, 12, 31)
    assert candidate_disposition(1, later_q, bank_events, {1: [known[1][0], known_post]}) == "unresolved_identity"
    bank_events[1] = [_event(Q, status="failure_charter_uncertain",
                             link="future_reports_only")]
    assert candidate_disposition(1, Q, bank_events, known) == "unresolved_identity"
    bank_events[1] = [_event(Q + timedelta(days=1), link="post_event_report_conflict")]
    assert candidate_disposition(1, Q, bank_events, known) == "eligible"


def test_no_available_report_yields_no_report_derived_candidate():
    table, summary = build_prediction_quarter(Q, [_report(repdte=Q)], [], _schema())
    assert table.num_rows == 0
    assert summary["candidates"] == summary["no_eligible_history"] == 0
    assert summary["report_age_days"] == {"min": None, "p50": None,
                                           "p90": None, "max": None}


def test_later_verified_revision_cannot_change_earlier_prediction():
    first = _report(published=date(2023, 5, 1), asset=10.0)
    later = _report(published=date(2023, 7, 1), asset=99.0)
    table, summary = build_prediction_quarter(Q, [first, later], [], _schema())
    assert summary["candidates"] == 1
    assert table.to_pylist()[0]["ASSET"] == 10.0
    assert table.to_pylist()[0]["AVAILABILITY_DATE"] == date(2023, 5, 1)
