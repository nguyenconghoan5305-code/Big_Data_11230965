"""Known-count synthetic tests for lossless P2.1 issue treatment."""

from datetime import date
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.acquisition.storage import arrow_schema
from bft65b.config import EXTENDED_FINANCIAL_FIELDS, ProjectConfig
from bft65b.manifest import RunManifest
from bft65b.quality.treatment import (
    POLICY, _flag_counts, _logical_sha, _write_verified, derive_table, treatment_log,
)
from bft65b.quality.audit import _sha


Q1 = date(2023, 3, 31)
Q2 = date(2023, 6, 30)


def financial_table(rows):
    return pa.Table.from_pylist(rows, schema=arrow_schema(
        "financials", "20230331", financial_fields=EXTENDED_FINANCIAL_FIELDS))


def failure_table(rows):
    return pa.Table.from_pylist(rows, schema=arrow_schema("failures"))


def institution_table(rows):
    return pa.Table.from_pylist(rows, schema=arrow_schema("institutions"))


def test_overlap_flags_count_one_source_key_and_preserve_null_vs_absent():
    raw = financial_table([
        {"CERT": 1, "REPDTE": Q1, "ASSET": Decimal("10.0000"),
         "P3ASSET": None, "BRO": None, "SOURCE_MISSING_FIELDS": ["BRO"]},
        {"CERT": 2, "REPDTE": Q1, "ASSET": Decimal("20.0000"),
         "P3ASSET": 1.0, "BRO": Decimal("0.0000"), "SOURCE_MISSING_FIELDS": []},
    ])
    fields = ("CERT", "REPDTE", "ASSET", "P3ASSET", "BRO")
    derived, scan = derive_table("financials", raw, fields, "20230331", institution_certs={2})
    assert raw.num_rows == derived.num_rows == 2
    assert derived.select(raw.column_names).equals(raw, check_metadata=False)
    assert scan.summary["affected_records_unique"] == 1
    assert scan.summary["missing_or_null_records_unique"] == 1
    assert scan.summary["field_counts"]["BRO"] == {
        "absent": 1, "explicit_null": 0, "non_null": 1, "denominator": 2}
    assert scan.summary["field_counts"]["P3ASSET"]["explicit_null"] == 1
    assert set(scan.row_flags[0]) == {
        "source_field_absent:BRO", "source_explicit_null:P3ASSET",
        "financial_cert_absent_from_institutions:CERT"}
    assert scan.row_flags[1] == ()
    counts = _flag_counts("financials", raw.to_pylist(), scan.row_flags, "20230331")
    assert counts[("financials", "20230331", "BRO", "source_field_absent")] == 1
    assert counts[("financials", "20230331", "P3ASSET", "source_explicit_null")] == 1


def test_failures_remain_one_event_per_source_id_with_multiple_prior_reports():
    raw = failure_table([
        {"ID": "a", "CERT": 1, "FAILDATE": date(2023, 5, 1),
         "COST": None, "SOURCE_MISSING_FIELDS": []},
        {"ID": "b", "CERT": None, "FAILDATE": date(2023, 5, 2),
         "COST": 0.0, "SOURCE_MISSING_FIELDS": []},
        {"ID": "c", "CERT": 3, "FAILDATE": date(2023, 5, 3),
         "COST": 2.0, "SOURCE_MISSING_FIELDS": []},
        {"ID": "d", "CERT": 4, "FAILDATE": date(2023, 5, 4),
         "COST": 3.0, "SOURCE_MISSING_FIELDS": []},
    ])
    reports = {1: [date(2022, 12, 31), Q1], 4: [Q2]}
    derived, scan = derive_table("failures", raw, ("CERT", "FAILDATE", "COST"),
                                 "all", financial_reports=reports)
    assert derived.num_rows == 4
    assert [r["ID"] for r in derived.to_pylist()] == ["a", "b", "c", "d"]
    assert scan.summary["affected_records_unique"] == 4
    assert scan.summary["field_counts"]["COST"]["explicit_null"] == 1
    assert derived.to_pylist()[1]["COST"] == 0.0
    assert "failure_multiple_prior_reports:CERT,FAILDATE" in scan.row_flags[0]
    assert "source_explicit_null:COST" in scan.row_flags[0]
    assert "failure_cert_null:CERT" in scan.row_flags[1]
    assert "failure_cert_absent_from_financials:CERT" in scan.row_flags[2]
    assert "failure_only_future_reports:CERT,FAILDATE" in scan.row_flags[3]
    assert _flag_counts("failures", raw.to_pylist(), scan.row_flags, None)[
        ("failures", "2023", "CERT,FAILDATE", "failure_multiple_prior_reports")] == 1


def test_institutions_absent_change_code_preserved_with_provenance():
    raw = institution_table([
        {"CERT": 1, "ACTIVE": 1, "CHANGEC1": None,
         "SOURCE_MISSING_FIELDS": ["CHANGEC1"]},
        {"CERT": 2, "ACTIVE": 0, "CHANGEC1": None,
         "SOURCE_MISSING_FIELDS": []},
    ])
    derived, scan = derive_table("institutions", raw,
                                 ("CERT", "ACTIVE", "CHANGEC1"), "snapshot")
    assert derived.num_rows == 2
    assert scan.summary["field_counts"]["CHANGEC1"]["absent"] == 1
    assert scan.summary["field_counts"]["CHANGEC1"]["explicit_null"] == 1
    assert scan.row_flags == [
        ("source_field_absent:CHANGEC1",), ("source_explicit_null:CHANGEC1",)]
    assert [r["SOURCE_MISSING_FIELDS"] for r in derived.to_pylist()] == [["CHANGEC1"], []]


def test_unknown_definite_violation_fails_closed_without_silent_retention():
    raw = financial_table([{"CERT": 1, "REPDTE": Q1,
                            "ASSET": Decimal("-1.0000"), "SOURCE_MISSING_FIELDS": []}])
    with pytest.raises(ValueError, match="Unreviewed quality rules"):
        derive_table("financials", raw, ("CERT", "REPDTE", "ASSET"), "20230331")


def test_duplicate_source_key_fails_closed():
    row = {"CERT": 1, "REPDTE": Q1, "ASSET": Decimal("1.0000"),
           "SOURCE_MISSING_FIELDS": []}
    raw = financial_table([row, row])
    with pytest.raises(ValueError, match="Source keys are not unique"):
        derive_table("financials", raw, ("CERT", "REPDTE", "ASSET"), "20230331")


def test_treatment_log_has_exact_denominators_counts_and_actions():
    audit = {"accepted_attempt": "real20261111b", "period_findings": [
        {"dataset": "financials", "period": "20230331", "field": "P3ASSET",
         "rule": "source_explicit_null", "severity": "review", "count": 2, "denominator": 3},
        {"dataset": "failures", "period": "2023", "field": "CERT,FAILDATE",
         "rule": "failure_multiple_prior_reports", "severity": "review", "count": 1, "denominator": 4},
    ]}
    log = treatment_log(audit)
    assert log["period_issue_entries"] == 2
    assert log["entries"][0]["prevalence"] == 2 / 3
    assert (log["entries"][0]["affected_records_retained"],
            log["entries"][0]["raw_values_changed"],
            log["entries"][0]["records_excluded"],
            log["entries"][0]["before_count"],
            log["entries"][0]["after_count"]) == (2, 0, 0, 2, 2)
    assert log["entries"][1]["action"] == "retain_one_source_event_and_flag"
    assert set(POLICY) == {"source_explicit_null", "source_field_absent",
        "financial_cert_absent_from_institutions", "failure_cert_null",
        "failure_cert_absent_from_financials", "failure_only_future_reports",
        "failure_multiple_prior_reports"}


def test_treatment_log_rejects_unhandled_or_duplicate_finding():
    item = {"dataset": "financials", "period": "20230331", "field": "ASSET",
            "rule": "negative_total_assets", "severity": "violation", "count": 1,
            "denominator": 2}
    with pytest.raises(ValueError, match="No justified treatment"):
        treatment_log({"accepted_attempt": "x", "period_findings": [item]})
    item["rule"] = "source_explicit_null"
    with pytest.raises(ValueError, match="Duplicate period finding"):
        treatment_log({"accepted_attempt": "x", "period_findings": [item, item]})


def test_atomic_derived_roundtrip_and_repeat_logical_hash(tmp_path):
    raw = failure_table([{"ID": "f1", "CERT": None,
                          "FAILDATE": date(2023, 5, 1), "COST": None,
                          "SOURCE_MISSING_FIELDS": []}])
    derived, _ = derive_table("failures", raw, ("CERT", "FAILDATE", "COST"),
                              "all", financial_reports={})
    one = tmp_path / "one" / "failures" / "part-000.parquet"
    two = tmp_path / "two" / "failures" / "part-000.parquet"
    first_sha, first_logical = _write_verified(one, derived, raw)
    second_sha, second_logical = _write_verified(two, derived, raw)
    assert first_logical == second_logical == _logical_sha(derived)
    assert first_sha == second_sha
    assert pq.ParquetFile(one).read().equals(derived, check_metadata=True)
    with pytest.raises(FileExistsError):
        _write_verified(one, derived, raw)
    assert pq.ParquetFile(one).read().num_rows == 1


def _show_fixture(root: Path, run_id: str = "synthetic", *,
                  manifest_status: str = "complete", metadata_status: str = "complete",
                  summary_status: str = "complete", failure_reason: str = "") -> Path:
    evidence = root / "runs" / "task_2_1_treatment" / run_id
    evidence.mkdir(parents=True)
    summary = {"status": summary_status, "source_files": 1, "derived_files": 1,
        "period_issue_entries": 2, "datasets": {"failures": {
            "before_rows": 1, "after_rows": 1, "flagged_distinct_source_keys": 1,
            "raw_values_changed": 0, "records_excluded": 0}}}
    summary_path = evidence / "summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    (evidence / "metadata.json").write_text(json.dumps({
        "status": metadata_status, "summary_sha256": _sha(summary_path)}) + "\n", encoding="utf-8")
    manifest = RunManifest.new(f"p2_1_treatment_{run_id}", ProjectConfig())
    manifest.record_input("synthetic.parquet", "a" * 64, 1, 1)
    manifest.record_output(path="synthetic_derived.parquet", dataset="failures", rows=1,
        sha256="b" * 64, compression="ZSTD", schema_version="flag_only_v1",
        partition=None, source_total=1, source_query=None)
    manifest.record_validation("synthetic_gate", manifest_status != "failed",
                               failure_reason or "Synthetic validation passed")
    manifest.set_status(manifest_status)
    manifest.write(evidence / "run_manifest.json")
    return evidence


def _invoke_show(root: Path, run_id: str = "synthetic") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    source = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "bft65b.quality.treatment", "show",
                           "--project-root", str(root), "--run-id", run_id],
                          cwd=source.parent, env=env, capture_output=True, text=True,
                          check=False)


def _assert_rejected(result: subprocess.CompletedProcess[str], run_id: str,
                     diagnostic: str) -> None:
    assert result.returncode == 2
    assert result.stdout == ""
    assert f"Treatment run '{run_id}':" in result.stderr
    assert diagnostic in result.stderr
    assert '"status": "complete"' not in result.stderr


def test_show_valid_saved_run_preserves_success_output_fields(tmp_path):
    _show_fixture(tmp_path)
    result = _invoke_show(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    shown = json.loads(result.stdout)
    assert set(shown) == {"status", "source_files", "derived_files",
                          "period_issue_entries", "datasets", "report"}
    assert shown["status"] == "complete"
    assert (shown["source_files"], shown["derived_files"],
            shown["period_issue_entries"]) == (1, 1, 2)
    assert shown["datasets"]["failures"] == {
        "before_rows": 1, "after_rows": 1, "flagged_distinct_source_keys": 1,
        "raw_values_changed": 0, "records_excluded": 0}
    assert shown["report"].endswith("report.md")


def test_show_rejects_complete_summary_after_later_write_failure(tmp_path):
    evidence = _show_fixture(tmp_path, manifest_status="failed",
                             failure_reason="OSError: simulated report write failure")
    (evidence / "metadata.json").unlink()  # Report failed before metadata and final commit.
    result = _invoke_show(tmp_path)
    _assert_rejected(result, "synthetic", "manifest status=failed")
    assert "OSError: simulated report write failure" in result.stderr
    assert json.loads((evidence / "summary.json").read_text())["status"] == "complete"


def test_show_rejects_missing_manifest_even_with_complete_summary(tmp_path):
    evidence = _show_fixture(tmp_path)
    (evidence / "run_manifest.json").unlink()
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "missing run_manifest.json")


def test_show_surfaces_failed_manifest_reason(tmp_path):
    _show_fixture(tmp_path, manifest_status="failed", failure_reason="Checksum validation failed")
    result = _invoke_show(tmp_path)
    _assert_rejected(result, "synthetic", "manifest status=failed")
    assert "Checksum validation failed" in result.stderr


def test_show_rejects_running_manifest(tmp_path):
    _show_fixture(tmp_path, manifest_status="running")
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "manifest status=running")


def test_show_rejects_unreadable_manifest(tmp_path):
    evidence = _show_fixture(tmp_path)
    (evidence / "run_manifest.json").write_text("{", encoding="utf-8")
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "unreadable run_manifest.json")


def test_show_rejects_manifest_for_different_run(tmp_path):
    evidence = _show_fixture(tmp_path)
    manifest = RunManifest.read(evidence / "run_manifest.json")
    manifest.run_id = "p2_1_treatment_other"
    manifest.write(evidence / "run_manifest.json")
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "manifest run ID mismatch")


@pytest.mark.parametrize("metadata_problem,diagnostic", [
    ("missing", "missing metadata.json"),
    ("unreadable", "unreadable metadata.json"),
    ("running", "metadata status='running'"),
])
def test_show_rejects_incomplete_metadata(tmp_path, metadata_problem, diagnostic):
    evidence = _show_fixture(tmp_path)
    path = evidence / "metadata.json"
    if metadata_problem == "missing":
        path.unlink()
    elif metadata_problem == "unreadable":
        path.write_text("{", encoding="utf-8")
    else:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["status"] = "running"
        path.write_text(json.dumps(metadata), encoding="utf-8")
    _assert_rejected(_invoke_show(tmp_path), "synthetic", diagnostic)


def test_show_rejects_summary_checksum_mismatch(tmp_path):
    evidence = _show_fixture(tmp_path)
    path = evidence / "summary.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "summary.json SHA-256 differs")


def test_show_rejects_incomplete_summary_with_matching_checksum(tmp_path):
    _show_fixture(tmp_path, summary_status="running")
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "summary status='running'")


def test_show_rejects_missing_or_unreadable_summary(tmp_path):
    evidence = _show_fixture(tmp_path)
    summary_path = evidence / "summary.json"
    summary_path.unlink()
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "missing summary.json")
    summary_path.write_text("{", encoding="utf-8")
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "unreadable summary.json")


def test_show_rejects_empty_validations_and_inconsistent_counts(tmp_path):
    evidence = _show_fixture(tmp_path)
    path = evidence / "run_manifest.json"
    manifest = RunManifest.read(path)
    manifest.validations.clear()
    manifest.write(path)
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "manifest validations failed or missing")
    manifest.record_validation("synthetic_gate", True)
    manifest.outputs.clear()
    manifest.write(path)
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "summary counts or manifest output statuses disagree")


def test_show_rejects_malformed_manifest_validation_records(tmp_path):
    evidence = _show_fixture(tmp_path)
    path = evidence / "run_manifest.json"
    manifest = RunManifest.read(path)
    manifest.validations = ["invalid record"]
    manifest.write(path)
    _assert_rejected(_invoke_show(tmp_path), "synthetic", "invalid manifest validation records")


def test_show_rejects_malformed_summary_without_traceback(tmp_path):
    evidence = _show_fixture(tmp_path)
    path = evidence / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["datasets"] = ["invalid"]
    path.write_text(json.dumps(summary), encoding="utf-8")
    metadata_path = evidence / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["summary_sha256"] = _sha(path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = _invoke_show(tmp_path)
    _assert_rejected(result, "synthetic", "invalid summary counts or dataset fields")
    assert "Traceback" not in result.stderr
