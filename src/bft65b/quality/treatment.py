"""Flag-only, lossless treatment of the accepted P2.1 raw-quality findings.

The derived tables retain every raw column and source row. DQ_FLAGS records
review and missingness signals without imputing, excluding, or joining data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.config import load_config
from bft65b.manifest import RunManifest, dependency_versions
from bft65b.quality.audit import (
    ATTEMPT, CONFIG, _read_json, _sha, _write_json, audit_snapshot,
)
from bft65b.quality.rules import PeriodAudit, inspect_period


# Each observed rule must have a declared disposition. New rules fail closed.
POLICY: dict[str, tuple[str, str]] = {
    "source_explicit_null": (
        "retain_null_and_flag", "Preserve the reported null and its distinction from zero; defer any model-specific imputation."),
    "source_field_absent": (
        "retain_absence_and_flag", "Preserve null plus SOURCE_MISSING_FIELDS provenance; do not infer an unreported value."),
    "financial_cert_absent_from_institutions": (
        "retain_row_and_flag", "The Institutions endpoint is a snapshot, not evidence that the historical Financials row is invalid."),
    "failure_cert_null": (
        "retain_source_event_and_flag", "Keep the unique failure source ID; no official bank identifier is inferred."),
    "failure_cert_absent_from_financials": (
        "retain_source_event_and_flag", "The accepted Financials coverage cannot establish a bank link for this event."),
    "failure_only_future_reports": (
        "retain_source_event_and_flag", "A later REPDTE cannot establish a pre-failure observation."),
    "failure_multiple_prior_reports": (
        "retain_one_source_event_and_flag", "Longitudinal reports are not duplicate failure events; defer any single-row join."),
}

FLAG_FIELD = pa.field("DQ_FLAGS", pa.list_(pa.field("element", pa.string())), nullable=False)


def _key(dataset: str, row: Mapping[str, Any]) -> Any:
    if dataset == "financials":
        return row["CERT"], row["REPDTE"]
    return row["ID"] if dataset == "failures" else row["CERT"]


def derive_table(dataset: str, raw: pa.Table, fields: tuple[str, ...], period: str, *,
                 institution_certs: set[int] | None = None,
                 financial_reports: Mapping[int, list[date]] | None = None) -> tuple[pa.Table, PeriodAudit]:
    """Return the unchanged raw columns plus sorted row-level quality flags."""
    rows = raw.to_pylist()
    scan = inspect_period(dataset, rows, fields, period,
        partition=period if dataset == "financials" else None,
        institution_certs=institution_certs, financial_reports=financial_reports)
    if len(scan.row_flags) != raw.num_rows:
        raise ValueError(f"Flag count differs from source rows: {dataset}:{period}")
    flagged = sum(bool(flags) for flags in scan.row_flags)
    if flagged != scan.summary["affected_records_unique"]:
        raise ValueError(f"Overlapping issue union does not reconcile: {dataset}:{period}")
    keys = [_key(dataset, row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Source keys are not unique: {dataset}:{period}")
    unknown = {flag.split(":", 1)[0] for flags in scan.row_flags for flag in flags} - set(POLICY)
    if unknown:
        raise ValueError(f"Unreviewed quality rules require treatment policy: {sorted(unknown)}")
    derived = raw.append_column(FLAG_FIELD, pa.array(scan.row_flags, type=FLAG_FIELD.type))
    metadata = dict(raw.schema.metadata or {})
    metadata[b"bft65b.quality_treatment"] = b"flag_only_v1"
    derived = derived.replace_schema_metadata(metadata)
    if not derived.select(raw.column_names).equals(raw, check_metadata=False):
        raise ValueError(f"Raw values changed while flagging: {dataset}:{period}")
    return derived, scan


def treatment_log(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Assign a checked action to every nonzero, period-specific audit finding."""
    entries = []
    seen = set()
    for finding in audit["period_findings"]:
        dataset, period, field, rule = (finding[k] for k in ("dataset", "period", "field", "rule"))
        identity = (dataset, period, field, rule)
        if identity in seen:
            raise ValueError(f"Duplicate period finding: {identity}")
        seen.add(identity)
        if rule not in POLICY:
            raise ValueError(f"No justified treatment for observed rule: {identity}")
        count, denominator = finding["count"], finding["denominator"]
        if not (isinstance(count, int) and isinstance(denominator, int) and 0 < count <= denominator):
            raise ValueError(f"Invalid finding population: {identity}")
        action, rationale = POLICY[rule]
        entries.append({"dataset": dataset, "period": period, "field": field,
            "rule": rule, "severity": finding["severity"],
            "affected_records": count, "denominator": denominator,
            "prevalence": count / denominator, "action": action, "rationale": rationale,
            "affected_records_retained": count, "raw_values_changed": 0,
            "records_excluded": 0,
            "before_count": count, "after_count": count,
            "count_meaning": "null/absence count for source missingness; rule-hit count otherwise"})
    return {"policy_version": "flag_only_v1", "accepted_attempt": audit["accepted_attempt"],
            "period_issue_entries": len(entries), "entries": entries,
            "overlap_note": "Rule counts overlap; dataset source-key unions, not their sum, count affected records."}


def _period_for(dataset: str, row: Mapping[str, Any], quarter: str | None) -> str:
    if dataset == "financials":
        return str(quarter)
    if dataset == "institutions":
        return "snapshot"
    day = row.get("FAILDATE")
    return str(day.year) if isinstance(day, date) else "undated"


def _flag_counts(dataset: str, rows: list[dict], flags: list[tuple[str, ...]],
                 quarter: str | None) -> Counter[tuple[str, str, str, str]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row, row_flags in zip(rows, flags, strict=True):
        period = _period_for(dataset, row, quarter)
        for flag in row_flags:
            rule, field = flag.split(":", 1)
            counts[(dataset, period, field, rule)] += 1
    return counts


def _logical_sha(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _write_verified(path: Path, derived: pa.Table, raw: pa.Table) -> tuple[str, str]:
    """Validate a temporary file before one atomic promotion in a fresh run root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Derived destination already exists: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(derived, temporary, compression="zstd", compression_level=3)
        with pq.ParquetFile(temporary) as parquet:
            observed = parquet.read()
            if parquet.metadata.num_rows != derived.num_rows:
                raise ValueError(f"Derived row-count mismatch: {path}")
            for group in range(parquet.metadata.num_row_groups):
                for column in range(parquet.metadata.row_group(group).num_columns):
                    if parquet.metadata.row_group(group).column(column).compression != "ZSTD":
                        raise ValueError(f"Derived codec mismatch: {path}")
        if (not observed.schema.equals(derived.schema, check_metadata=True)
                or not observed.equals(derived, check_metadata=True)
                or not observed.select(raw.column_names).equals(raw, check_metadata=False)):
            raise ValueError(f"Derived read-back changed schema, values, or flags: {path}")
        digest, logical = _sha(Path(temporary)), _logical_sha(observed)
        os.replace(temporary, path)
        if _sha(path) != digest:
            raise ValueError(f"Derived checksum changed during promotion: {path}")
        return digest, logical
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _relative_path(dataset: str, quarter: str | None) -> Path:
    if dataset == "financials":
        return Path("financials") / f"REPDTE={quarter}" / "part-000.parquet"
    return Path(dataset) / "part-000.parquet"


def _report(summary: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    lines = ["# P2.1 Raw-Quality Treatment: Report-Ready Results", "",
        "Every source record and raw field value is retained. DQ_FLAGS is an added list of row-level observations; no imputation, exclusion, panel join, exit classification, or label is performed.", "",
        "| Dataset | Before rows | After rows | Distinct keys before/after | Flagged distinct keys | Raw values changed | Excluded |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dataset, item in sorted(summary["datasets"].items()):
        lines.append(f"| {dataset} | {item['before_rows']:,} | {item['after_rows']:,} | {item['unique_source_keys_before']:,}/{item['unique_source_keys_after']:,} | {item['flagged_distinct_source_keys']:,} | 0 | 0 |")
    lines += ["", "## Every observed dataset-field-rule issue", "",
        f"The full {summary['period_issue_entries']:,}-entry period log gives each denominator and action. Aggregate rule counts below overlap and must not be added to estimate affected source records.", "",
        "| Dataset / field | Rule | Before | After | Denominator | Action |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for dataset, item in sorted(audit["datasets"].items()):
        for finding in item["rule_findings"]:
            action = POLICY[finding["rule"]][0]
            lines.append(f"| {dataset} / {finding['field']} | {finding['rule']} | {finding['count']:,} | {finding['count']:,} | {finding['denominator']:,} | {action} |")
    lines += ["", "Counts intentionally remain unchanged: the accepted snapshot has review signals and unavailable values, but no verified defect justifying raw-value replacement or row removal at this milestone. The Institutions endpoint is a snapshot; its absence cannot exclude historical Financials. Failure source ID remains the event key, and multiple earlier reports do not multiply an event.", "",
        "Model-specific missing-value treatment, historical institution validity, dated bank linkage, point-in-time availability, panel construction, and labels remain separate milestones.", ""]
    return "\n".join(lines)


def run_treatment(root: Path, config_file: Path, audit_run: str, run_id: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    for value in (audit_run, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", value):
            raise ValueError("Run IDs must be short safe alphanumeric identifiers")
    config_path = config_file if config_file.is_absolute() else root / config_file
    config = load_config(config_path)
    audit_dir = root / "runs" / "task_2_1_quality" / audit_run
    accepted = _read_json(audit_dir / "findings.json")
    audit_meta = _read_json(audit_dir / "metadata.json")
    audit_manifest = RunManifest.read(audit_dir / "run_manifest.json")
    if (accepted.get("status") != "complete" or accepted.get("accepted_attempt") != ATTEMPT
            or audit_meta.get("status") != "complete" or not audit_meta.get("input_hashes_unchanged")
            or audit_meta.get("findings_sha256") != _sha(audit_dir / "findings.json")
            or audit_manifest.status != "complete" or len(audit_manifest.inputs) != 172
            or any(not item["passed"] for item in audit_manifest.validations)):
        raise ValueError("Verified P2.1 audit evidence is required before treatment")
    bundle_path = root / "runs/source_vintage_bundle.json"
    if bundle_path.exists():
        if audit_manifest.config.get("source_vintage_bundle_sha256") != _sha(bundle_path):
            raise ValueError("Audit source-vintage lineage differs")
    fresh, inventory = audit_snapshot(root, config, ATTEMPT)
    if fresh != accepted or len(inventory) != 172:
        raise ValueError("Accepted raw snapshot no longer reproduces the P2.1 audit")
    log = treatment_log(accepted)
    expected_flags = Counter({(entry["dataset"], entry["period"], entry["field"], entry["rule"]):
                              entry["affected_records"] for entry in log["entries"]})
    data_root = root / "data_processed" / "task_2_1_treatment" / run_id
    evidence_root = root / "runs" / "task_2_1_treatment" / run_id
    if data_root.exists() or evidence_root.exists():
        raise FileExistsError("Treatment run ID already has data or evidence; use a fresh ID")
    data_root.mkdir(parents=True)
    evidence_root.mkdir(parents=True)
    manifest = RunManifest.new(f"p2_1_treatment_{run_id}", config)
    manifest.config.update({"source_audit_run": audit_run,
        "source_audit_findings_sha256": audit_meta["findings_sha256"]})
    if bundle_path.exists():
        manifest.config["source_vintage_bundle_sha256"] = _sha(bundle_path)
    manifest.set_status("running")
    for item in inventory:
        manifest.record_input(item["path"], item["sha256"], item["rows"], item["source_total"])
    indexed = {(item["dataset"], item["quarter"]): item for item in inventory}
    datasets = {name: {"before_rows": 0, "after_rows": 0, "unique_source_keys_before": 0,
        "unique_source_keys_after": 0, "flagged_distinct_source_keys": 0,
        "raw_values_changed": 0, "records_excluded": 0,
        "before_field_counts": accepted["datasets"][name]["field_counts"],
        "after_field_counts": accepted["datasets"][name]["field_counts"]}
        for name in ("financials", "failures", "institutions")}
    reports: dict[int, list[date]] = defaultdict(list)
    flag_counts: Counter[tuple[str, str, str, str]] = Counter()
    outputs = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for dataset, quarter, fields in [
            ("institutions", None, config.institution_fields),
            *(("financials", period, config.financial_fields) for period in
              sorted(item["quarter"] for item in inventory if item["dataset"] == "financials")),
            ("failures", None, config.failure_fields),
        ]:
            item = indexed[(dataset, quarter)]
            source_path = Path(item["path"])
            if _sha(source_path) != item["sha256"]:
                raise ValueError(f"Raw input checksum differs before treatment: {dataset}:{quarter}")
            with pq.ParquetFile(source_path) as parquet:
                raw = parquet.read()
            period = quarter if dataset == "financials" else ("snapshot" if dataset == "institutions" else "all")
            derived, scan = derive_table(dataset, raw, fields, period,
                institution_certs=institution_certs if dataset == "financials" else None,
                financial_reports=reports if dataset == "failures" else None)
            rows = raw.to_pylist()
            if dataset == "institutions":
                institution_certs = scan.certs
            elif dataset == "financials":
                for cert, dates in scan.cert_dates.items():
                    reports[cert].extend(dates)
            flag_counts.update(_flag_counts(dataset, rows, scan.row_flags, quarter))
            target_rel = _relative_path(dataset, quarter)
            target = data_root / target_rel
            output_sha, logical_sha = _write_verified(target, derived, raw)
            if _sha(source_path) != item["sha256"]:
                raise ValueError(f"Raw input checksum differs after treatment: {dataset}:{quarter}")
            n = raw.num_rows
            section = datasets[dataset]
            section["before_rows"] += n
            section["after_rows"] += derived.num_rows
            section["unique_source_keys_before"] += n
            section["unique_source_keys_after"] += derived.num_rows
            section["flagged_distinct_source_keys"] += sum(bool(flags) for flags in scan.row_flags)
            outputs.append({"dataset": dataset, "quarter": quarter,
                "relative_path": target_rel.as_posix(), "rows": n,
                "source_sha256": item["sha256"], "derived_sha256": output_sha,
                "logical_sha256": logical_sha, "source_schema_version": item["schema_version"],
                "quality_schema_version": "flag_only_v1", "codec": "ZSTD"})
            manifest.record_output(path=str(target), dataset=dataset, rows=n,
                sha256=output_sha, compression="ZSTD", schema_version="flag_only_v1",
                partition=quarter, source_total=n, source_query=None)
        if flag_counts != expected_flags:
            raise ValueError(f"Row flags do not reconcile to all period audit findings: {flag_counts - expected_flags}; {expected_flags - flag_counts}")
        if len(outputs) != 172:
            raise ValueError("Treatment did not produce all 172 expected derived files")
        for name, section in datasets.items():
            source = accepted["datasets"][name]
            if (section["before_rows"] != source["rows"]
                    or section["after_rows"] != source["rows"]
                    or section["flagged_distinct_source_keys"] != source["affected_records_unique"]
                    or section["unique_source_keys_before"] != source["rows"]
                    or section["unique_source_keys_after"] != source["rows"]):
                raise ValueError(f"Distinct-source-key reconciliation failed: {name}")
        for item in inventory:
            if _sha(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Raw input changed during treatment: {item['dataset']}:{item['quarter']}")
        summary = {"status": "complete", "accepted_attempt": ATTEMPT,
            "audit_findings_sha256": audit_meta["findings_sha256"],
            "policy_version": "flag_only_v1", "source_files": len(inventory),
            "derived_files": len(outputs), "period_issue_entries": len(log["entries"]),
            "datasets": datasets, "outputs": outputs, "raw_hashes_unchanged": True,
            "period_flag_counts_reconciled": True,
            "before_after_policy": "All original values, nulls, provenance, source keys and rows retained; DQ_FLAGS added."}
        _write_json(evidence_root / "treatment_log.json", log)
        _write_json(evidence_root / "summary.json", summary)
        report_path = evidence_root / "report.md"
        report_path.write_text(_report(summary, accepted), encoding="utf-8")
        metadata = {"status": "complete", "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": f"python -m bft65b.quality.treatment run --project-root {root} --config {config_file} --audit-run {audit_run} --run-id {run_id}",
            "os": platform.platform(), "python": platform.python_version(),
            "dependencies": dependency_versions(), "config": config.to_dict(),
            "audit_findings_path": str(audit_dir / "findings.json"),
            "audit_findings_sha256": audit_meta["findings_sha256"],
            "input_hashes": inventory, "data_root": str(data_root),
            "summary_sha256": _sha(evidence_root / "summary.json"),
            "treatment_log_sha256": _sha(evidence_root / "treatment_log.json"),
            "report_sha256": _sha(report_path)}
        _write_json(evidence_root / "metadata.json", metadata)
        manifest.record_validation("accepted_audit_and_raw_snapshot", True, "Fresh 172-file audit equals saved findings")
        manifest.record_validation("all_period_flags_reconciled", True, "Every issue/period count matches distinct source rows")
        manifest.record_validation("derived_read_back", True, "All 172 files preserve source columns and row-level flags")
        manifest.record_validation("raw_hashes_unchanged", True, "All accepted SHA-256 values match after treatment")
        manifest.set_status("complete")
        manifest.write(evidence_root / "run_manifest.json")
        return {"status": "PASS", "run_id": run_id, "evidence_root": str(evidence_root),
            "data_root": str(data_root), "source_files": len(inventory),
            "derived_files": len(outputs), "period_issue_entries": len(log["entries"]),
            "datasets": {name: {"before": item["before_rows"], "after": item["after_rows"],
                               "flagged_keys": item["flagged_distinct_source_keys"]}
                         for name, item in datasets.items()},
            "summary_sha256": metadata["summary_sha256"],
            "treatment_log_sha256": metadata["treatment_log_sha256"]}
    except Exception as exc:
        manifest.record_validation("p2_1_treatment", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence_root / "run_manifest.json")
        raise


class ShowEvidenceError(ValueError):
    """Saved treatment evidence cannot establish a completed run."""


def _completed_summary(evidence: Path, run_id: str) -> dict[str, Any]:
    """Check saved commit evidence without touching datasets or changing files."""
    manifest_path = evidence / "run_manifest.json"
    if not manifest_path.is_file():
        raise ShowEvidenceError("missing run_manifest.json")
    try:
        manifest = RunManifest.read(manifest_path)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ShowEvidenceError(f"unreadable run_manifest.json ({type(exc).__name__})") from exc
    expected_id = f"p2_1_treatment_{run_id}"
    if manifest.run_id != expected_id:
        raise ShowEvidenceError(f"manifest run ID mismatch: {manifest.run_id!r}")
    if (not isinstance(manifest.validations, list)
            or any(not isinstance(item, dict) for item in manifest.validations)):
        raise ShowEvidenceError("invalid manifest validation records")
    if manifest.status != "complete":
        failures = [item for item in manifest.validations if item.get("passed") is False]
        reason = failures[-1].get("detail", "") if failures else ""
        suffix = f"; reason: {reason}" if reason else ""
        raise ShowEvidenceError(f"manifest status={manifest.status}{suffix}")
    if not manifest.validations or any(item.get("passed") is not True for item in manifest.validations):
        failures = [item for item in manifest.validations if item.get("passed") is not True]
        reason = failures[-1].get("detail", "") if failures else "no recorded validations"
        raise ShowEvidenceError(f"manifest validations failed or missing: {reason}")
    if (not isinstance(manifest.inputs, list) or not isinstance(manifest.outputs, list)
            or any(not isinstance(item, dict) for item in manifest.outputs)):
        raise ShowEvidenceError("invalid manifest input or output records")

    metadata_path = evidence / "metadata.json"
    if not metadata_path.is_file():
        raise ShowEvidenceError("missing metadata.json")
    try:
        metadata = _read_json(metadata_path)
    except (OSError, ValueError, TypeError) as exc:
        raise ShowEvidenceError(f"unreadable metadata.json ({type(exc).__name__})") from exc
    if metadata.get("status") != "complete":
        raise ShowEvidenceError(f"metadata status={metadata.get('status')!r}")
    expected_sha = metadata.get("summary_sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ShowEvidenceError("metadata summary_sha256 missing or invalid")

    summary_path = evidence / "summary.json"
    if not summary_path.is_file():
        raise ShowEvidenceError("missing summary.json")
    try:
        actual_sha = _sha(summary_path)
        summary = _read_json(summary_path)
    except (OSError, ValueError, TypeError) as exc:
        raise ShowEvidenceError(f"unreadable summary.json ({type(exc).__name__})") from exc
    if actual_sha != expected_sha:
        raise ShowEvidenceError("summary.json SHA-256 differs from metadata summary_sha256")
    if summary.get("status") != "complete":
        raise ShowEvidenceError(f"summary status={summary.get('status')!r}")
    if (any(type(summary.get(key)) is not int or summary[key] < 0
            for key in ("source_files", "derived_files", "period_issue_entries"))
            or summary["source_files"] == 0 or summary["derived_files"] == 0
            or not isinstance(summary.get("datasets"), dict) or not summary["datasets"]
            or any(not isinstance(item, dict) or any(key not in item for key in (
                "before_rows", "after_rows", "flagged_distinct_source_keys",
                "raw_values_changed", "records_excluded"))
                for item in summary["datasets"].values())):
        raise ShowEvidenceError("invalid summary counts or dataset fields")
    if (summary.get("source_files") != len(manifest.inputs)
            or summary.get("derived_files") != len(manifest.outputs)
            or any(item.get("status") != "complete" for item in manifest.outputs)):
        raise ShowEvidenceError("summary counts or manifest output statuses disagree")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lossless, flag-only P2.1 treatment of accepted FDIC raw data")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--project-root", type=Path, default=Path.cwd())
    run.add_argument("--config", type=Path, default=CONFIG)
    run.add_argument("--audit-run", default="audit20261114a")
    run.add_argument("--run-id", required=True)
    show = sub.add_parser("show")
    show.add_argument("--project-root", type=Path, default=Path.cwd())
    show.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_treatment(args.project_root, args.config, args.audit_run, args.run_id)
        print(json.dumps(result, indent=2))
    else:
        evidence = args.project_root.resolve() / "runs" / "task_2_1_treatment" / args.run_id
        try:
            summary = _completed_summary(evidence, args.run_id)
        except ShowEvidenceError as exc:
            parser.exit(status=2, message=f"Treatment run {args.run_id!r}: {exc}\n")
        print(json.dumps({"status": summary["status"], "source_files": summary["source_files"],
                          "derived_files": summary["derived_files"],
                          "period_issue_entries": summary["period_issue_entries"],
                          "datasets": {name: {key: item[key] for key in (
                              "before_rows", "after_rows", "flagged_distinct_source_keys",
                              "raw_values_changed", "records_excluded")}
                              for name, item in summary["datasets"].items()},
                          "report": str(evidence / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
