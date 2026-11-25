"""Read-only P2.1 audit of the accepted FDIC raw snapshot.

This stage reports source quality and join coverage. It does not transform raw
records, select an exit reason, build a panel, or create modelling labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping

import pyarrow.parquet as pq

from bft65b.acquisition.extended import _entry, _verify_complete, full_plan
from bft65b.acquisition.full import full_paths
from bft65b.acquisition.pilot import _query
from bft65b.acquisition.storage import ParquetStore
from bft65b.config import EXTENDED_FINANCIAL_FIELDS, ProjectConfig, load_config
from bft65b.manifest import RunManifest, dependency_versions
from bft65b.quality.rules import PeriodAudit, RULES, inspect_period


ATTEMPT = "real20261111b"
CONFIG = Path("config/financials_full_v2.json")
ACCEPTANCE = Path("runs/task_1_5_full_real20261111b_acceptance.json")
RULE_LIMITATIONS = [
    "No full assets = liabilities + equity check: the selected fields omit a complete liability decomposition.",
    "No LNRE + LNCI <= LNLSGR assertion: category overlap and historical accounting scope require official reconciliation.",
    "No universal bounds for ROA, NIMY, RBC1AAJ, RBCRWAJ, P3ASSET, P9ASSET, or NAASSET: negative values, annualisation, or unresolved scale may be legitimate.",
    "DEPDOM > DEP is a review observation, not a definite violation, until domestic/total reporting bases are confirmed for each historical period.",
    "Typed Parquet cannot recover every original JSON numeric representation; source_type_counts come from the accepted acquisition evidence.",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _load_snapshot(root: Path, config: ProjectConfig, attempt: str) -> tuple:
    if attempt != ATTEMPT or config.financial_fields != EXTENDED_FINANCIAL_FIELDS:
        raise ValueError("P2.1 is pinned to accepted real20261111b and the verified Financials v2 fields")
    paths = full_paths(root, config, attempt)
    plan = full_plan(config)
    manifest = RunManifest.read(paths.manifest)
    summary = _read_json(paths.summary)
    bundle_path = root / "runs/source_vintage_bundle.json"
    if bundle_path.exists():
        from bft65b.quality.source_vintage import relocated_view
        manifest, summary = relocated_view(root, manifest, summary,
            ParquetStore(paths.raw_root, config))
    jobs = plan.jobs(config)
    if (manifest.status != "complete" or summary.get("status") != "complete"
            or manifest.run_id != summary.get("run_id") or len(jobs) != 172
            or len(manifest.outputs) != 172 or len(manifest.sources) != 172
            or len(summary.get("quarters", [])) != 170 or len(summary.get("small_tables", [])) != 2
            or any(not validation.get("passed") for validation in manifest.validations)
            or not any(v.get("name") == "plan:all_outputs" and v.get("passed")
                       for v in manifest.validations)
            or json.dumps(manifest.config, sort_keys=True) != json.dumps(config.to_dict(), sort_keys=True)):
        raise ValueError("Accepted source manifest/summary completion gate failed")
    store = ParquetStore(paths.raw_root, config)
    _verify_complete(plan, config, store, manifest, summary)
    return paths, plan, manifest, summary, store, jobs


def _job_metadata(endpoint: str, quarter: str | None, fields: tuple[str, ...], limit: int,
                  manifest: RunManifest, summary: dict, store: ParquetStore) -> tuple[dict, dict, dict, Path]:
    details = _entry(summary, endpoint, quarter)
    query = _query(endpoint, fields, quarter, limit)
    sources = [item for item in manifest.sources
               if item.get("endpoint") == endpoint and item.get("query") == query]
    path = store.target_path(endpoint, quarter)
    outputs = [item for item in manifest.outputs
               if item.get("dataset") == endpoint and item.get("path") == str(path)]
    if details is None or len(sources) != 1 or len(outputs) != 1:
        raise ValueError(f"Missing/duplicate source or output evidence: {endpoint}:{quarter}")
    source, output = sources[0], outputs[0]
    if (details.get("query") != query or source.get("api_total") != output.get("rows")
            or source.get("downloaded_rows") != output.get("rows")
            or details.get("api_total") != output.get("rows")
            or details.get("downloaded_rows") != output.get("rows")
            or output.get("source_total") != output.get("rows")
            or output.get("source_query") != query
            or output.get("sha256") != details.get("sha256")
            or details.get("index") != source.get("index")
            or details.get("missing_fields") != source.get("missing_fields")
            or details.get("null_fields") != source.get("null_fields")
            or set(details.get("missing_fields", {})) != set(fields)
            or set(details.get("null_fields", {})) != set(fields)):
        raise ValueError(f"Source query/total/provenance disagrees: {endpoint}:{quarter}")
    return source, output, details, path


def _reconcile_fields(scan: Mapping[str, Any], source: Mapping[str, Any], fields: tuple[str, ...],
                      endpoint: str, period: str) -> None:
    for field in fields:
        observed = scan["field_counts"][field]
        if (observed["absent"] != source["missing_fields"][field]
                or observed["explicit_null"] != source["null_fields"][field]
                or observed["absent"] + observed["explicit_null"] + observed["non_null"] != scan["rows"]):
            raise ValueError(f"Physical/source null-versus-absent mismatch: {endpoint}:{period}:{field}")


def _aggregate(dataset: str, scans: list[PeriodAudit], fields: tuple[str, ...]) -> dict:
    rows = sum(scan.summary["rows"] for scan in scans)
    counts: dict[str, Counter] = {field: Counter() for field in fields}
    rules: Counter[tuple[str, str]] = Counter()
    certs: set[int] = set()
    for scan in scans:
        certs.update(scan.certs)
        for field in fields:
            values = scan.summary["field_counts"][field]
            counts[field]["absent"] += values["absent"]
            counts[field]["explicit_null"] += values["explicit_null"]
            counts[field]["non_null"] += values["non_null"]
        for finding in scan.summary["rule_findings"]:
            rules[(finding["rule"], finding["field"])] += finding["count"]
    field_counts = {field: {"absent": counts[field]["absent"],
                            "explicit_null": counts[field]["explicit_null"],
                            "non_null": counts[field]["non_null"], "denominator": rows}
                    for field in fields}
    findings = []
    for field in fields:
        for kind, label, interpretation in (
            ("absent", "source_field_absent", "Requested field was absent from the source response."),
            ("explicit_null", "source_explicit_null", "Requested field was present with a null value."),
        ):
            count = field_counts[field][kind]
            if count:
                findings.append({"dataset": dataset, "period": "all" if dataset != "institutions" else "snapshot",
                    "field": field, "rule": label, "count": count, "denominator": rows,
                    "prevalence": count / rows if rows else None, "severity": "review",
                    "interpretation": interpretation})
    for (rule, field), count in sorted(rules.items()):
        severity, interpretation = RULES[rule]
        findings.append({"dataset": dataset, "period": "all" if dataset != "institutions" else "snapshot",
            "field": field, "rule": rule, "count": count, "denominator": rows,
            "prevalence": count / rows if rows else None, "severity": severity,
            "interpretation": interpretation})
    return {"rows": rows, "periods": len(scans), "distinct_nonnull_certs": len(certs),
            "field_counts": field_counts, "rule_findings": sorted(findings, key=lambda x: (x["field"], x["rule"])),
            "affected_records_unique": sum(x.summary["affected_records_unique"] for x in scans),
            "missing_or_null_records_unique": sum(x.summary["missing_or_null_records_unique"] for x in scans)}


def _period_findings(scan: PeriodAudit) -> list[dict]:
    summary = scan.summary
    findings = list(summary["rule_findings"])
    n = summary["rows"]
    for field, values in summary["field_counts"].items():
        for key, rule, interpretation in (
            ("absent", "source_field_absent", "Requested field was absent from the source response."),
            ("explicit_null", "source_explicit_null", "Requested field was present with a null value."),
        ):
            count = values[key]
            if count:
                findings.append({"dataset": summary["dataset"], "period": summary["period"],
                    "field": field, "rule": rule, "count": count, "denominator": n,
                    "prevalence": count / n if n else None, "severity": "review",
                    "interpretation": interpretation})
    return sorted(findings, key=lambda x: (x["dataset"], x["period"], x["field"], x["rule"]))


def _cross_check(result: dict, root: Path) -> dict:
    accepted = _read_json(root / ACCEPTANCE)
    datasets = result["datasets"]
    checks = {
        "rows_by_endpoint": {name: datasets[name]["rows"] for name in datasets} == accepted["rows_by_endpoint"],
        "failures_null_cert": datasets["failures"]["field_counts"]["CERT"]["explicit_null"]
            == accepted["explicit_null_counts"]["failures.CERT"],
        "institutions_absent_changec1": datasets["institutions"]["field_counts"]["CHANGEC1"]["absent"]
            == accepted["field_absence_counts"]["institutions.CHANGEC1"],
        "financials_null_p3asset": datasets["financials"]["field_counts"]["P3ASSET"]["explicit_null"]
            == accepted["explicit_null_counts"]["financials.P3ASSET"],
        "financials_all_null_p3asset_quarters": result["p3asset_all_explicit_null_quarters"]
            == accepted["p3asset_source_explicit_all_null_quarters"],
    }
    result["acceptance_cross_check"] = {"checks": checks, "all_passed": all(checks.values()),
        "acceptance_report_sha256": _sha(root / ACCEPTANCE)}
    if not all(checks.values()):
        raise ValueError(f"Independent Task 1.5 acceptance cross-check differs: {checks}")
    return result["acceptance_cross_check"]


def audit_snapshot(root: Path, config: ProjectConfig, attempt: str = ATTEMPT) -> tuple[dict, list[dict]]:
    """Return deterministic findings and the verified 172-file input inventory."""
    paths, plan, manifest, summary, store, jobs = _load_snapshot(root, config, attempt)
    inventory = []
    details_by_job = {}
    source_types: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    source_conversion_anomalies = Counter()
    for endpoint, quarter, fields, limit in jobs:
        source, output, details, path = _job_metadata(endpoint, quarter, fields, limit, manifest, summary, store)
        inventory.append({"dataset": endpoint, "quarter": quarter, "path": str(path),
                          "rows": output["rows"], "schema_version": output["schema_version"],
                          "compression": output["compression"], "sha256": output["sha256"],
                          "source_total": source["api_total"], "source_index": source["index"]})
        details_by_job[(endpoint, quarter)] = (source, details, path)
        source_conversion_anomalies[endpoint] += details.get("conversion_anomalies", 0)
        for field, type_counts in details.get("source_type_counts", {}).items():
            source_types[endpoint][field].update(type_counts)

    institutions_fields = config.institution_fields
    inst_source, _, inst_path = details_by_job[("institutions", None)]
    with pq.ParquetFile(inst_path) as parquet:
        inst_rows = parquet.read().to_pylist()
    inst_scan = inspect_period("institutions", inst_rows, institutions_fields, "snapshot")
    _reconcile_fields(inst_scan.summary, inst_source, institutions_fields, "institutions", "snapshot")
    institution_certs = inst_scan.certs

    financial_scans: list[PeriodAudit] = []
    reports: dict[int, list[date]] = defaultdict(list)
    financial_certs: set[int] = set()
    financial_unmatched_certs: set[int] = set()
    p3_all_null: list[str] = []
    for quarter in plan.quarters:
        source, _, path = details_by_job[("financials", quarter)]
        with pq.ParquetFile(path) as parquet:
            rows = parquet.read().to_pylist()
        scan = inspect_period("financials", rows, config.financial_fields, quarter,
                              partition=quarter, institution_certs=institution_certs)
        _reconcile_fields(scan.summary, source, config.financial_fields, "financials", quarter)
        financial_scans.append(scan)
        financial_certs.update(scan.certs)
        financial_unmatched_certs.update(scan.unmatched_certs)
        for cert, dates in scan.cert_dates.items():
            reports[cert].extend(dates)
        p3 = scan.summary["field_counts"]["P3ASSET"]
        if p3["explicit_null"] == scan.summary["rows"] and p3["absent"] == 0:
            p3_all_null.append(quarter)

    fail_source, _, fail_path = details_by_job[("failures", None)]
    with pq.ParquetFile(fail_path) as parquet:
        fail_rows = parquet.read().to_pylist()
    failure_all = inspect_period("failures", fail_rows, config.failure_fields, "all",
                                 financial_reports=reports)
    _reconcile_fields(failure_all.summary, fail_source, config.failure_fields, "failures", "all")
    failures_by_year: dict[str, list[dict]] = defaultdict(list)
    for row in fail_rows:
        day = row.get("FAILDATE")
        failures_by_year[str(day.year) if isinstance(day, date) else "undated"].append(row)
    failure_year_scans = [inspect_period("failures", failures_by_year[year], config.failure_fields,
                                         year, financial_reports=reports)
                          for year in sorted(failures_by_year)]

    datasets = {
        "financials": _aggregate("financials", financial_scans, config.financial_fields),
        "failures": _aggregate("failures", [failure_all], config.failure_fields),
        "institutions": _aggregate("institutions", [inst_scan], config.institution_fields),
    }
    datasets["failures"]["event_years"] = len(failure_year_scans)
    financial_join = {
        "record_denominator": datasets["financials"]["rows"],
        "matched_records": sum(x.summary["institution_snapshot_join"]["matched_records"] for x in financial_scans),
        "unmatched_records": sum(x.summary["institution_snapshot_join"]["unmatched_records"] for x in financial_scans),
        "distinct_financial_cert_denominator": len(financial_certs),
        "matched_distinct_financial_certs": len(financial_certs & institution_certs),
        "unmatched_distinct_financial_certs": len(financial_certs - institution_certs),
        "distinct_institution_cert_denominator": len(institution_certs),
        "institution_certs_with_any_financial": len(institution_certs & financial_certs),
        "restriction": "CERT equality over the accepted 1984Q1-2026Q2 Financials range and one current Institutions endpoint snapshot; no historical institution validity claim",
    }
    fail_join = dict(failure_all.summary["failure_linkage"])
    fail_join["distinct_failure_source_ids"] = len({row["ID"] for row in fail_rows
                                                     if isinstance(row.get("ID"), str)})
    fail_join["non_null_cert_records"] = sum(1 for row in fail_rows if row.get("CERT") is not None)
    fail_join["missing_cert_records"] = fail_join["record_categories"].get("missing_cert", 0)

    period_scans = financial_scans + failure_year_scans + [inst_scan]
    result = {
        "status": "complete", "requirement": "P2.1", "accepted_attempt": attempt,
        "scope": {"financial_quarters": len(plan.quarters), "first_quarter": plan.quarters[0],
                  "last_quarter": plan.quarters[-1], "files": len(inventory)},
        "datasets": datasets,
        "periods": [scan.summary for scan in period_scans],
        "period_findings": [finding for scan in period_scans for finding in _period_findings(scan)],
        "joins": {"financials_to_institutions": financial_join,
                  "failures_to_financials": fail_join},
        "p3asset_all_explicit_null_quarters": p3_all_null,
        "source_type_counts": {dataset: {field: dict(sorted(types.items()))
                             for field, types in sorted(fields.items())}
                             for dataset, fields in sorted(source_types.items())},
        "source_conversion_anomalies": dict(sorted(source_conversion_anomalies.items())),
        "unit_and_accounting_limitations": RULE_LIMITATIONS,
        "method_limits": [
            "Institutions is an endpoint snapshot, not a dated institution-history table.",
            "Failure-to-Financials chronology uses REPDTE <= FAILDATE; it does not establish report publication availability, bank identity, or a model observation.",
            "Financials begin at 1984Q1; an older failure can have no eligible Financials even with a genuine bank relationship.",
            "Null and absent counts are separate; affected_records_unique is a union over overlapping row checks, never a sum of field counts.",
        ],
    }
    _cross_check(result, root)
    for item in inventory:
        if _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Accepted raw file changed during audit: {item['dataset']}:{item['quarter']}")
    return result, inventory


def _report(result: dict) -> str:
    datasets = result["datasets"]
    selected = []
    priorities = {
        ("financials", "P3ASSET", "source_explicit_null"),
        ("failures", "CERT", "source_explicit_null"),
        ("failures", "COST", "source_explicit_null"),
        ("institutions", "CHANGEC1", "source_field_absent"),
        ("financials", "CERT", "financial_cert_absent_from_institutions"),
        ("failures", "CERT", "failure_cert_null"),
        ("failures", "CERT", "failure_cert_absent_from_financials"),
        ("failures", "CERT,FAILDATE", "failure_only_future_reports"),
        ("failures", "CERT,REPDTE", "failure_ambiguous_latest_report"),
        ("financials", "ASSET", "negative_total_assets"),
        ("financials", "DEPDOM,DEP", "domestic_deposits_above_total"),
    }
    for dataset, summary in datasets.items():
        for finding in summary["rule_findings"]:
            if (dataset, finding["field"], finding["rule"]) in priorities:
                selected.append(finding)
    selected.sort(key=lambda x: (x["dataset"], x["field"], x["rule"]))
    lines = ["# P2.1 Raw-Data Quality Audit: Report-Ready Findings", "",
        "The accepted Task 1.5 snapshot was audited read-only. Counts are measured from typed raw Parquet and SOURCE_MISSING_FIELDS, then reconciled to acquisition evidence. No row was cleaned, imputed, excluded, or labelled.", "",
        "| Dataset / field | Period | Rule | Count | Denominator | Prevalence | Interpretation |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for x in selected:
        lines.append(f"| {x['dataset']} / {x['field']} | {x['period']} | {x['rule']} | {x['count']:,} | {x['denominator']:,} | {100*x['prevalence']:.3f}% | {x['interpretation']} |")
    if not selected:
        lines.append("| None | all | No selected material finding | 0 | - | - | See the machine-readable audit for complete zero and positive checks. |")
    fjoin = result["joins"]["financials_to_institutions"]
    failure = result["joins"]["failures_to_financials"]
    lines += ["", "## Join coverage and interpretation", "",
        f"Financials-to-Institutions CERT equality matched {fjoin['matched_records']:,} of {fjoin['record_denominator']:,} Financials records and {fjoin['matched_distinct_financial_certs']:,} of {fjoin['distinct_financial_cert_denominator']:,} distinct Financials CERT values. The Institutions table is a current endpoint snapshot; this is not a historically dated institution join.", "",
        f"Failures have {failure['record_denominator']:,} distinct source-record candidates by ID. Chronological linkage categories (denominator: all failure records) are: "
        + ", ".join(f"{name}={count:,}" for name, count in sorted(failure["record_categories"].items()))
        + ". A CERT match or REPDTE <= FAILDATE does not establish that a report was available to a supervisor on the failure date.", "",
        f"P3ASSET is explicitly null in all rows of {len(result['p3asset_all_explicit_null_quarters'])} quarters. Its period detail and all other field counts are in findings.json.", "",
        "## Rule and method boundaries", ""]
    lines += [f"- {item}" for item in result["unit_and_accounting_limitations"]]
    lines += ["", "The audit is the P2.1 measurement gate only. Treatment decisions, before/after counts, panel construction, exit classification, availability joins, features, and labels remain pending.", ""]
    return "\n".join(lines)


def run_audit(root: Path, config_file: Path, attempt: str, run_id: str) -> dict:
    root = root.resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", run_id):
        raise ValueError("Run ID must be a short safe alphanumeric identifier")
    config_path = config_file if config_file.is_absolute() else root / config_file
    config = load_config(config_path)
    output = root / "runs" / "task_2_1_quality" / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest.new(f"p2_1_{run_id}", config)
    manifest.config["source_attempt"] = attempt
    bundle_path = root / "runs/source_vintage_bundle.json"
    if bundle_path.exists():
        manifest.config["source_vintage_bundle_sha256"] = _sha(bundle_path)
    manifest.set_status("running")
    started = datetime.now(timezone.utc).isoformat()
    try:
        result, inventory = audit_snapshot(root, config, attempt)
        for item in inventory:
            manifest.record_input(item["path"], item["sha256"], item["rows"], item["source_total"])
        findings_path = output / "findings.json"
        report_path = output / "report.md"
        _write_json(findings_path, result)
        report_path.write_text(_report(result), encoding="utf-8")
        metadata = {"status": "complete", "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": f"python -m bft65b.quality.audit audit --project-root {root} --config {config_file} --attempt {attempt} --run-id {run_id}",
            "os": platform.platform(), "python": platform.python_version(),
            "dependencies": dependency_versions(), "config": config.to_dict(),
            "config_sha256": _sha(config_path),
            "source_manifest_sha256": _sha(root / f"runs/task_1_5_full_{attempt}.json"),
            "source_summary_sha256": _sha(root / f"runs/task_1_5_full_{attempt}_summary.json"),
            "input_files": inventory, "input_hashes_unchanged": True,
            "findings_sha256": _sha(findings_path), "report_sha256": _sha(report_path),
            "findings_path": str(findings_path), "report_path": str(report_path)}
        _write_json(output / "metadata.json", metadata)
        manifest.record_validation("accepted_snapshot_172_files", True,
                                   "All source schema, key, row, codec, and SHA-256 checks passed")
        manifest.record_validation("source_missingness_reconciliation", True,
                                   "Physical absent and explicit-null counts match accepted acquisition evidence")
        manifest.record_validation("task_1_5_independent_cross_check", True,
                                   "Rows and four known acceptance findings reconcile")
        manifest.record_validation("source_hashes_unchanged_after_audit", True,
                                   "All 172 SHA-256 values match the accepted manifest")
        manifest.set_status("complete")
        manifest.write(output / "run_manifest.json")
        return {"status": "PASS", "output": str(output), "rows": {k: v["rows"] for k, v in result["datasets"].items()},
                "periods": len(result["periods"]), "period_findings": len(result["period_findings"]),
                "findings_sha256": metadata["findings_sha256"], "source_hashes_unchanged": True}
    except Exception as exc:
        manifest.record_validation("p2_1_audit", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(output / "run_manifest.json")
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read-only P2.1 audit of accepted FDIC raw data")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--project-root", type=Path, default=Path.cwd())
    audit.add_argument("--config", type=Path, default=CONFIG)
    audit.add_argument("--attempt", default=ATTEMPT)
    audit.add_argument("--run-id", required=True)
    show = sub.add_parser("show")
    show.add_argument("--project-root", type=Path, default=Path.cwd())
    show.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = run_audit(args.project_root, args.config, args.attempt, args.run_id)
        print(json.dumps(result, indent=2))
    else:
        output = args.project_root.resolve() / "runs" / "task_2_1_quality" / args.run_id
        result = _read_json(output / "findings.json")
        print(json.dumps({"status": result["status"], "scope": result["scope"],
                          "datasets": {name: {"rows": data["rows"],
                                             "affected_records_unique": data["affected_records_unique"]}
                                       for name, data in result["datasets"].items()},
                          "report": str(output / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
