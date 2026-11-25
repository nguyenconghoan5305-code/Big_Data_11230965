"""Read-only Task 1.5 acceptance audit for one completed real full attempt.

This is not collected by pytest. It never requests FDIC data or changes raw files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

from bft65b.acquisition.extended import _entry, _verify_complete, full_plan
from bft65b.acquisition.full import full_paths
from bft65b.acquisition.pilot import _linkage_audit_valid, _query
from bft65b.acquisition.storage import ParquetStore
from bft65b.config import load_config
from bft65b.manifest import RunManifest


def audit(root: Path, config_file: Path, attempt: str) -> dict:
    config = load_config(config_file)
    plan = full_plan(config)
    paths = full_paths(root, config, attempt)
    manifest = RunManifest.read(paths.manifest)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    store = ParquetStore(paths.raw_root, config)
    jobs = plan.jobs(config)
    if (manifest.status != "complete" or summary.get("status") != "complete"
            or summary.get("run_id") != manifest.run_id or len(jobs) != 172
            or len(manifest.sources) != 172 or len(manifest.outputs) != 172
            or len(summary.get("quarters", [])) != 170
            or len(summary.get("small_tables", [])) != 2
            or any(not check.get("passed") for check in manifest.validations)
            or not any(check.get("name") == "plan:all_outputs" and check.get("passed")
                       for check in manifest.validations)):
        raise ValueError("Full-run manifest/summary scope or completion gate failed")
    _verify_complete(plan, config, store, manifest, summary)
    totals = Counter()
    missing = Counter()
    explicit_null = Counter()
    p3_all_null = []
    p3_source_all_null = []
    p3_source_absent = []
    indexes = set()
    for endpoint, quarter, fields, limit in jobs:
        details = _entry(summary, endpoint, quarter)
        query = _query(endpoint, fields, quarter, limit)
        source = [s for s in manifest.sources
                  if s.get("endpoint") == endpoint and s.get("query") == query]
        output_path = store.target_path(endpoint, quarter)
        output = [o for o in manifest.outputs if o.get("path") == str(output_path)]
        if len(source) != 1 or len(output) != 1:
            raise ValueError(f"Source/output multiplicity failed: {endpoint}:{quarter}")
        source, output = source[0], output[0]
        if (details["query"] != query or details["api_total"] != details["downloaded_rows"]
                or details["api_total"] != source["api_total"]
                or details["downloaded_rows"] != source["downloaded_rows"]
                or details["downloaded_rows"] != output["rows"]
                or details["index"] != source["index"]
                or details["missing_fields"] != source["missing_fields"]
                or details["null_fields"] != source["null_fields"]
                or details["conversion_anomalies"] != 0
                or details["duplicate_keys"] != 0
                or details["compression"] != "ZSTD"
                or set(details["missing_fields"]) != set(fields)
                or set(details["null_fields"]) != set(fields)):
            raise ValueError(f"Source/storage reconciliation failed: {endpoint}:{quarter}")
        physical = store.validate_file(output_path, endpoint, quarter=quarter,
                                       expected_rows=details["api_total"],
                                       expected_sha256=details["sha256"])
        if physical.rows != output["rows"] or physical.compression != "ZSTD":
            raise ValueError(f"Physical Parquet mismatch: {endpoint}:{quarter}")
        totals[endpoint] += physical.rows
        for field in fields:
            missing[f"{endpoint}.{field}"] += details["missing_fields"][field]
            explicit_null[f"{endpoint}.{field}"] += details["null_fields"][field]
        if endpoint == "financials":
            indexes.add(tuple(details["index"]))
            p3_null_count = pq.ParquetFile(output_path).read(columns=["P3ASSET"]).column("P3ASSET").null_count
            if p3_null_count == physical.rows:
                p3_all_null.append(quarter)
            if details["null_fields"]["P3ASSET"] == physical.rows:
                p3_source_all_null.append(quarter)
            if details["missing_fields"]["P3ASSET"]:
                p3_source_absent.append(quarter)
    if len(indexes) != 1:
        raise ValueError("Financials source index differs across quarters")
    failure_details = _entry(summary, "failures", None)
    if not _linkage_audit_valid(failure_details, store):
        raise ValueError("Failures nullable-CERT audit does not reconcile")
    return {
        "attempt": attempt, "status": "PASS", "jobs": len(jobs),
        "financial_quarters": len(plan.quarters), "first_quarter": plan.quarters[0],
        "last_quarter": plan.quarters[-1], "rows_by_endpoint": dict(totals),
        "field_absence_counts": dict(sorted(missing.items())),
        "explicit_null_counts": dict(sorted(explicit_null.items())),
        "conversion_anomalies": 0,
        "p3asset_physical_all_null_quarters": p3_all_null,
        "p3asset_source_explicit_all_null_quarters": p3_source_all_null,
        "p3asset_source_absent_quarters": p3_source_absent,
        "failures_unlinked_source_ids": failure_details["unresolved_linkage"]["count"],
        "manifest": str(paths.manifest), "summary": str(paths.summary),
        "linkage_audit": str(paths.linkage_audit),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only real Task 1.5 acceptance audit")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.project_root, args.config, args.attempt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
