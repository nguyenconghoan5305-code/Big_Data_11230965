"""Bounded 2023 FDIC acquisition pilot with fail-closed acceptance gates."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlencode

import pyarrow.parquet as pq

from bft65b.acquisition.client import BASE_URL, FDICClient, FDICContractError, MAX_PAGE_SIZE, Page
from bft65b.acquisition.storage import ParquetStore, SCHEMA_VERSIONS, StorageValidationError
from bft65b.config import ProjectConfig, load_config
from bft65b.manifest import RunManifest


PILOT_QUARTERS = ("20230331", "20230630", "20230930", "20231231")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query(endpoint: str, fields: tuple[str, ...], quarter: str | None, limit: int) -> str:
    params = {
        "fields": ",".join(fields), "limit": limit, "sort_by": "CERT",
        "sort_order": "ASC", "format": "json",
    }
    if quarter is not None:
        params["filters"] = f"REPDTE:{quarter}"
    return f"{BASE_URL}/{endpoint}?{urlencode(params)}"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _jobs(config: ProjectConfig) -> list[tuple[str, str | None, tuple[str, ...], int]]:
    return [
        ("financials", quarter, config.financial_fields, config.api_page_size)
        for quarter in PILOT_QUARTERS
    ] + [
        ("failures", None, config.failure_fields, MAX_PAGE_SIZE),
        ("institutions", None, config.institution_fields, config.api_page_size),
    ]


def _summary_entry(summary: dict[str, Any], endpoint: str, quarter: str | None) -> dict[str, Any] | None:
    matches = [item for item in summary["quarters" if quarter else "small_tables"]
               if item.get("endpoint") == endpoint and item.get("quarter") == quarter]
    if len(matches) > 1:
        raise StorageValidationError("Duplicate pilot summary entry")
    return matches[0] if matches else None


def _upsert_summary(summary: dict[str, Any], details: dict[str, Any]) -> None:
    collection = summary["quarters" if details["quarter"] else "small_tables"]
    collection[:] = [item for item in collection if not (
        item.get("endpoint") == details["endpoint"] and
        item.get("quarter") == details["quarter"]
    )]
    collection.append(details)


def _linkage_audit_valid(details: dict[str, Any], store: ParquetStore) -> bool:
    linkage = details.get("unresolved_linkage", {})
    path = Path(linkage.get("source_ids_path", ""))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != linkage.get("sha256"):
        return False
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
        ids = audit["source_ids"]
        rows = pq.ParquetFile(store.target_path("failures")).read(columns=["ID", "CERT"]).to_pylist()
    except (OSError, ValueError, KeyError):
        return False
    return (audit.get("source_query") == details.get("query") and
            audit.get("api_total") == details.get("api_total") and
            audit.get("unlinked_count") == linkage.get("count") and
            isinstance(ids, list) and len(ids) == linkage.get("count") and
            len(ids) == len(set(ids)) and
            set(ids) == {row["ID"] for row in rows if row["CERT"] is None})


def _verify_completion(
    config: ProjectConfig, store: ParquetStore, manifest: RunManifest,
    summary: dict[str, Any],
) -> None:
    """Check all six physical outputs before the pilot run can be complete."""
    jobs = _jobs(config)
    if len(summary["quarters"]) != 4 or len(summary["small_tables"]) != 2:
        raise StorageValidationError("Pilot summary is missing one or more required datasets")
    for endpoint, quarter, fields, limit in jobs:
        details = _summary_entry(summary, endpoint, quarter)
        target = store.target_path(endpoint, quarter)
        query = _query(endpoint, fields, quarter, limit)
        outputs = [item for item in manifest.outputs
                   if item.get("path") == str(target) and item.get("dataset") == endpoint]
        if details is None or not outputs:
            raise StorageValidationError(f"Pilot output missing: {endpoint}:{quarter}")
        output = outputs[-1]
        if (output.get("status") != "complete" or output.get("source_query") != query or
            output.get("source_total") != details.get("api_total") or
            output.get("rows") != details.get("downloaded_rows") or
            output.get("schema_version") != SCHEMA_VERSIONS[endpoint] or
            output.get("sha256") != details.get("sha256") or
            details.get("completeness") != "pass" or details.get("duplicate_keys") != 0):
            raise StorageValidationError(f"Pilot output metadata differs: {endpoint}:{quarter}")
        store.validate_file(target, endpoint, quarter=quarter,
                            expected_rows=output["source_total"],
                            expected_sha256=output["sha256"])
    failure_details = _summary_entry(summary, "failures", None)
    if not _linkage_audit_valid(failure_details, store):
        raise StorageValidationError("Failures unresolved-linkage audit is missing or changed")


def _collect(
    client: FDICClient, endpoint: str, fields: tuple[str, ...],
    quarter: str | None, limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = _now()
    pages: list[Page] = []
    rows: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    nulls: Counter[str] = Counter()
    seen: set[tuple[str, ...]] = set()
    duplicate_keys = 0
    filters = f"REPDTE:{quarter}" if quarter else None
    for page in client.iter_pages(endpoint, filters=filters, limit=limit, require_nonempty=True):
        if endpoint == "failures" and page.total > MAX_PAGE_SIZE:
            raise FDICContractError(
                "Failures exceed one API page; verify an official bulk or partitioned strategy"
            )
        pages.append(page)
        for row, absent, explicit_null in zip(page.rows, page.missing_fields, page.null_fields):
            key = client._row_key(endpoint, row)
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
            missing.update(absent)
            nulls.update(explicit_null)
            rows.append(row)
    if not pages:
        raise FDICContractError("No API pages returned")
    total = pages[0].total
    if len(rows) != total or duplicate_keys:
        raise FDICContractError(
            f"{endpoint} count/key gate failed: total={total}, rows={len(rows)}, duplicates={duplicate_keys}"
        )
    if endpoint == "failures" and len(pages) != 1:
        raise FDICContractError("Failures must fit in one verified API page")
    details = {
        "endpoint": endpoint, "quarter": quarter, "query": _query(endpoint, fields, quarter, limit),
        "api_total": total, "downloaded_rows": len(rows), "duplicate_keys": duplicate_keys,
        "pages": len(pages), "page_offsets": [page.offset for page in pages],
        "index": list(pages[0].index) if pages[0].index is not None else None,
        "started_at_utc": started, "finished_at_utc": _now(),
        "missing_fields": {field: missing[field] for field in fields},
        "null_fields": {field: nulls[field] for field in fields},
    }
    return rows, details


def run_pilot(
    *, config: ProjectConfig, client: FDICClient, store: ParquetStore,
    manifest_path: Path, summary_path: Path, resume: bool = False,
) -> dict[str, Any]:
    """Acquire the four approved quarters and audit both small endpoints."""
    if client.config != config or store.config != config:
        raise ValueError("Pilot, client, and store configuration must agree")
    manifest_path, summary_path = Path(manifest_path), Path(summary_path)
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Pilot manifest exists; use --resume or a new path")
        manifest = RunManifest.read(manifest_path)
        if json.dumps(manifest.config, sort_keys=True) != json.dumps(config.to_dict(), sort_keys=True):
            raise StorageValidationError("Existing pilot manifest configuration differs")
        if manifest.status not in {"running", "complete"}:
            raise StorageValidationError("Failed or pending pilot manifest cannot be resumed; use a new run path")
        summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                   if summary_path.exists() else {
                       "run_id": manifest.run_id, "quarters": [], "small_tables": [],
                       "started_at_utc": manifest.created_at_utc,
                   })
        if summary.get("run_id") != manifest.run_id:
            raise StorageValidationError("Pilot summary and manifest run IDs differ")
        if manifest.status == "complete":
            if not summary_path.exists() or summary.get("status") != "complete":
                raise StorageValidationError("Completed pilot summary is missing or incomplete")
            _verify_completion(config, store, manifest, summary)
            for endpoint, quarter, fields, limit in _jobs(config):
                details = _summary_entry(summary, endpoint, quarter)
                probe = client.fetch_page(
                    endpoint, filters=f"REPDTE:{quarter}" if quarter else None, limit=1,
                )
                probe_index = list(probe.index) if probe.index is not None else None
                if probe.total != details["api_total"] or probe_index != details["index"]:
                    raise FDICContractError(
                        f"Source total or index changed for {endpoint}:{quarter or 'all'}; "
                        "start a new run"
                    )
            return summary
        manifest.set_status("running")
    else:
        if resume and summary_path.exists():
            raise StorageValidationError("Summary exists without a persisted pilot manifest")
        manifest = RunManifest.new(f"fdic-pilot-2023-{_now()}", config)
        manifest.set_status("running")
        summary = {
            "run_id": manifest.run_id, "quarters": [], "small_tables": [],
            "started_at_utc": manifest.created_at_utc,
        }
    manifest.write(manifest_path)
    summary["status"] = "running"
    _atomic_json(summary_path, summary)
    label = "initialization"
    try:
        for endpoint, quarter, fields, limit in _jobs(config):
            label = f"{endpoint}:{quarter or 'all'}"
            query = _query(endpoint, fields, quarter, limit)
            previous = _summary_entry(summary, endpoint, quarter)
            if (resume and previous is not None and previous.get("query") == query
                and previous.get("index") is not None and
                store.can_skip(endpoint, manifest_path, quarter=quarter,
                               source_total=previous.get("api_total"), source_query=query)):
                filter_value = f"REPDTE:{quarter}" if quarter else None
                probe = client.fetch_page(endpoint, filters=filter_value, limit=1)
                probe_index = list(probe.index) if probe.index is not None else None
                if probe.total != previous["api_total"] or probe_index != previous["index"]:
                    raise FDICContractError(f"Source total or index changed for {label}; cannot skip")
                if endpoint != "failures" or _linkage_audit_valid(previous, store):
                    details = dict(previous)
                    details.update(storage_skipped=True, storage_reconciled=False,
                                   recovery_action="skipped")
                    manifest.record_validation(f"resume:{label}", True,
                                               "Persisted source metadata and Parquet checks passed")
                    manifest.write(manifest_path)
                    _upsert_summary(summary, details)
                    _atomic_json(summary_path, summary)
                    continue
            manifest.stages[f"acquisition:{label}"] = "running"
            manifest.write(manifest_path)
            rows, details = _collect(client, endpoint, fields, quarter, limit)
            manifest.record_source(
                endpoint=endpoint, query=details["query"], total=details["api_total"],
                downloaded=details["downloaded_rows"], pages=details["pages"],
                index=details["index"], started_at_utc=details["started_at_utc"],
                finished_at_utc=details["finished_at_utc"],
                missing_fields=details["missing_fields"], null_fields=details["null_fields"],
            )
            manifest.stages[f"acquisition:{label}"] = "complete"
            manifest.record_validation(
                f"acquisition:{label}", True,
                f"api_total={details['api_total']}; downloaded={len(rows)}; duplicate_keys=0; pages={details['pages']}",
            )
            manifest.write(manifest_path)
            result = store.write(
                endpoint, rows, quarter=quarter, source_total=details["api_total"],
                manifest=manifest, manifest_path=manifest_path, source_query=details["query"],
            )
            check = store.validate_file(
                result.path, endpoint, quarter=quarter,
                expected_rows=details["api_total"], expected_sha256=result.sha256,
            )
            details.update({
                "parquet_path": str(result.path), "parquet_bytes": result.path.stat().st_size,
                "sha256": check.sha256, "read_back": "pass", "compression": check.compression,
                "completeness": "pass", "storage_skipped": result.skipped,
                "storage_reconciled": result.reconciled,
                "recovery_action": "reconciled" if result.reconciled else "written",
            })
            if endpoint == "failures":
                unresolved = sorted(row["ID"] for row in rows if row.get("CERT") is None)
                if len(unresolved) != (details["null_fields"]["CERT"] +
                                       details["missing_fields"]["CERT"]):
                    raise FDICContractError("Unlinked Failures count disagrees with field audit")
                linkage_path = summary_path.with_name(summary_path.stem + "_unlinked_ids.json")
                _atomic_json(linkage_path, {
                    "source_query": details["query"], "api_total": details["api_total"],
                    "unlinked_count": len(unresolved), "source_ids": unresolved,
                })
                linkage_sha = hashlib.sha256(linkage_path.read_bytes()).hexdigest()
                details["unresolved_linkage"] = {
                    "count": len(unresolved), "source_ids_path": str(linkage_path),
                    "sha256": linkage_sha,
                }
                manifest.record_validation(
                    "failures:unresolved_linkage_audit", True,
                    f"count={len(unresolved)}; path={linkage_path}; sha256={linkage_sha}",
                )
            manifest.record_validation(f"pilot:{label}", True, "API total, keys, typed read-back and SHA-256 pass")
            manifest.write(manifest_path)
            _upsert_summary(summary, details)
            _atomic_json(summary_path, summary)
        _verify_completion(config, store, manifest, summary)
        manifest.record_validation("pilot:all_six_outputs", True,
                                   "Four Financials quarters, Failures, and Institutions validated")
        manifest.set_status("complete")
        manifest.write(manifest_path)
        summary["status"] = "complete"
        summary["finished_at_utc"] = _now()
        _atomic_json(summary_path, summary)
        return summary
    except Exception as exc:
        manifest.stages[f"acquisition:{label}"] = "failed"
        manifest.record_validation("pilot:failed_gate", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(manifest_path)
        summary["status"] = "failed"
        summary["failed_stage"] = label
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["finished_at_utc"] = _now()
        _atomic_json(summary_path, summary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded FDIC 2023 four-quarter pilot")
    parser.add_argument("--config", type=Path, help="Optional JSON project configuration")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, help="Ignored manifest path; use a new path to preserve prior runs")
    parser.add_argument("--summary", type=Path, help="Ignored summary path")
    parser.add_argument("--resume", action="store_true", help="Resume a running or completed manifest after verification")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config.paths(args.project_root)
    summary = run_pilot(
        config=config, client=FDICClient(config), store=ParquetStore(paths["raw"], config),
        manifest_path=args.manifest or paths["runs"] / "task_1_3_pilot.json",
        summary_path=args.summary or paths["runs"] / "task_1_3_pilot_summary.json",
        resume=args.resume,
    )
    print(json.dumps({"status": summary["status"], "summary": str(args.summary or paths["runs"] / "task_1_3_pilot_summary.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
