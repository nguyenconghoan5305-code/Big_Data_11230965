"""Versioned Financials acquisition with bounded Task 1.5A execution modes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from bft65b.acquisition.client import FDICClient, FDICContractError, MAX_PAGE_SIZE
from bft65b.acquisition.pilot import (
    _atomic_json, _collect, _linkage_audit_valid, _now, _query,
)
from bft65b.acquisition.storage import ParquetStore, StorageValidationError
from bft65b.config import EXTENDED_FINANCIAL_FIELDS, ProjectConfig, load_config
from bft65b.manifest import RunManifest


def quarter_ends(start: str, end: str) -> tuple[str, ...]:
    """Enumerate inclusive calendar quarter ends in source YYYYMMDD format."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("Quarter range is reversed")
    result = []
    for year in range(first.year, last.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            quarter = date(year, month, day)
            if first <= quarter <= last:
                result.append(quarter.strftime("%Y%m%d"))
    return tuple(result)


@dataclass(frozen=True)
class AcquisitionPlan:
    name: str
    quarters: tuple[str, ...]
    include_small_tables: bool = False

    def jobs(self, config: ProjectConfig) -> tuple[tuple[str, str | None, tuple[str, ...], int], ...]:
        if not self.quarters or len(self.quarters) != len(set(self.quarters)):
            raise ValueError("Acquisition plan needs unique Financials quarters")
        if tuple(sorted(self.quarters)) != self.quarters:
            raise ValueError("Acquisition quarters must be chronological")
        for quarter in self.quarters:
            store_date = date.fromisoformat(f"{quarter[:4]}-{quarter[4:6]}-{quarter[6:]}")
            if store_date.strftime("%Y%m%d") != quarter or (store_date.month, store_date.day) not in {
                (3, 31), (6, 30), (9, 30), (12, 31)
            } or not config.financial_start <= store_date.isoformat() <= config.financial_end:
                raise ValueError(f"Invalid acquisition quarter: {quarter}")
        jobs = [("financials", quarter, config.financial_fields, config.api_page_size)
                for quarter in self.quarters]
        if self.include_small_tables:
            jobs.extend((
                ("failures", None, config.failure_fields, MAX_PAGE_SIZE),
                ("institutions", None, config.institution_fields, config.api_page_size),
            ))
        return tuple(jobs)


EXTENDED_PILOT = AcquisitionPlan(
    "extended_pilot_2023", ("20230331", "20230630", "20230930", "20231231")
)
HISTORICAL_PROFILE = AcquisitionPlan(
    "historical_profile", ("19840331", "19891231", "20091231", "20260630")
)


def full_plan(config: ProjectConfig) -> AcquisitionPlan:
    """Define, but do not invoke, the assignment's 170-quarter full scope."""
    if config.financial_fields != EXTENDED_FINANCIAL_FIELDS:
        raise ValueError("Full acquisition requires the verified extended Financials field set")
    if config.financial_start != "1984-03-31" or config.financial_end != "2026-06-30":
        raise ValueError("Full acquisition dates must match the official assignment")
    plan = AcquisitionPlan("full_1984q1_2026q2", quarter_ends(config.financial_start, config.financial_end), True)
    if len(plan.quarters) != 170 or len(plan.jobs(config)) != 172:
        raise ValueError("Full acquisition scope differs from the assignment")
    return plan


def _entry(summary: dict[str, Any], endpoint: str, quarter: str | None) -> dict[str, Any] | None:
    group = summary["quarters" if quarter else "small_tables"]
    matches = [item for item in group if item.get("endpoint") == endpoint and item.get("quarter") == quarter]
    if len(matches) > 1:
        raise StorageValidationError("Duplicate acquisition summary entry")
    return matches[0] if matches else None


def _upsert(summary: dict[str, Any], details: dict[str, Any]) -> None:
    group = summary["quarters" if details["quarter"] else "small_tables"]
    group[:] = [item for item in group if not (
        item.get("endpoint") == details["endpoint"] and item.get("quarter") == details["quarter"]
    )]
    group.append(details)


def _profile(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    types: dict[str, dict[str, int]] = {}
    for field in fields:
        counts = Counter("absent" if field not in row else
                         "null" if row[field] is None else type(row[field]).__name__
                         for row in rows)
        types[field] = dict(sorted(counts.items()))
    return {"source_type_counts": types, "conversion_anomalies": 0}


def _verify_complete(
    plan: AcquisitionPlan, config: ProjectConfig, store: ParquetStore,
    manifest: RunManifest, summary: dict[str, Any],
    on_job: Callable[[str, str | None], None] | None = None,
) -> None:
    jobs = plan.jobs(config)
    if (summary.get("plan") != plan.name or
        summary.get("planned_quarters") != list(plan.quarters) or
        len(summary.get("quarters", [])) != len(plan.quarters) or
        len(summary.get("small_tables", [])) != (2 if plan.include_small_tables else 0)):
        raise StorageValidationError("Acquisition summary does not cover the exact plan")
    for endpoint, quarter, fields, limit in jobs:
        if on_job is not None:
            on_job(endpoint, quarter)
        details = _entry(summary, endpoint, quarter)
        target = store.target_path(endpoint, quarter)
        outputs = [item for item in manifest.outputs if item.get("path") == str(target)
                   and item.get("dataset") == endpoint]
        if details is None or not outputs:
            raise StorageValidationError(f"Missing acquisition output: {endpoint}:{quarter}")
        output = outputs[-1]
        if (output.get("status") != "complete" or
            output.get("source_query") != _query(endpoint, fields, quarter, limit) or
            output.get("source_total") != details.get("api_total") or
            output.get("rows") != details.get("downloaded_rows") or
            output.get("schema_version") != store.schema_version(endpoint) or
            output.get("sha256") != details.get("sha256") or
            details.get("duplicate_keys") != 0 or details.get("completeness") != "pass"):
            raise StorageValidationError(f"Acquisition metadata differs: {endpoint}:{quarter}")
        store.validate_file(target, endpoint, quarter=quarter,
                            expected_rows=output["source_total"], expected_sha256=output["sha256"])
    if plan.include_small_tables:
        if on_job is not None:
            on_job("failures", None)
        if not _linkage_audit_valid(_entry(summary, "failures", None), store):
            raise StorageValidationError("Failures linkage audit is missing or changed")


def run_plan(
    *, plan: AcquisitionPlan, config: ProjectConfig, client: FDICClient,
    store: ParquetStore, manifest_path: Path, summary_path: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one bounded plan; the full plan remains disabled in Task 1.5A."""
    if plan.name == "full_1984q1_2026q2":
        raise ValueError("Full 170-quarter acquisition is disabled during Task 1.5A")
    bounded_quarters = set(EXTENDED_PILOT.quarters) | set(HISTORICAL_PROFILE.quarters)
    if len(plan.quarters) > 4 or not set(plan.quarters).issubset(bounded_quarters):
        raise ValueError("Only the eight approved bounded quarters may run during Task 1.5A")
    if config.financial_fields != EXTENDED_FINANCIAL_FIELDS or client.config != config or store.config != config:
        raise ValueError("Extended acquisition requires one matching verified configuration")
    jobs = plan.jobs(config)
    if "fdic_extended_v2" not in store.target_path("financials", plan.quarters[0]).parts:
        raise ValueError("Extended acquisition must use a separate schema namespace")
    manifest_path, summary_path = Path(manifest_path), Path(summary_path)
    if manifest_path.name.startswith("task_1_3") or summary_path.name.startswith("task_1_3"):
        raise ValueError("Accepted Task 1.3 manifests are protected")
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Manifest exists; use resume or a new run path")
        manifest = RunManifest.read(manifest_path)
        if (json.dumps(manifest.config, sort_keys=True) != json.dumps(config.to_dict(), sort_keys=True)
                or manifest.status not in {"running", "complete"}):
            raise StorageValidationError("Manifest configuration/status cannot be resumed")
        if not summary_path.exists():
            raise StorageValidationError("Resume summary is missing")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("run_id") != manifest.run_id or summary.get("plan") != plan.name:
            raise StorageValidationError("Summary and manifest differ")
        if manifest.status == "complete":
            if summary.get("status") != "complete":
                raise StorageValidationError("Completed manifest has incomplete summary")
            _verify_complete(plan, config, store, manifest, summary)
            for endpoint, quarter, _, _ in jobs:
                previous = _entry(summary, endpoint, quarter)
                probe = client.fetch_page(endpoint, filters=f"REPDTE:{quarter}" if quarter else None, limit=1)
                if probe.total != previous["api_total"] or list(probe.index or ()) != previous["index"]:
                    raise FDICContractError("Source total or index changed; start a new run")
            return summary
    else:
        if resume and summary_path.exists():
            raise StorageValidationError("Summary exists without a manifest")
        manifest = RunManifest.new(f"fdic-{plan.name}-{_now()}", config)
        manifest.set_status("running")
        summary = {"run_id": manifest.run_id, "plan": plan.name,
                   "planned_quarters": list(plan.quarters), "quarters": [], "small_tables": [],
                   "started_at_utc": manifest.created_at_utc}
    manifest.write(manifest_path)
    summary["status"] = "running"
    _atomic_json(summary_path, summary)
    label = "initialization"
    try:
        for endpoint, quarter, fields, limit in jobs:
            label = f"{endpoint}:{quarter or 'all'}"
            query = _query(endpoint, fields, quarter, limit)
            previous = _entry(summary, endpoint, quarter)
            if (resume and previous is not None and previous.get("query") == query
                and previous.get("index") is not None and
                store.can_skip(endpoint, manifest_path, quarter=quarter,
                               source_total=previous.get("api_total"), source_query=query)):
                probe = client.fetch_page(endpoint, filters=f"REPDTE:{quarter}" if quarter else None, limit=1)
                if probe.total != previous["api_total"] or list(probe.index or ()) != previous["index"]:
                    raise FDICContractError(f"Source total or index changed for {label}")
                if endpoint != "failures" or _linkage_audit_valid(previous, store):
                    details = dict(previous, recovery_action="skipped")
                    manifest.record_validation(f"resume:{label}", True, "Source and stored output passed")
                    manifest.write(manifest_path)
                    _upsert(summary, details)
                    _atomic_json(summary_path, summary)
                    continue
            manifest.stages[f"acquisition:{label}"] = "running"
            manifest.write(manifest_path)
            rows, details = _collect(client, endpoint, fields, quarter, limit)
            offsets = details["page_offsets"]
            details["page_row_counts"] = [end - start for start, end in
                                          zip(offsets, offsets[1:] + [details["api_total"]])]
            details.update(_profile(rows, fields))
            details["explicit_null_rates"] = {
                field: details["null_fields"][field] / details["api_total"] for field in fields
            }
            manifest.record_source(
                endpoint=endpoint, query=query, total=details["api_total"],
                downloaded=details["downloaded_rows"], pages=details["pages"],
                index=details["index"], started_at_utc=details["started_at_utc"],
                finished_at_utc=details["finished_at_utc"],
                missing_fields=details["missing_fields"], null_fields=details["null_fields"],
            )
            manifest.stages[f"acquisition:{label}"] = "complete"
            manifest.record_validation(f"acquisition:{label}", True,
                                       f"total={details['api_total']}; rows={len(rows)}; pages={details['pages']}")
            manifest.write(manifest_path)
            result = store.write(endpoint, rows, quarter=quarter,
                                 source_total=details["api_total"], manifest=manifest,
                                 manifest_path=manifest_path, source_query=query)
            check = store.validate_file(result.path, endpoint, quarter=quarter,
                                        expected_rows=details["api_total"],
                                        expected_sha256=result.sha256)
            details.update(parquet_path=str(result.path), parquet_bytes=result.path.stat().st_size,
                           sha256=check.sha256, compression=check.compression,
                           read_back="pass", completeness="pass", duplicate_keys=0,
                           recovery_action="skipped" if result.skipped else
                           "reconciled" if result.reconciled else "written")
            if endpoint == "failures":
                unresolved = sorted(row["ID"] for row in rows if row.get("CERT") is None)
                if len(unresolved) != details["null_fields"]["CERT"] + details["missing_fields"]["CERT"]:
                    raise FDICContractError("Failures linkage audit count differs")
                linkage_path = summary_path.with_name(summary_path.stem + "_unlinked_ids.json")
                _atomic_json(linkage_path, {"source_query": query,
                                           "api_total": details["api_total"],
                                           "unlinked_count": len(unresolved),
                                           "source_ids": unresolved})
                details["unresolved_linkage"] = {
                    "count": len(unresolved), "source_ids_path": str(linkage_path),
                    "sha256": hashlib.sha256(linkage_path.read_bytes()).hexdigest(),
                }
                manifest.record_validation("failures:unresolved_linkage_audit", True,
                                           f"count={len(unresolved)}; sha256={details['unresolved_linkage']['sha256']}")
            manifest.record_validation(f"plan:{label}", True, "Source, schema, read-back and SHA-256 pass")
            manifest.write(manifest_path)
            _upsert(summary, details)
            _atomic_json(summary_path, summary)
        _verify_complete(plan, config, store, manifest, summary)
        manifest.record_validation("plan:all_outputs", True, f"{len(jobs)} outputs validated")
        manifest.set_status("complete")
        manifest.write(manifest_path)
        summary["status"] = "complete"
        summary["finished_at_utc"] = _now()
        _atomic_json(summary_path, summary)
        return summary
    except Exception as exc:
        manifest.stages[f"acquisition:{label}"] = "failed"
        manifest.record_validation("plan:failed_gate", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(manifest_path)
        summary.update(status="failed", failed_stage=label,
                       error=f"{type(exc).__name__}: {exc}", finished_at_utc=_now())
        _atomic_json(summary_path, summary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only bounded extended Financials checks")
    parser.add_argument("mode", choices=("extended-pilot", "historical-profile"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = (load_config(args.config) if args.config else ProjectConfig(
        financial_fields=EXTENDED_FINANCIAL_FIELDS, data_root="data_raw/task_1_5a",
    ))
    plan = EXTENDED_PILOT if args.mode == "extended-pilot" else HISTORICAL_PROFILE
    paths = config.paths(args.project_root)
    default_name = "task_1_5a_" + plan.name
    summary = run_plan(
        plan=plan, config=config, client=FDICClient(config),
        store=ParquetStore(paths["raw"], config),
        manifest_path=args.manifest or paths["runs"] / f"{default_name}.json",
        summary_path=args.summary or paths["runs"] / f"{default_name}_summary.json",
        resume=args.resume,
    )
    print(json.dumps({"plan": plan.name, "status": summary["status"],
                      "quarters": len(summary["quarters"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
