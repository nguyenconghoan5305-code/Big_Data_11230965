"""Fail-closed, attempt-versioned full FDIC acquisition runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

from bft65b.acquisition.client import FDICClient, FDICContractError
from bft65b.acquisition.extended import (
    AcquisitionPlan, _entry, _profile, _upsert, _verify_complete, full_plan,
)
from bft65b.acquisition.pilot import (
    _atomic_json, _collect, _linkage_audit_valid, _now, _query,
)
from bft65b.acquisition.storage import (
    CorruptParquetError, ParquetStore, StorageConflictError, StorageValidationError,
)
from bft65b.config import EXTENDED_FINANCIAL_FIELDS, ProjectConfig, load_config
from bft65b.manifest import RunManifest


# Operational pacing, not a claim about FDIC's unpublished rate-limit window.
FULL_PROBE_INTERVAL_SECONDS = 3.2
FULL_MAX_ATTEMPTS = 5
FULL_BACKOFF_SECONDS = 5.0
FULL_MAX_BACKOFF_SECONDS = 60.0


class _ProbePacer:
    """Space one-row source probes without changing their validation contract."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.next_at: float | None = None

    def fetch(self, client: FDICClient, endpoint: str, quarter: str | None):
        if self.next_at is not None:
            delay = self.next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        try:
            return client.fetch_page(endpoint,
                                     filters=f"REPDTE:{quarter}" if quarter else None,
                                     limit=1)
        finally:
            self.next_at = time.monotonic() + self.interval_seconds


@dataclass(frozen=True)
class FullPaths:
    project_root: Path
    raw_root: Path
    namespace_root: Path
    manifest: Path
    summary: Path
    linkage_audit: Path

    @property
    def lock_path(self) -> Path:
        return self.raw_root.parent / ".task_1_5_full.lock"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_config(manifest: RunManifest, config: ProjectConfig) -> bool:
    return json.dumps(manifest.config, sort_keys=True) == json.dumps(config.to_dict(), sort_keys=True)


def _require_index(index: Any, label: str) -> list[str]:
    if (not isinstance(index, (tuple, list)) or len(index) != 2
            or any(not isinstance(value, str) or not value.strip() for value in index)):
        raise FDICContractError(f"Missing or incomplete source index fingerprint: {label}")
    return list(index)


def full_paths(project_root: Path, config: ProjectConfig, attempt: str) -> FullPaths:
    """Resolve and reject any path that could alias an accepted pilot artifact."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", attempt):
        raise ValueError("Attempt ID must be 1-64 letters, digits, underscores, or hyphens")
    if config.financial_fields != EXTENDED_FINANCIAL_FIELDS or not config.fields_verified:
        raise ValueError("Full run requires the exact verified 26-field Financials configuration")
    plan = full_plan(config)
    if len(plan.jobs(config)) != 172:
        raise ValueError("Full acquisition requires exactly 172 jobs")
    if (Path(config.data_root).parts != ("data_raw", "task_1_5_full")
            or Path(config.run_root).parts != ("runs",)):
        raise ValueError("Full run requires dedicated data_raw/task_1_5_full and runs roots")
    project_root = Path(project_root).resolve(strict=True)
    expected_raw = (project_root / "data_raw" / "task_1_5_full").resolve()
    actual_raw = config.paths(project_root)["raw"]
    if actual_raw != expected_raw:
        raise ValueError("BFT65B_DATA_ROOT or path resolution redirects the protected full raw root")
    namespace = (actual_raw / "fdic_extended_v2").resolve()
    protected_roots = (
        (project_root / "data_raw" / "fdic").resolve(),
        (project_root / "data_raw" / "task_1_5a").resolve(),
    )
    if any(namespace == protected or namespace.is_relative_to(protected)
           or protected.is_relative_to(namespace) for protected in protected_roots):
        raise ValueError("Full raw output aliases an accepted pilot root")
    runs = (project_root / "runs").resolve()
    prefix = f"task_1_5_full_{attempt}"
    manifest = runs / f"{prefix}.json"
    summary = runs / f"{prefix}_summary.json"
    audit = runs / f"{prefix}_summary_unlinked_ids.json"
    protected_files = {
        (runs / name).resolve() for name in (
            "task_1_3_pilot_resume3.json",
            "task_1_3_pilot_resume3_summary.json",
            "task_1_3_pilot_resume3_summary_unlinked_ids.json",
            "task_1_5a_historical_profile.json",
            "task_1_5a_historical_profile_summary.json",
            "task_1_5a_extended_pilot_2023.json",
            "task_1_5a_extended_pilot_2023_summary.json",
        )
    }
    if any(path.resolve() in protected_files or path.resolve().parent != runs
           for path in (manifest, summary, audit)):
        raise ValueError("Full run manifest/summary/audit path aliases a protected artifact")
    return FullPaths(project_root, actual_raw, namespace, manifest, summary, audit)


def dry_run(project_root: Path, config: ProjectConfig, attempt: str,
            recover_from: str | None = None) -> dict[str, Any]:
    """Return the entire exact plan and resolved destinations without network or writes."""
    paths = full_paths(project_root, config, attempt)
    plan = full_plan(config)
    previous = None
    if recover_from is not None:
        previous = full_paths(project_root, config, recover_from)
        if previous.manifest == paths.manifest:
            raise ValueError("Recovery attempt must use a new attempt ID")
    store = ParquetStore(paths.raw_root, config)
    jobs = plan.jobs(config)
    return {
        "mode": "dry-run", "network_requests": 0, "plan": plan.name,
        "probe_interval_seconds": FULL_PROBE_INTERVAL_SECONDS,
        "api_retry": {"max_attempts": FULL_MAX_ATTEMPTS,
                      "backoff_seconds": FULL_BACKOFF_SECONDS,
                      "max_backoff_seconds": FULL_MAX_BACKOFF_SECONDS},
        "financial_quarters": len(plan.quarters), "total_jobs": len(jobs),
        "first_quarter": plan.quarters[0], "last_quarter": plan.quarters[-1],
        "modelling_end": config.modelling_end,
        "field_sets": {"financials": list(config.financial_fields),
                       "failures": list(config.failure_fields),
                       "institutions": list(config.institution_fields)},
        "raw_root": str(paths.raw_root), "namespace_root": str(paths.namespace_root),
        "first_output": str(store.target_path("financials", plan.quarters[0])),
        "last_output": str(store.target_path("financials", plan.quarters[-1])),
        "failures_output": str(store.target_path("failures")),
        "institutions_output": str(store.target_path("institutions")),
        "manifest": str(paths.manifest), "summary": str(paths.summary),
        "linkage_audit": str(paths.linkage_audit),
        "lock_path": str(paths.lock_path),
        "recovery_from": str(previous.manifest) if previous else None,
        "jobs": [{"endpoint": endpoint, "quarter": quarter, "limit": limit}
                 for endpoint, quarter, _, limit in jobs],
    }


def _load_previous(
    previous_paths: FullPaths | None, config: ProjectConfig, plan: AcquisitionPlan,
) -> tuple[RunManifest | None, dict[str, Any] | None]:
    if previous_paths is None:
        return None, None
    if not previous_paths.manifest.is_file():
        raise FileNotFoundError("Recovery source manifest does not exist")
    previous = RunManifest.read(previous_paths.manifest)
    if not _same_config(previous, config) or previous.status not in {"running", "failed", "complete"}:
        raise StorageValidationError("Recovery source manifest configuration/status differs")
    summary = (json.loads(previous_paths.summary.read_text(encoding="utf-8"))
               if previous_paths.summary.is_file() else None)
    if summary is not None and (
        summary.get("run_id") != previous.run_id or
        summary.get("plan") != plan.name or
        summary.get("planned_quarters") != list(plan.quarters)
    ):
        raise StorageValidationError("Recovery source summary scope/run ID differs")
    return previous, summary


def _previous_candidate(
    *, previous: RunManifest | None, previous_summary: dict[str, Any] | None,
    previous_paths: FullPaths | None, client: FDICClient, pacer: _ProbePacer,
    store: ParquetStore,
    endpoint: str, quarter: str | None, fields: tuple[str, ...], limit: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Inspect one old output independently; a failed run is never a skip permit."""
    if previous is None or previous_summary is None:
        return None
    details = _entry(previous_summary, endpoint, quarter)
    target = store.target_path(endpoint, quarter)
    outputs = [item for item in previous.outputs if isinstance(item, dict)
               and item.get("path") == str(target) and item.get("dataset") == endpoint]
    if details is None or not outputs:
        return None  # A file promoted before manifest persistence must be re-fetched.
    output = outputs[-1]
    query = _query(endpoint, fields, quarter, limit)
    sources = [item for item in previous.sources if isinstance(item, dict)
               and item.get("endpoint") == endpoint and item.get("query") == query]
    if not sources:
        return None
    source = sources[-1]
    label = f"{endpoint}:{quarter or 'all'}"
    required_validations = {
        f"acquisition:{label}", f"storage:{endpoint}:{quarter or 'all'}", f"plan:{label}",
    }
    passing = {item.get("name") for item in previous.validations
               if isinstance(item, dict) and item.get("passed")}
    failed_for_job = any(
        isinstance(item, dict) and item.get("name") in required_validations
        and not item.get("passed") for item in previous.validations
    )
    if (failed_for_job or not required_validations.issubset(passing)
            or previous.stages.get(f"acquisition:{label}") != "complete"
            or previous.stages.get(f"storage:{endpoint}:{f'REPDTE={quarter}' if quarter else 'all'}") != "complete"):
        return None
    if (output.get("status") != "complete" or output.get("source_query") != query or
        output.get("schema_version") != store.schema_version(endpoint) or
        output.get("compression") != store.config.parquet_compression.upper() or
        output.get("partition") != (f"REPDTE={quarter}" if quarter else None) or
        type(output.get("source_total")) is not int or output["source_total"] < 1 or
        output.get("rows") != output["source_total"] or
        details.get("query") != query or details.get("api_total") != output["source_total"] or
        details.get("downloaded_rows") != output["rows"] or
        details.get("sha256") != output.get("sha256") or
        details.get("completeness") != "pass" or details.get("duplicate_keys") != 0 or
        source.get("api_total") != output["source_total"] or
        source.get("downloaded_rows") != output["rows"] or
        source.get("index") != details.get("index") or
        source.get("pages") != details.get("pages") or
        source.get("missing_fields") != details.get("missing_fields") or
        source.get("null_fields") != details.get("null_fields") or
        details.get("parquet_path") != str(target) or
        details.get("read_back") != "pass" or
        details.get("compression") != output.get("compression") or
        type(details.get("pages")) is not int or details["pages"] < 1 or
        not isinstance(details.get("page_offsets"), list) or
        len(details["page_offsets"]) != details.get("pages") or
        not isinstance(details.get("page_row_counts"), list) or
        len(details["page_row_counts"]) != details.get("pages") or
        any(type(n) is not int or n < 1 for n in details["page_row_counts"]) or
        sum(details["page_row_counts"]) != details.get("api_total") or
        details["page_offsets"] != [sum(details["page_row_counts"][:i])
                                    for i in range(details["pages"])] or
        set(details.get("missing_fields", {})) != set(fields) or
        set(details.get("null_fields", {})) != set(fields) or
        not isinstance(details.get("source_type_counts"), dict) or
        set(details["source_type_counts"]) != set(fields) or
        details.get("conversion_anomalies") != 0):
        raise StorageConflictError(f"Recovery manifest/source metadata conflict: {label}")
    _require_index(details.get("index"), label)
    if not target.is_file():
        return None
    try:
        check = store.validate_file(target, endpoint, quarter=quarter,
                                    expected_rows=output["rows"])
    except CorruptParquetError:
        return None  # The later full re-fetch repairs through atomic promotion.
    except StorageValidationError as exc:
        raise StorageConflictError(f"Readable recovery file violates schema/keys/codec: {label}") from exc
    if check.sha256 != output["sha256"]:
        raise StorageConflictError(f"Recovery checksum conflicts with healthy file: {label}")
    if target.stat().st_size != details.get("parquet_bytes"):
        raise StorageConflictError(f"Recovery file size differs from manifest: {label}")
    if endpoint == "failures":
        audit = details.get("unresolved_linkage", {})
        if Path(audit.get("source_ids_path", "")).resolve() != previous_paths.linkage_audit.resolve():
            raise StorageConflictError("Failures linkage audit path differs from recovery attempt")
        if not _linkage_audit_valid(details, store):
            return None  # Re-fetch and regenerate the audit, preserving prior evidence.
    probe = pacer.fetch(client, endpoint, quarter)
    if probe.total != details["api_total"] or _require_index(probe.index, label) != details["index"]:
        raise FDICContractError(f"Source total or index changed for {label}; cannot inherit")
    return details, source, output


def _record_inherited(
    *, manifest: RunManifest, summary: dict[str, Any], manifest_path: Path,
    summary_path: Path, audit_path: Path, previous_paths: FullPaths,
    store: ParquetStore, endpoint: str, quarter: str | None,
    candidate: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    details, source, output = candidate
    label = f"{endpoint}:{quarter or 'all'}"
    inherited = dict(details, recovery_action="inherited",
                     inherited_from=str(previous_paths.manifest))
    if endpoint == "failures":
        previous_audit = json.loads(previous_paths.linkage_audit.read_text(encoding="utf-8"))
        _atomic_json(audit_path, previous_audit)
        inherited["unresolved_linkage"] = {
            "count": previous_audit["unlinked_count"], "source_ids_path": str(audit_path),
            "sha256": _sha256(audit_path),
        }
        if not _linkage_audit_valid(inherited, store):
            raise StorageValidationError("Inherited Failures linkage audit does not reconcile")
        manifest.record_validation("failures:unresolved_linkage_audit", True,
                                   f"count={previous_audit['unlinked_count']}; sha256={_sha256(audit_path)}")
    manifest.record_source(
        endpoint=endpoint, query=source["query"], total=source["api_total"],
        downloaded=source["downloaded_rows"], pages=source["pages"],
        index=source["index"], started_at_utc=source["started_at_utc"],
        finished_at_utc=source["finished_at_utc"],
        missing_fields=source["missing_fields"], null_fields=source["null_fields"],
    )
    manifest.stages[f"acquisition:{label}"] = "complete"
    manifest.record_validation(f"acquisition:{label}", True,
                               "Inherited after independent source and Parquet verification")
    manifest.record_output(
        path=output["path"], dataset=endpoint, rows=output["rows"],
        sha256=output["sha256"], compression=output["compression"],
        schema_version=output["schema_version"], partition=output["partition"],
        source_total=output["source_total"], source_query=output["source_query"],
    )
    manifest.record_validation(f"storage:{endpoint}:{quarter or 'all'}", True,
                               "Inherited healthy immutable bytes with matching SHA-256")
    manifest.record_validation(f"plan:{label}", True, "Source and output inherited safely")
    manifest.write(manifest_path)
    _upsert(summary, inherited)
    _atomic_json(summary_path, summary)
    return inherited


def _fetch_job(
    *, config: ProjectConfig, client: FDICClient, store: ParquetStore,
    manifest: RunManifest, manifest_path: Path, summary: dict[str, Any],
    summary_path: Path, audit_path: Path, endpoint: str, quarter: str | None,
    fields: tuple[str, ...], limit: int,
) -> dict[str, Any]:
    label = f"{endpoint}:{quarter or 'all'}"
    query = _query(endpoint, fields, quarter, limit)
    manifest.stages[f"acquisition:{label}"] = "running"
    manifest.write(manifest_path)
    rows, details = _collect(client, endpoint, fields, quarter, limit)
    _require_index(details["index"], label)
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
                               f"total={details['api_total']}; pages={details['pages']}")
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
            raise FDICContractError("Failures unresolved linkage count differs")
        _atomic_json(audit_path, {"source_query": query, "api_total": details["api_total"],
                                  "unlinked_count": len(unresolved), "source_ids": unresolved})
        details["unresolved_linkage"] = {
            "count": len(unresolved), "source_ids_path": str(audit_path),
            "sha256": _sha256(audit_path),
        }
        if not _linkage_audit_valid(details, store):
            raise StorageValidationError("Failures linkage audit does not reconcile")
        manifest.record_validation("failures:unresolved_linkage_audit", True,
                                   f"count={len(unresolved)}; sha256={_sha256(audit_path)}")
    manifest.record_validation(f"plan:{label}", True, "Source and physical output validated")
    manifest.write(manifest_path)
    _upsert(summary, details)
    _atomic_json(summary_path, summary)
    return details


def _source_sweep(plan: AcquisitionPlan, config: ProjectConfig,
                  client: FDICClient, summary: dict[str, Any],
                  pacer: _ProbePacer,
                  on_job: Callable[[str, str | None], None] | None = None) -> None:
    """Recheck all source totals/indexes immediately before completion."""
    for endpoint, quarter, _, _ in plan.jobs(config):
        if on_job is not None:
            on_job(endpoint, quarter)
        details = _entry(summary, endpoint, quarter)
        if details is None:
            raise StorageValidationError(f"Source sweep missing {endpoint}:{quarter}")
        probe = pacer.fetch(client, endpoint, quarter)
        if probe.total != details["api_total"] or _require_index(probe.index, endpoint) != details["index"]:
            raise FDICContractError(f"Source total/index changed for {endpoint}:{quarter or 'all'}")


def run_attempt(
    *, plan: AcquisitionPlan, config: ProjectConfig, client: FDICClient,
    store: ParquetStore, paths: FullPaths, previous_paths: FullPaths | None = None,
    allow_bounded_test_plan: bool = False,
    probe_interval_seconds: float | None = None,
) -> dict[str, Any]:
    """Execute one immutable attempt; bounded plans require an explicit test-only flag."""
    if not allow_bounded_test_plan and plan != full_plan(config):
        raise ValueError("Production runner requires the exact 170-quarter/172-job full plan")
    if client.config != config or store.config != config or store.raw_root != paths.raw_root:
        raise ValueError("Client, store, and attempt paths must use the same configuration")
    interval = (probe_interval_seconds if probe_interval_seconds is not None else
                0.0 if allow_bounded_test_plan else FULL_PROBE_INTERVAL_SECONDS)
    if interval < 0:
        raise ValueError("Probe interval cannot be negative")
    with store.full_namespace_ownership():
        return _run_attempt_owned(plan=plan, config=config, client=client,
                                  store=store, paths=paths, previous_paths=previous_paths,
                                  probe_interval_seconds=interval)


def _run_attempt_owned(
    *, plan: AcquisitionPlan, config: ProjectConfig, client: FDICClient,
    store: ParquetStore, paths: FullPaths, previous_paths: FullPaths | None,
    probe_interval_seconds: float,
) -> dict[str, Any]:
    """All attempt reads and writes occur while the namespace lock is held."""
    jobs = plan.jobs(config)
    if paths.manifest.exists() or paths.summary.exists() or paths.linkage_audit.exists():
        raise FileExistsError("Attempt artifacts already exist; choose a new attempt ID")
    previous, old_summary = _load_previous(previous_paths, config, plan)
    manifest = RunManifest.new(f"fdic-{plan.name}-{_now()}", config)
    manifest.set_status("running")
    summary: dict[str, Any] = {
        "run_id": manifest.run_id, "plan": plan.name,
        "planned_quarters": list(plan.quarters), "quarters": [], "small_tables": [],
        "started_at_utc": manifest.created_at_utc, "status": "running",
        "previous_attempt": str(previous_paths.manifest) if previous_paths else None,
        "execution": {"probe_interval_seconds": probe_interval_seconds,
                      "api_max_attempts": client.max_attempts,
                      "api_backoff_seconds": client.backoff_seconds,
                      "api_max_backoff_seconds": client.max_backoff_seconds},
    }
    if previous_paths is not None:
        manifest.record_input(str(previous_paths.manifest), _sha256(previous_paths.manifest))
        if previous_paths.summary.exists():
            manifest.record_input(str(previous_paths.summary), _sha256(previous_paths.summary))
    manifest.write(paths.manifest)
    _atomic_json(paths.summary, summary)
    failure_context: dict[str, Any] = {
        "phase": "initialization", "endpoint": None, "quarter": None,
    }

    def current_job(endpoint: str, quarter: str | None) -> None:
        failure_context.update(endpoint=endpoint, quarter=quarter)

    manifest_committed = False
    pacer = _ProbePacer(probe_interval_seconds)
    try:
        financial_index: list[Any] | None = None
        for endpoint, quarter, fields, limit in jobs:
            failure_context.update(phase="acquisition", endpoint=endpoint, quarter=quarter)
            candidate = _previous_candidate(
                previous=previous, previous_summary=old_summary, previous_paths=previous_paths,
                client=client, pacer=pacer, store=store, endpoint=endpoint, quarter=quarter,
                fields=fields, limit=limit,
            )
            if candidate is not None:
                details = _record_inherited(
                    manifest=manifest, summary=summary, manifest_path=paths.manifest,
                    summary_path=paths.summary, audit_path=paths.linkage_audit,
                    previous_paths=previous_paths, store=store, endpoint=endpoint,
                    quarter=quarter, candidate=candidate,
                )
            else:
                details = _fetch_job(
                    config=config, client=client, store=store, manifest=manifest,
                    manifest_path=paths.manifest, summary=summary,
                    summary_path=paths.summary, audit_path=paths.linkage_audit,
                    endpoint=endpoint, quarter=quarter, fields=fields, limit=limit,
                )
            if endpoint == "financials":
                failure_context["phase"] = "source_consistency"
                if financial_index is None:
                    financial_index = details["index"]
                elif details["index"] != financial_index:
                    raise FDICContractError("Financials source indexes differ across quarters")
        failure_context.update(phase="final_parquet", endpoint=None, quarter=None)
        _verify_complete(plan, config, store, manifest, summary, on_job=current_job)
        failure_context.update(phase="final_source", endpoint=None, quarter=None)
        _source_sweep(plan, config, client, summary, pacer, on_job=current_job)
        failure_context.update(phase="plan_commit", endpoint=None, quarter=None)
        manifest.record_validation("plan:all_outputs", True,
                                   f"{len(jobs)} physical outputs and source probes passed")
        manifest.set_status("complete")
        manifest.write(paths.manifest)
        manifest_committed = True
        summary["status"] = "complete"
        summary["finished_at_utc"] = _now()
        _atomic_json(paths.summary, summary)
        return summary
    except Exception as exc:
        # A complete manifest plus incomplete summary is intentional crash evidence.
        # A new attempt independently verifies every output before completing.
        if not manifest_committed:
            phase = failure_context["phase"]
            endpoint = failure_context["endpoint"]
            quarter = failure_context["quarter"]
            job = f"{endpoint}:{quarter or 'all'}" if endpoint else "plan"
            failed_stage = (f"acquisition:{job}" if phase == "acquisition"
                            else f"validation:{phase}:{job}")
            manifest.stages[failed_stage] = "failed"
            if phase != "acquisition":
                manifest.stages["plan:completion"] = "failed"
            failure = {
                "phase": phase, "endpoint": endpoint, "quarter": quarter,
                "error_type": type(exc).__name__, "message": str(exc),
            }
            manifest.record_validation(f"{phase}:{job}", False,
                                       f"{type(exc).__name__}: {exc}")
            manifest.record_validation("plan:failed_gate", False,
                                       f"{type(exc).__name__}: {exc}")
            manifest.set_status("failed")
            manifest.write(paths.manifest)
            summary.update(status="failed", failed_stage=failed_stage,
                           failure=failure,
                           error=f"{type(exc).__name__}: {exc}", finished_at_utc=_now())
            _atomic_json(paths.summary, summary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Full FDIC acquisition: 170 quarters and 2 complete endpoints")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--recover-from", help="Prior attempt ID; evidence remains immutable")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    info = dry_run(args.project_root, config, args.attempt, args.recover_from)
    if args.dry_run:
        print(json.dumps(info, indent=2))
        return 0
    paths = full_paths(args.project_root, config, args.attempt)
    previous = (full_paths(args.project_root, config, args.recover_from)
                if args.recover_from else None)
    summary = run_attempt(
        plan=full_plan(config), config=config,
        client=FDICClient(config, max_attempts=FULL_MAX_ATTEMPTS,
                          backoff_seconds=FULL_BACKOFF_SECONDS,
                          max_backoff_seconds=FULL_MAX_BACKOFF_SECONDS),
        store=ParquetStore(paths.raw_root, config), paths=paths,
        previous_paths=previous,
    )
    print(json.dumps({"plan": summary["plan"], "status": summary["status"],
                      "quarters": len(summary["quarters"]),
                      "small_tables": len(summary["small_tables"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
