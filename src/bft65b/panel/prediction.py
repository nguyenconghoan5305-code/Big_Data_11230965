"""Apply the accepted provisional as-of rule to the observed FDIC panel.

This stage creates prediction candidates and audit dispositions, not labels or
model features fitted from outcomes. Accepted source files are read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.acquisition.extended import full_plan
from bft65b.config import EXTENDED_FINANCIAL_FIELDS, load_config
from bft65b.manifest import RunManifest, dependency_versions
from bft65b.panel.asof import (
    AVAILABILITY_LAG_DAYS, PREDICTOR_FIELDS, FinancialReport,
    availability_date, population_from_available_reports, select_asof,
)
from bft65b.panel.build import _write_verified, _accepted_inputs
from bft65b.panel.repair import _accepted_panel
from bft65b.quality.audit import _load_snapshot, _read_json, _sha, _write_json


PANEL_RUN = "panel20261115b"
REPAIR_RUN = "repair20261115f"
TREATMENT_RUN = "treatment20261114a"
RAW_RUN = "real20261111b"
RUN_NAMESPACE = "task_2_5_prediction"
DATA_NAMESPACE = "task_2_5_prediction"
EXCLUSION_REASONS = ("confirmed_failure_before_or_on_q", "unresolved_identity")
SOURCE_KEYS = ("raw", "treatment", "panel", "repair")
DEFAULT_SOURCES = {"raw": RAW_RUN, "treatment": TREATMENT_RUN,
                   "panel": PANEL_RUN, "repair": REPAIR_RUN}


def select_sources(root: Path, *, raw_run: str | None = None,
                   treatment_run: str | None = None, panel_run: str | None = None,
                   repair_run: str | None = None) -> dict[str, str]:
    """Require explicit, complete lineage for a relocated source-vintage replay."""
    requested = {"raw": raw_run, "treatment": treatment_run,
                 "panel": panel_run, "repair": repair_run}
    portable = (root / "runs/source_vintage_bundle.json").exists()
    if portable and any(value is None for value in requested.values()):
        raise ValueError("Portable PIT replay requires all four explicit upstream run IDs")
    selected = {key: DEFAULT_SOURCES[key] if requested[key] is None else requested[key]
                for key in SOURCE_KEYS}
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", value)
           for value in selected.values()):
        raise ValueError("Unsafe PIT upstream run ID")
    return selected


def _portable_chain_contract(sources: Mapping[str, str], bundle_sha: str,
        audit: RunManifest, treatment: RunManifest, panel: RunManifest,
        repair: RunManifest, repair_summary: Mapping[str, Any],
        repair_metadata: Mapping[str, Any], panel_hashes: Mapping[str, str],
        *, audit_findings_sha: str, treatment_summary_sha: str,
        panel_summary_sha: str) -> None:
    """Reject any cross-vintage or cross-run link before a portable PIT write."""
    if (audit.status != "complete"
            or audit.config.get("source_attempt") != sources["raw"]
            or audit.config.get("source_vintage_bundle_sha256") != bundle_sha
            or audit.run_id != f"p2_1_{treatment.config.get('source_audit_run')}"
            or treatment.config.get("source_audit_findings_sha256") != audit_findings_sha
            or treatment.config.get("source_vintage_bundle_sha256") != bundle_sha
            or panel.config.get("source_treatment_run") != sources["treatment"]
            or panel.config.get("source_treatment_summary_sha256") != treatment_summary_sha
            or panel.config.get("source_vintage_bundle_sha256") != bundle_sha
            or repair.config.get("source_panel_run") != sources["panel"]
            or repair.config.get("source_treatment_run") != sources["treatment"]
            or repair.config.get("source_panel_summary_sha256") != panel_summary_sha
            or repair.config.get("source_vintage_bundle_sha256") != bundle_sha
            or repair_summary.get("source_panel_run") != sources["panel"]
            or repair_metadata.get("protected_hashes") != panel_hashes):
        raise ValueError("Portable PIT audit/treatment/panel/repair source-vintage lineage differs")


def _verified_raw_paths(root: Path, raw: RunManifest, *, portable: bool) -> dict[str, str]:
    if raw.status != "complete" or len(raw.outputs) != 172 or not all(
            v["passed"] for v in raw.validations):
        raise ValueError("Accepted full acquisition manifest is incomplete")
    protected = {}
    for item in raw.outputs:
        path = Path(item["path"])
        if portable and not path.resolve().is_relative_to(root):
            raise ValueError("Relocated raw input escapes the portable source root")
        if _sha(path) != item["sha256"]:
            raise ValueError(f"Accepted raw input checksum changed: {path}")
        protected[str(path)] = item["sha256"]
    return protected


def _day(value: str) -> date:
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def read_panel_file(path: Path, columns: list[str] | None = None) -> pa.Table:
    """Read a physical file without inferring a conflicting REPDTE directory type."""
    with pq.ParquetFile(path) as parquet:
        return parquet.read(columns=columns)


def prediction_quarters(config: Any) -> tuple[str, ...]:
    """Use the assignment's 1984Q1--2025Q2 prediction observation scope."""
    if (config.financial_start != "1984-03-31"
            or config.financial_end != "2026-06-30"
            or config.modelling_end != "2025-06-30"
            or config.availability_lag_days != AVAILABILITY_LAG_DAYS
            or config.financial_fields != EXTENDED_FINANCIAL_FIELDS):
        raise ValueError("Prediction scope differs from verified assignment/configuration")
    quarters = tuple(q for q in full_plan(config).quarters if q <= "20250630")
    if len(quarters) != 166 or quarters[0] != "19840331" or quarters[-1] != "20250630":
        raise ValueError("Prediction quarters do not cover 1984Q1--2025Q2")
    return quarters


def candidate_disposition(cert: int, q: date,
                          events_by_cert: Mapping[int, Sequence[Mapping[str, Any]]],
                          reports_by_cert: Mapping[int, Sequence[FinancialReport]]) -> str:
    """Use dated, repaired events for risk-set exclusions, never for predictors.

    A conflicting identity/transaction takes precedence over a valid event.
    Assistance alone cannot establish an exit. Null-CERT events cannot be
    assigned to a bank and remain in the source-event audit. Historical
    BANK_LINK_DISPOSITION is retrospective, so recompute chronology using only
    reports provisionally available by q.
    """
    prior = [event for event in events_by_cert.get(cert, ())
             if event["FAILDATE"] <= q]
    known = [report.repdte for report in reports_by_cert.get(cert, ())
             if availability_date(report) <= q]
    unresolved = False
    confirmed = False
    for event in prior:
        status = event["TRANSACTION_DISPOSITION"]
        if status in {"assistance_transaction", "open_bank_assistance_code_unverified"}:
            continue
        has_prior = any(repdte <= event["FAILDATE"] for repdte in known)
        has_post = any(repdte > event["FAILDATE"] for repdte in known)
        if (status != "verified_failure" or event.get("SUPPLEMENT_MATCH", "matched") != "matched"
                or not has_prior or has_post):
            unresolved = True
        else:
            confirmed = True
    if unresolved:
        return "unresolved_identity"
    if confirmed:
        return "confirmed_failure_before_or_on_q"
    return "eligible"


def _percentile(histogram: Counter[int], fraction: float) -> int | None:
    n = sum(histogram.values())
    if not n:
        return None
    rank = max(1, int((n * fraction) + (0 if n * fraction == int(n * fraction) else 1)))
    seen = 0
    for age, count in sorted(histogram.items()):
        seen += count
        if seen >= rank:
            return age
    raise AssertionError("Age histogram is inconsistent")


def _age_summary(ages: Counter[int]) -> dict[str, int | None]:
    return {"min": min(ages) if ages else None,
            "p50": _percentile(ages, 0.5),
            "p90": _percentile(ages, 0.9),
            "max": max(ages) if ages else None}


def build_prediction_quarter(q: date, reports: Sequence[FinancialReport],
                             events: Sequence[Mapping[str, Any]],
                             output_schema: pa.Schema) -> tuple[pa.Table, dict[str, Any]]:
    """Select one row per available-report candidate and reconcile dispositions."""
    roster = population_from_available_reports(q, reports)
    selections = select_asof(roster, reports)
    events_by_cert: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if event["CERT"] is not None:
            events_by_cert[event["CERT"]].append(event)
    reports_by_cert: dict[int, list[FinancialReport]] = defaultdict(list)
    for report in reports:
        reports_by_cert[report.cert].append(report)
    rows = []
    counts: Counter[str] = Counter()
    ages: Counter[int] = Counter()
    selected_dates = []
    for item in selections:
        disposition = candidate_disposition(item.cert, q, events_by_cert, reports_by_cert)
        if item.status != "selected" or item.source_repdte is None or item.predictors is None:
            raise ValueError("Available-report roster produced a no-history candidate")
        if item.source_availability_date is None or item.source_availability_date > q:
            raise ValueError("Future-available report entered prediction candidates")
        counts[disposition] += 1
        ages[item.report_age_days] += 1
        selected_dates.append(item.source_repdte)
        rows.append({"CERT": item.cert, "Q": q,
            "POPULATION_KNOWN_AT": item.population_known_at,
            "SELECTED_REPDTE": item.source_repdte,
            "AVAILABILITY_DATE": item.source_availability_date,
            "REPORT_AGE_DAYS": item.report_age_days,
            "ELIGIBILITY_DISPOSITION": disposition,
            "AVAILABILITY_BASIS": item.availability_basis,
            "SOURCE_ID_DIAGNOSTIC": item.diagnostics["source_id"],
            "SOURCE_MISSING_FIELDS_DIAGNOSTIC": list(
                item.diagnostics["source_missing_fields"]),
            **item.predictors})
    table = pa.Table.from_pylist(rows, schema=output_schema)
    keys = [(row["CERT"], row["Q"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate CERT/q prediction key")
    result = {"quarter": q.strftime("%Y%m%d"),
        "candidates": len(rows), "with_eligible_history": len(rows),
        "no_eligible_history": 0,
        "eligible": counts["eligible"],
        "excluded_confirmed_failure_before_or_on_q": counts[
            "confirmed_failure_before_or_on_q"],
        "excluded_unresolved_identity": counts["unresolved_identity"],
        "report_age_days": _age_summary(ages),
        "earliest_selected_repdte": min(selected_dates).isoformat() if selected_dates else None,
        "latest_selected_repdte": max(selected_dates).isoformat() if selected_dates else None}
    if sum(counts.values()) != len(rows):
        raise ValueError("Candidate dispositions do not reconcile")
    return table, result


def _upstream(root: Path, config: Any, sources: Mapping[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Check the selected chain and relocate only verified source-vintage paths."""
    bundle_path = root / "runs/source_vintage_bundle.json"
    portable = bundle_path.exists()
    _, _, raw, _, _, _ = _load_snapshot(root, config, sources["raw"])
    protected = _verified_raw_paths(root, raw, portable=portable)
    bundle_sha = _sha(bundle_path) if portable else None
    if portable:
        protected[str(bundle_path)] = bundle_sha
    _, treated, treatment_manifest = _accepted_inputs(root, config, sources["treatment"])
    if treatment_manifest.status != "complete" or len(treated) != 172:
        raise ValueError("Accepted treatment evidence is incomplete")
    if portable:
        audit_run = treatment_manifest.config.get("source_audit_run")
        if not isinstance(audit_run, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", audit_run):
            raise ValueError("Portable treatment lacks a selected audit run")
        audit_root = root / "runs/task_2_1_quality" / audit_run
        audit_manifest = RunManifest.read(audit_root / "run_manifest.json")
    for item in treated:
        protected[item["path"]] = item["derived_sha256"]
    panel_summary, _, _, _, _, panel_hashes = _accepted_panel(root, sources["panel"])
    protected.update(panel_hashes)
    panel_root = root / "runs" / "task_2_2_panel" / sources["panel"]
    panel_manifest = RunManifest.read(panel_root / "run_manifest.json")
    if (panel_manifest.status != "complete" or len(panel_manifest.inputs) != 172
            or len(panel_manifest.outputs) != 173
            or any(not v["passed"] for v in panel_manifest.validations)
            or panel_manifest.config.get("source_treatment_run") != sources["treatment"]):
        raise ValueError("Accepted panel manifest is incomplete")
    for item in panel_manifest.inputs:
        if protected.get(item["path"]) != item["sha256"]:
            raise ValueError("Panel input checksum differs from treatment")
    repair_root = root / "runs" / "task_2_3_transaction_repair" / sources["repair"]
    repair_summary = _read_json(repair_root / "summary.json")
    repair_metadata = _read_json(repair_root / "metadata.json")
    repair_manifest = RunManifest.read(repair_root / "run_manifest.json")
    if (repair_manifest.status != "complete" or repair_summary.get("status") != "complete"
            or repair_metadata.get("status") != "complete"
            or repair_metadata.get("summary_sha256") != _sha(repair_root / "summary.json")
            or len(repair_manifest.outputs) != 2
            or any(not v["passed"] for v in repair_manifest.validations)
            or repair_summary.get("source_panel_run") != sources["panel"]
            or repair_manifest.config.get("source_panel_run") != sources["panel"]
            or panel_summary["panel_overall"]["output_rows"] != 1682616):
        raise ValueError("Accepted P2.3 repair evidence is incomplete")
    if portable:
        _portable_chain_contract(sources, bundle_sha, audit_manifest,
            treatment_manifest, panel_manifest, repair_manifest, repair_summary,
            repair_metadata, panel_hashes,
            audit_findings_sha=_sha(audit_root / "findings.json"),
            treatment_summary_sha=_sha(root / "runs/task_2_1_treatment" /
                sources["treatment"] / "summary.json"),
            panel_summary_sha=_sha(panel_root / "summary.json"))
    for item in repair_manifest.inputs:
        path = Path(item["path"])
        if (portable and not path.resolve().is_relative_to(root)) or _sha(path) != item["sha256"]:
            raise ValueError("Selected repaired input path/checksum differs")
    for name in ("summary.json", "metadata.json", "run_manifest.json", "report.md"):
        path = repair_root / name
        protected[str(path)] = _sha(path)
    for item in repair_manifest.outputs:
        path = Path(item["path"])
        if portable and not path.resolve().is_relative_to(root):
            raise ValueError("Selected repaired output escapes portable root")
        if _sha(path) != item["sha256"]:
            raise ValueError(f"Accepted repaired output checksum changed: {path}")
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != item["rows"]:
                raise ValueError("Repaired output row count differs")
        protected[str(path)] = item["sha256"]
    financial = [item for item in panel_summary["outputs"] if item["dataset"] == "financials"]
    if len(financial) != 170:
        raise ValueError("Accepted panel does not have 170 Financials quarters")
    events_path = next(Path(item["path"]) for item in repair_manifest.outputs
                       if item["dataset"] == "failure_events")
    events = pq.read_table(events_path).to_pylist()
    if len(events) != 4117 or len({row["ID"] for row in events}) != 4117:
        raise ValueError("Repaired source events do not reconcile")
    return financial, events, protected


def _real_perturbation(root: Path, financial: Sequence[Mapping[str, Any]],
                       events: Sequence[Mapping[str, Any]], panel_run: str = PANEL_RUN) -> dict[str, Any]:
    """On a bounded real sample, future reports and audit flags cannot change q."""
    by_q = {row["quarter"]: row for row in financial}
    q = date(2023, 6, 30)
    data_root = root / "data_processed" / "task_2_2_panel" / panel_run
    earlier = read_panel_file(data_root / by_q["20221231"]["relative_path"]).to_pylist()
    later = read_panel_file(data_root / by_q["20230630"]["relative_path"]).to_pylist()
    chosen = earlier[:100]
    if len(chosen) != 100:
        raise ValueError("Real perturbation fixture lacks 100 prior reports")
    certs = {row["CERT"] for row in chosen}
    later = [row for row in later if row["CERT"] in certs]
    before_reports = [FinancialReport.from_panel_row(row) for row in chosen]
    before_population = population_from_available_reports(q, before_reports)
    before = select_asof(before_population, before_reports)
    mutated = []
    for row in chosen + later:
        changed = dict(row)
        changed.update({"DQ_FLAGS": ["future_perturbation"],
            "EXIT_CATEGORY": "changed_after_q",
            "TRANSACTION_DISPOSITION": "changed_after_q",
            "INSTITUTION_ACTIVE_SNAPSHOT": 0})
        mutated.append(FinancialReport.from_panel_row(changed))
    after_population = population_from_available_reports(q, mutated)
    after = select_asof(after_population, mutated)
    if before_population != after_population or before != after:
        raise ValueError("Future data or retrospective flags changed earlier selection")
    by_cert: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if event["CERT"] in certs:
            by_cert[event["CERT"]].append(event)
    before_by_cert = {report.cert: [report] for report in before_reports}
    original = {cert: candidate_disposition(cert, q, by_cert, before_by_cert)
                for cert in certs}
    future_event = {"CERT": chosen[0]["CERT"], "FAILDATE": date(2023, 7, 1),
        "TRANSACTION_DISPOSITION": "verified_failure",
        "BANK_LINK_DISPOSITION": "chronological_candidate"}
    by_cert[future_event["CERT"]].append(future_event)
    after_by_cert: dict[int, list[FinancialReport]] = defaultdict(list)
    for report in mutated:
        after_by_cert[report.cert].append(report)
    if original != {cert: candidate_disposition(cert, q, by_cert, after_by_cert)
                    for cert in certs}:
        raise ValueError("Future failure event changed earlier risk-set disposition")
    return {"q": q.isoformat(), "report_quarters": ["2022Q4", "2023Q2"],
        "banks_compared": len(certs), "later_q_reports_appended": len(later),
        "future_event_date": "2023-07-01",
        "compared": ["population membership", "selected REPDTE", "availability date",
                     "report age", "all 23 predictor columns", "risk-set disposition"],
        "perturbed_audit_columns": ["DQ_FLAGS", "EXIT_CATEGORY",
            "TRANSACTION_DISPOSITION", "INSTITUTION_ACTIVE_SNAPSHOT"],
        "result": "identical"}


def run(root: Path, run_id: str, *, raw_run: str | None = None,
        treatment_run: str | None = None, panel_run: str | None = None,
        repair_run: str | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", run_id):
        raise ValueError("Unsafe run ID")
    sources = select_sources(root, raw_run=raw_run, treatment_run=treatment_run,
                             panel_run=panel_run, repair_run=repair_run)
    evidence_root = root / "runs" / RUN_NAMESPACE / run_id
    data_root = root / "data_processed" / DATA_NAMESPACE / run_id
    if evidence_root.exists() or data_root.exists():
        raise FileExistsError("Prediction run already exists")
    config = load_config(root / "config" / "financials_full_v2.json")
    quarters = prediction_quarters(config)
    evidence_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    manifest = RunManifest.new(run_id, config)
    manifest.config.update({"source_raw_run": sources["raw"],
        "source_treatment_run": sources["treatment"],
        "source_panel_run": sources["panel"], "source_repair_run": sources["repair"],
        "source_vintage_bundle_sha256": _sha(root / "runs/source_vintage_bundle.json")
            if (root / "runs/source_vintage_bundle.json").exists() else None,
        "population_policy": "available_report_roster", "availability_assumption_days": 120,
        "outcome_window_unchanged": "(q, q + 4 quarters]", "predictor_fields": list(PREDICTOR_FIELDS)})
    manifest.set_status("running")
    manifest.write(evidence_root / "run_manifest.json")
    try:
        financial, events, protected = _upstream(root, config, sources)
        for path, digest in protected.items():
            manifest.record_input(path, digest)
        source = {item["quarter"]: item for item in financial}
        if len(source) != 170 or set(source) != set(full_plan(config).quarters):
            raise ValueError("Accepted panel quarter inventory differs")
        schema_path = (root / "data_processed" / "task_2_2_panel" / sources["panel"] /
                       source["19840331"]["relative_path"])
        input_schema = pq.read_schema(schema_path)
        if any(field not in input_schema.names for field in PREDICTOR_FIELDS):
            raise ValueError("Accepted Financials schema lacks selected predictors")
        output_schema = pa.schema([
            pa.field("CERT", pa.int64(), nullable=False),
            pa.field("Q", pa.date32(), nullable=False),
            pa.field("POPULATION_KNOWN_AT", pa.date32(), nullable=False),
            pa.field("SELECTED_REPDTE", pa.date32()),
            pa.field("AVAILABILITY_DATE", pa.date32()),
            pa.field("REPORT_AGE_DAYS", pa.int32()),
            pa.field("ELIGIBILITY_DISPOSITION", pa.string(), nullable=False),
            pa.field("AVAILABILITY_BASIS", pa.string()),
            pa.field("SOURCE_ID_DIAGNOSTIC", pa.string()),
            pa.field("SOURCE_MISSING_FIELDS_DIAGNOSTIC",
                     pa.list_(pa.field("element", pa.string()))),
            *(input_schema.field(name) for name in PREDICTOR_FIELDS),
        ], metadata={b"bft65b.pit": b"provisional_120_day_v1",
                     b"bft65b.predictor_fields": ",".join(PREDICTOR_FIELDS).encode()})
        pending = list(financial)
        first: dict[int, FinancialReport] = {}
        latest: dict[int, FinancialReport] = {}
        rows_by_quarter = []
        overall_ages: Counter[int] = Counter()
        outputs = []
        (evidence_root / "progress").mkdir()
        for quarter in quarters:
            q = _day(quarter)
            releaseable = [item for item in pending
                           if _day(item["quarter"]) + timedelta(days=120) <= q]
            pending = [item for item in pending if item not in releaseable]
            for item in releaseable:
                path = (root / "data_processed" / "task_2_2_panel" / sources["panel"] /
                        item["relative_path"])
                source_rows = read_panel_file(path).to_pylist()
                for row in source_rows:
                    report = FinancialReport.from_panel_row(row)
                    if report.cert not in first:
                        first[report.cert] = report
                    if (report.cert not in latest
                            or report.repdte > latest[report.cert].repdte):
                        latest[report.cert] = report
            rule_reports = []
            for cert in sorted(latest):
                rule_reports.append(first[cert])
                if latest[cert] is not first[cert]:
                    rule_reports.append(latest[cert])
            table, stats = build_prediction_quarter(q, rule_reports, events, output_schema)
            observed = read_panel_file(root / "data_processed" / "task_2_2_panel" /
                sources["panel"] / source[quarter]["relative_path"], ["CERT"])["CERT"].to_pylist()
            stats["observed_q_reports_diagnostic"] = len(observed)
            stats["observed_q_without_prior_available_history_diagnostic"] = sum(
                cert not in first for cert in observed)
            if stats["candidates"] != len(first):
                raise ValueError(f"Candidate roster does not reconcile at {quarter}")
            # Full row-level read-back is performed by the accepted atomic writer.
            output = data_root / "predictions" / f"Q={quarter}" / "part-000.parquet"
            digest = _write_verified(output, table)
            outputs.append({"quarter": quarter, "path": str(output), "rows": table.num_rows,
                            "sha256": digest})
            manifest.record_output(path=str(output), dataset="prediction_candidates",
                rows=table.num_rows, sha256=digest, compression="ZSTD",
                schema_version="pit_provisional_120_day_v1", partition=quarter,
                source_total=table.num_rows, source_query=None)
            rows_by_quarter.append(stats)
            overall_ages.update(table["REPORT_AGE_DAYS"].to_pylist())
            # Immutable per-quarter progress avoids replacing a manifest that a
            # Windows observer may have open while the run is still active.
            _write_json(evidence_root / "progress" / f"Q={quarter}.json",
                        {"quarter": quarter, "rows": table.num_rows,
                         "sha256": digest, "status": "output_verified"})
        for item in outputs:
            path = Path(item["path"])
            if _sha(path) != item["sha256"]:
                raise ValueError(f"Prediction output checksum differs: {item['quarter']}")
            with pq.ParquetFile(path) as parquet:
                if (parquet.metadata.num_rows != item["rows"]
                        or not parquet.schema_arrow.equals(output_schema, check_metadata=True)
                        or any(parquet.metadata.row_group(g).column(c).compression != "ZSTD"
                               for g in range(parquet.metadata.num_row_groups)
                               for c in range(parquet.metadata.row_group(g).num_columns))):
                    raise ValueError(f"Prediction output read-back differs: {item['quarter']}")
        if any(_sha(Path(path)) != digest for path, digest in protected.items()):
            raise ValueError("Accepted upstream hashes changed after prediction run")
        perturbation = _real_perturbation(root, financial, events, sources["panel"])
        total = Counter()
        for item in rows_by_quarter:
            for key in ("candidates", "with_eligible_history", "no_eligible_history", "eligible",
                        "excluded_confirmed_failure_before_or_on_q",
                        "excluded_unresolved_identity", "observed_q_reports_diagnostic",
                        "observed_q_without_prior_available_history_diagnostic"):
                total[key] += item[key]
        if (len(outputs) != 166 or total["candidates"] !=
                total["eligible"] + total["excluded_confirmed_failure_before_or_on_q"]
                + total["excluded_unresolved_identity"]):
            raise ValueError("Full prediction reconciliation differs")
        unlinked = Counter(event["BANK_LINK_DISPOSITION"] for event in events
                           if event["BANK_LINK_DISPOSITION"] != "chronological_candidate")
        result = {"status": "complete", "run_id": run_id,
            "source_runs": dict(sources),
            "prediction_scope": {"first_q": quarters[0], "last_q": quarters[-1],
                                 "quarters": len(quarters), "availability_lag_days": 120,
                                 "target_window_unchanged": "(q, q + 4 quarters]"},
            "population_policy": "report-derived candidate roster; stale banks retained unless a dated repaired failure excludes them",
            "provenance_limits": ["120-day lag is assumed, not verified historical publication time",
                "historical revised statement vintages unavailable",
                "FDIC failure event publication timestamp at q unverified",
                "null-CERT events cannot be assigned to prediction banks"],
            "by_quarter": rows_by_quarter,
            "overall": {**dict(total), "report_age_days": _age_summary(overall_ages),
                "earliest_selected_repdte": next((v["earliest_selected_repdte"] for v in
                    rows_by_quarter if v["earliest_selected_repdte"]), None),
                "latest_selected_repdte": max((v["latest_selected_repdte"] for v in
                    rows_by_quarter if v["latest_selected_repdte"]), default=None)},
            "unlinked_source_events_by_disposition": dict(sorted(unlinked.items())),
            "future_perturbation": perturbation,
            "outputs": outputs, "upstream_file_hash_count": len(protected)}
        _write_json(evidence_root / "summary.json", result)
        lines = ["# P2.4 Full Prediction-Date Reconciliation", "",
            "Candidate roster: at least one report provisionally available by q. Counts are bank-quarter candidates, not failures or labels. Current-quarter report existence never creates membership at that q. Stale candidates remain unless dated repaired failure evidence excludes them.", "",
            "| Prediction quarter | Candidates | Eligible | No history | Excluded: prior failure | Excluded: unresolved identity | Observed-q reports (diagnostic) | Observed-q without prior history (diagnostic) | Age min/p50/p90/max days | Earliest selected REPDTE | Latest selected REPDTE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"]
        for item in rows_by_quarter:
            age = item["report_age_days"]
            lines.append(f"| {item['quarter']} | {item['candidates']:,} | {item['eligible']:,} | {item['no_eligible_history']:,} | {item['excluded_confirmed_failure_before_or_on_q']:,} | {item['excluded_unresolved_identity']:,} | {item['observed_q_reports_diagnostic']:,} | {item['observed_q_without_prior_available_history_diagnostic']:,} | {age['min']}/{age['p50']}/{age['p90']}/{age['max']} | {item['earliest_selected_repdte'] or '—'} | {item['latest_selected_repdte'] or '—'} |")
        overall = result["overall"]
        age = overall["report_age_days"]
        lines.append(f"| Overall | {overall['candidates']:,} | {overall['eligible']:,} | {overall['no_eligible_history']:,} | {overall['excluded_confirmed_failure_before_or_on_q']:,} | {overall['excluded_unresolved_identity']:,} | {overall['observed_q_reports_diagnostic']:,} | {overall['observed_q_without_prior_available_history_diagnostic']:,} | {age['min']}/{age['p50']}/{age['p90']}/{age['max']} | {overall['earliest_selected_repdte']} | {overall['latest_selected_repdte']} |")
        lines += ["", "The 120-day lag is a conservative eligibility assumption. It delays the first possible prediction for a newly reporting bank. A report-derived roster therefore has zero candidate rows without eligible history by construction; the observed-q diagnostic counts show banks that would be wrongly added using a later-published q report. No historical vintage reconstruction is claimed.", "",
            "Repaired FDIC verified failure with a valid chronological CERT link excludes a bank only when FAILDATE <= q. Conflicting/uncertain non-assistance evidence linked to an observed CERT is excluded as unresolved identity. Assistance alone is never an exit. Null-CERT events remain source-level unresolved cases and cannot be assigned to a bank. No outcome label is created; the official later target remains (q, q + 4 quarters].", ""]
        (evidence_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
        metadata = {"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "dependencies": dependency_versions(), "config": config.to_dict(),
            "summary_sha256": _sha(evidence_root / "summary.json"),
            "report_sha256": _sha(evidence_root / "report.md"),
            "upstream_hashes": protected}
        _write_json(evidence_root / "metadata.json", metadata)
        manifest.record_validation("upstream_complete_and_hashes", True,
            f"{len(protected)} accepted upstream paths verified before and after")
        manifest.record_validation("prediction_scope_and_keys", True,
            "166 quarters; unique CERT/q and source availability <= q")
        manifest.record_validation("outputs_readback", True,
            "166 ZSTD Parquet files rechecked for schema, rows and SHA-256")
        manifest.record_validation("future_perturbation", True,
            "Bounded real snapshot: 100 banks, all 23 predictors and dated exclusions unchanged")
        manifest.set_status("complete")
        manifest.write(evidence_root / "run_manifest.json")
        return result
    except Exception as exc:
        manifest.record_validation("p2_4_full_prediction", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence_root / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str, *, check_upstream: bool = True) -> dict[str, Any]:
    """Independently recheck persisted candidates, evidence and source hashes."""
    root = root.resolve(strict=True)
    evidence = root / "runs" / RUN_NAMESPACE / run_id
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if (manifest.run_id != run_id or manifest.status != "complete"
            or summary.get("run_id") != run_id or summary.get("status") != "complete"
            or metadata.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or metadata.get("report_sha256") != _sha(evidence / "report.md")
            or not manifest.validations or any(not row["passed"] for row in manifest.validations)
            or len(manifest.outputs) != 166 or len(summary.get("outputs", [])) != 166
            or len(summary.get("by_quarter", [])) != 166):
        raise ValueError("Saved prediction manifest, summary, or metadata is incomplete")
    recorded = summary.get("source_runs")
    if (not isinstance(recorded, dict) or set(recorded) != set(SOURCE_KEYS)
            or any(manifest.config.get(f"source_{key}_run") != recorded[key]
                   for key in SOURCE_KEYS)):
        raise ValueError("Saved prediction upstream run selection differs")
    sources = select_sources(root, raw_run=recorded["raw"],
        treatment_run=recorded["treatment"], panel_run=recorded["panel"],
        repair_run=recorded["repair"])
    bundle_path = root / "runs/source_vintage_bundle.json"
    if bundle_path.exists() and manifest.config.get("source_vintage_bundle_sha256") != _sha(bundle_path):
        raise ValueError("Saved prediction source-vintage bundle checksum differs")
    recorded_hashes = metadata.get("upstream_hashes")
    if (not isinstance(recorded_hashes, dict)
            or len(manifest.inputs) != len(recorded_hashes)
            or {item["path"]: item["sha256"] for item in manifest.inputs} != recorded_hashes):
        raise ValueError("Saved prediction manifest/input-hash binding differs")
    data_root = (root / "data_processed" / DATA_NAMESPACE / run_id).resolve()
    by_output = {row["partition"]: row for row in manifest.outputs}
    by_summary = {row["quarter"]: row for row in summary["by_quarter"]}
    if len(by_output) != 166 or len(by_summary) != 166:
        raise ValueError("Prediction quarter inventory has duplicates")
    totals: Counter[str] = Counter()
    for item in summary["outputs"]:
        quarter = item["quarter"]
        recorded = by_output.get(quarter)
        stats = by_summary.get(quarter)
        path = Path(item["path"]).resolve()
        if (not recorded or not stats or quarter != stats["quarter"]
                or not path.is_relative_to(data_root)
                or path != Path(recorded["path"]).resolve()
                or item["sha256"] != recorded["sha256"]
                or item["rows"] != recorded["rows"]
                or _sha(path) != item["sha256"]):
            raise ValueError(f"Prediction output inventory/checksum differs: {quarter}")
        with pq.ParquetFile(path) as parquet:
            schema = parquet.schema_arrow
            if (parquet.metadata.num_rows != item["rows"]
                    or schema.metadata.get(b"bft65b.pit") != b"provisional_120_day_v1"
                    or schema.metadata.get(b"bft65b.predictor_fields") !=
                    ",".join(PREDICTOR_FIELDS).encode()
                    or any(parquet.metadata.row_group(g).column(c).compression != "ZSTD"
                           for g in range(parquet.metadata.num_row_groups)
                           for c in range(parquet.metadata.row_group(g).num_columns))):
                raise ValueError(f"Prediction output schema/codec differs: {quarter}")
            columns = parquet.read(columns=["CERT", "Q", "SELECTED_REPDTE",
                "AVAILABILITY_DATE", "REPORT_AGE_DAYS", "ELIGIBILITY_DISPOSITION"])
        rows = columns.to_pylist()
        q = _day(quarter)
        if (len(rows) != len({row["CERT"] for row in rows})
                or any(row["Q"] != q or row["SELECTED_REPDTE"] is None
                       or row["AVAILABILITY_DATE"] is None
                       or row["AVAILABILITY_DATE"] > q
                       or row["SELECTED_REPDTE"] + timedelta(days=120) > q
                       or row["REPORT_AGE_DAYS"] != (q - row["SELECTED_REPDTE"]).days
                       for row in rows)):
            raise ValueError(f"Prediction keys or availability differ: {quarter}")
        dispositions = Counter(row["ELIGIBILITY_DISPOSITION"] for row in rows)
        if (len(rows) != stats["candidates"]
                or dispositions["eligible"] != stats["eligible"]
                or dispositions["confirmed_failure_before_or_on_q"] !=
                stats["excluded_confirmed_failure_before_or_on_q"]
                or dispositions["unresolved_identity"] != stats["excluded_unresolved_identity"]
                or sum(dispositions.values()) != len(rows)):
            raise ValueError(f"Prediction dispositions differ: {quarter}")
        for key in ("candidates", "eligible", "excluded_confirmed_failure_before_or_on_q",
                    "excluded_unresolved_identity"):
            totals[key] += stats[key]
    if any(totals[key] != summary["overall"][key] for key in totals):
        raise ValueError("Prediction overall reconciliation differs")
    checked_upstream = 0
    if check_upstream:
        _, _, current_hashes = _upstream(root, load_config(root / "config/financials_full_v2.json"), sources)
        if current_hashes != recorded_hashes:
            raise ValueError("Saved prediction selected upstream chain differs")
        for path, digest in recorded_hashes.items():
            if _sha(Path(path)) != digest:
                raise ValueError(f"Accepted upstream checksum changed: {path}")
            checked_upstream += 1
    return {"status": "verified", "run_id": run_id, "quarters": 166,
            "candidate_rows": totals["candidates"], "upstream_hashes_checked": checked_upstream}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build provisional as-of FDIC prediction candidates")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("run")
    build.add_argument("--project-root", type=Path, default=Path.cwd())
    build.add_argument("--run-id", required=True)
    for option in ("raw", "treatment", "panel", "repair"):
        build.add_argument(f"--{option}-run")
    show = sub.add_parser("show")
    show.add_argument("--project-root", type=Path, default=Path.cwd())
    show.add_argument("--run-id", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--project-root", type=Path, default=Path.cwd())
    verify.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.project_root, args.run_id, raw_run=args.raw_run,
            treatment_run=args.treatment_run, panel_run=args.panel_run,
            repair_run=args.repair_run)
        print(json.dumps({"status": result["status"], "run_id": args.run_id,
            "overall": result["overall"], "report": str(args.project_root / "runs" /
                RUN_NAMESPACE / args.run_id / "report.md")}, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_saved(args.project_root, args.run_id), indent=2))
    else:
        evidence = args.project_root / "runs" / RUN_NAMESPACE / args.run_id
        try:
            manifest = RunManifest.read(evidence / "run_manifest.json")
            summary = _read_json(evidence / "summary.json")
            metadata = _read_json(evidence / "metadata.json")
            if (manifest.run_id != args.run_id or manifest.status != "complete"
                    or summary.get("run_id") != args.run_id or summary.get("status") != "complete"
                    or metadata.get("status") != "complete"
                    or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
                    or not manifest.validations or any(not v["passed"] for v in manifest.validations)):
                raise ValueError("manifest, summary, metadata, or checksum is incomplete")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(2, f"P2.4 prediction run {args.run_id}: {exc}\n")
        print(json.dumps({"status": "complete", "run_id": args.run_id,
            "overall": summary["overall"], "report": str(evidence / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
