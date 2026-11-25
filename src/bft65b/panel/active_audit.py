"""Supplementary FDIC membership-source acquisition and retrospective audit.

This stage never promotes retrospective intervals to a historical as-of roster.
It preserves accepted snapshots and writes into an isolated ignored namespace.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import pyarrow.parquet as pq
import requests

from bft65b.manifest import RunManifest
from bft65b.config import load_config
from bft65b.panel.repair import _accepted_panel
from bft65b.panel.prediction import verify_saved as verify_prediction
from bft65b.features.label import _verify_repair
from bft65b.quality.audit import _load_snapshot, _sha, _read_json, _write_json

BASE = "https://api.fdic.gov/banks"
INST_FIELDS = "CERT,STNAME,ACTIVE,CHANGEC1,INSDATE,ESTYMD,ENDEFYMD,DATEUPDT"
HIST_FIELDS = "CERT,EFFDATE,PROCDATE,CHANGECODE,ORG_ROLE_CDE,TRANSNUM"
SCOPE_YEARS = range(1984, 2026)


def _date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"Unexpected source date type: {type(value).__name__}")
    for fmt in ("%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized source date: {value!r}")


def _cert(value: Any) -> int:
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(f"Invalid source CERT: {value!r}")
    return int(value)


def _request(session: requests.Session, endpoint: str, params: dict) -> dict:
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(f"{BASE}/{endpoint}", params=params, timeout=(10, 60))
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("meta"), dict) or not isinstance(payload.get("data"), list):
                raise ValueError("FDIC response missing meta/data")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt == 3:
                break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"FDIC {endpoint} request failed: {last}")


def _atomic_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Supplementary source file already exists: {path}")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".source.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        if _read_json(Path(temporary)) != payload:
            raise ValueError("Supplementary JSON read-back failed")
        os.replace(temporary, path)
        return _sha(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fetch_partition(session: requests.Session, endpoint: str, params: dict,
                     path: Path, expected_index: tuple | None) -> tuple[list[dict], tuple, dict]:
    payload = _request(session, endpoint, params)
    meta = payload["meta"]
    index = meta.get("index") or {}
    fingerprint = (index.get("name"), index.get("createTimestamp"))
    if not all(fingerprint) or (expected_index is not None and fingerprint != expected_index):
        raise ValueError(f"{endpoint} source index changed: {fingerprint}")
    total = meta.get("total")
    rows = [item["data"] for item in payload["data"]]
    if type(total) is not int or total < 0 or len(rows) != total or total > int(params["limit"]):
        raise ValueError(f"{endpoint} bounded partition count differs: {total} vs {len(rows)}")
    digest = _atomic_json(path, payload)
    return rows, fingerprint, {"path": str(path), "sha256": digest, "rows": len(rows),
                               "total": total, "query": params, "index": list(fingerprint)}


def _fetch_institutions(session: requests.Session, root: Path) -> tuple[list[dict], list[dict], tuple]:
    rows: list[dict] = []
    records: list[dict] = []
    index = None
    total = None
    for offset in (0, 10000, 20000):
        params = {"fields": INST_FIELDS, "limit": 10000, "offset": offset,
                  "sort_by": "CERT", "sort_order": "ASC", "format": "json"}
        payload = _request(session, "institutions", params)
        meta = payload["meta"]
        fingerprint = (meta["index"]["name"], meta["index"]["createTimestamp"])
        if index is None:
            index, total = fingerprint, meta["total"]
        if fingerprint != index or meta["total"] != total:
            raise ValueError("Institutions source index or total changed across pages")
        page = [item["data"] for item in payload["data"]]
        if offset < total and not page:
            raise ValueError("Empty Institutions intermediate page")
        if offset + len(page) < total and len(page) != 10000:
            raise ValueError("Short Institutions intermediate page")
        path = root / "institutions" / f"offset={offset:05d}.json"
        digest = _atomic_json(path, payload)
        records.append({"path": str(path), "sha256": digest, "rows": len(page),
                        "total": total, "query": params, "index": list(index)})
        rows.extend(page)
        if len(rows) == total:
            break
    certs = [_cert(row["CERT"]) for row in rows]
    if (len(rows) != total or len(certs) != len(set(certs))
            or certs != sorted(certs)):
        raise ValueError("Institutions total/key/order gate failed")
    return rows, records, index


def _fetch_history(session: requests.Session, root: Path) -> tuple[list[dict], list[dict], tuple]:
    rows: list[dict] = []
    records: list[dict] = []
    index = None
    for year in SCOPE_YEARS:
        filter_ = f"ORG_ROLE_CDE:FI AND EFFDATE:[{year}-01-01 TO {year}-12-31]"
        params = {"fields": HIST_FIELDS, "filters": filter_, "limit": 10000,
                  "offset": 0, "format": "json"}
        payload = _request(session, "history", params)
        meta = payload["meta"]
        fingerprint = (meta["index"]["name"], meta["index"]["createTimestamp"])
        if index is None:
            index = fingerprint
        if fingerprint != index:
            raise ValueError(f"History source index changed at {year}")
        part = [item["data"] for item in payload["data"]]
        if (type(meta["total"]) is not int or meta["total"] != len(part)
                or len(part) == 0 or len(part) > 10000):
            raise ValueError(f"History year {year} requires verified subpartition: total={meta.get('total')}")
        if any(row.get("ORG_ROLE_CDE") != "FI" or _date(row.get("EFFDATE")).year != year
               for row in part):
            raise ValueError(f"History role/date filter failed at {year}")
        path = root / "history_fi" / f"year={year}.json"
        digest = _atomic_json(path, payload)
        records.append({"path": str(path), "sha256": digest, "rows": len(part),
                        "total": meta["total"], "query": params, "index": list(index)})
        rows.extend(part)
    return rows, records, index


def _audit(root: Path, institutions: list[dict], history: list[dict],
           inst_records: list[dict], hist_records: list[dict], *,
           panel_run: str = "panel20261115b", pit_run: str = "pit20261115f",
           repair_run: str = "repair20261115f") -> dict:
    for row in institutions:
        if (type(row.get("CERT")) is not int or row.get("ACTIVE") not in (0, 1)
                or type(row.get("INSDATE")) is not str
                or type(row.get("ESTYMD")) is not str
                or type(row.get("ENDEFYMD")) is not str):
            raise ValueError("Institutions identity/status/date source type differs")
        for field in ("INSDATE", "ESTYMD", "ENDEFYMD", "DATEUPDT"):
            _date(row.get(field))
    for row in history:
        if (not isinstance(row.get("ID"), str)
                or (row.get("CERT") is not None and type(row["CERT"]) is not int)
                or type(row.get("CHANGECODE")) is not int
                or row.get("ORG_ROLE_CDE") != "FI"
                or type(row.get("TRANSNUM")) is not int):
            raise ValueError("FI History source schema differs")
        _date(row.get("EFFDATE"))
        _date(row.get("PROCDATE"))
    _, _, accepted, _, _, _ = _load_snapshot(root,
        load_config(root / "config/financials_full_v2.json"), "real20261111b")
    old = next(item for item in accepted.outputs if item["dataset"] == "institutions")
    if _sha(Path(old["path"])) != old["sha256"]:
        raise ValueError("Accepted Institutions checksum differs")
    with pq.ParquetFile(old["path"]) as file:
        prior = {row["CERT"]: row for row in file.read(columns=[
            "CERT", "STNAME", "ACTIVE", "CHANGEC1"]).to_pylist()}
    current = {_cert(row["CERT"]): row for row in institutions}
    if len(prior) != len(set(prior)) or len(current) != len(institutions):
        raise ValueError("Institutions CERT identity is not unique")
    shared = set(prior) & set(current)
    revisions = Counter()
    for cert in shared:
        for field in ("STNAME", "ACTIVE", "CHANGEC1"):
            before, after = prior[cert][field], current[cert].get(field)
            if field == "CHANGEC1" and after is not None:
                after = str(after)
            if before != after:
                revisions[field] += 1
    panel, _, _, _, _, panel_hashes = _accepted_panel(root, panel_run)
    verify_prediction(root, pit_run, check_upstream=True)
    panel_certs = set()
    for item in panel["outputs"]:
        if item["dataset"] == "financials":
            path = root / "data_processed" / "task_2_2_panel" / panel_run / item["relative_path"]
            with pq.ParquetFile(path) as file:
                panel_certs.update(file.read(columns=["CERT"])["CERT"].to_pylist())
    coverage = Counter()
    for cert in panel_certs:
        row = current.get(cert)
        if row is None:
            coverage["missing_in_current_institutions"] += 1
            continue
        start, end = _date(row.get("INSDATE")), _date(row.get("ENDEFYMD"))
        established = _date(row.get("ESTYMD"))
        coverage["matched_cert"] += 1
        if start is None: coverage["missing_insurance_entry"] += 1
        if established is None: coverage["missing_established_date"] += 1
        if start is not None and end is not None and end <= start:
            coverage["conflicting_interval"] += 1
        if row.get("ACTIVE") == 0 and end is None:
            coverage["inactive_without_end_date"] += 1
        if start is not None and (row.get("ACTIVE") == 1 or end is not None) and not (
                end is not None and end <= start):
            coverage["retrospective_interval_candidate"] += 1
    ids = [row.get("ID") for row in history]
    history_issues = {"rows": len(history), "missing_source_id": sum(not isinstance(x, str) or not x.strip() for x in ids),
                      "duplicate_source_ids": len(ids)-len(set(ids)),
                      "absent_cert": sum("CERT" not in row for row in history),
                      "explicit_null_cert": sum("CERT" in row and row["CERT"] is None for row in history),
                      "missing_cert": sum(row.get("CERT") is None for row in history),
                      "missing_process_date": sum(_date(row.get("PROCDATE")) is None for row in history),
                      "process_after_effective": sum(_date(row.get("PROCDATE")) is not None and
                          _date(row["PROCDATE"]) > _date(row["EFFDATE"]) for row in history),
                      "event_codes": dict(sorted(Counter(str(row.get("CHANGECODE")) for row in history).items()))}
    failure_summary, failures, _ = _verify_repair(root, repair_run)
    if len(failures) != 4117:
        raise ValueError("Accepted repaired Failures count changed")
    failure_certs = {event.cert for event in failures if event.cert is not None}
    history_certs = {_cert(row["CERT"]) for row in history if row.get("CERT") is not None}
    coverage["missing_in_institutions_but_in_history_fi"] = len((panel_certs - set(current)) & history_certs)
    coverage["missing_in_both_institutions_and_history_fi"] = len(panel_certs - set(current) - history_certs)
    event_reconciliation = {
        "failure_source_events": len(failures),
        "failure_null_cert_events": sum(event.cert is None for event in failures),
        "failure_events_cert_in_current_institutions": sum(event.cert in current for event in failures if event.cert is not None),
        "failure_events_cert_in_observed_panel": sum(event.cert in panel_certs for event in failures if event.cert is not None),
        "failure_events_cert_in_history_fi": sum(event.cert in history_certs for event in failures if event.cert is not None),
        "failure_distinct_nonnull_certs": len(failure_certs),
        "failure_certs_in_history_fi": len(failure_certs & history_certs),
        "failure_certs_absent_from_history_fi": len(failure_certs - history_certs),
        "history_fi_distinct_nonnull_certs": len(history_certs),
        "history_certs_in_observed_panel": len(history_certs & panel_certs),
        "history_source_id_matches_failure_ids": len(set(ids) & {event.source_id for event in failures}),
        "failure_latest_null_cert_date": max((event.effective_date.isoformat() for event in failures
            if event.cert is None), default=None),
    }
    predictions = _read_json(root / "runs" / "task_2_5_prediction" /
                             pit_run / "summary.json")
    # Parse snapshot dates once per CERT, not once per candidate bank-quarter.
    intervals = {cert: (_date(row.get("INSDATE")), _date(row.get("ENDEFYMD")),
                        row.get("ACTIVE") == 1) for cert, row in current.items()}
    by_quarter = []
    all_counts = Counter()
    all_diagnostics = Counter()
    for item in predictions["outputs"]:
        with pq.ParquetFile(item["path"]) as file:
            table = file.read(columns=["CERT", "ELIGIBILITY_DISPOSITION", "REPORT_AGE_DAYS"])
            certs = table["CERT"].to_pylist()
            dispositions = table["ELIGIBILITY_DISPOSITION"].to_pylist()
            ages = table["REPORT_AGE_DAYS"].to_pylist()
        if len(certs) != item["rows"] or len(certs) != len(set(certs)):
            raise ValueError(f"Candidate roster key/count gate failed: {item['quarter']}")
        q = date(int(item["quarter"][:4]), int(item["quarter"][4:6]),
                 int(item["quarter"][6:]))
        counts = Counter()
        diagnostics = Counter()
        for cert, disposition, age in zip(certs, dispositions, ages):
            interval = intervals.get(cert)
            if interval is None:
                category = "no_institution_identity"
            else:
                start, end, active_now = interval
                if start is None or (not active_now and end is None) or (
                        end is not None and end <= start):
                    category = "unresolved_activity_interval"
                elif start > q or (end is not None and end <= q):
                    category = "outside_retrospective_interval"
                else:
                    category = "inside_retrospective_interval"
            counts[category] += 1
            if disposition == "eligible":
                diagnostics["p24_eligible"] += 1
                diagnostics[f"p24_eligible_{category}"] += 1
                if age is not None and age > 365:
                    diagnostics["p24_eligible_report_older_than_365d"] += 1
                    diagnostics[f"p24_eligible_older_than_365d_{category}"] += 1
        if sum(counts.values()) != len(certs):
            raise ValueError("Quarterly coverage count differs")
        all_counts.update(counts)
        all_diagnostics.update(diagnostics)
        by_quarter.append({"quarter": item["quarter"], "candidate_bank_quarters": len(certs),
                           **dict(counts), "diagnostics": dict(diagnostics)})
    if sum(all_counts.values()) != predictions["overall"]["candidates"]:
        raise ValueError("Full candidate coverage does not reconcile")
    if sum(item["rows"] for item in inst_records) != len(institutions) or sum(
            item["rows"] for item in hist_records) != len(history):
        raise ValueError("Supplementary source totals differ")
    return {"accepted_institutions_index": next(x["index"] for x in accepted.sources if x["endpoint"] == "institutions"),
            "current_institutions_index": inst_records[0]["index"],
            "snapshot_reconciliation": {"accepted_certs": len(prior), "current_certs": len(current),
                "shared_certs": len(shared), "added_certs": len(set(current)-set(prior)),
                "removed_certs": len(set(prior)-set(current)), "shared_field_revisions": dict(revisions),
                "current_changec1_absent": sum("CHANGEC1" not in row for row in current.values()),
                "current_changec1_explicit_null": sum("CHANGEC1" in row and row["CHANGEC1"] is None for row in current.values())},
            "panel_cert_coverage": {"panel_distinct_certs": len(panel_certs), **dict(coverage)},
            "history_fi": history_issues, "failure_reconciliation": event_reconciliation,
            "candidate_retrospective_coverage_by_quarter": by_quarter,
            "candidate_retrospective_coverage_overall": dict(all_counts),
            "candidate_diagnostics_overall": dict(all_diagnostics),
            "candidate_2025q2": by_quarter[-1],
            "panel_hashes_verified": len(panel_hashes),
            "failure_disposition_source": failure_summary["run_id"]}


def run(root: Path, run_id: str) -> dict:
    root = root.resolve(strict=True)
    if not run_id.isalnum() or len(run_id) > 60:
        raise ValueError("Unsafe audit run ID")
    evidence = root / "runs" / "task_4_2_active_membership" / run_id
    if evidence.exists():
        raise FileExistsError("Audit run already exists")
    evidence.mkdir(parents=True)
    manifest = RunManifest.new(run_id, load_config(root / "config" / "financials_full_v2.json"))
    manifest.set_status("running")
    session = requests.Session()
    session.headers.update({"User-Agent": "BFT65B-membership-audit/1.0"})
    try:
        institutions, ir, ii = _fetch_institutions(session, evidence)
        history, hr, hi = _fetch_history(session, evidence)
        for record in ir + hr:
            manifest.record_input(record["path"], record["sha256"], record["rows"], record["total"])
        result = _audit(root, institutions, history, ir, hr)
        result.update({"run_id": run_id, "status": "audit_complete",
                       "production_membership_status": "BLOCKED",
                       "blocker": "No verified historical public-knowledge dates or justified bound for entry/exit events",
                       "institutions_pages": ir, "history_years": hr,
                       "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                       "history_index": list(hi)})
        _write_json(evidence / "summary.json", result)
        manifest.record_validation("source_completeness_and_snapshot_reconciliation", True,
                                   "Complete Institutions and yearly institution-role History partitions")
        manifest.record_validation("historical_knowability_assessed", True,
                                   "Production membership blocked: first public availability not established")
        manifest.record_output(path=str(evidence / "summary.json"), dataset="membership_audit",
            rows=len(hr), sha256=_sha(evidence / "summary.json"), compression="none",
            schema_version="1", partition=None, source_total=len(hr), source_query=None)
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return {"status": result["status"], "production_membership_status": "BLOCKED",
                "institutions": len(institutions), "history_fi": len(history),
                "summary": str(evidence / "summary.json")}
    except Exception as exc:
        manifest.record_validation("membership_audit", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def analyze_saved(root: Path, source_run_id: str, run_id: str, *,
                  panel_run: str | None = None, pit_run: str | None = None,
                  repair_run: str | None = None) -> dict:
    """Reanalyse immutable, checksummed source responses without network I/O."""
    root = root.resolve(strict=True)
    if any(not name.isalnum() or len(name) > 60 for name in (source_run_id, run_id)):
        raise ValueError("Unsafe audit run ID")
    source = root / "runs" / "task_4_2_active_membership" / source_run_id
    portable = (root / "runs/membership_source_bundle_v1.json").exists()
    if portable:
        if (source_run_id != "activeaudit20261119a"
                or any(value is None for value in (panel_run, pit_run, repair_run))):
            raise ValueError("Portable membership audit requires explicit captured source, panel, PIT and repair runs")
        from bft65b.panel.membership_source_vintage import relocated_capture
        prior, source_manifest, member_bundle_sha = relocated_capture(root)
    else:
        prior = _read_json(source / "summary.json")
        source_manifest = RunManifest.read(source / "run_manifest.json")
        member_bundle_sha = None
    panel_run = panel_run or "panel20261115b"
    pit_run = pit_run or "pit20261115f"
    repair_run = repair_run or "repair20261115f"
    if source_manifest.status != "complete" or prior.get("status") != "audit_complete":
        raise ValueError("Source audit is incomplete")
    if portable:
        pit_manifest = RunManifest.read(root / "runs/task_2_5_prediction" / pit_run / "run_manifest.json")
        repair_manifest = RunManifest.read(root / "runs/task_2_3_transaction_repair" / repair_run / "run_manifest.json")
        raw_bundle_sha = _sha(root / "runs/source_vintage_bundle.json")
        if (pit_manifest.config.get("source_panel_run") != panel_run
                or pit_manifest.config.get("source_repair_run") != repair_run
                or repair_manifest.config.get("source_panel_run") != panel_run
                or pit_manifest.config.get("source_vintage_bundle_sha256") != raw_bundle_sha
                or repair_manifest.config.get("source_vintage_bundle_sha256") != raw_bundle_sha):
            raise ValueError("Portable membership audit upstream lineage differs")
    evidence = root / "runs" / "task_4_2_active_membership" / run_id
    if evidence.exists():
        raise FileExistsError("Audit run already exists")
    institutions, history = [], []
    for key, target in (("institutions_pages", institutions), ("history_years", history)):
        for record in prior[key]:
            path = Path(record["path"]).resolve(strict=True)
            if not path.is_relative_to(source.resolve()) or _sha(path) != record["sha256"]:
                raise ValueError(f"Saved source path/checksum differs: {path}")
            payload = _read_json(path)
            if (payload["meta"]["total"] != record["total"] or
                    len(payload["data"]) != record["rows"] or
                    [payload["meta"]["index"][part] for part in ("name", "createTimestamp")]
                    != record["index"]):
                raise ValueError(f"Saved source metadata differs: {path}")
            target.extend(item["data"] for item in payload["data"])
    evidence.mkdir(parents=True)
    manifest = RunManifest.new(run_id, load_config(root / "config" / "financials_full_v2.json"))
    manifest.set_status("running")
    try:
        for record in prior["institutions_pages"] + prior["history_years"]:
            manifest.record_input(record["path"], record["sha256"], record["rows"], record["total"])
        result = _audit(root, institutions, history,
                        prior["institutions_pages"], prior["history_years"],
                        panel_run=panel_run, pit_run=pit_run, repair_run=repair_run)
        if result["history_fi"]["missing_source_id"] or result["history_fi"]["duplicate_source_ids"]:
            raise ValueError("History source identity incomplete or duplicated")
        result.update({"run_id": run_id, "status": "audit_complete",
            "source_run_id": source_run_id, "production_membership_status": "BLOCKED",
            "blocker": "Current Institution/History snapshots lack verified first public availability or a defensible historical upper bound for entries and exits",
            "institutions_pages": prior["institutions_pages"], "history_years": prior["history_years"],
            "retrieved_at_utc": prior["retrieved_at_utc"], "reanalyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "history_index": prior["history_index"]})
        manifest.config.update({"source_capture_run": source_run_id,
            "source_panel_run": panel_run, "source_pit_run": pit_run,
            "source_repair_run": repair_run})
        if portable:
            from bft65b.panel.membership_source_vintage import BUNDLE
            manifest.config.update({"membership_source_bundle_sha256": member_bundle_sha,
                "source_vintage_bundle_sha256": _sha(root / "runs/source_vintage_bundle.json")})
            manifest.record_input(str(root / BUNDLE), member_bundle_sha)
        result["source_runs"] = {"capture": source_run_id, "panel": panel_run,
                                 "pit": pit_run, "repair": repair_run}
        _write_json(evidence / "summary.json", result)
        manifest.record_validation("saved_source_and_reconciliation", True,
            "All 45 saved source files and 166 candidate quarters reconciled")
        manifest.record_output(path=str(evidence / "summary.json"), dataset="membership_audit",
            rows=len(result["candidate_retrospective_coverage_by_quarter"]),
            sha256=_sha(evidence / "summary.json"), compression="none",
            schema_version="2", partition=None,
            source_total=len(result["candidate_retrospective_coverage_by_quarter"]), source_query=None)
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return {"status": result["status"], "production_membership_status": "BLOCKED",
                "institutions": len(institutions), "history_fi": len(history),
                "summary": str(evidence / "summary.json")}
    except Exception as exc:
        manifest.record_validation("membership_audit", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str) -> dict:
    """Read back a complete audit and its saved source hashes; no API calls."""
    evidence = root.resolve(strict=True) / "runs" / "task_4_2_active_membership" / run_id
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary_path = evidence / "summary.json"
    summary = _read_json(summary_path)
    if (manifest.status != "complete" or not manifest.validations
            or any(not check["passed"] for check in manifest.validations)
            or summary.get("status") != "audit_complete"
            or summary.get("run_id") != run_id
            or len(manifest.outputs) != 1
            or manifest.outputs[0]["sha256"] != _sha(summary_path)):
        raise ValueError(f"Incomplete or inconsistent audit: {run_id}")
    portable = (root / "runs/membership_source_bundle_v1.json").exists()
    if portable:
        from bft65b.panel.membership_source_vintage import BUNDLE, inventory
        inventory(root)
        sources = summary.get("source_runs", {})
        if (manifest.config.get("membership_source_bundle_sha256") != _sha(root / BUNDLE)
                or manifest.config.get("source_vintage_bundle_sha256") != _sha(root / "runs/source_vintage_bundle.json")
                or any(manifest.config.get(f"source_{key}_run") != sources.get(name)
                       for key, name in (("capture", "capture"), ("panel", "panel"),
                                         ("pit", "pit"), ("repair", "repair")))):
            raise ValueError("Portable membership audit saved lineage differs")
    for item in manifest.inputs:
        if portable and not Path(item["path"]).resolve().is_relative_to(root.resolve()):
            raise ValueError("Portable membership audit input escapes root")
        if _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Saved source checksum differs: {item['path']}")
    if (len(summary["candidate_retrospective_coverage_by_quarter"]) != 166
            or sum(summary["candidate_retrospective_coverage_overall"].values()) != 3697094):
        raise ValueError("Saved candidate coverage does not reconcile")
    return {"status": "complete", "run_id": run_id, "source_files_verified": len(manifest.inputs),
            "candidate_bank_quarters": 3697094,
            "production_membership_status": summary["production_membership_status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official active-bank sources without publishing membership")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--from-run", help="Reanalyse a saved complete source run without network requests")
    parser.add_argument("--panel-run")
    parser.add_argument("--pit-run")
    parser.add_argument("--repair-run")
    parser.add_argument("--verify", action="store_true", help="Verify saved audit evidence without network requests")
    args = parser.parse_args()
    result = (verify_saved(args.project_root, args.run_id) if args.verify else
              analyze_saved(args.project_root, args.from_run, args.run_id,
                  panel_run=args.panel_run, pit_run=args.pit_run, repair_run=args.repair_run)
              if args.from_run else run(args.project_root, args.run_id))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
