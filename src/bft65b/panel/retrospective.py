"""Reconstruct an auditable historical activity cohort from accepted FDIC snapshots.

Membership is retrospective study design, not a claim of a public registry
available at each prediction quarter. Predictor selection remains the accepted
120-day as-of rule and never reads these structural fields as features.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.acquisition.extended import full_plan
from bft65b.config import load_config
from bft65b.manifest import RunManifest
from bft65b.panel.active_audit import verify_saved as verify_audit
from bft65b.panel.build import _write_verified
from bft65b.panel.membership_rule import (
    RETROSPECTIVE_POLICY_VERSION, ORDINARY_MERGER_CODES,
    OTHER_EXIT_CODES, FAILURE_RESOLUTION_CODES, BankEvidence,
    derive_bank_evidence, retrospective_membership, source_date,
)
from bft65b.panel.prediction import verify_saved as verify_prediction
from bft65b.panel.repair import _accepted_panel
from bft65b.quality.audit import _read_json, _sha, _write_json


AUDIT_RUN = "activeaudit20261119e"
PANEL_RUN = "panel20261115b"
CANDIDATE_RUN = "pit20261115f"
CODE_PROBE = "codeprobe20261119a"
CODE_PROBE_SHA256 = "cec7f7d7792bd81b509fcd8a60c5c08d1103c42e94bf89395e3aff265b846268"
RUN_NAMESPACE = "task_4_2_membership"
DATA_NAMESPACE = "task_4_2_membership"

EXPECTED_CODE_DESCRIPTIONS = {
    110: "New Institution",
    150: "Cert Changed from Old Cert by Administrative Order",
    211: "Failure - Whole Institution",
    213: "Merger - Assisted",
    215: "Failure Multiple Acquirer",
    216: "Bridge Bank Resolution",
    217: "Passthrough Receivorship/Conservatorship Resolution",
    221: "Absorbtion - Without Assistance",
    222: "Consolidated - Without Assistance",
    223: "Merger -Without Assistance",
    224: "Affiliated Institution Merger (Pooling) (i.e. Corporate Reorganization",
    225: "Partial Purchase & Assumption - Without Assistance",
    230: "Closing - Failure Payoff",
    235: "Rtc Supervised Payoffs, Liquidations, and Closings",
    240: "Closing - Voluntary",
    260: "Bank Closed by Chartering Agency Pending Sale",
    410: "Change in Insurance Status - ONI Becomes Insured",
    810: "Participated in Absorbtion/Consolidation/Merger",
    811: "Participated in FDIC Assisted Merger",
    812: "Rtc Assisted Merger",
    830: "Open Bank Assistance",
}

BANK_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("BANK_DISPOSITION", pa.string(), nullable=False),
    pa.field("EVIDENCE_SOURCE", pa.string(), nullable=False),
    pa.field("START_DATE", pa.date32()), pa.field("END_DATE", pa.date32()),
    pa.field("INSURED_DATE", pa.date32()),
    pa.field("INSURANCE_DATE_STATUS", pa.string(), nullable=False),
    pa.field("FIRST_REPORT", pa.date32(), nullable=False),
    pa.field("LAST_REPORT", pa.date32(), nullable=False),
    pa.field("REPORT_COUNT", pa.int32(), nullable=False),
    pa.field("HISTORY_SOURCE_IDS", pa.list_(pa.string())),
    pa.field("CERT_CHANGE_DATES", pa.list_(pa.date32())),
    pa.field("CONFLICT_START", pa.date32()),
    pa.field("HISTORY_EXIT_CONFLICT_START", pa.date32()),
    pa.field("HISTORY_EXIT_CONFLICT_END", pa.date32()),
    pa.field("BANK_REASON", pa.string(), nullable=False),
], metadata={b"bft65b.membership": RETROSPECTIVE_POLICY_VERSION.encode()})

QUARTER_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("Q", pa.date32(), nullable=False),
    pa.field("MEMBERSHIP", pa.string(), nullable=False),
    pa.field("MEMBERSHIP_REASON", pa.string(), nullable=False),
    pa.field("EVIDENCE_SOURCE", pa.string(), nullable=False),
    pa.field("START_DATE", pa.date32()), pa.field("END_DATE", pa.date32()),
    pa.field("INSURANCE_DATE_STATUS", pa.string(), nullable=False),
    pa.field("FEATURE_HISTORY_ELIGIBLE", pa.bool_(), nullable=False),
    pa.field("SELECTED_REPDTE", pa.date32()),
    pa.field("AVAILABILITY_DATE", pa.date32()),
    pa.field("P2_4_DISPOSITION_DIAGNOSTIC", pa.string()),
], metadata={b"bft65b.membership": RETROSPECTIVE_POLICY_VERSION.encode()})


def _root(root: Path) -> Path:
    resolved = root.resolve(strict=True)
    if not (resolved / "final_report_bft65b.pdf").is_file():
        raise ValueError("Project root lacks the authoritative assignment")
    return resolved


def _run_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", value):
        raise ValueError("Unsafe run ID")


def _quarter(value: str) -> date:
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def load_verified_sources(root: Path, audit_run: str = AUDIT_RUN,
                          code_probe: str = CODE_PROBE) -> tuple[dict[int, dict], dict[int, list[dict]], dict, dict[str, str]]:
    """Verify the saved audit, every captured API response and code probe."""
    checked = verify_audit(root, audit_run)
    from bft65b.panel.membership_source_vintage import BUNDLE
    portable = (root / BUNDLE).is_file()
    if checked["source_files_verified"] != (46 if portable else 45):
        raise ValueError("Captured source inventory differs")
    evidence = root / "runs" / "task_4_2_active_membership" / audit_run
    summary = _read_json(evidence / "summary.json")
    probe_path = root / "runs" / "task_4_2_membership_code_probe" / f"{code_probe}.json"
    if _sha(probe_path) != CODE_PROBE_SHA256:
        raise ValueError("Official code-description probe checksum differs")
    probe = _read_json(probe_path)
    if probe["source_index"] != summary["history_index"]:
        raise ValueError("History code probe source index differs")
    by_code = {int(item["code"]): item for item in probe["rows"]}
    if len(by_code) != 36:
        raise ValueError("History code probe coverage differs")
    for code, description in EXPECTED_CODE_DESCRIPTIONS.items():
        item = by_code.get(code)
        if (item is None or item["descriptions"] != [description]
                or item["missing_description_samples"] != 0
                or not 1 <= item["sample_count"] <= 5):
            raise ValueError(f"Official code-description evidence differs: {code}")
    institutions: dict[int, dict] = {}
    history: dict[int, list[dict]] = defaultdict(list)
    hashes = {str(evidence / "summary.json"): _sha(evidence / "summary.json"),
              str(evidence / "run_manifest.json"): _sha(evidence / "run_manifest.json"),
              str(probe_path): CODE_PROBE_SHA256}
    allowed_source = (root / "runs" / "task_4_2_active_membership" /
                      "activeaudit20261119a").resolve() if portable else None
    for key in ("institutions_pages", "history_years"):
        for item in summary[key]:
            path = Path(item["path"]).resolve(strict=True)
            if portable and not path.is_relative_to(allowed_source):
                raise ValueError("Membership capture escaped portable source root")
            if _sha(path) != item["sha256"]:
                raise ValueError(f"Saved FDIC source checksum differs: {path}")
            hashes[str(path)] = item["sha256"]
            payload = _read_json(path)
            if (payload["meta"]["total"] != item["total"]
                    or len(payload["data"]) != item["rows"]
                    or [payload["meta"]["index"][part] for part in ("name", "createTimestamp")]
                    != item["index"]):
                raise ValueError(f"Saved FDIC source count/index differs: {path}")
            for wrapped in payload["data"]:
                row = wrapped["data"]
                if key == "institutions_pages":
                    cert = row["CERT"]
                    if cert in institutions:
                        raise ValueError("Repeated Institutions CERT")
                    institutions[cert] = row
                elif row.get("CERT") is not None:
                    history[row["CERT"]].append(row)
    if len(institutions) != 27835 or sum(map(len, history.values())) != 116867:
        raise ValueError("Supplementary Institutions or linked FI History scope differs")
    return institutions, history, summary, hashes


def build_banks(institutions: dict[int, dict], history: dict[int, list[dict]],
                reports: dict[int, list[date]]) -> dict[int, BankEvidence]:
    result = {cert: derive_bank_evidence(cert, institutions.get(cert),
                                        history.get(cert, []), dates)
              for cert, dates in reports.items()}
    if len(result) != 24064:
        raise ValueError("Observed bank identity count differs")
    return result


def bank_table(banks: dict[int, BankEvidence]) -> pa.Table:
    rows = []
    for bank in sorted(banks.values(), key=lambda item: item.cert):
        interval = bank.intervals[0] if bank.intervals else None
        conflict = bank.history_end_conflict
        rows.append({"CERT": bank.cert, "BANK_DISPOSITION": bank.disposition,
            "EVIDENCE_SOURCE": bank.source,
            "START_DATE": interval.start if interval else None,
            "END_DATE": interval.end if interval else None,
            "INSURED_DATE": bank.insured_date,
            "INSURANCE_DATE_STATUS": bank.insurance_date_status,
            "FIRST_REPORT": bank.report_dates[0], "LAST_REPORT": bank.report_dates[-1],
            "REPORT_COUNT": len(bank.report_dates),
            "HISTORY_SOURCE_IDS": list(bank.history_ids),
            "CERT_CHANGE_DATES": list(bank.cert_change_dates),
            "CONFLICT_START": bank.conflict_start,
            "HISTORY_EXIT_CONFLICT_START": conflict[0] if conflict else None,
            "HISTORY_EXIT_CONFLICT_END": conflict[1] if conflict else None,
            "BANK_REASON": bank.reason})
    return pa.Table.from_pylist(rows, schema=BANK_SCHEMA)


def quarter_table(q: date, banks: dict[int, BankEvidence],
                  candidates: dict[int, dict]) -> tuple[pa.Table, dict]:
    """Cover every observed CERT, including banks with no available features."""
    if set(candidates) - set(banks):
        raise ValueError("Prediction candidate lacks observed-bank identity")
    rows = []
    counts = Counter()
    insurance_sensitivity = 0
    for cert in sorted(banks):
        bank = banks[cert]
        decision = retrospective_membership(bank, q)
        candidate = candidates.get(cert)
        available = candidate is not None
        if candidate is not None and (candidate["Q"] != q
                or candidate["AVAILABILITY_DATE"] is None
                or candidate["AVAILABILITY_DATE"] > q):
            raise ValueError(f"Future or wrong-quarter feature history: {cert}, {q}")
        counts[decision.status] += 1
        counts["candidate" if available else "no_feature_history"] += 1
        if decision.status == "eligible":
            counts["eligible_with_feature_history" if available else "eligible_no_feature_history"] += 1
            if (bank.insured_date is None or bank.insured_date > q):
                insurance_sensitivity += 1
        rows.append({"CERT": cert, "Q": q, "MEMBERSHIP": decision.status,
            "MEMBERSHIP_REASON": decision.reason,
            "EVIDENCE_SOURCE": decision.evidence_source,
            "START_DATE": decision.start, "END_DATE": decision.end,
            "INSURANCE_DATE_STATUS": bank.insurance_date_status,
            "FEATURE_HISTORY_ELIGIBLE": available,
            "SELECTED_REPDTE": candidate["SELECTED_REPDTE"] if available else None,
            "AVAILABILITY_DATE": candidate["AVAILABILITY_DATE"] if available else None,
            "P2_4_DISPOSITION_DIAGNOSTIC": candidate["ELIGIBILITY_DISPOSITION"] if available else None})
    if counts["candidate"] != len(candidates) or sum(counts[k] for k in (
            "eligible", "ineligible", "uncertain")) != len(banks):
        raise ValueError("Membership quarter reconciliation differs")
    return pa.Table.from_pylist(rows, schema=QUARTER_SCHEMA), {
        "quarter": q.strftime("%Y%m%d"), "observed_bank_identities": len(banks),
        **dict(counts), "insurance_start_sensitivity_eligible_bank_quarters": insurance_sensitivity}


def run(root: Path, run_id: str, *, audit_run: str = AUDIT_RUN,
        panel_run: str = PANEL_RUN, candidate_run: str = CANDIDATE_RUN,
        code_probe: str = CODE_PROBE) -> dict:
    root = _root(root)
    _run_id(run_id)
    evidence = root / "runs" / RUN_NAMESPACE / run_id
    data = root / "data_processed" / DATA_NAMESPACE / run_id
    if evidence.exists() or data.exists():
        raise FileExistsError("Membership run ID already exists")
    config = load_config(root / "config" / "financials_full_v2.json")
    quarters = tuple(q for q in full_plan(config).quarters if q <= "20250630")
    if len(quarters) != 166:
        raise ValueError("Membership prediction-quarter scope differs")
    from bft65b.panel.membership_source_vintage import BUNDLE
    portable = (root / BUNDLE).is_file()
    if portable and (audit_run == AUDIT_RUN or panel_run == PANEL_RUN or
                     candidate_run == CANDIDATE_RUN):
        raise ValueError("Portable cohort requires explicit upstream run selection")
    audit_manifest = RunManifest.read(root / "runs" / "task_4_2_active_membership" /
                                      audit_run / "run_manifest.json")
    if portable and (audit_manifest.config.get("source_panel_run") != panel_run or
                     audit_manifest.config.get("source_pit_run") != candidate_run):
        raise ValueError("Cohort audit/PIT/panel lineage differs")
    institutions, history, audit, source_hashes = load_verified_sources(root, audit_run, code_probe)
    panel, _, _, reports, _, panel_hashes = _accepted_panel(root, panel_run)
    if panel["panel_overall"]["output_rows"] != 1682616:
        raise ValueError("Accepted panel row total differs")
    verify_prediction(root, candidate_run, check_upstream=True)
    candidate_summary = _read_json(root / "runs" / "task_2_5_prediction" /
                                   candidate_run / "summary.json")
    if len(candidate_summary["outputs"]) != 166:
        raise ValueError("Accepted candidate quarter count differs")
    banks = build_banks(institutions, history, reports)
    manifest = RunManifest.new(run_id, config)
    manifest.config["membership_policy"] = RETROSPECTIVE_POLICY_VERSION
    manifest.config["design"] = "retrospective_historical_cohort_not_operational_registry"
    manifest.config.update({"source_audit_run": audit_run, "source_panel_run": panel_run,
        "source_candidate_run": candidate_run, "source_code_probe": code_probe})
    if portable:
        manifest.config["membership_source_bundle_sha256"] = _sha(root / BUNDLE)
        manifest.config["source_vintage_bundle_sha256"] = _sha(root / "runs/source_vintage_bundle.json")
        source_hashes[str(root / BUNDLE)] = manifest.config["membership_source_bundle_sha256"]
    for path, digest in sorted({**source_hashes, **panel_hashes}.items()):
        manifest.record_input(path, digest)
    outputs = []
    by_quarter = []
    evidence.mkdir(parents=True)
    data.mkdir(parents=True)
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    try:
        table = bank_table(banks)
        bank_path = data / "banks" / "part-000.parquet"
        bank_hash = _write_verified(bank_path, table)
        manifest.record_output(path=str(bank_path), dataset="bank_evidence", rows=table.num_rows,
            sha256=bank_hash, compression="ZSTD", schema_version=RETROSPECTIVE_POLICY_VERSION,
            partition=None, source_total=table.num_rows, source_query=None)
        outputs.append({"path": str(bank_path), "sha256": bank_hash, "rows": table.num_rows,
                        "dataset": "bank_evidence"})
        old_missing = set(reports) - set(institutions)
        old_conflicts = {cert for cert in reports if cert in institutions
            and source_date(institutions[cert]["ENDEFYMD"]) <= source_date(institutions[cert]["INSDATE"])}
        bank_dispositions = {"missing_institution": dict(Counter(
                banks[cert].disposition + ":" + banks[cert].source for cert in old_missing)),
            "prior_insurance_interval_conflict": dict(Counter(
                banks[cert].disposition + ":" + banks[cert].source for cert in old_conflicts)),
            "all_observed_banks": dict(Counter(bank.disposition + ":" + bank.source
                for bank in banks.values()))}
        for qtext, item in zip(quarters, candidate_summary["outputs"]):
            if item["quarter"] != qtext:
                raise ValueError("Candidate quarter order differs")
            q = _quarter(qtext)
            candidate_path = Path(item["path"])
            if _sha(candidate_path) != item["sha256"]:
                raise ValueError(f"Candidate checksum differs: {qtext}")
            manifest.record_input(str(candidate_path), item["sha256"], item["rows"])
            with pq.ParquetFile(candidate_path) as parquet:
                candidate_rows = parquet.read(columns=["CERT", "Q", "AVAILABILITY_DATE",
                    "SELECTED_REPDTE", "ELIGIBILITY_DISPOSITION"]).to_pylist()
            candidates = {row["CERT"]: row for row in candidate_rows}
            if len(candidates) != item["rows"]:
                raise ValueError(f"Duplicate candidate key: {qtext}")
            quarter, stats = quarter_table(q, banks, candidates)
            target = data / "quarters" / f"Q={qtext}" / "part-000.parquet"
            digest = _write_verified(target, quarter)
            outputs.append({"path": str(target), "sha256": digest,
                            "rows": quarter.num_rows, "quarter": qtext, "dataset": "membership"})
            manifest.record_output(path=str(target), dataset="membership", rows=quarter.num_rows,
                sha256=digest, compression="ZSTD", schema_version=RETROSPECTIVE_POLICY_VERSION,
                partition=qtext, source_total=quarter.num_rows, source_query=None)
            by_quarter.append(stats)
            if len(by_quarter) % 20 == 0:
                manifest.write(evidence / "run_manifest.json")
        overall = dict(Counter({key: sum(row.get(key, 0) for row in by_quarter)
                                for key in set().union(*(row.keys() for row in by_quarter))
                                if key not in {"quarter", "observed_bank_identities"}}))
        if (len(outputs) != 167 or overall["candidate"] != 3697094
                or overall["eligible"] + overall["ineligible"] + overall["uncertain"]
                != 24064 * 166):
            raise ValueError("Full membership output reconciliation differs")
        summary = {"status": "complete", "run_id": run_id,
                   "policy_version": RETROSPECTIVE_POLICY_VERSION,
                   "design": "retrospective_historical_activity_cohort",
                   "not_operational_historical_registry": True,
                   "source_runs": {"audit": audit_run, "panel": panel_run,
                                   "candidates": candidate_run, "code_probe": code_probe},
                   "bank_dispositions": bank_dispositions,
                   "overall": overall, "by_quarter": by_quarter,
                   "outputs": outputs,
                   "source_hash_count": len(source_hashes),
                   "panel_hash_count": len(panel_hashes),
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "limitations": ["Current structural snapshot defines a retrospective cohort; public historical vintages are unavailable.",
                                   "Missing identities and inconsistent dates remain uncertain.",
                                   "Insurance date is audited separately from operational establishment."]}
        _write_json(evidence / "summary.json", summary)
        metadata = {"status": "complete", "summary_sha256": _sha(evidence / "summary.json"),
                    "output_count": len(outputs), "source_hash_count": len(source_hashes)}
        _write_json(evidence / "metadata.json", metadata)
        manifest.record_validation("accepted_inputs_and_45_source_hashes", True,
                                   "Verified audit, code probe, panel and 166 as-of candidate files")
        manifest.record_validation("all_bank_quarters_and_uncertainty", True,
                                   "166 quarters x 24,064 observed CERTs; unresolved decisions retained")
        manifest.record_validation("atomic_parquet_and_summary", True,
                                   "167 files read back, schema/codec/key/count/checksum reconciled")
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return {"run_id": run_id, "status": "complete", "overall": overall,
                "summary": str(evidence / "summary.json")}
    except Exception as exc:
        manifest.record_validation("membership_run", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str) -> dict:
    root = _root(root)
    _run_id(run_id)
    evidence = root / "runs" / RUN_NAMESPACE / run_id
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    sources = summary.get("source_runs", {})
    if (sources.get("audit") != manifest.config.get("source_audit_run", AUDIT_RUN) or
            sources.get("panel") != manifest.config.get("source_panel_run", PANEL_RUN) or
            sources.get("candidates") != manifest.config.get("source_candidate_run", CANDIDATE_RUN) or
            sources.get("code_probe") != manifest.config.get("source_code_probe", CODE_PROBE)):
        raise ValueError("Membership recorded source lineage differs")
    from bft65b.panel.membership_source_vintage import BUNDLE, inventory
    if (root / BUNDLE).is_file():
        inventory(root)
        if (manifest.config.get("membership_source_bundle_sha256") != _sha(root / BUNDLE) or
                manifest.config.get("source_vintage_bundle_sha256") != _sha(root / "runs/source_vintage_bundle.json")):
            raise ValueError("Membership portable source bundle differs")
    if (manifest.status != "complete" or not manifest.validations
            or any(not item["passed"] for item in manifest.validations)
            or summary.get("run_id") != run_id or summary.get("status") != "complete"
            or summary.get("policy_version") != RETROSPECTIVE_POLICY_VERSION
            or metadata.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or len(summary["outputs"]) != 167 or len(manifest.outputs) != 167):
        raise ValueError("Membership manifest/summary/metadata incomplete")
    for item in manifest.inputs:
        if ((root / "runs/source_vintage_bundle.json").is_file() and
                not Path(item["path"]).resolve().is_relative_to(root)):
            raise ValueError("Portable membership input escaped root")
        if item["sha256"] and _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Membership upstream checksum differs: {item['path']}")
    recorded = {item["path"]: item for item in manifest.outputs}
    seen_quarters = set()
    for item in summary["outputs"]:
        path = Path(item["path"])
        expected_schema = BANK_SCHEMA if item["dataset"] == "bank_evidence" else QUARTER_SCHEMA
        if (not path.resolve().is_relative_to((root / "data_processed" / DATA_NAMESPACE / run_id).resolve())
                or recorded.get(str(path), {}).get("sha256") != item["sha256"]
                or _sha(path) != item["sha256"]):
            raise ValueError(f"Membership output path/checksum differs: {path}")
        with pq.ParquetFile(path) as parquet:
            if (parquet.schema_arrow != expected_schema
                    or parquet.metadata.num_rows != item["rows"]
                    or any(parquet.metadata.row_group(g).column(c).compression != "ZSTD"
                           for g in range(parquet.metadata.num_row_groups)
                           for c in range(parquet.metadata.row_group(g).num_columns))):
                raise ValueError(f"Membership output schema/codec/rows differ: {path}")
            if item["dataset"] == "membership":
                quarter = item["quarter"]
                seen_quarters.add(quarter)
                keys = parquet.read(columns=["CERT", "Q"])
                if (len(set(keys["CERT"].to_pylist())) != item["rows"]
                        or any(day != _quarter(quarter) for day in keys["Q"].to_pylist())):
                    raise ValueError(f"Membership keys differ: {quarter}")
    if (len(seen_quarters) != 166 or
            summary["overall"]["eligible"] + summary["overall"]["ineligible"] +
            summary["overall"]["uncertain"] != 24064 * 166):
        raise ValueError("Membership complete-quarter reconciliation differs")
    return {"status": "complete", "run_id": run_id,
            "output_files": len(summary["outputs"]), "quarters": len(seen_quarters),
            "overall": summary["overall"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospective active-bank cohort")
    parser.add_argument("mode", choices=("run", "verify", "show"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audit-run", default=AUDIT_RUN)
    parser.add_argument("--panel-run", default=PANEL_RUN)
    parser.add_argument("--candidate-run", default=CANDIDATE_RUN)
    parser.add_argument("--code-probe", default=CODE_PROBE)
    args = parser.parse_args()
    if args.mode == "run":
        result = run(args.project_root, args.run_id, audit_run=args.audit_run,
                     panel_run=args.panel_run, candidate_run=args.candidate_run,
                     code_probe=args.code_probe)
    elif args.mode == "verify":
        result = verify_saved(args.project_root, args.run_id)
    else:
        verify_saved(args.project_root, args.run_id)
        result = _read_json(_root(args.project_root) / "runs" / RUN_NAMESPACE /
                            args.run_id / "summary.json")
        result = {"run_id": result["run_id"], "status": result["status"],
                  "overall": result["overall"], "q2025q2": result["by_quarter"][-1],
                  "bank_dispositions": result["bank_dispositions"]}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
