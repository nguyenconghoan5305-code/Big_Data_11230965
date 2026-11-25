"""Production four-quarter labels for the retrospective activity cohort.

The historical cohort is reconstructed from effective structural dates. It is
not an operational claim that these registry vintages were public at q.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.config import load_config
from bft65b.features.label import (
    LABEL_RULE_VERSION, MembershipEvidence, OutcomeEvent, classify_label,
    _verify_repair,
)
from bft65b.manifest import RunManifest
from bft65b.panel.build import _write_verified
from bft65b.panel.retrospective import (
    RETROSPECTIVE_POLICY_VERSION, RUN_NAMESPACE as MEMBERSHIP_RUN_NAMESPACE,
    _quarter, _root, _run_id, verify_saved as verify_membership,
)
from bft65b.quality.audit import _read_json, _sha, _write_json


RUN_NAMESPACE = "task_4_2_production_labels"
DATA_NAMESPACE = "task_4_2_production_labels"
LABEL_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("Q", pa.date32(), nullable=False),
    pa.field("RIGHT_ENDPOINT", pa.date32(), nullable=False),
    pa.field("MEMBERSHIP", pa.string(), nullable=False),
    pa.field("MEMBERSHIP_REASON", pa.string(), nullable=False),
    pa.field("FEATURE_HISTORY_ELIGIBLE", pa.bool_(), nullable=False),
    pa.field("LABEL_DISPOSITION", pa.string(), nullable=False),
    pa.field("Y", pa.int8()),
    pa.field("LABEL_REASON", pa.string(), nullable=False),
    pa.field("QUALIFYING_SOURCE_IDS", pa.list_(pa.string()), nullable=False),
], metadata={b"bft65b.label": LABEL_RULE_VERSION.encode(),
             b"bft65b.membership": RETROSPECTIVE_POLICY_VERSION.encode()})


def label_quarter(memberships: list[dict], events_by_cert: dict[int, list[OutcomeEvent]],
                  unassigned: list[OutcomeEvent], q: date,
                  *, outcome_complete_through: date) -> tuple[pa.Table, dict]:
    """Apply the accepted label rule to every membership decision once."""
    rows = []
    counts = Counter()
    seen: set[int] = set()
    positive_ids: set[str] = set()
    positive_banks: set[int] = set()
    right = date(q.year + 1, q.month, q.day)
    relevant_unassigned = [event for event in unassigned
                           if q < event.effective_date <= right]
    for row in memberships:
        cert = row["CERT"]
        if cert in seen or row["Q"] != q:
            raise ValueError(f"Duplicate or wrong-quarter membership key: {cert}, {q}")
        seen.add(cert)
        evidence = MembershipEvidence(
            cert=cert, q=q, status=row["MEMBERSHIP"],
            policy=RETROSPECTIVE_POLICY_VERSION, known_at=None,
            source_id=f"{row['EVIDENCE_SOURCE']}:{cert}",
            identity_resolved=row["MEMBERSHIP"] != "uncertain")
        decision = classify_label(evidence,
                                  events_by_cert.get(cert, []) + relevant_unassigned,
                                  outcome_complete_through=outcome_complete_through)
        counts[decision.disposition] += 1
        counts[f"reason:{decision.reason}"] += 1
        if row["FEATURE_HISTORY_ELIGIBLE"]:
            counts["feature_history_eligible"] += 1
            counts[f"feature_history_and_{decision.disposition}"] += 1
        if decision.disposition == "positive":
            positive_banks.add(cert)
            positive_ids.update(decision.qualifying_source_ids)
        rows.append({"CERT": cert, "Q": q, "RIGHT_ENDPOINT": decision.right_endpoint,
                     "MEMBERSHIP": row["MEMBERSHIP"],
                     "MEMBERSHIP_REASON": row["MEMBERSHIP_REASON"],
                     "FEATURE_HISTORY_ELIGIBLE": row["FEATURE_HISTORY_ELIGIBLE"],
                     "LABEL_DISPOSITION": decision.disposition, "Y": decision.y,
                     "LABEL_REASON": decision.reason,
                     "QUALIFYING_SOURCE_IDS": list(decision.qualifying_source_ids)})
    if sum(counts[k] for k in ("positive", "negative", "unresolved", "excluded")) != len(rows):
        raise ValueError("Label dispositions fail to reconcile")
    table = pa.Table.from_pylist(rows, schema=LABEL_SCHEMA)
    return table, {"quarter": q.strftime("%Y%m%d"), "decisions": len(rows),
                   **dict(counts), "distinct_positive_banks": len(positive_banks),
                   "distinct_qualifying_source_events": len(positive_ids)}


def run(root: Path, run_id: str, membership_run: str,
        repair_run: str = "repair20261115f") -> dict:
    root = _root(root)
    _run_id(run_id)
    _run_id(membership_run)
    _run_id(repair_run)
    evidence = root / "runs" / RUN_NAMESPACE / run_id
    data = root / "data_processed" / DATA_NAMESPACE / run_id
    if evidence.exists() or data.exists():
        raise FileExistsError("Production label run ID already exists")
    accepted = verify_membership(root, membership_run)
    membership_evidence = root / "runs" / MEMBERSHIP_RUN_NAMESPACE / membership_run
    membership_manifest = RunManifest.read(membership_evidence / "run_manifest.json")
    audit_run = membership_manifest.config.get("source_audit_run")
    portable = (root / "runs/source_vintage_bundle.json").is_file()
    if portable:
        if (not audit_run or repair_run == "repair20261115f"):
            raise ValueError("Portable labels require explicit selected repair lineage")
        audit_manifest = RunManifest.read(root / "runs/task_4_2_active_membership" /
                                          audit_run / "run_manifest.json")
        if audit_manifest.config.get("source_repair_run") != repair_run:
            raise ValueError("Membership audit and label repair lineage differ")
    repair_summary, events, repair_hashes = _verify_repair(root, repair_run)
    if portable and any(not Path(path).resolve().is_relative_to(root)
                        for path in repair_hashes):
        raise ValueError("Portable label repair input escaped root")
    if accepted["quarters"] != 166 or len(events) != 4117:
        raise ValueError("Membership or repaired failure scope differs")
    membership_summary = _read_json(membership_evidence / "summary.json")
    by_cert: dict[int, list[OutcomeEvent]] = defaultdict(list)
    unassigned = []
    for event in events:
        (unassigned if event.cert is None else by_cert[event.cert]).append(event)
    config = load_config(root / "config" / "financials_full_v2.json")
    manifest = RunManifest.new(run_id, config)
    manifest.config["label_rule"] = LABEL_RULE_VERSION
    manifest.config["membership_run"] = membership_run
    manifest.config["repair_run"] = repair_run
    manifest.config["membership_policy"] = RETROSPECTIVE_POLICY_VERSION
    for path, digest in sorted(repair_hashes.items()):
        manifest.record_input(path, digest)
    for name in ("summary.json", "metadata.json", "run_manifest.json"):
        path = membership_evidence / name
        manifest.record_input(str(path), _sha(path))
    evidence.mkdir(parents=True)
    data.mkdir(parents=True)
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    outputs = []
    by_quarter = []
    positive_banks: set[int] = set()
    positive_events: set[str] = set()
    try:
        for item in membership_summary["outputs"]:
            if item["dataset"] != "membership":
                continue
            path = Path(item["path"])
            if _sha(path) != item["sha256"]:
                raise ValueError(f"Membership quarter checksum differs: {path}")
            manifest.record_input(str(path), item["sha256"], item["rows"])
            q = _quarter(item["quarter"])
            with pq.ParquetFile(path) as file:
                membership_rows = file.read().to_pylist()
            if len(membership_rows) != item["rows"]:
                raise ValueError(f"Membership quarter rows differ: {q}")
            table, stats = label_quarter(membership_rows, by_cert, unassigned, q,
                                         outcome_complete_through=date(2026, 6, 30))
            if stats["decisions"] != 24064:
                raise ValueError(f"Membership-to-label cardinality differs: {q}")
            for row in table.select(["CERT", "LABEL_DISPOSITION", "QUALIFYING_SOURCE_IDS"]).to_pylist():
                if row["LABEL_DISPOSITION"] == "positive":
                    positive_banks.add(row["CERT"])
                    positive_events.update(row["QUALIFYING_SOURCE_IDS"])
            target = data / "quarters" / f"Q={item['quarter']}" / "part-000.parquet"
            digest = _write_verified(target, table)
            manifest.record_output(path=str(target), dataset="production_labels",
                rows=table.num_rows, sha256=digest, compression="ZSTD",
                schema_version=LABEL_RULE_VERSION, partition=item["quarter"],
                source_total=table.num_rows, source_query=None)
            outputs.append({"path": str(target), "sha256": digest,
                            "rows": table.num_rows, "quarter": item["quarter"]})
            by_quarter.append(stats)
            if len(by_quarter) % 20 == 0:
                manifest.write(evidence / "run_manifest.json")
        overall = {key: sum(row.get(key, 0) for row in by_quarter)
                   for key in set().union(*(row.keys() for row in by_quarter))
                   if key not in {"quarter", "distinct_positive_banks",
                                  "distinct_qualifying_source_events"}}
        overall["distinct_positive_banks"] = len(positive_banks)
        overall["distinct_qualifying_source_events"] = len(positive_events)
        if (len(outputs) != 166 or overall["decisions"] != 24064 * 166
                or overall["feature_history_eligible"] != membership_summary["overall"]["candidate"]
                or overall["positive"] + overall["negative"] + overall["unresolved"] + overall["excluded"]
                != overall["decisions"]):
            raise ValueError("Complete production label reconciliation differs")
        summary = {"status": "complete", "run_id": run_id,
                   "membership_run": membership_run,
                   "membership_policy": RETROSPECTIVE_POLICY_VERSION,
                   "label_rule": LABEL_RULE_VERSION,
                   "outcome_complete_through": "2026-06-30",
                   "repaired_event_run": repair_run,
                   "repaired_source_events": len(events),
                   "unassigned_source_events": len(unassigned),
                   "overall": overall, "by_quarter": by_quarter, "outputs": outputs,
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "limitations": ["Retrospective membership is a historical study cohort, not a reconstructed public registry at q.",
                                   "Unresolved identity and structural-date cases retain unresolved decisions.",
                                   "Feature-history eligibility is separate from outcome classification."]}
        _write_json(evidence / "summary.json", summary)
        _write_json(evidence / "metadata.json", {"status": "complete",
                    "summary_sha256": _sha(evidence / "summary.json"),
                    "output_count": len(outputs)})
        manifest.record_validation("upstream_membership_and_repaired_events", True,
            "166 verified membership quarters and 4,117 distinct repaired source events")
        manifest.record_validation("quarterly_label_cardinality_and_followup", True,
            "166 quarters x 24,064 decisions; full right endpoint 2026-06-30")
        manifest.record_validation("atomic_readback_schema_codec_checksum", True,
            "166 Parquet outputs independently read back on write")
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return {"status": "complete", "run_id": run_id, "overall": overall,
                "summary": str(evidence / "summary.json")}
    except Exception as exc:
        manifest.record_validation("production_label_run", False, f"{type(exc).__name__}: {exc}")
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
    if (manifest.config.get("membership_run") != summary.get("membership_run") or
            manifest.config.get("repair_run", "repair20261115f") !=
            summary.get("repaired_event_run")):
        raise ValueError("Production label membership/repair lineage differs")
    if (root / "runs/source_vintage_bundle.json").is_file():
        membership_manifest = RunManifest.read(root / "runs" / MEMBERSHIP_RUN_NAMESPACE /
                                               summary["membership_run"] / "run_manifest.json")
        audit_run = membership_manifest.config.get("source_audit_run")
        audit_manifest = RunManifest.read(root / "runs/task_4_2_active_membership" /
                                          audit_run / "run_manifest.json")
        if (audit_manifest.config.get("source_repair_run") != summary["repaired_event_run"]
                or summary["repaired_event_run"] == "repair20261115f"):
            raise ValueError("Production label portable lineage differs")
    if (manifest.status != "complete" or not manifest.validations
            or any(not item["passed"] for item in manifest.validations)
            or summary.get("status") != "complete" or summary.get("run_id") != run_id
            or summary.get("label_rule") != LABEL_RULE_VERSION
            or metadata.get("status") != "complete"
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or len(summary["outputs"]) != 166 or len(manifest.outputs) != 166):
        raise ValueError("Production label manifest/summary/metadata incomplete")
    for item in manifest.inputs:
        if ((root / "runs/source_vintage_bundle.json").is_file() and
                not Path(item["path"]).resolve().is_relative_to(root)):
            raise ValueError("Portable label input escaped root")
        if item["sha256"] and _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Production label upstream checksum differs: {item['path']}")
    recorded = {item["path"]: item for item in manifest.outputs}
    quarters = set()
    for item in summary["outputs"]:
        path = Path(item["path"])
        if (not path.resolve().is_relative_to((root / "data_processed" / DATA_NAMESPACE / run_id).resolve())
                or _sha(path) != item["sha256"]
                or recorded.get(str(path), {}).get("sha256") != item["sha256"]):
            raise ValueError(f"Production label path/checksum differs: {path}")
        with pq.ParquetFile(path) as file:
            if (file.schema_arrow != LABEL_SCHEMA or file.metadata.num_rows != item["rows"]
                    or any(file.metadata.row_group(g).column(c).compression != "ZSTD"
                           for g in range(file.metadata.num_row_groups)
                           for c in range(file.metadata.row_group(g).num_columns))):
                raise ValueError(f"Production label schema/codec/row count differs: {path}")
            keys = file.read(columns=["CERT", "Q"])
            if (len(set(keys["CERT"].to_pylist())) != item["rows"]
                    or any(day != _quarter(item["quarter"]) for day in keys["Q"].to_pylist())):
                raise ValueError(f"Production label key/quarter differs: {path}")
        quarters.add(item["quarter"])
    if len(quarters) != 166 or summary["overall"]["decisions"] != 24064 * 166:
        raise ValueError("Production label coverage differs")
    return {"status": "complete", "run_id": run_id, "quarters": len(quarters),
            "outputs": len(summary["outputs"]), "overall": summary["overall"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Production retrospective-cohort labels")
    parser.add_argument("mode", choices=("run", "verify", "show"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--membership-run", default="membership20261120a")
    parser.add_argument("--repair-run", default="repair20261115f")
    args = parser.parse_args()
    if args.mode == "run":
        result = run(args.project_root, args.run_id, args.membership_run,
                     args.repair_run)
    elif args.mode == "verify":
        result = verify_saved(args.project_root, args.run_id)
    else:
        verify_saved(args.project_root, args.run_id)
        saved = _read_json(_root(args.project_root) / "runs" / RUN_NAMESPACE /
                           args.run_id / "summary.json")
        result = {"status": saved["status"], "run_id": saved["run_id"],
                  "overall": saved["overall"], "q2025q2": saved["by_quarter"][-1]}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
