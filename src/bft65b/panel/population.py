"""Audit a recent-reporter prediction population without asserting bank activity.

The accepted P2.4 candidate files supply one as-of selected report per CERT/q.
This layer never reads predictors to decide membership and never treats a
retrospective failure, merger, or institution snapshot as known at q.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.manifest import RunManifest, dependency_versions
from bft65b.panel.build import _write_verified
from bft65b.panel.prediction import _day, verify_saved as verify_candidates
from bft65b.quality.audit import _read_json, _sha, _write_json


POLICY_VERSION = "recent_reporter_4_quarters_v1"
SOURCE_RUN = "pit20261115f"
RUN_NAMESPACE = "task_2_6_population"
WINDOWS = (2, 4, 8)
SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("Q", pa.date32(), nullable=False),
    pa.field("SELECTED_REPDTE", pa.date32(), nullable=False),
    pa.field("AVAILABILITY_DATE", pa.date32(), nullable=False),
    pa.field("REPORT_AGE_DAYS", pa.int32(), nullable=False),
    pa.field("CANDIDATE_DISPOSITION", pa.string(), nullable=False),
    pa.field("MEMBERSHIP_DISPOSITION", pa.string(), nullable=False),
    pa.field("MEMBERSHIP_REASON", pa.string(), nullable=False),
    pa.field("EVIDENCE_SOURCE", pa.string(), nullable=False),
    pa.field("EVIDENCE_DATE", pa.date32()),
    pa.field("EVIDENCE_KNOWN_AT", pa.date32()),
    pa.field("POLICY_VERSION", pa.string(), nullable=False),
], metadata={b"bft65b.population_policy": POLICY_VERSION.encode()})


def quarter_distance(older: date, newer: date) -> int:
    """Count quarter-end steps; an exact four-quarter gap is included."""
    if older > newer or (older.month, older.day) not in {
            (3, 31), (6, 30), (9, 30), (12, 31)}:
        raise ValueError("Selected report must be a quarter-end no later than q")
    return (newer.year - older.year) * 4 + (newer.month - older.month) // 3


@dataclass(frozen=True)
class DatedEvent:
    cert: int
    kind: str
    effective_date: date
    known_at: date | None
    source_id: str
    knowledge_verified: bool = False


def classify(row: Mapping[str, Any], *, window_quarters: int = 4,
             events: Iterable[DatedEvent] = ()) -> dict[str, Any]:
    """Classify membership from as-of reports and explicitly verified knowledge.

    The accepted candidate disposition is retained solely as a diagnostic.
    An event with unknown or later knowledge time cannot alter membership.
    """
    q, repdte, available = row["Q"], row["SELECTED_REPDTE"], row["AVAILABILITY_DATE"]
    if available > q or repdte > q or row["REPORT_AGE_DAYS"] != (q - repdte).days:
        raise ValueError("Candidate report is not as-of consistent")
    if window_quarters < 2:
        raise ValueError("Recency window is shorter than the provisional release lag")
    known = [e for e in events if e.cert == row["CERT"] and e.knowledge_verified
             and e.known_at is not None and e.known_at <= q and e.effective_date <= q]
    decisive = [e for e in known if e.kind in {"failure", "merger", "other_exit", "identity_conflict"}]
    if len({e.kind for e in decisive}) > 1 or any(e.kind == "identity_conflict" for e in decisive):
        disposition, reason, event = "unresolved_identity", "conflicting_known_event", decisive[0]
    elif decisive:
        event = min(decisive, key=lambda e: (e.effective_date, e.source_id))
        disposition, reason = "known_exit", f"known_{event.kind}_by_q"
    elif quarter_distance(repdte, q) <= window_quarters:
        disposition, reason, event = "recent_reporter", "selected_report_within_window", None
    else:
        disposition, reason, event = "activity_uncertain_stale", "selected_report_outside_window", None
    return {"CERT": row["CERT"], "Q": q, "SELECTED_REPDTE": repdte,
        "AVAILABILITY_DATE": available, "REPORT_AGE_DAYS": row["REPORT_AGE_DAYS"],
        "CANDIDATE_DISPOSITION": row["ELIGIBILITY_DISPOSITION"],
        "MEMBERSHIP_DISPOSITION": disposition, "MEMBERSHIP_REASON": reason,
        "EVIDENCE_SOURCE": event.source_id if event else "Financials:as_of_selected_report",
        "EVIDENCE_DATE": event.effective_date if event else repdte,
        "EVIDENCE_KNOWN_AT": event.known_at if event else available,
        "POLICY_VERSION": POLICY_VERSION}


def _output_path(root: Path, run_id: str, quarter: str) -> Path:
    return root / "data_processed" / RUN_NAMESPACE / run_id / f"Q={quarter}" / "part-000.parquet"


def _evidence_path(root: Path, run_id: str) -> Path:
    return root / "runs" / RUN_NAMESPACE / run_id


def _read_candidates(path: Path, quarter: str) -> list[dict[str, Any]]:
    columns = ["CERT", "Q", "SELECTED_REPDTE", "AVAILABILITY_DATE",
               "REPORT_AGE_DAYS", "ELIGIBILITY_DISPOSITION"]
    with pq.ParquetFile(path) as file:
        rows = file.read(columns=columns).to_pylist()
    if len(rows) != len({r["CERT"] for r in rows}) or any(
            r["Q"] != _day(quarter) for r in rows):
        raise ValueError(f"Candidate keys differ: {quarter}")
    return rows


def _age_stats(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    ages = sorted(r["REPORT_AGE_DAYS"] for r in rows)
    return {"min": ages[0] if ages else None,
            "median": ages[(len(ages) - 1) // 2] if ages else None,
            "max": ages[-1] if ages else None}


def _quarter_stats(rows: list[dict[str, Any]], classified: list[dict[str, Any]],
                   previous_recent: set[int]) -> dict[str, Any]:
    counts = Counter(r["MEMBERSHIP_DISPOSITION"] for r in classified)
    baseline = Counter(r["ELIGIBILITY_DISPOSITION"] for r in rows)
    recent = {r["CERT"] for r in classified if r["MEMBERSHIP_DISPOSITION"] == "recent_reporter"}
    sensitivity = {str(w): sum(quarter_distance(r["SELECTED_REPDTE"], r["Q"]) <= w
                               for r in rows) for w in WINDOWS}
    eligible = [r for r in rows if r["ELIGIBILITY_DISPOSITION"] == "eligible"]
    return {"quarter": rows[0]["Q"].strftime("%Y%m%d") if rows else None,
            "candidate_bank_quarters": len(rows), "candidate_distinct_banks": len({r["CERT"] for r in rows}),
            "baseline_eligible": baseline["eligible"],
            "baseline_prior_failure": baseline["confirmed_failure_before_or_on_q"],
            "baseline_unresolved_identity": baseline["unresolved_identity"],
            "recent_reporter": counts["recent_reporter"],
            "activity_uncertain_stale": counts["activity_uncertain_stale"],
            "known_exit": counts["known_exit"], "unresolved_identity": counts["unresolved_identity"],
            "recent_entrants_vs_previous_quarter": len(recent - previous_recent),
            "baseline_eligible_older_than_365_days": sum(r["REPORT_AGE_DAYS"] > 365 for r in eligible),
            "baseline_eligible_recent_reporter": sum(r["MEMBERSHIP_DISPOSITION"] == "recent_reporter"
                and r["CANDIDATE_DISPOSITION"] == "eligible" for r in classified),
            "all_candidate_report_age_days": _age_stats(rows),
            "baseline_eligible_report_age_days": _age_stats(eligible),
            "sensitivity_recent_by_window_quarters": sensitivity}


def run(root: Path, run_id: str) -> dict[str, Any]:
    """Read accepted candidates; write a separate audited membership layer."""
    root = root.resolve(strict=True)
    if not run_id or not all(c.isalnum() or c in "_-" for c in run_id):
        raise ValueError("Invalid run ID")
    evidence = _evidence_path(root, run_id)
    data_root = root / "data_processed" / RUN_NAMESPACE / run_id
    if evidence.exists() or data_root.exists():
        raise FileExistsError(f"Population run already exists: {run_id}")
    source_evidence = root / "runs" / "task_2_5_prediction" / SOURCE_RUN
    source_summary_path = source_evidence / "summary.json"
    source_manifest_path = source_evidence / "run_manifest.json"
    source_metadata_path = source_evidence / "metadata.json"
    verified = verify_candidates(root, SOURCE_RUN, check_upstream=True)
    if verified["candidate_rows"] != 3697094:
        raise ValueError("Accepted candidate count differs")
    source = _read_json(source_summary_path)
    protected = {str(p): _sha(p) for p in
                 (source_summary_path, source_manifest_path, source_metadata_path,
                  source_evidence / "report.md")}
    protected.update({item["path"]: item["sha256"] for item in source["outputs"]})
    evidence.mkdir(parents=True)
    manifest = RunManifest(run_id=run_id, config={"policy_version": POLICY_VERSION,
        "source_run": SOURCE_RUN, "window_quarters": 4,
        "sensitivity_windows_quarters": list(WINDOWS)})
    manifest.set_status("running")
    manifest.record_input(str(source_manifest_path), protected[str(source_manifest_path)])
    manifest.write(evidence / "run_manifest.json")
    outputs = []
    by_quarter = []
    previous_recent: set[int] = set()
    all_banks: set[int] = set()
    recent_banks: set[int] = set()
    totals: Counter[str] = Counter()
    try:
        for item in source["outputs"]:
            quarter = item["quarter"]
            rows = _read_candidates(Path(item["path"]), quarter)
            classified = [classify(row) for row in rows]
            stats = _quarter_stats(rows, classified, previous_recent)
            stats["quarter"] = quarter
            if stats["candidate_bank_quarters"] != item["rows"] or sum(
                    stats[k] for k in ("recent_reporter", "activity_uncertain_stale",
                                         "known_exit", "unresolved_identity")) != item["rows"]:
                raise ValueError(f"Population reconciliation differs: {quarter}")
            table = pa.Table.from_pylist(classified, schema=SCHEMA)
            path = _output_path(root, run_id, quarter)
            digest = _write_verified(path, table)
            with pq.ParquetFile(path) as file:
                if not file.read().equals(table, check_metadata=False):
                    raise ValueError(f"Population read-back differs: {quarter}")
            manifest.record_output(path=str(path), dataset="population", rows=len(rows),
                sha256=digest, compression="ZSTD", schema_version=POLICY_VERSION,
                partition=quarter, source_total=len(rows), source_query=None)
            outputs.append({"quarter": quarter, "path": str(path), "rows": len(rows), "sha256": digest})
            by_quarter.append(stats)
            previous_recent = {r["CERT"] for r in classified
                               if r["MEMBERSHIP_DISPOSITION"] == "recent_reporter"}
            all_banks.update(r["CERT"] for r in rows)
            recent_banks.update(previous_recent)
            totals.update({k: stats[k] for k in ("candidate_bank_quarters", "baseline_eligible",
                "baseline_prior_failure", "baseline_unresolved_identity", "recent_reporter",
                "activity_uncertain_stale", "known_exit", "unresolved_identity",
                "recent_entrants_vs_previous_quarter", "baseline_eligible_older_than_365_days",
                "baseline_eligible_recent_reporter")})
        if len(outputs) != 166 or totals["candidate_bank_quarters"] != source["overall"]["candidates"]:
            raise ValueError("Population output scope differs from accepted candidate roster")
        if any(_sha(Path(p)) != digest for p, digest in protected.items()):
            raise ValueError("Accepted P2.4 evidence changed")
        result = {"status": "complete", "run_id": run_id, "policy_version": POLICY_VERSION,
            "source_run": SOURCE_RUN, "scope": "recent reporters, not verified active banks",
            "source_evidence": {"manifest_sha256": protected[str(source_manifest_path)],
                "summary_sha256": protected[str(source_summary_path)],
                "protected_path_count": len(protected)},
            "official_history_limitation": "History PROCDATE is processing date, not verified public availability; no history event is applied to real membership",
            "by_quarter": by_quarter,
            "overall": {**dict(totals), "distinct_candidate_banks": len(all_banks),
                "distinct_recent_reporter_banks": len(recent_banks)},
            "outputs": outputs}
        _write_json(evidence / "summary.json", result)
        last = by_quarter[-1]
        lines = ["# P2 Prediction Population Reconciliation", "",
            "Unit: bank-quarter at each q; within a quarter, bank-quarter and distinct-bank counts coincide. Overall distinct banks are separately reported. Membership is a recent-reporter proxy, not verified activity.", "",
            "| Period | Candidate bank-quarters | Baseline eligible | Recent reporters (4 quarters) | Stale activity uncertain | Recent entrants vs prior q | Eligible older than 365 days |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for s in (by_quarter[0], last):
            lines.append(f"| {s['quarter']} | {s['candidate_bank_quarters']:,} | {s['baseline_eligible']:,} | {s['recent_reporter']:,} | {s['activity_uncertain_stale']:,} | {s['recent_entrants_vs_previous_quarter']:,} | {s['baseline_eligible_older_than_365_days']:,} |")
        lines.append(f"| Overall (bank-quarters) | {totals['candidate_bank_quarters']:,} | {totals['baseline_eligible']:,} | {totals['recent_reporter']:,} | {totals['activity_uncertain_stale']:,} | {totals['recent_entrants_vs_previous_quarter']:,} | {totals['baseline_eligible_older_than_365_days']:,} |")
        lines += ["", f"2025Q2: {last['baseline_eligible']:,} baseline eligible includes {last['baseline_eligible_older_than_365_days']:,} selected reports older than 365 days. These are descriptive counts, not closure evidence.",
            "", "The four-quarter window follows quarterly reporting and the provisional 120-day availability delay. Two- and eight-quarter alternatives are in summary.json. An effective-date event learned after q is not an as-of exclusion. The accepted P2.4 retrospective event disposition is carried only as a diagnostic; verified historical publication/vintage timing remains unavailable.", ""]
        (evidence / "report.md").write_text("\n".join(lines), encoding="utf-8")
        _write_json(evidence / "metadata.json", {"status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "dependencies": dependency_versions(), "source_hashes": protected,
            "summary_sha256": _sha(evidence / "summary.json"),
            "report_sha256": _sha(evidence / "report.md")})
        manifest.record_validation("accepted_candidate_and_upstream_hashes", True,
            "166 candidates, 527 upstream hashes verified before; source evidence unchanged after")
        manifest.record_validation("reconciliation_and_readback", True,
            "166 unique-key population outputs passed logical read-back")
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return result
    except Exception as exc:
        manifest.record_validation("population_run", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    evidence = _evidence_path(root, run_id)
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if (manifest.status != "complete" or manifest.run_id != run_id
            or summary.get("status") != "complete" or summary.get("run_id") != run_id
            or metadata.get("status") != "complete"
            or _sha(evidence / "summary.json") != metadata["summary_sha256"]
            or _sha(evidence / "report.md") != metadata["report_sha256"]
            or len(manifest.outputs) != 166 or len(summary["outputs"]) != 166
            or any(not v["passed"] for v in manifest.validations)):
        raise ValueError("Population evidence is incomplete")
    data_root = root / "data_processed" / RUN_NAMESPACE / run_id
    count = 0
    for item, recorded in zip(summary["outputs"], manifest.outputs, strict=True):
        path = Path(item["path"]).resolve()
        if (not path.is_relative_to(data_root.resolve()) or str(path) != recorded["path"]
                or item["sha256"] != recorded["sha256"] or _sha(path) != item["sha256"]):
            raise ValueError(f"Population checksum/path differs: {item['quarter']}")
        with pq.ParquetFile(path) as file:
            if (file.metadata.num_rows != item["rows"]
                    or not file.schema_arrow.equals(SCHEMA, check_metadata=True)
                    or any(file.metadata.row_group(g).column(c).compression != "ZSTD"
                           for g in range(file.metadata.num_row_groups)
                           for c in range(file.metadata.row_group(g).num_columns))):
                raise ValueError(f"Population schema/codec differs: {item['quarter']}")
            rows = file.read(columns=["CERT", "Q", "MEMBERSHIP_DISPOSITION"]).to_pylist()
        if (len(rows) != len({r["CERT"] for r in rows})
                or any(r["Q"] != _day(item["quarter"]) for r in rows)):
            raise ValueError(f"Population keys differ: {item['quarter']}")
        count += len(rows)
    if count != summary["overall"]["candidate_bank_quarters"]:
        raise ValueError("Population total differs")
    for p, digest in metadata["source_hashes"].items():
        if _sha(Path(p)) != digest:
            raise ValueError(f"Accepted source changed: {p}")
    return {"status": "verified", "run_id": run_id, "quarters": 166,
            "candidate_bank_quarters": count, "protected_hashes_checked": len(metadata["source_hashes"])}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit recent-reporter membership separately from P2.4 predictors")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "verify", "show"):
        p = sub.add_parser(command)
        p.add_argument("--project-root", type=Path, default=Path.cwd())
        p.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.project_root, args.run_id)
        print(json.dumps({"status": result["status"], "run_id": args.run_id,
                          "overall": result["overall"]}, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_saved(args.project_root, args.run_id), indent=2))
    else:
        evidence = _evidence_path(args.project_root, args.run_id)
        try:
            manifest = RunManifest.read(evidence / "run_manifest.json")
            summary = _read_json(evidence / "summary.json")
            metadata = _read_json(evidence / "metadata.json")
            if (manifest.status != "complete" or summary["status"] != "complete"
                    or metadata["status"] != "complete"
                    or metadata["summary_sha256"] != _sha(evidence / "summary.json")):
                raise ValueError("manifest or evidence is incomplete")
        except (OSError, ValueError, KeyError) as exc:
            parser.exit(2, f"Population run {args.run_id}: {exc}\n")
        print(json.dumps({"status": "complete", "run_id": args.run_id,
            "overall": summary["overall"]}, indent=2))


if __name__ == "__main__":
    main()
