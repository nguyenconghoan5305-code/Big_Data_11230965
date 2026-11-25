"""Four-quarter outcome rule and historical aggregate preflight.

Membership at q, identity/linkage, and future outcome are separate inputs.
The production runner consumes an independently verified membership artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Sequence

import pyarrow.parquet as pq

from bft65b.config import load_config
from bft65b.manifest import RunManifest
from bft65b.panel.population import verify_saved as verify_population
from bft65b.panel.prediction import verify_saved as verify_candidates
from bft65b.quality.audit import _read_json, _sha, _write_json


LABEL_RULE_VERSION = "four_quarter_failure_v1"
MAX_Q = date(2025, 6, 30)
TARGET_RIGHT_ENDPOINT = date(2026, 6, 30)
REPAIR_RUN = "repair20261115f"
CANDIDATE_RUN = "pit20261115f"
POPULATION_RUN = "population20261115b"
ALLOWED_POPULATION_POLICIES = frozenset({
    "verified_active_v1", "approved_recent_reporter_v1", "retrospective_operational_v1"})


@dataclass(frozen=True)
class MembershipEvidence:
    """Externally established population decision; never inferred by labels."""

    cert: int
    q: date
    status: str  # eligible, ineligible, uncertain
    policy: str
    known_at: date | None
    source_id: str
    identity_resolved: bool

    def __post_init__(self) -> None:
        if (type(self.cert) is not int or self.cert <= 0
                or not _quarter_end(self.q) or self.status not in {
                    "eligible", "ineligible", "uncertain"}
                or self.policy not in ALLOWED_POPULATION_POLICIES
                or (self.policy == "retrospective_operational_v1" and self.known_at is not None)
                or (self.policy != "retrospective_operational_v1" and
                    (type(self.known_at) is not date or self.known_at > self.q))
                or not self.source_id or type(self.identity_resolved) is not bool):
            raise ValueError("Membership must have an approved policy and evidence known by q or explicitly retrospective timing")


@dataclass(frozen=True)
class OutcomeEvent:
    """One FDIC source ID or one independently dated non-failure exit event."""

    source_id: str
    cert: int | None
    effective_date: date
    transaction: str
    bank_link: str
    supplement_match: str = "matched"

    @classmethod
    def from_repaired_row(cls, row: dict) -> "OutcomeEvent":
        return cls(row["ID"], row["CERT"], row["FAILDATE"],
                   row["TRANSACTION_DISPOSITION"],
                   row["BANK_LINK_DISPOSITION"], row["SUPPLEMENT_MATCH"])

    def __post_init__(self) -> None:
        if (not isinstance(self.source_id, str) or not self.source_id.strip()
                or (self.cert is not None and (type(self.cert) is not int or self.cert <= 0))
                or type(self.effective_date) is not date or not self.transaction
                or not self.bank_link or not self.supplement_match):
            raise ValueError("Outcome event requires source identity, date and dispositions")


@dataclass(frozen=True)
class LabelDecision:
    cert: int
    q: date
    right_endpoint: date
    disposition: str  # positive, negative, unresolved, excluded
    y: int | None
    reason: str
    qualifying_source_ids: tuple[str, ...]


def _quarter_end(day: date) -> bool:
    return type(day) is date and (day.month, day.day) in {
        (3, 31), (6, 30), (9, 30), (12, 31)}


def _qualifies(event: OutcomeEvent) -> bool:
    return (event.transaction == "verified_failure"
            and event.bank_link == "chronological_candidate"
            and event.supplement_match == "matched" and event.cert is not None)


def classify_label(membership: MembershipEvidence, events: Sequence[OutcomeEvent],
                   *, outcome_complete_through: date) -> LabelDecision:
    """Classify a synthetic/approved member without changing the risk-set policy.

    A positive needs one corroborated FDIC failure transaction and the P2.3
    chronological link. Assistance alone is neither failure nor exit. A
    negative needs full follow-up and no linked, censored or unassignable
    failure evidence in the target window.
    """
    if type(outcome_complete_through) is not date:
        raise TypeError("Outcome coverage date must be a date")
    ids = [event.source_id for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate outcome source IDs")
    q, cert = membership.q, membership.cert
    right = date(q.year + 1, q.month, q.day)  # exactly four calendar quarters

    def result(disposition: str, reason: str,
               qualifying: Sequence[OutcomeEvent] = ()) -> LabelDecision:
        y = 1 if disposition == "positive" else (0 if disposition == "negative" else None)
        return LabelDecision(cert, q, right, disposition, y, reason,
                             tuple(sorted(item.source_id for item in qualifying)))

    if q > MAX_Q:
        return result("excluded", "prediction_after_2025q2")
    same_bank = [event for event in events if event.cert == cert]
    prior = [event for event in same_bank if event.effective_date <= q]
    if any(_qualifies(event) for event in prior):
        return result("excluded", "verified_failure_before_or_on_q")
    if membership.status == "ineligible":
        return result("excluded", "not_in_approved_population_at_q")
    if membership.status == "uncertain":
        return result("unresolved", "population_membership_uncertain_at_q")
    if not membership.identity_resolved:
        return result("unresolved", "identity_unresolved_at_q")
    if any(event.transaction not in {"assistance_transaction"} for event in prior):
        return result("unresolved", "prior_event_or_linkage_conflict")

    in_window = [event for event in same_bank if q < event.effective_date <= right]
    qualifying = [event for event in in_window if _qualifies(event)]
    nonfailure_exits = [event for event in in_window
                        if event.transaction in {"documented_merger", "documented_other_exit"}]
    conflicting = [event for event in in_window if not _qualifies(event)
                   and event.transaction not in {"assistance_transaction",
                                                 "documented_merger", "documented_other_exit"}]
    if len(qualifying) > 1 or conflicting or (qualifying and nonfailure_exits):
        return result("unresolved", "multiple_or_conflicting_events_in_window", qualifying)
    if qualifying:
        return result("positive", "one_verified_linked_failure_in_window", qualifying)
    unassigned = [event for event in events if event.cert is None
                  and q < event.effective_date <= right
                  and event.transaction in {"verified_failure", "failure_charter_uncertain",
                                            "unmapped_transaction", "unknown_transaction"}]
    if unassigned:
        return result("unresolved", "unassignable_failure_source_record_in_window")
    if outcome_complete_through < right:
        return result("unresolved", "incomplete_four_quarter_followup")
    if nonfailure_exits:
        return result("negative", "complete_followup_with_nonfailure_exit")
    return result("negative", "complete_followup_without_qualifying_or_censoring_event")


def label_batch(memberships: Sequence[MembershipEvidence], events: Sequence[OutcomeEvent],
                *, outcome_complete_through: date) -> list[LabelDecision]:
    """Preserve one decision per approved (CERT, q), without deduplicating events."""
    keys = [(item.cert, item.q) for item in memberships]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate bank-quarter population key")
    return [classify_label(item, events,
                           outcome_complete_through=outcome_complete_through)
            for item in memberships]


def _verify_repair(root: Path, repair_run: str = REPAIR_RUN) -> tuple[dict, list[OutcomeEvent], dict[str, str]]:
    evidence = root / "runs" / "task_2_3_transaction_repair" / repair_run
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    manifest = RunManifest.read(evidence / "run_manifest.json")
    if (summary.get("run_id") != repair_run or summary.get("status") != "complete"
            or metadata.get("status") != "complete" or manifest.status != "complete"
            or len(manifest.outputs) != 2 or len(summary.get("outputs", [])) != 2
            or not manifest.validations or any(not check["passed"] for check in manifest.validations)
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or metadata.get("report_sha256") != _sha(evidence / "report.md")):
        raise ValueError("Repaired P2.3 evidence is incomplete")
    output_by_path = {row["path"]: row for row in manifest.outputs}
    protected: dict[str, str] = {}
    for item in summary["outputs"]:
        path = Path(item["path"]).resolve(strict=True)
        if (not path.is_relative_to((root / "data_processed" /
             "task_2_3_transaction_repair" / repair_run).resolve())
                or _sha(path) != item["sha256"]
                or output_by_path.get(str(path), {}).get("sha256") != item["sha256"]):
            raise ValueError("Repaired P2.3 output checksum differs")
        protected[str(path)] = item["sha256"]
    for item in manifest.inputs:
        if item["sha256"] and _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Repaired P2.3 input checksum differs: {item['path']}")
    event_path = (root / "data_processed" / "task_2_3_transaction_repair" /
                  repair_run / "failure_events" / "part-000.parquet")
    with pq.ParquetFile(event_path) as file:
        if file.metadata.num_rows != 4117:
            raise ValueError("Repaired event count differs")
        events = [OutcomeEvent.from_repaired_row(row) for row in file.read().to_pylist()]
    if (len({event.source_id for event in events}) != 4117
            or Counter(event.transaction for event in events) !=
            Counter(summary["transaction_dispositions"])):
        raise ValueError("Repaired event IDs or dispositions differ")
    for name in ("summary.json", "metadata.json", "run_manifest.json", "report.md"):
        path = evidence / name
        protected[str(path)] = _sha(path)
    return summary, events, protected


def preflight(root: Path, run_id: str) -> dict:
    """Save aggregate evidence only; no bank-quarter labels or production rows."""
    root = root.resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}", run_id):
        raise ValueError("Unsafe label-preflight run ID")
    evidence = root / "runs" / "task_4_2_label_preflight" / run_id
    if evidence.exists():
        raise FileExistsError("Label-preflight run already exists")
    repair, events, hashes = _verify_repair(root)
    candidates = verify_candidates(root, CANDIDATE_RUN, check_upstream=True)
    population = verify_population(root, POPULATION_RUN)
    candidate_summary = _read_json(root / "runs" / "task_2_5_prediction" /
                                   CANDIDATE_RUN / "summary.json")
    population_summary = _read_json(root / "runs" / "task_2_6_population" /
                                    POPULATION_RUN / "summary.json")
    if (candidates["candidate_rows"] != population["candidate_bank_quarters"]
            or candidates["candidate_rows"] != candidate_summary["overall"]["candidates"]
            or candidates["candidate_rows"] != population_summary["overall"]["candidate_bank_quarters"]
            or len(candidate_summary["by_quarter"]) != 166
            or len(population_summary["by_quarter"]) != 166):
        raise ValueError("Candidate and population denominators differ")
    summary_by_q = {row["quarter"]: row for row in candidate_summary["by_quarter"]}
    choices = []
    for row in population_summary["by_quarter"]:
        q = row["quarter"]
        base = summary_by_q[q]
        recent_eligible = row["baseline_eligible_recent_reporter"]
        eligible = base["eligible"]
        if (row["candidate_bank_quarters"] != base["candidates"]
                or row["baseline_eligible"] != eligible
                or recent_eligible > eligible
                or row["recent_reporter"] + row["activity_uncertain_stale"] != base["candidates"]):
            raise ValueError(f"Population reconciliation differs at {q}")
        choices.append({"q": q, "candidate_bank_quarters": base["candidates"],
                        "p2_4_eligible_diagnostic": eligible,
                        "recent_reporter_proxy": row["recent_reporter"],
                        "stale_activity_uncertain": row["activity_uncertain_stale"],
                        "recent_and_p2_4_eligible_diagnostic_only": recent_eligible,
                        "p2_4_eligible_outside_recent_proxy": eligible - recent_eligible,
                        "prior_failure_diagnostic": base["excluded_confirmed_failure_before_or_on_q"],
                        "unresolved_identity_diagnostic": base["excluded_unresolved_identity"]})
    if (sum(item["candidate_bank_quarters"] for item in choices) != 3697094
            or sum(item["p2_4_eligible_diagnostic"] for item in choices) != 3335808):
        raise ValueError("Quarter-to-overall candidate reconciliation differs")
    source_retrieved = datetime.fromisoformat(repair["source_retrieved_utc"]).date()
    index_created = datetime.fromisoformat(
        repair["source_index"]["createTimestamp"].replace("Z", "+00:00")).date()
    if source_retrieved <= TARGET_RIGHT_ENDPOINT or index_created <= TARGET_RIGHT_ENDPOINT:
        raise ValueError("Failures source fingerprint does not extend past target endpoint")
    event_quarters: Counter[str] = Counter()
    for event in events:
        period = f"{event.effective_date.year}Q{(event.effective_date.month - 1)//3 + 1}"
        event_quarters[period] += 1
    missing_cert = [event for event in events if event.cert is None]
    result = {
        "preflight_status": "complete", "production_label_status": "BLOCKED",
        "blocker": "No approved point-in-time modelling population or complete verified active-bank history",
        "run_id": run_id, "label_rule_version": LABEL_RULE_VERSION,
        "source_runs": {"repair": REPAIR_RUN, "candidates": CANDIDATE_RUN,
                        "recent_reporter_proxy": POPULATION_RUN},
        "verified_inputs": {"repair_output_and_evidence_hashes": hashes,
                            "candidate_validation": candidates,
                            "population_validation": population},
        "outcome_scope": {"last_prediction_q": MAX_Q.isoformat(),
                          "latest_right_endpoint": TARGET_RIGHT_ENDPOINT.isoformat(),
                          "source_retrieved_date": source_retrieved.isoformat(),
                          "source_index_created_date": index_created.isoformat(),
                          "max_source_event_date": max(event.effective_date for event in events).isoformat(),
                          "complete_endpoint_source_rows": len(events),
                          "unique_source_ids": len({event.source_id for event in events}),
                          "transaction_dispositions": dict(sorted(Counter(
                              event.transaction for event in events).items())),
                          "verified_failure_links": dict(sorted(Counter(
                              event.bank_link for event in events
                              if event.transaction == "verified_failure").items())),
                          "null_cert_events": len(missing_cert),
                          "latest_null_cert_event_date": max(
                              (event.effective_date for event in missing_cert),
                              default=None).isoformat() if missing_cert else None,
                          "source_event_counts_by_event_quarter": dict(sorted(event_quarters.items()))},
        "population_scope": {"unit": "candidate bank-quarters; proxy counts are diagnostics only",
                             "by_prediction_quarter": choices,
                             "overall": {
                                 "candidate_bank_quarters": sum(x["candidate_bank_quarters"] for x in choices),
                                 "p2_4_eligible_diagnostic": sum(x["p2_4_eligible_diagnostic"] for x in choices),
                                 "recent_reporter_proxy": sum(x["recent_reporter_proxy"] for x in choices),
                                 "stale_activity_uncertain": sum(x["stale_activity_uncertain"] for x in choices),
                                 "recent_and_p2_4_eligible_diagnostic_only": sum(
                                     x["recent_and_p2_4_eligible_diagnostic_only"] for x in choices),
                                 "p2_4_eligible_outside_recent_proxy": sum(
                                     x["p2_4_eligible_outside_recent_proxy"] for x in choices)}},
        "production_label_counts": None,
    }
    config = load_config(root / "config" / "financials_full_v2.json")
    manifest = RunManifest.new(run_id, config)
    for path, digest in sorted(hashes.items()):
        manifest.record_input(path, digest)
    for namespace, selected in (("task_2_5_prediction", CANDIDATE_RUN),
                                ("task_2_6_population", POPULATION_RUN)):
        for name in ("summary.json", "metadata.json", "run_manifest.json"):
            path = root / "runs" / namespace / selected / name
            manifest.record_input(str(path), _sha(path))
    manifest.record_validation("accepted_repair_candidate_population_evidence", True,
                               "Accepted manifests, physical outputs, and source hashes verified")
    manifest.record_validation("aggregate_reconciliation", True,
                               "166 candidate quarters and 4,117 FDIC source IDs reconcile")
    manifest.record_validation("population_decision_assessed", True,
                               "Production labels remain BLOCKED: " + result["blocker"])
    manifest.config["production_label_status"] = "BLOCKED"
    # The preflight is complete; its manifest does not represent a label dataset.
    evidence.mkdir(parents=True)
    _write_json(evidence / "preflight.json", result)
    manifest.record_output(path=str(evidence / "preflight.json"),
        dataset="aggregate_label_preflight", rows=len(choices),
        sha256=_sha(evidence / "preflight.json"), compression="none",
        schema_version=LABEL_RULE_VERSION, partition=None,
        source_total=len(choices), source_query=None)
    manifest.set_status("complete")
    manifest.write(evidence / "run_manifest.json")
    return {"preflight_status": "complete", "production_label_status": "BLOCKED",
            "run_id": run_id, "evidence": str(evidence / "preflight.json"),
            "candidate_bank_quarters": result["population_scope"]["overall"]["candidate_bank_quarters"],
            "source_events": len(events)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit four-quarter label readiness without publishing labels")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(preflight(args.project_root, args.run_id), indent=2))


if __name__ == "__main__":
    main()
