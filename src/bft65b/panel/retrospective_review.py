"""Read-only P4.2 reconciliation of saved membership, labels and PIT candidates."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path

import pyarrow.parquet as pq

from bft65b.features.production import verify_saved as verify_labels
from bft65b.panel.retrospective import verify_saved as verify_membership
from bft65b.quality.audit import _read_json, _sha, _write_json


def review(root: Path, membership_run: str, label_run: str, review_id: str,
           repeat_membership_run: str | None = None,
           repeat_label_run: str | None = None,
           candidate_run: str = "pit20261115f") -> dict:
    root = root.resolve(strict=True)
    verify_membership(root, membership_run)
    verify_labels(root, label_run)
    membership_path = root / "runs" / "task_4_2_membership" / membership_run / "summary.json"
    label_path = root / "runs" / "task_4_2_production_labels" / label_run / "summary.json"
    memberships = _read_json(membership_path)
    labels = _read_json(label_path)
    if labels["membership_run"] != membership_run:
        raise ValueError("Label/membership lineage differs")
    from bft65b.manifest import RunManifest
    member_manifest = RunManifest.read(root / "runs/task_4_2_membership" /
                                       membership_run / "run_manifest.json")
    if ((root / "runs/source_vintage_bundle.json").is_file()
            and (candidate_run == "pit20261115f" or
                 member_manifest.config.get("source_candidate_run") != candidate_run)):
        raise ValueError("Portable review requires the cohort's selected PIT run")
    candidate_path = root / "runs" / "task_2_5_prediction" / candidate_run / "summary.json"
    candidates = _read_json(candidate_path)
    m_path = next(item["path"] for item in memberships["outputs"]
                  if item.get("quarter") == "20250630")
    l_path = next(item["path"] for item in labels["outputs"]
                  if item["quarter"] == "20250630")
    p_path = candidates["outputs"][-1]["path"]
    if candidates["outputs"][-1]["quarter"] != "20250630":
        raise ValueError("PIT final quarter differs")
    with pq.ParquetFile(m_path) as file:
        m_rows = file.read().to_pylist()
    with pq.ParquetFile(l_path) as file:
        l_rows = file.read().to_pylist()
    with pq.ParquetFile(p_path) as file:
        p_rows = file.read(columns=["CERT", "ELIGIBILITY_DISPOSITION",
                                    "REPORT_AGE_DAYS"]).to_pylist()
    m_by_cert = {row["CERT"]: row for row in m_rows}
    l_by_cert = {row["CERT"]: row for row in l_rows}
    if (len(m_by_cert) != 24064 or len(l_by_cert) != 24064
            or len(p_rows) != 24051 or set(m_by_cert) != set(l_by_cert)):
        raise ValueError("2025Q2 bank-quarter key reconciliation differs")
    cross_all = Counter()
    cross_baseline = Counter()
    old_report_baseline = Counter()
    for row in p_rows:
        cert = row["CERT"]
        status = m_by_cert[cert]["MEMBERSHIP"]
        label = l_by_cert[cert]["LABEL_DISPOSITION"]
        cross_all[(status, label)] += 1
        if row["ELIGIBILITY_DISPOSITION"] == "eligible":
            cross_baseline[(status, label)] += 1
            if row["REPORT_AGE_DAYS"] > 365:
                old_report_baseline[status] += 1
    if sum(cross_baseline.values()) != 21214 or sum(old_report_baseline.values()) != 16600:
        raise ValueError("2025Q2 baseline cross-tab does not reconcile")
    q_membership = memberships["by_quarter"][-1]
    q_label = labels["by_quarter"][-1]
    if (sum(cross_all.values()) != 24051 or q_membership["quarter"] != "20250630"
            or q_label["quarter"] != "20250630" or q_label["decisions"] != 24064):
        raise ValueError("2025Q2 final quarter totals differ")
    bank_path = next(item["path"] for item in memberships["outputs"]
                     if item["dataset"] == "bank_evidence")
    with pq.ParquetFile(bank_path) as file:
        banks = file.read().to_pylist()
    exit_on_quarter = Counter(row["END_DATE"].strftime("%Y%m%d") for row in banks
                              if row["BANK_DISPOSITION"] == "supported"
                              and row["END_DATE"] is not None
                              and date(1984, 3, 31) <= row["END_DATE"] <= date(2025, 6, 30)
                              and (row["END_DATE"].month, row["END_DATE"].day)
                              in {(3, 31), (6, 30), (9, 30), (12, 31)})
    output = {"membership_run": membership_run, "label_run": label_run,
              "candidate_run": candidate_run,
              "source_summary_hashes": {str(path): _sha(path) for path in
                                        (membership_path, label_path, candidate_path)},
              "all_bank_resolution": memberships["bank_dispositions"],
              "overall_membership": memberships["overall"],
              "overall_labels": labels["overall"],
              "by_quarter_membership": memberships["by_quarter"],
              "by_quarter_labels": labels["by_quarter"],
              "q2025q2": {"all_observed_bank_identities": 24064,
                          "candidate_bank_quarters": len(p_rows),
                          "no_feature_history": 24064 - len(p_rows),
                          "p2_4_eligible_diagnostic": sum(cross_baseline.values()),
                          "p2_4_eligible_older_than_365_days": sum(old_report_baseline.values()),
                          "candidate_membership_label_cross": [
                              {"membership": m, "label": l, "count": count}
                              for (m, l), count in sorted(cross_all.items())],
                          "baseline_eligible_membership_label_cross": [
                              {"membership": m, "label": l, "count": count}
                              for (m, l), count in sorted(cross_baseline.items())],
                          "older_report_baseline_membership": dict(sorted(old_report_baseline.items())),
                          "membership": q_membership, "labels": q_label},
              "sensitivity": {
                  "require_verified_insurance_by_q_removed_eligible_bank_quarters":
                      memberships["overall"]["insurance_start_sensitivity_eligible_bank_quarters"],
                  "require_verified_insurance_by_q_removed_at_2025q2":
                      q_membership["insurance_start_sensitivity_eligible_bank_quarters"],
                  "treat_exit_effective_on_q_as_still_active_maximum_additional_cases_by_quarter":
                      dict(sorted(exit_on_quarter.items())),
                  "interpretation": "Scenarios only; neither insurance-date coverage nor inclusive exit is adopted."}}
    if (repeat_membership_run is None) != (repeat_label_run is None):
        raise ValueError("Both repeat run IDs are required")
    if repeat_membership_run is not None and repeat_label_run is not None:
        verify_membership(root, repeat_membership_run)
        verify_labels(root, repeat_label_run)
        repeated_membership = _read_json(root / "runs" / "task_4_2_membership" /
                                         repeat_membership_run / "summary.json")
        repeated_labels = _read_json(root / "runs" / "task_4_2_production_labels" /
                                     repeat_label_run / "summary.json")
        membership_equal = (memberships["bank_dispositions"] == repeated_membership["bank_dispositions"]
                            and memberships["overall"] == repeated_membership["overall"]
                            and memberships["by_quarter"] == repeated_membership["by_quarter"])
        labels_equal = (labels["overall"] == repeated_labels["overall"]
                        and labels["by_quarter"] == repeated_labels["by_quarter"])
        membership_hash_differences = sum(a["sha256"] != b["sha256"]
            for a, b in zip(memberships["outputs"], repeated_membership["outputs"]))
        label_hash_differences = sum(a["sha256"] != b["sha256"]
            for a, b in zip(labels["outputs"], repeated_labels["outputs"]))
        if (not membership_equal or not labels_equal
                or len(memberships["outputs"]) != len(repeated_membership["outputs"])
                or len(labels["outputs"]) != len(repeated_labels["outputs"])
                or membership_hash_differences or label_hash_differences):
            raise ValueError("Independent P4.2 repeat differs logically or physically")
        output["repeat_comparison"] = {
            "repeat_membership_run": repeat_membership_run,
            "repeat_label_run": repeat_label_run,
            "membership_summary_counts_equal": membership_equal,
            "label_summary_counts_equal": labels_equal,
            "membership_output_hash_differences": membership_hash_differences,
            "label_output_hash_differences": label_hash_differences,
            "membership_files_compared": len(memberships["outputs"]),
            "label_files_compared": len(labels["outputs"])}
    target = root / "runs" / "task_4_2_review" / review_id / "summary.json"
    if target.exists():
        raise FileExistsError("Review ID already exists")
    target.parent.mkdir(parents=True, exist_ok=False)
    _write_json(target, output)
    return {"review": str(target), "sha256": _sha(target),
            "membership_run": membership_run, "label_run": label_run,
            "q2025q2_baseline": sum(cross_baseline.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Review saved P4.2 evidence")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--membership-run", required=True)
    parser.add_argument("--label-run", required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--repeat-membership-run")
    parser.add_argument("--repeat-label-run")
    parser.add_argument("--candidate-run", default="pit20261115f")
    args = parser.parse_args()
    print(json.dumps(review(args.project_root, args.membership_run,
                            args.label_run, args.review_id,
                            args.repeat_membership_run,
                            args.repeat_label_run, args.candidate_run), indent=2))


if __name__ == "__main__":
    main()
