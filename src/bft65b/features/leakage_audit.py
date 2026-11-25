"""Read-only P4.4 leakage audit of accepted feature, label and source runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, timedelta, timezone, datetime
from decimal import Decimal
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bft65b.config import load_config
from bft65b.features.contract import FEATURE_NAMES, compute_features
from bft65b.features.label import MembershipEvidence, OutcomeEvent, classify_label
from bft65b.features.preprocessing import PREDICTOR_COLUMNS, make_preprocessing_pipeline
from bft65b.features.table import (
    CONTRACT_RUN, LABEL_RUN, MEMBERSHIP_RUN, PANEL_RUN, PIT_RUN, REVIEW_RUN,
    SOURCE_COLUMNS, ReportHistoryState, _day, _report_from_panel, _run_id,
    build_quarter_tables,
    verify_saved as verify_feature_table,
)
from bft65b.manifest import RunManifest
from bft65b.panel.asof import (FinancialReport, PREDICTOR_FIELDS,
                               PredictionRequest, availability_date)
from bft65b.quality.audit import _read_json, _sha


AUDIT_VERSION = "temporal_leakage_audit_v1"
NAMESPACE = "task_4_4_leakage_audit"
FEATURE_RUN = "featuretable20261120a"
REPEAT_RUN = "featuretable20261120b"
SAMPLE_SEED = "P4.4|2026-11-20|v1"
SAMPLE_PER_CATEGORY = 8
CATEGORIES = ("ordinary", "calendar_gap", "other_missing")
DEFAULT_RUNS = {"panel": PANEL_RUN, "pit": PIT_RUN,
    "feature_contract": CONTRACT_RUN, "membership": MEMBERSHIP_RUN,
    "labels": LABEL_RUN, "feature_table": FEATURE_RUN,
    "independent_repeat": REPEAT_RUN, "p42_review": REVIEW_RUN}


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary evidence file already exists: {temporary}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _accepted_paths(root: Path, selected: dict[str, str] | None = None) -> tuple[dict, dict, dict, dict, list[Path]]:
    """Run accepted read-only validators before sampling or creating evidence."""
    selected = dict(DEFAULT_RUNS if selected is None else selected)
    if set(selected) != set(DEFAULT_RUNS):
        raise ValueError("Leakage audit upstream run selection is incomplete")
    portable = (root / "runs/source_vintage_bundle.json").is_file()
    if portable and any(selected.get(key) in (None, historical) for key, historical in (
            ("panel", PANEL_RUN), ("pit", PIT_RUN),
            ("feature_contract", CONTRACT_RUN), ("membership", MEMBERSHIP_RUN),
            ("labels", LABEL_RUN), ("feature_table", FEATURE_RUN),
            ("independent_repeat", REPEAT_RUN), ("p42_review", REVIEW_RUN))):
        raise ValueError("Portable leakage audit requires explicit upstream run selection")
    primary = verify_feature_table(root, selected["feature_table"], check_upstream=True)
    repeat = verify_feature_table(root, selected["independent_repeat"], check_upstream=True)
    a = _read_json(root / "runs" / "task_4_3_feature_table" / selected["feature_table"] / "summary.json")
    b = _read_json(root / "runs" / "task_4_3_feature_table" / selected["independent_repeat"] / "summary.json")
    labels = _read_json(root / "runs" / "task_4_2_production_labels" / selected["labels"] / "summary.json")
    panel = _read_json(root / "runs" / "task_2_2_panel" / selected["panel"] / "summary.json")
    comparison_path = root / "runs" / "task_4_3_feature_table" / selected["independent_repeat"] / "repeat_comparison.json"
    comparison = _read_json(comparison_path) if not portable else {
        "output_count_a": len(a["outputs"]), "output_count_b": len(b["outputs"]),
        "output_sha256_differences": sum(x["sha256"] != y["sha256"] for x, y in zip(a["outputs"], b["outputs"])),
        "output_keys_equal": [(x["dataset"], x["quarter"]) for x in a["outputs"]] ==
                             [(x["dataset"], x["quarter"]) for x in b["outputs"]],
        "by_quarter_equal": a["by_quarter"] == b["by_quarter"],
        "overall_equal": a["overall"] == b["overall"],
        "null_counts_equal": a["candidate_nulls"] == b["candidate_nulls"] and
                             a["model_nulls"] == b["model_nulls"],
        "exclusion_counts_equal": a["exclusion_reasons"] == b["exclusion_reasons"]}
    if (primary["quarters"] != repeat["quarters"]
            or comparison.get("output_count_a") != 332
            or comparison.get("output_count_b") != 332
            or comparison.get("output_sha256_differences") != 0
            or not all(comparison.get(name) for name in (
                "output_keys_equal", "by_quarter_equal", "overall_equal",
                "null_counts_equal", "exclusion_counts_equal"))
            or len(labels["outputs"]) != 166 or len(panel["outputs"]) < 170):
        raise ValueError("Accepted P4.3 repeat, label, or panel evidence differs")
    a_hashes = {(item["dataset"], item["quarter"]): item["sha256"]
                for item in a["outputs"]}
    b_hashes = {(item["dataset"], item["quarter"]): item["sha256"]
                for item in b["outputs"]}
    if a_hashes != b_hashes:
        raise ValueError("Accepted independent Parquet output hashes differ")
    paths: set[Path] = set()
    if portable:
        expected = {key: selected[key] for key in (
            "panel", "pit", "feature_contract", "membership", "labels")}
        expected["review"] = selected["p42_review"]
        if a.get("source_runs") != expected or b.get("source_runs") != expected:
            raise ValueError("Portable leakage audit feature-table lineage differs")
    for run in (selected["feature_table"], selected["independent_repeat"]):
        directory = root / "runs" / "task_4_3_feature_table" / run
        manifest = RunManifest.read(directory / "run_manifest.json")
        paths.update(Path(item["path"]) for item in manifest.inputs)
        paths.update(Path(item["path"]) for item in manifest.outputs)
        paths.update(directory / name for name in ("run_manifest.json", "summary.json", "metadata.json"))
    for name, folder in ((selected["labels"], "task_4_2_production_labels"),
                         (selected["membership"], "task_4_2_membership"),
                         (selected["pit"], "task_2_5_prediction"),
                         (selected["panel"], "task_2_2_panel")):
        directory = root / "runs" / folder / name
        paths.update(directory / item for item in ("run_manifest.json", "summary.json", "metadata.json"))
    paths.update((root / "runs" / "task_4_1_feature_contract" / selected["feature_contract"] / "contract.json",
                  root / "runs" / "task_4_2_review" / selected["p42_review"] / "summary.json"))
    if not portable:
        paths.add(comparison_path)
    if portable and any(not path.resolve().is_relative_to(root) for path in paths):
        raise ValueError("Portable leakage-audit input escaped root")
    return a, b, labels, panel, sorted(paths, key=str)


def _inventory(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha(path) for path in paths}


def _sample_category(row: dict) -> str:
    if all(row[name] is not None for name in FEATURE_NAMES):
        return "ordinary"
    if any("missing_or_unavailable_calendar_quarter" in reason
           for reason in row["FEATURE_MISSING_REASONS"]):
        return "calendar_gap"
    return "other_missing"


def _offer_sample(heaps: dict[str, list], row: dict, quarter: str, path: str) -> None:
    category = _sample_category(row)
    score = int.from_bytes(hashlib.sha256(
        f"{SAMPLE_SEED}|{category}|{row['CERT']}|{quarter}".encode()).digest()[:8], "big")
    entry = (-score, row["CERT"], quarter, path)
    heap = heaps[category]
    if len(heap) < SAMPLE_PER_CATEGORY:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _full_alignment(primary: dict, labels: dict) -> tuple[dict, list[dict]]:
    """Independently compare every saved model key and value with labels/candidates."""
    outputs = {(item["quarter"], item["dataset"]): item
               for item in primary["outputs"]}
    if len(outputs) != 332:
        raise ValueError("Duplicate or missing feature output partition")
    totals = Counter()
    nulls = Counter()
    heaps = {name: [] for name in CATEGORIES}
    by_quarter = []
    for label_item in labels["outputs"]:
        quarter = label_item["quarter"]
        candidate_item = outputs[(quarter, "feature_candidates")]
        model_item = outputs[(quarter, "model_table")]
        with pq.ParquetFile(label_item["path"]) as file:
            label_rows = file.read(columns=["CERT", "Q", "MEMBERSHIP",
                "FEATURE_HISTORY_ELIGIBLE", "LABEL_DISPOSITION", "Y"]).to_pylist()
        with pq.ParquetFile(candidate_item["path"]) as file:
            candidate_rows = file.read(columns=["CERT", "Q", *FEATURE_NAMES,
                "SELECTED_REPDTE", "AVAILABILITY_DATE", "REPORT_AGE_DAYS",
                "FEATURE_MISSING_REASONS"]).to_pylist()
        with pq.ParquetFile(model_item["path"]) as file:
            model_rows = file.read(columns=["CERT", "Q", "Y", *FEATURE_NAMES]).to_pylist()
        label_by_cert = {row["CERT"]: row for row in label_rows}
        candidate_by_cert = {row["CERT"]: row for row in candidate_rows}
        model_by_cert = {row["CERT"]: row for row in model_rows}
        if (len(label_by_cert) != len(label_rows)
                or len(candidate_by_cert) != len(candidate_rows)
                or len(model_by_cert) != len(model_rows)):
            raise ValueError(f"Duplicate bank-quarter key: {quarter}")
        candidate_keys = set(candidate_by_cert)
        expected_candidate = {cert for cert, row in label_by_cert.items()
                              if row["FEATURE_HISTORY_ELIGIBLE"]}
        expected_model = {cert for cert, row in label_by_cert.items()
                          if row["MEMBERSHIP"] == "eligible"
                          and row["FEATURE_HISTORY_ELIGIBLE"]
                          and row["LABEL_DISPOSITION"] in {"positive", "negative"}}
        if candidate_keys != expected_candidate or set(model_by_cert) != expected_model:
            raise ValueError(f"Feature-history or resolved-label subset differs: {quarter}")
        q = _day(quarter)
        for cert, row in candidate_by_cert.items():
            if (row["Q"] != q or row["AVAILABILITY_DATE"] > q
                    or row["REPORT_AGE_DAYS"] != (q - row["SELECTED_REPDTE"]).days):
                raise ValueError(f"Candidate availability or date lineage differs: {cert}, {quarter}")
            if cert in expected_model:
                _offer_sample(heaps, row, quarter, candidate_item["path"])
        for cert, row in model_by_cert.items():
            candidate = candidate_by_cert[cert]
            label = label_by_cert[cert]
            if (row["Q"] != q or row["Y"] != label["Y"]
                    or any(row[name] != candidate[name] for name in FEATURE_NAMES)):
                raise ValueError(f"Model value or outcome alignment differs: {cert}, {quarter}")
            totals["positive" if row["Y"] == 1 else "negative"] += 1
            for name in FEATURE_NAMES:
                nulls[name] += row[name] is None
        totals["label_keys"] += len(label_rows)
        totals["candidate_keys"] += len(candidate_rows)
        totals["model_keys"] += len(model_rows)
        totals["label_only"] += len(set(label_by_cert) - candidate_keys)
        by_quarter.append({"quarter": quarter, "label_keys": len(label_rows),
                           "candidate_keys": len(candidate_rows), "model_keys": len(model_rows),
                           "positive": sum(row["Y"] == 1 for row in model_rows),
                           "negative": sum(row["Y"] == 0 for row in model_rows)})
    if (totals["candidate_keys"] != primary["overall"]["candidate_keys"]
            or totals["label_keys"] != primary["overall"]["label_keys"]
            or totals["label_only"] != primary["overall"]["label_only"]
            or totals["positive"] != primary["overall"]["model_positive"]
            or totals["negative"] != primary["overall"]["model_negative"]
            or {name: nulls[name] for name in FEATURE_NAMES} != primary["model_nulls"]):
        raise ValueError("Independent full-table counts or nulls differ")
    sample = [{"category": category, "cert": cert, "quarter": quarter,
               "candidate_path": path}
              for category in CATEGORIES
              for _, cert, quarter, path in sorted(heaps[category], reverse=True)]
    if any(sum(item["category"] == category for item in sample) != SAMPLE_PER_CATEGORY
           for category in CATEGORIES):
        raise ValueError("Deterministic real sample lacks a required category")
    return {"totals": dict(totals), "model_nulls": {name: nulls[name] for name in FEATURE_NAMES},
            "by_quarter": by_quarter}, sample


def _next_quarter(q: date) -> date:
    return date(q.year + (q.month == 12), 3 if q.month == 12 else q.month + 3,
                31 if q.month in (12, 9) else 30)


def _sample_recomputation(root: Path, panel: dict, sample: list[dict],
                          panel_run: str = PANEL_RUN) -> dict:
    selected_certs = {item["cert"] for item in sample}
    reports: dict[int, list[FinancialReport]] = defaultdict(list)
    for source in sorted((item for item in panel["outputs"]
                          if item["dataset"] == "financials"), key=lambda item: item["quarter"]):
        path = root / "data_processed" / "task_2_2_panel" / panel_run / source["relative_path"]
        with pq.ParquetFile(path) as file:
            for row in file.read(columns=list(SOURCE_COLUMNS)).to_pylist():
                if row["CERT"] in selected_certs:
                    reports[row["CERT"]].append(_report_from_panel(row))
    details = []
    for item in sample:
        cert, q = item["cert"], _day(item["quarter"])
        with pq.ParquetFile(item["candidate_path"]) as file:
            candidates = file.read(columns=["CERT", "Q", *FEATURE_NAMES,
                "SELECTED_REPDTE", "AVAILABILITY_DATE", "FEATURE_LINEAGE_JSON"]).to_pylist()
        saved = next(row for row in candidates if row["CERT"] == cert)
        request = PredictionRequest(cert, q, "verified_registry", q, "P4.4 sampled recomputation")
        result = compute_features(request, reports[cert])
        selected = next(report for report in reports[cert]
                        if report.repdte == result.source_repdte)
        if (result.status != "selected" or result.source_repdte != saved["SELECTED_REPDTE"]
                or availability_date(selected) != saved["AVAILABILITY_DATE"]
                or any(result.values[name] != saved[name] for name in FEATURE_NAMES)
                or result.diagnostics != json.loads(saved["FEATURE_LINEAGE_JSON"])):
            raise ValueError(f"Real source-history feature or lineage differs: {cert}, {q}")
        future_date = _next_quarter(q)
        future_values = dict(reports[cert][-1].values)
        future_values.update({"ASSET": Decimal("999999999.0000"),
                              "ROA": 999.0, "FAILDATE": q,
                              "MEMBERSHIP": "changed", "DQ_FLAGS": ["future"]})
        future = FinancialReport(cert, future_date, future_values,
                                 source_id=f"audit-future:{cert}:{q}")
        revised_values = dict(selected.values)
        revised_values["ASSET"] = Decimal("888888888.0000")
        revised_values["FAILDATE"] = q
        revision = FinancialReport(cert, selected.repdte, revised_values,
            source_id=f"audit-revision:{cert}:{q}",
            revision_published_at=q + timedelta(days=1), revision_time_verified=True)
        perturbed = compute_features(request, [*reports[cert], future, revision])
        if (perturbed.values != result.values
                or perturbed.diagnostics != result.diagnostics
                or perturbed.source_repdte != result.source_repdte):
            raise ValueError(f"Future data altered an earlier sampled feature: {cert}, {q}")
        details.append({"category": item["category"], "cert": cert,
                        "q": q.isoformat(), "selected_repdte": result.source_repdte.isoformat(),
                        "availability_date": saved["AVAILABILITY_DATE"].isoformat(),
                        "feature_values_equal": True, "lineage_equal": True,
                        "future_report_revision_and_metadata_invariant": True})
    return {"sample_seed": SAMPLE_SEED, "per_category": SAMPLE_PER_CATEGORY,
            "sampled_rows": len(details), "distinct_banks": len(selected_certs),
            "category_counts": dict(sorted(Counter(item["category"] for item in details).items())),
            "first_q": min(item["q"] for item in details),
            "last_q": max(item["q"] for item in details),
            "details": details}


def _synthetic_controls() -> dict:
    """Exercise the production components with deliberate leakage controls."""
    q = date(2023, 6, 30)
    repdte = date(2022, 12, 31)
    values = dict.fromkeys(PREDICTOR_FIELDS)
    values.update(ASSET=Decimal("100.0000"), DEP=Decimal("80.0000"),
                  EQ=Decimal("10.0000"), ROA=1.0, RBC1AAJ=8.0)
    report = FinancialReport(11, repdte, values, source_id="fixture-original")
    row = {"CERT": 11, "REPDTE": repdte, "ID": "fixture-original",
           "SOURCE_MISSING_FIELDS": [], **values}
    state = ReportHistoryState([{"quarter": "20221231", "rows": 1}])
    state.advance(date(2023, 3, 31), lambda source: [row])
    if state.pointer != 0:
        raise ValueError("Negative control: future report entered the history cache")
    state.advance(q, lambda source: [row])
    anchor = state.anchors[11]
    candidate = {"CERT": 11, "Q": q, "SELECTED_REPDTE": repdte,
        "AVAILABILITY_DATE": q + timedelta(days=1),
        "REPORT_AGE_DAYS": (q - repdte).days,
        "AVAILABILITY_BASIS": "provisional_120_day",
        "SOURCE_ID_DIAGNOSTIC": report.source_id,
        "SOURCE_MISSING_FIELDS_DIAGNOSTIC": [], **anchor.source_values}
    label = {"CERT": 11, "Q": q, "MEMBERSHIP": "eligible",
             "FEATURE_HISTORY_ELIGIBLE": True, "LABEL_DISPOSITION": "negative", "Y": 0}
    try:
        build_quarter_tables(q, [candidate], [label], state.anchors)
    except ValueError as exc:
        if "availability" not in str(exc):
            raise
    else:
        raise ValueError("Negative control: a future-available candidate was accepted")

    def frame(assets: list[str], growth: list[object], *, y: int) -> pd.DataFrame:
        rows = []
        for asset, change in zip(assets, growth):
            item = {name: 1.0 for name in FEATURE_NAMES}
            item.update(asset_size=Decimal(asset), asset_growth_qoq_pct=change,
                        CERT=11, Q=q, Y=y, FAILDATE="future-only-metadata")
            rows.append(item)
        return pd.DataFrame(rows)

    train = frame(["100", "200"], [None, None], y=0)
    heldout = frame(["1000000"], [Decimal("50")], y=1)
    local = make_preprocessing_pipeline().fit(train)
    stats = local.named_steps["imputer"].statistics_.copy()
    mean = local.named_steps["scaler"].mean_.copy()
    variance = local.named_steps["scaler"].var_.copy()
    scale = local.named_steps["scaler"].scale_.copy()
    first = local.transform(heldout)
    changed = heldout.copy()
    changed["asset_size"] = Decimal("999999999")
    changed["Y"] = 0
    changed["FAILDATE"] = "changed"
    second = local.transform(changed)
    if (not np.array_equal(stats, local.named_steps["imputer"].statistics_)
            or not np.array_equal(mean, local.named_steps["scaler"].mean_)
            or not np.array_equal(variance, local.named_steps["scaler"].var_)
            or not np.array_equal(scale, local.named_steps["scaler"].scale_)
            or first[0, 0] == second[0, 0]
            or not np.array_equal(first[0, 1:], second[0, 1:])
            or list(local.get_feature_names_out()) != list(PREDICTOR_COLUMNS)):
        raise ValueError("Fold-local fit or metadata perturbation control failed")
    contaminated = make_preprocessing_pipeline().fit(pd.concat([train, heldout], ignore_index=True))
    if contaminated.named_steps["scaler"].mean_[0] == mean[0]:
        raise ValueError("Negative control failed to detect combined train/held-out fitting")

    member = MembershipEvidence(11, q, "eligible", "retrospective_operational_v1",
                                None, "fixture", True)
    event = lambda day, transaction="verified_failure": OutcomeEvent(
        f"event-{day}-{transaction}", 11, day, transaction,
        "chronological_candidate")
    right = date(2024, 6, 30)
    outcomes = {
        "on_q_excluded": classify_label(member, [event(q)], outcome_complete_through=right).disposition,
        "after_q_positive": classify_label(member, [event(q + timedelta(days=1))],
                                           outcome_complete_through=right).disposition,
        "right_endpoint_positive": classify_label(member, [event(right)],
                                                  outcome_complete_through=right).disposition,
        "after_endpoint_negative": classify_label(member, [event(right + timedelta(days=1))],
                                                  outcome_complete_through=right).disposition,
        "incomplete_unresolved": classify_label(member, [],
                                                 outcome_complete_through=right - timedelta(days=1)).disposition,
        "assistance_negative": classify_label(member, [event(right, "assistance_transaction")],
                                              outcome_complete_through=right).disposition,
    }
    if outcomes != {"on_q_excluded": "excluded", "after_q_positive": "positive",
                    "right_endpoint_positive": "positive", "after_endpoint_negative": "negative",
                    "incomplete_unresolved": "unresolved", "assistance_negative": "negative"}:
        raise ValueError(f"Official outcome-window fixture differs: {outcomes}")
    return {"future_report_not_cached_before_availability": True,
            "deliberate_early_candidate_rejected": True,
            "fold_statistics_unchanged_by_heldout_perturbation": True,
            "combined_fit_negative_control_detected": True,
            "outcome_boundaries": outcomes}


def run(root: Path, run_id: str, *, selected: dict[str, str] | None = None) -> dict:
    root = root.resolve(strict=True)
    _run_id(run_id)
    evidence = root / "runs" / NAMESPACE / run_id
    if evidence.exists():
        raise FileExistsError("P4.4 run ID already exists")
    selected = dict(DEFAULT_RUNS if selected is None else selected)
    primary, repeat, labels, panel, paths = _accepted_paths(root, selected)
    before = _inventory(paths)
    config = load_config(root / "config" / "financials_full_v2.json")
    manifest = RunManifest.new(run_id, config)
    manifest.config.update({"audit_version": AUDIT_VERSION, "sample_seed": SAMPLE_SEED,
                            "sample_per_category": SAMPLE_PER_CATEGORY,
                            "accepted_runs": selected})
    for path, digest in before.items():
        manifest.record_input(path, digest)
    evidence.mkdir(parents=True)
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    try:
        controls = _synthetic_controls()
        alignment, sample = _full_alignment(primary, labels)
        recomputed = _sample_recomputation(root, panel, sample, selected["panel"])
        after = _inventory(paths)
        if before != after:
            changed = [path for path in before if before[path] != after.get(path)]
            raise ValueError(f"Accepted upstream hash changed during audit: {changed[:3]}")
        inventory = {"status": "complete", "run_id": run_id,
                     "before": before, "after": after, "changed_paths": []}
        _write_json_atomic(evidence / "hash_inventory.json", inventory)
        _write_json_atomic(evidence / "sample_checks.json", recomputed)
        checks = [
            {"name": "accepted_manifest_schema_key_codec_sha_and_source", "passed": True,
             "scope": "both accepted P4.3 runs and their validated ancestors"},
            {"name": "independent_repeat", "passed": True,
             "scope": "332 matching output SHA-256 values"},
            {"name": "synthetic_future_admission_and_fold_fit_negative_controls", "passed": True,
             "scope": "deliberate early admission and combined-period fit detected"},
            {"name": "synthetic_official_label_window", "passed": True,
             "scope": "q, after q, right endpoint, after endpoint, assistance, incomplete follow-up"},
            {"name": "full_model_label_candidate_alignment", "passed": True,
             "scope": "all 166 prediction quarters and all model bank-quarter keys"},
            {"name": "full_model_predictor_values_and_missingness", "passed": True,
             "scope": "all model rows and seven predictor values"},
            {"name": "availability_and_age_lineage", "passed": True,
             "scope": "all candidate bank-quarters"},
            {"name": "sampled_source_history_and_future_perturbations", "passed": True,
             "scope": f"{recomputed['sampled_rows']} deterministic real bank-quarters"},
            {"name": "unchanged_upstream_hashes", "passed": True,
             "scope": f"{len(paths)} accepted files before and after"},
        ]
        summary = {"status": "complete", "run_id": run_id,
                   "audit_version": AUDIT_VERSION,
                   "source_runs": selected,
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "checks": checks, "full_alignment": alignment,
                   "synthetic_controls": controls,
                   "sample": {key: value for key, value in recomputed.items() if key != "details"},
                   "upstream_file_count": len(paths), "upstream_hashes_unchanged": True,
                   "artifacts": {"sample": str(evidence / "sample_checks.json"),
                                 "hash_inventory": str(evidence / "hash_inventory.json")},
                   "limitations": [
                       "Sampled source-history value recomputation is not an exhaustive feature-value audit.",
                       "The accepted Financials panel has one version per bank-quarter and lacks historical revision vintages.",
                       "REPDTE plus 120 days is an availability assumption, not a historical publication timestamp.",
                       "Retrospective cohort activity evidence is not a predictor or a real-time registry claim."]}
        _write_json_atomic(evidence / "summary.json", summary)
        _write_json_atomic(evidence / "metadata.json", {
            "status": "complete", "summary_sha256": _sha(evidence / "summary.json"),
            "sample_sha256": _sha(evidence / "sample_checks.json"),
            "hash_inventory_sha256": _sha(evidence / "hash_inventory.json")})
        for check in checks:
            manifest.record_validation(check["name"], True, check["scope"])
        manifest.record_output(path=str(evidence / "summary.json"), dataset="leakage_audit_summary",
            rows=len(checks), sha256=_sha(evidence / "summary.json"), compression="none",
            schema_version=AUDIT_VERSION, partition=None, source_total=len(checks), source_query=None)
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return {"status": "complete", "run_id": run_id, "checks": len(checks),
                "sampled_rows": recomputed["sampled_rows"],
                "upstream_files": len(paths), "totals": alignment["totals"]}
    except Exception as exc:
        manifest.record_validation("leakage_audit", False, f"{type(exc).__name__}: {exc}")
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str, *, rehash_inputs: bool = True) -> dict:
    root = root.resolve(strict=True)
    _run_id(run_id)
    evidence = root / "runs" / NAMESPACE / run_id
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    inventory = _read_json(evidence / "hash_inventory.json")
    sample = _read_json(evidence / "sample_checks.json")
    recorded_inputs = {item["path"]: item["sha256"] for item in manifest.inputs}
    recorded_checks = {item["name"]: item["passed"] for item in manifest.validations}
    if (manifest.status != "complete" or any(not item["passed"] for item in manifest.validations)
            or summary.get("status") != "complete" or summary.get("run_id") != run_id
            or summary.get("audit_version") != AUDIT_VERSION
            or len(summary.get("checks", [])) != 9
            or len(manifest.validations) != 9
            or {item["name"]: item["passed"] for item in summary["checks"]} != recorded_checks
            or len(recorded_inputs) != len(manifest.inputs)
            or recorded_inputs != inventory.get("before")
            or len(manifest.outputs) != 1
            or manifest.outputs[0]["sha256"] != _sha(evidence / "summary.json")
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or metadata.get("sample_sha256") != _sha(evidence / "sample_checks.json")
            or metadata.get("hash_inventory_sha256") != _sha(evidence / "hash_inventory.json")
            or inventory.get("before") != inventory.get("after")
            or summary.get("sample", {}).get("sampled_rows") != sample.get("sampled_rows")
            or not all(item.get("passed") for item in summary.get("checks", []))):
        raise ValueError("P4.4 manifest or saved evidence is incomplete/inconsistent")
    if summary.get("source_runs", manifest.config.get("accepted_runs")) != manifest.config.get("accepted_runs"):
        raise ValueError("P4.4 saved upstream lineage differs")
    if (root / "runs/source_vintage_bundle.json").is_file() and any(
            not Path(path).resolve().is_relative_to(root) for path in inventory["after"]):
        raise ValueError("Portable leakage-audit input escaped root")
    if rehash_inputs and _inventory([Path(path) for path in inventory["after"]]) != inventory["after"]:
        raise ValueError("P4.4 accepted input hash changed after the audit")
    return {"status": "complete", "run_id": run_id,
            "checks": len(summary["checks"]),
            "sampled_rows": sample["sampled_rows"],
            "upstream_files": len(inventory["before"]),
            "totals": summary["full_alignment"]["totals"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only P4.4 temporal leakage audit")
    parser.add_argument("mode", choices=("run", "verify", "show"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    for key in DEFAULT_RUNS:
        parser.add_argument("--" + key.replace("_", "-"))
    args = parser.parse_args()
    if args.mode == "run":
        selected = {key: getattr(args, key) for key in DEFAULT_RUNS}
        if any(selected.values()) and not all(selected.values()):
            parser.error("All upstream run IDs are required when selecting a replay chain")
        result = run(args.project_root, args.run_id,
                     selected=selected if any(selected.values()) else None)
    elif args.mode == "verify":
        result = verify_saved(args.project_root, args.run_id)
    else:
        result = verify_saved(args.project_root, args.run_id, rehash_inputs=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
