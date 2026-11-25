"""Freeze one development-OOF alarm threshold under the accepted COST proxy."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import platform

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bft65b.config import load_config
from bft65b.manifest import RunManifest
from bft65b.modelling.cost_audit import (
    fold_training_costs, representative_training_cost, scenario_spec,
    verify_saved as verify_cost, _write_text_atomic,
)
from bft65b.modelling.development import verify_saved as verify_development
from bft65b.modelling.split import _write_json_atomic, verify_saved as verify_split
from bft65b.panel.build import _write_verified
from bft65b.panel.retrospective import _root, _run_id
from bft65b.quality.audit import _read_json, _sha


VERSION = "development_oof_threshold_v1"
NAMESPACE = "task_5_5_threshold"
SPLIT_RUN = "split20261120b"
DEV_RUNS = ("dev20261121a", "dev20261121b")
COST_RUNS = ("costaudit20261122d", "costaudit20261122e")
DEFAULT_RUNS = {"split": SPLIT_RUN, "development_primary": DEV_RUNS[0],
                "development_repeat": DEV_RUNS[1], "cost_primary": COST_RUNS[0],
                "cost_repeat": COST_RUNS[1]}
SELECTED = "bounded_hist_gradient_boosting"
REFERENCE_ALPHA = 0.10
REFERENCE_BETA = 0.01
COMPARE_ATOL = 1e-10

CURVE_SCHEMA = pa.schema([
    pa.field("POLICY_INDEX", pa.int64(), nullable=False),
    pa.field("POLICY_KIND", pa.string(), nullable=False),
    pa.field("THRESHOLD", pa.float64()),
    pa.field("N", pa.int64(), nullable=False),
    pa.field("TP", pa.int64(), nullable=False),
    pa.field("FP", pa.int64(), nullable=False),
    pa.field("TN", pa.int64(), nullable=False),
    pa.field("FN", pa.int64(), nullable=False),
    pa.field("ALARMS", pa.int64(), nullable=False),
    pa.field("ALARM_RATE", pa.float64(), nullable=False),
    pa.field("PRECISION", pa.float64()),
    pa.field("RECALL", pa.float64()),
    pa.field("FALSE_POSITIVE_RATE", pa.float64()),
    pa.field("FALSE_ALARM_COST_THOUSANDS", pa.float64(), nullable=False),
    pa.field("MISSED_COST_THOUSANDS", pa.float64(), nullable=False),
    pa.field("OBJECTIVE_THOUSANDS_PER_ROW", pa.float64(), nullable=False),
    pa.field("NORMALIZED_OBJECTIVE", pa.float64(), nullable=False),
], metadata={b"bft65b.threshold": VERSION.encode()})


def _paths(root: Path, run_id: str) -> tuple[Path, Path]:
    _run_id(run_id)
    return root / "runs" / NAMESPACE / run_id, root / "data_processed" / NAMESPACE / run_id


def _positive_median(value: object, fold: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid representative COST median for {fold}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Representative COST median must be finite and positive for {fold}")
    return number


def _quarter_index(value: date) -> int:
    return value.year * 4 + value.month // 3 - 1


def validate_oof_contract(rows: list[dict], expected: dict[tuple[int, date], int],
                          positive_links: dict[tuple[int, date], str],
                          fold_by_q: dict[date, str], fold_ids: list[str]) -> dict:
    """Reject incomplete, duplicated, nondevelopment, or relabelled OOF rows."""
    seen = set()
    score_values, labels, fold_index = [], [], []
    fold_lookup = {fold_id: i for i, fold_id in enumerate(fold_ids)}
    positive_ids = set()
    for row in rows:
        key = row["CERT"], row["Q"]
        if (key in seen or key not in expected or expected[key] != row["Y"]
                or row["FOLD_ID"] != fold_by_q.get(row["Q"])
                or row["FOLD_ID"] not in fold_lookup
                or row["CANDIDATE_ID"] != SELECTED or row["Y"] not in (0, 1)
                or not isinstance(row["SCORE"], (float, int))
                or not math.isfinite(row["SCORE"]) or not 0 <= row["SCORE"] <= 1):
            raise ValueError(f"Duplicate, missing, nondevelopment or mismatched OOF score/label key: {key}")
        seen.add(key)
        if row["Y"] == 1:
            if key not in positive_links:
                raise ValueError(f"Positive OOF key lacks verified FDIC source ID: {key}")
            positive_ids.add(positive_links[key])
        elif key in positive_links:
            raise ValueError(f"Negative OOF key has positive source ID: {key}")
        score_values.append(row["SCORE"])
        labels.append(row["Y"])
        fold_index.append(fold_lookup[row["FOLD_ID"]])
    if seen != set(expected) or {key for key in positive_links} != {key for key in seen if expected[key] == 1}:
        raise ValueError("OOF keys or positive source linkage do not cover the frozen validation set")
    return {"scores": np.asarray(score_values, dtype=np.float64),
            "labels": np.asarray(labels, dtype=np.int8),
            "fold_index": np.asarray(fold_index, dtype=np.int8),
            "positive_rows": int(np.count_nonzero(labels)),
            "distinct_positive_events": len(positive_ids)}


def _preflight(root: Path, selected: dict[str, str] | None = None) -> dict:
    """Verify saved evidence and read only the selected development OOF scores."""
    selected = selected or DEFAULT_RUNS.copy()
    portable = (root / "runs/source_vintage_bundle.json").is_file()
    if set(selected) != set(DEFAULT_RUNS) or any(not value for value in selected.values()):
        raise ValueError("All threshold upstream run IDs are required")
    if portable and any(selected[key] == default for key, default in DEFAULT_RUNS.items()):
        raise ValueError("Portable threshold cannot use historical default upstream runs")
    verify_split(root, selected["split"], rehash_inputs=True)
    dev_runs = (selected["development_primary"], selected["development_repeat"])
    cost_runs = (selected["cost_primary"], selected["cost_repeat"])
    for run_id in dev_runs:
        checked = verify_development(root, run_id, rehash_inputs=True)
        if checked["selected_candidate"] != SELECTED:
            raise ValueError(f"Accepted candidate changed in {run_id}")
    for run_id in cost_runs:
        verify_cost(root, run_id, rehash_inputs=True)
    split_dir = root / "runs/task_5_1_split" / selected["split"]
    dev_dir = root / "runs/task_5_2_development" / dev_runs[0]
    cost_dir = root / "runs/task_5_4_cost" / cost_runs[0]
    split = _read_json(split_dir / "summary.json")
    development = _read_json(dev_dir / "summary.json")
    frozen = _read_json(dev_dir / "frozen_plan.json")
    cost = _read_json(cost_dir / "summary.json")
    if portable:
        expected_cost = {"split": selected["split"], "development_primary": dev_runs[0],
                         "development_repeat": dev_runs[1],
                         "repair": split.get("source_runs", {}).get("repaired_events")}
        if (frozen.get("split_run") != selected["split"]
                or cost.get("source_runs") != expected_cost
                or _read_json(root / "runs/task_5_4_cost" / cost_runs[1] / "summary.json").get("source_runs") != expected_cost):
            raise ValueError("Threshold split, development and COST lineage is inconsistent")
    scenario = _read_json(cost_dir / "scenario.json")
    if scenario != scenario_spec() or development["selected_candidate"] != SELECTED:
        raise ValueError("Accepted scenario or model selection changed")
    if (scenario["reference"] != {"preventable_fraction": REFERENCE_ALPHA,
                                 "false_alarm_fraction_of_training_median": REFERENCE_BETA}
            or scenario["sensitivity"]["preventable_fractions"] != [0.05, 0.10, 0.25]
            or scenario["sensitivity"]["false_alarm_fractions_of_training_median"] != [0.001, 0.01, 0.05]):
        raise ValueError("Accepted reference or sensitivity grid changed")
    oof_item = next(item for item in development["outputs"] if item["candidate_id"] == SELECTED)
    repeat = _read_json(root / "runs/task_5_2_development" / dev_runs[1] / "summary.json")
    repeat_item = next(item for item in repeat["outputs"] if item["candidate_id"] == SELECTED)
    if oof_item["sha256"] != repeat_item["sha256"]:
        raise ValueError("Accepted selected-model OOF repeat differs")
    oof_path = Path(oof_item["path"])
    oof = pq.read_table(oof_path, columns=["CERT", "Q", "Y", "FOLD_ID", "CANDIDATE_ID", "SCORE"])
    rows = oof.to_pylist()
    folds = frozen["folds"]
    fold_by_q: dict[date, str] = {}
    for fold in folds:
        start, end = date.fromisoformat(fold["validation_start"]), date.fromisoformat(fold["validation_end"])
        expected_quarters = _quarter_index(end) - _quarter_index(start) + 1
        observed_quarters = 0
        for row in split["by_quarter"]:
            q = date(int(row["quarter"][:4]), int(row["quarter"][4:6]), int(row["quarter"][6:]))
            if start <= q <= end:
                if q in fold_by_q or row["period"] != "development":
                    raise ValueError("Frozen validation quarters overlap or leave development")
                fold_by_q[q] = fold["id"]
                observed_quarters += 1
        if observed_quarters != expected_quarters:
            raise ValueError(f"Frozen validation quarter coverage differs: {fold['id']}")
    assignment_path = Path(next(item["path"] for item in split["outputs"] if item["dataset"] == "assignments"))
    assignments = pq.read_table(assignment_path, columns=["CERT", "Q", "Y", "PERIOD"])
    allowed_q = pa.array(sorted(fold_by_q), type=pa.date32())
    assignments = assignments.filter(pc.is_in(assignments["Q"], value_set=allowed_q))
    expected = {(row["CERT"], row["Q"]): row["Y"] for row in assignments.to_pylist()
                if row["PERIOD"] == "development"}
    if len(expected) != assignments.num_rows:
        raise ValueError("Duplicate or nondevelopment split assignment in OOF range")
    link_path = Path(next(item["path"] for item in split["outputs"] if item["dataset"] == "positive_linkage"))
    links = pq.read_table(link_path).to_pylist()
    positive_links = {(row["CERT"], row["Q"]): row["SOURCE_ID"] for row in links
                      if row["PERIOD"] == "development" and row["Q"] in fold_by_q}
    if len(positive_links) != sum(row["PERIOD"] == "development" and row["Q"] in fold_by_q for row in links):
        raise ValueError("Duplicate positive OOF linkage key")
    validated = validate_oof_contract(rows, expected, positive_links, fold_by_q,
                                      [fold["id"] for fold in folds])
    score_values, labels, fold_index = (validated[name] for name in ("scores", "labels", "fold_index"))
    for i, fold in enumerate(folds):
        count = sum(k == i for k in fold_index)
        positive = sum(labels[j] == 1 for j, k in enumerate(fold_index) if k == i)
        if count != fold["validation_rows"] or positive != fold["validation_positive_rows"]:
            raise ValueError(f"Frozen fold OOF counts differ: {fold['id']}")
    if (validated["distinct_positive_events"] != cost["measured"]["oof_validation"]["distinct_events"]
            or len(positive_links) != cost["measured"]["oof_validation"]["positive_bank_quarter_rows"]):
        raise ValueError("OOF positive event identity or row coverage differs from accepted COST audit")
    cost_events_path = Path(cost["output"]["path"])
    event_rows = pq.read_table(cost_events_path).to_pylist()
    fitted = fold_training_costs(links, event_rows, folds)
    if fitted != cost["measured"]["fold_training_costs"]:
        raise ValueError("Fold-local COST estimates changed")
    medians = [_positive_median(item["training_only_median_nominal_thousands_usd"], item["fold_id"])
               for item in fitted]
    full_dev_median = representative_training_cost({row["SOURCE_ID"] for row in links
        if row["PERIOD"] == "development"}, event_rows)
    input_paths = [oof_path, Path(repeat_item["path"]), assignment_path, link_path,
                   cost_events_path, split_dir / "summary.json", split_dir / "run_manifest.json",
                   dev_dir / "summary.json", dev_dir / "frozen_plan.json",
                   cost_dir / "summary.json", cost_dir / "scenario.json",
                   cost_dir / "run_manifest.json"]
    for run_id in dev_runs[1:]:
        input_paths.extend([root / "runs/task_5_2_development" / run_id / "summary.json",
                            root / "runs/task_5_2_development" / run_id / "run_manifest.json"])
    for run_id in cost_runs[1:]:
        input_paths.extend([root / "runs/task_5_4_cost" / run_id / "summary.json",
                            root / "runs/task_5_4_cost" / run_id / "run_manifest.json"])
    hashes = {str(path): _sha(path) for path in input_paths}
    if portable:
        bundle = root / "runs/source_vintage_bundle.json"
        hashes[str(bundle)] = _sha(bundle)
        if any(not Path(path).resolve().is_relative_to(root) for path in hashes):
            raise ValueError("Portable threshold input path escaped root")
    positive_count = validated["positive_rows"]
    if positive_count != cost["measured"]["oof_validation"]["positive_bank_quarter_rows"]:
        raise ValueError("OOF positive count differs after numeric conversion")
    return {"scores": score_values,
            "labels": labels,
            "fold_index": fold_index,
            "folds": folds, "medians": medians, "scenario": scenario,
            "input_hashes": hashes, "oof_sha256": oof_item["sha256"],
            "oof_rows": len(rows), "oof_positive_rows": positive_count,
            "oof_events": validated["distinct_positive_events"],
            "initial_unscored_development_rows": sum(row["model_rows"] for row in split["by_quarter"]
                if row["period"] == "development" and row["quarter"] <
                   folds[0]["validation_start"].replace("-", "")),
            "full_development_representative_cost_thousands": full_dev_median,
            "selected_config": next(item for item in development["candidates"] if item["id"] == SELECTED),
            "source_runs": selected}


def frozen_policy(preflight: dict) -> dict:
    scenario = preflight["scenario"]
    return {
        "version": VERSION, "selected_candidate": SELECTED,
        "selected_configuration": preflight["selected_config"],
        "split_run": preflight["source_runs"]["split"],
        "development_runs": [preflight["source_runs"]["development_primary"], preflight["source_runs"]["development_repeat"]],
        "cost_runs": [preflight["source_runs"]["cost_primary"], preflight["source_runs"]["cost_repeat"]],
        "oof_sha256": preflight["oof_sha256"],
        "input_hashes": preflight["input_hashes"],
        "scenario_sha256": preflight["input_hashes"][next(path for path in preflight["input_hashes"]
            if Path(path).name == "scenario.json" and preflight["source_runs"]["cost_primary"] in Path(path).parts)],
        "folds": [{"id": row["id"], "validation_start": row["validation_start"],
                   "validation_end": row["validation_end"], "training_end": row["train_end"],
                   "training_target_right_end": row["training_target_right_end"],
                   "validation_rows": row["validation_rows"],
                   "validation_positive_rows": row["validation_positive_rows"]}
                  for row in preflight["folds"]],
        "training_only_fold_medians_nominal_thousands_usd": preflight["medians"],
        "oof_rows": preflight["oof_rows"],
        "oof_positive_rows": preflight["oof_positive_rows"],
        "oof_distinct_events": preflight["oof_events"],
        "initial_unscored_development_rows": preflight["initial_unscored_development_rows"],
        "reference": scenario["reference"], "sensitivity": scenario["sensitivity"],
        "decision_unit": scenario["decision_unit"],
        "false_alarm_policy": scenario["false_alarm_policy"],
        "miss_policy": scenario["miss_policy"],
        "candidates": "every distinct selected-model OOF score as SCORE >= threshold, grouping tied scores; explicit always_alarm and never_alarm boundaries",
        "score_comparison": "SCORE >= finite threshold; equality alarms; boundaries ignore numerical SCORE",
        "objective": "sum_f[M_f*(alpha*FN_f+beta*FP_f)]/sum_f[N_f], nominal thousands USD per bank-quarter",
        "normalized_diagnostic": "(alpha*sum_f FN_f+beta*sum_f FP_f)/sum_f N_f; not used for selection",
        "tie_break": "exact minimum float64 objective, then fewer overall false alarms, then higher threshold; rank never_alarm above finite thresholds above always_alarm",
        "undefined_metrics": "precision=null if no alarms; recall=null if no positives; false-positive rate=null if no negatives",
        "test_predictors_opened": 0, "test_scores_computed": 0,
    }


def build_curve(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray,
                medians: list[float]) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    folds = np.asarray(folds)
    if (len(scores) == 0 or len(scores) != len(labels) or len(scores) != len(folds)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
            or not np.isin(labels, [0, 1]).all()
            or not np.isin(folds, np.arange(len(medians))).all()):
        raise ValueError("Invalid score, label or fold vectors")
    medians = [_positive_median(m, f"fold_{i + 1}") for i, m in enumerate(medians)]
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_folds = folds[order]
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    thresholds = sorted_scores[group_ends]
    n_policy = len(thresholds) + 2
    fold_counts = []
    for i in range(len(medians)):
        is_fold = sorted_folds == i
        positives = int(np.sum(is_fold & (sorted_labels == 1)))
        negatives = int(np.sum(is_fold & (sorted_labels == 0)))
        tp = np.r_[0, np.cumsum(is_fold & (sorted_labels == 1))[group_ends], positives].astype(np.int64)
        fp = np.r_[0, np.cumsum(is_fold & (sorted_labels == 0))[group_ends], negatives].astype(np.int64)
        if len(tp) != n_policy:
            raise ValueError("Score grouping failed")
        fold_counts.append({"tp": tp, "fp": fp, "tn": negatives - fp,
                            "fn": positives - tp, "n": positives + negatives,
                            "median": medians[i]})
    return {"thresholds": np.r_[np.nan, thresholds, np.nan],
            "kinds": ["never_alarm"] + ["score_ge"] * len(thresholds) + ["always_alarm"],
            "fold_counts": fold_counts,
            "distinct_scores": len(thresholds), "n": len(scores)}


def _scenario_cost(curve: dict, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    n = curve["n"]
    raw = sum(item["median"] * (alpha * item["fn"] + beta * item["fp"])
              for item in curve["fold_counts"]) / n
    normalized = sum(alpha * item["fn"] + beta * item["fp"]
                     for item in curve["fold_counts"]) / n
    return raw, normalized


def _overall_counts(curve: dict) -> dict[str, np.ndarray]:
    return {name: sum(item[name] for item in curve["fold_counts"])
            for name in ("tp", "fp", "tn", "fn")}


def select_policy(curve: dict, alpha: float, beta: float) -> tuple[int, np.ndarray, np.ndarray]:
    objective, normalized = _scenario_cost(curve, alpha, beta)
    return _select_index(curve, objective), objective, normalized


def _select_index(curve: dict, objective: np.ndarray) -> int:
    fp = _overall_counts(curve)["fp"]
    ranking = np.where(np.isnan(curve["thresholds"]),
                       np.where(np.arange(len(objective)) == 0, np.inf, -np.inf),
                       curve["thresholds"])
    # Exact float64 objective order is part of the frozen policy; no hidden tolerance.
    ordered = np.lexsort((-ranking, fp, objective))
    return int(ordered[0])


def _ratio(numerator: np.ndarray, denominator: np.ndarray | int) -> np.ndarray:
    den = np.broadcast_to(denominator, np.shape(numerator))
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, den, out=result, where=den != 0)
    return result


def _metrics(curve: dict, index: int, alpha: float, beta: float) -> dict:
    counts = _overall_counts(curve)
    tp, fp, tn, fn = (int(counts[name][index]) for name in ("tp", "fp", "tn", "fn"))
    alarms = tp + fp
    cost, normalized = _scenario_cost(curve, alpha, beta)
    return {"policy_index": index, "policy_kind": curve["kinds"][index],
            "threshold": None if math.isnan(curve["thresholds"][index]) else float(curve["thresholds"][index]),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "alarms": alarms, "alarm_rate": alarms / curve["n"],
            "precision": tp / alarms if alarms else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "objective_nominal_thousands_usd_per_row": float(cost[index]),
            "normalized_diagnostic_per_row": float(normalized[index])}


def sensitivity(curve: dict, scenario: dict) -> tuple[dict, list[dict]]:
    ref = scenario["reference"]
    alpha = ref["preventable_fraction"]
    beta = ref["false_alarm_fraction_of_training_median"]
    reference_index, _, _ = select_policy(curve, alpha, beta)
    reference = _metrics(curve, reference_index, alpha, beta)
    rows = []
    for a in scenario["sensitivity"]["preventable_fractions"]:
        for b in scenario["sensitivity"]["false_alarm_fractions_of_training_median"]:
            best, _, _ = select_policy(curve, a, b)
            rows.append({"alpha": a, "beta": b, "alpha_beta_ratio": a / b,
                         "optimal": _metrics(curve, best, a, b),
                         "reference_policy": _metrics(curve, reference_index, a, b),
                         "always_alarm": _metrics(curve, len(curve["kinds"]) - 1, a, b),
                         "never_alarm": _metrics(curve, 0, a, b)})
    return reference, rows


def _curve_table(curve: dict, fold: int | None, alpha: float, beta: float) -> pa.Table:
    if fold is None:
        counts = _overall_counts(curve)
        n = curve["n"]
        false_cost = sum(item["median"] * beta * item["fp"] for item in curve["fold_counts"])
        missed_cost = sum(item["median"] * alpha * item["fn"] for item in curve["fold_counts"])
        objective, normalized = _scenario_cost(curve, alpha, beta)
    else:
        item = curve["fold_counts"][fold]
        counts = {name: item[name] for name in ("tp", "fp", "tn", "fn")}
        n = item["n"]
        false_cost = item["median"] * beta * item["fp"]
        missed_cost = item["median"] * alpha * item["fn"]
        objective = (false_cost + missed_cost) / n
        normalized = (beta * item["fp"] + alpha * item["fn"]) / n
    tp, fp, tn, fn = (counts[name] for name in ("tp", "fp", "tn", "fn"))
    alarms = tp + fp
    size = len(curve["kinds"])
    arrays = [
        pa.array(np.arange(size, dtype=np.int64)),
        pa.array(curve["kinds"]),
        pa.array(curve["thresholds"], mask=np.isnan(curve["thresholds"])),
        pa.array(np.full(size, n, dtype=np.int64)),
        *[pa.array(counts[name].astype(np.int64)) for name in ("tp", "fp", "tn", "fn")],
        pa.array(alarms.astype(np.int64)), pa.array(alarms / n),
        pa.array(_ratio(tp, alarms), mask=alarms == 0),
        pa.array(_ratio(tp, tp + fn), mask=tp + fn == 0),
        pa.array(_ratio(fp, fp + tn), mask=fp + tn == 0),
        pa.array(false_cost.astype(np.float64)), pa.array(missed_cost.astype(np.float64)),
        pa.array(objective.astype(np.float64)), pa.array(normalized.astype(np.float64)),
    ]
    return pa.Table.from_arrays(arrays, schema=CURVE_SCHEMA)


def _svg_curve(curve: dict, reference: dict) -> str:
    objective, _ = _scenario_cost(curve, REFERENCE_ALPHA, REFERENCE_BETA)
    counts = _overall_counts(curve)
    alarms = (counts["tp"] + counts["fp"]) / curve["n"]
    indices = np.unique(np.r_[0, np.linspace(0, len(objective) - 1,
                                             min(900, len(objective)), dtype=int),
                              reference["policy_index"], len(objective) - 1])
    lo, hi = float(np.min(objective)), float(np.max(objective))
    span = hi - lo or 1.0
    points = " ".join(f"{65 + 700 * float(alarms[i]):.2f},{370 - 310 * (float(objective[i])-lo)/span:.2f}"
                      for i in indices)
    j = reference["policy_index"]
    x = 65 + 700 * float(alarms[j])
    y = 370 - 310 * (float(objective[j])-lo)/span
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="440" viewBox="0 0 820 440">'
            f'<rect width="820" height="440" fill="white"/><text x="65" y="28" font-size="18">Reference surrogate cost versus alarm rate</text>'
            f'<path d="M65 60 V370 H765" fill="none" stroke="#333"/>'
            f'<polyline points="{points}" fill="none" stroke="#2867b2" stroke-width="2"/>'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#b22"/>'
            f'<text x="65" y="410" font-size="13">Alarm rate; all empirical policies evaluated, display sampled only</text>'
            f'<text x="70" y="52" font-size="12">Objective range: {lo:.6g} to {hi:.6g} nominal USD thousands per row</text></svg>')


def _svg_sensitivity(rows: list[dict]) -> str:
    cells = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="400" viewBox="0 0 820 400">',
             '<rect width="820" height="400" fill="white"/><text x="45" y="28" font-size="18">Scenario-optimal alarm rates (development OOF)</text>']
    alphas = sorted({row["alpha"] for row in rows})
    betas = sorted({row["beta"] for row in rows})
    for j, b in enumerate(betas):
        cells.append(f'<text x="12" y="{106+j*88}" font-size="14">β={b:g}</text>')
    for i, a in enumerate(alphas):
        cells.append(f'<text x="{160+i*210}" y="61" font-size="14">α={a:g}</text>')
    for row in rows:
        i, j = alphas.index(row["alpha"]), betas.index(row["beta"])
        x, y = 145 + i*210, 74 + j*88
        rate = row["optimal"]["alarm_rate"]
        shade = int(245 - min(rate, 0.20) / 0.20 * 130)
        cells.append(f'<rect x="{x}" y="{y}" width="180" height="70" fill="rgb({shade},{shade+4},{255})" stroke="#999"/>')
        cells.append(f'<text x="{x+9}" y="{y+27}" font-size="13">Alarm rate {rate:.4%}</text>')
        cells.append(f'<text x="{x+9}" y="{y+51}" font-size="12">{row["optimal"]["policy_kind"]}</text>')
    cells.append('</svg>')
    return "".join(cells)


def _report(summary: dict) -> str:
    ref = summary["reference"]
    lines = ["# Development OOF alarm-threshold selection", "",
             "The PDF requires development-only threshold selection from false-alarm and missed-failure costs. This transparent surrogate uses the accepted FDIC Estimated Loss proxy plus declared assumptions; it is not observed avoidable welfare loss.", "",
             f"Selected model: `{SELECTED}`. OOF bank-quarter rows: {summary['oof_rows']:,}; positive rows: {summary['positive_rows']:,}; distinct verified failure events: {summary['distinct_events']:,}. Initial development rows without OOF scores: {summary['initial_unscored_development_rows']:,} (excluded).", "",
             "For fold f, M_f > 0 and loss is M_f[alpha FN_f(t) + beta FP_f(t)]. Thus M_f cannot change that fold's minimizing threshold. The global objective sums monetary-scale losses across folds before dividing by all OOF rows, so different M_f values weight folds differently. The saved normalized diagnostic removes M_f but did not select the policy.", "",
             "Scores are uncalibrated ranking outputs; the threshold comes from the complete empirical OOF decision curve. Equality alarms (`SCORE >= threshold`), tied scores enter together, and explicit never/always policies are compared. Undefined precision is null when no alarms occur.", "",
             f"Reference assumptions: alpha={REFERENCE_ALPHA}, beta={REFERENCE_BETA}; unit is nominal USD thousands per bank-quarter surrogate. Selected policy: `{ref['policy_kind']}` with threshold `{ref['threshold']}`. Confusion: TP={ref['TP']:,}, FP={ref['FP']:,}, TN={ref['TN']:,}, FN={ref['FN']:,}; alarm rate={ref['alarm_rate']:.6%}; objective={ref['objective_nominal_thousands_usd_per_row']:.9g}.", "",
             f"Normalized diagnostic optimum (not selected): `{summary['normalized_diagnostic_optimum']['policy_kind']}`, threshold `{summary['normalized_diagnostic_optimum']['threshold']}`, normalized loss {summary['normalized_diagnostic_optimum']['normalized_diagnostic_per_row']:.9g}. The locked reference has normalized loss {ref['normalized_diagnostic_per_row']:.9g}.", "",
             "| α | β | Scenario-optimal policy | Optimal threshold | Optimal alarms | Optimal cost/row (nominal USD thousands) | Locked-reference cost/row | Always cost/row | Never cost/row |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in summary["sensitivity"]:
        best, locked = row["optimal"], row["reference_policy"]
        lines.append(f"| {row['alpha']:g} | {row['beta']:g} | {best['policy_kind']} | {best['threshold'] if best['threshold'] is not None else 'boundary'} | {best['alarms']:,} | {best['objective_nominal_thousands_usd_per_row']:.9g} | {locked['objective_nominal_thousands_usd_per_row']:.9g} | {row['always_alarm']['objective_nominal_thousands_usd_per_row']:.9g} | {row['never_alarm']['objective_nominal_thousands_usd_per_row']:.9g} |")
    lines.extend(["", "![Reference cost curve](reference_curve.svg)", "", "![Sensitivity alarm rates](sensitivity.svg)", "",
                  "Scenarios with the same alpha/beta ratio share the same optimal decision set under this common M_f structure, apart from exact numerical ties. Sensitivity never changes the locked reference policy. The COST audit keeps missing and negative source values explicit; the fold medians use only unique, usable training events.", "",
                  "Each alarmed negative bank-quarter incurs the assumed investigation charge; repeated missed positive bank-quarters for one failure incur a repeated surrogate. Their sum is not literal FDIC loss. Historical amounts are nominal, not inflation-adjusted. Development OOF served both model and threshold selection, so these numbers are selection results, not unbiased final performance. The test period remains unscored.", ""])
    return "\n".join(lines)


def plan(root: Path, run_id: str, selected: dict[str, str] | None = None) -> dict:
    root = _root(root)
    evidence, data = _paths(root, run_id)
    if evidence.exists() or data.exists():
        raise FileExistsError(f"Threshold run already exists: {run_id}")
    source = _preflight(root, selected)
    spec = frozen_policy(source)
    evidence.mkdir(parents=True)
    plan_path = evidence / "frozen_plan.json"
    plan_hash = _write_json_atomic(plan_path, spec)
    manifest = RunManifest.new(run_id, load_config(root / "config/financials_full_v2.json"))
    manifest.config.update({"threshold_version": VERSION, "frozen_plan_sha256": plan_hash,
                            "source_runs": source["source_runs"]})
    for path, digest in sorted(source["input_hashes"].items()):
        manifest.record_input(path, digest)
    manifest.record_validation("development_only_plan_frozen", True)
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    return {"status": "frozen", "run_id": run_id, "plan_sha256": plan_hash,
            "oof_rows": source["oof_rows"], "positive_rows": source["oof_positive_rows"],
            "distinct_events": source["oof_events"], "fold_medians": source["medians"]}


def execute(root: Path, run_id: str) -> dict:
    root = _root(root)
    evidence, data = _paths(root, run_id)
    plan_path = evidence / "frozen_plan.json"
    manifest = RunManifest.read(evidence / "run_manifest.json")
    if manifest.status != "running" or manifest.outputs or manifest.config["frozen_plan_sha256"] != _sha(plan_path):
        raise ValueError("Threshold plan is not frozen and pending execution")
    spec = _read_json(plan_path)
    selected = {"split": spec["split_run"],
                "development_primary": spec["development_runs"][0],
                "development_repeat": spec["development_runs"][1],
                "cost_primary": spec["cost_runs"][0], "cost_repeat": spec["cost_runs"][1]}
    source = _preflight(root, selected)
    if spec != frozen_policy(source):
        raise ValueError("Frozen plan no longer matches validated upstream inputs")
    try:
        curve = build_curve(source["scores"], source["labels"], source["fold_index"], source["medians"])
        reference, sensitivity_rows = sensitivity(curve, source["scenario"])
        _, _, normalized_objective = select_policy(curve, REFERENCE_ALPHA, REFERENCE_BETA)
        normalized_index = _select_index(curve, normalized_objective)
        normalized_diagnostic = _metrics(curve, normalized_index, REFERENCE_ALPHA, REFERENCE_BETA)
        outputs = []
        for label, fold in [("overall", None)] + [(f["id"], i) for i, f in enumerate(source["folds"])]:
            table = _curve_table(curve, fold, REFERENCE_ALPHA, REFERENCE_BETA)
            path = data / f"reference_curve_{label}.parquet"
            digest = _write_verified(path, table)
            item = {"name": label, "path": str(path), "rows": table.num_rows, "sha256": digest}
            outputs.append(item)
            manifest.record_output(path=str(path), dataset=f"curve_{label}", rows=table.num_rows,
                                   sha256=digest, compression="zstd", schema_version=VERSION,
                                   partition=label, source_total=table.num_rows, source_query=None)
        sensitivity_path = evidence / "sensitivity.json"
        sensitivity_hash = _write_json_atomic(sensitivity_path, {"version": VERSION, "rows": sensitivity_rows})
        full_dev = source["full_development_representative_cost_thousands"]
        _positive_median(full_dev, "full_development")
        locked = {"status": "locked", "version": VERSION, "run_id": run_id,
                  "locked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "candidate": SELECTED, "configuration": source["selected_config"],
                  "policy": {"kind": reference["policy_kind"], "threshold": reference["threshold"],
                             "score_comparison": "SCORE >= threshold"},
                  "reference": {"alpha": REFERENCE_ALPHA, "beta": REFERENCE_BETA},
                  "objective": spec["objective"], "tie_break": spec["tie_break"],
                  "selection_result": reference,
                  "full_development_representative_cost_nominal_thousands_usd": full_dev,
                  "full_development_cost_role": "descriptive/future fixed-policy scale; never used to reoptimize OOF threshold",
                  "frozen_plan_sha256": _sha(plan_path), "oof_sha256": source["oof_sha256"],
                  "scenario_sha256": spec["scenario_sha256"],
                  "sensitivity_sha256": sensitivity_hash,
                  "limitations": ["nominal current-vintage FDIC estimated loss", "assumed preventable fraction and false-alarm charge", "repeated bank-quarter surrogates are not literal total FDIC losses", "development OOF used for model and threshold selection", "scores are uncalibrated"]}
        lock_path = evidence / "locked_policy.json"
        lock_hash = _write_json_atomic(lock_path, locked)
        summary = {"status": "complete", "version": VERSION, "run_id": run_id,
                   "frozen_plan_sha256": _sha(plan_path), "locked_policy_sha256": lock_hash,
                   "sensitivity_sha256": sensitivity_hash,
                   "input_hashes": source["input_hashes"], "outputs": outputs,
                   "oof_rows": source["oof_rows"], "positive_rows": source["oof_positive_rows"],
                   "negative_rows": source["oof_rows"] - source["oof_positive_rows"],
                   "distinct_events": source["oof_events"],
                   "initial_unscored_development_rows": source["initial_unscored_development_rows"],
                   "distinct_score_thresholds": curve["distinct_scores"],
                   "policy_count": len(curve["kinds"]), "reference": reference,
                   "normalized_diagnostic_optimum": normalized_diagnostic,
                   "sensitivity": sensitivity_rows,
                   "fold_medians_nominal_thousands_usd": source["medians"],
                   "full_development_representative_cost_nominal_thousands_usd": full_dev,
                   "environment": {"python": platform.python_version(), "os": platform.platform(),
                                   "numpy": np.__version__, "pyarrow": pa.__version__},
                   "test_predictor_files_opened": 0, "test_predictions_computed": 0,
                   "model_refit": False}
        summary_path = evidence / "summary.json"
        summary_hash = _write_json_atomic(summary_path, summary)
        report_hash = _write_text_atomic(evidence / "report.md", _report(summary))
        plot_hashes = {
            "reference_curve.svg": _write_text_atomic(evidence / "reference_curve.svg", _svg_curve(curve, reference)),
            "sensitivity.svg": _write_text_atomic(evidence / "sensitivity.svg", _svg_sensitivity(sensitivity_rows)),
        }
        _write_json_atomic(evidence / "metadata.json", {"status": "complete", "summary_sha256": summary_hash,
            "report_sha256": report_hash, "plot_hashes": plot_hashes,
            "locked_policy_sha256": lock_hash, "sensitivity_sha256": sensitivity_hash,
            "output_hashes": {item["name"]: item["sha256"] for item in outputs}})
        for name in ("OOF_key_label_event_reconciliation", "fold_training_cost_only",
                     "complete_tied_score_curve", "scenario_grid_and_lock", "atomic_readback"):
            manifest.record_validation(name, True)
        manifest.record_validation("upstream_hashes_unchanged", all(
            _sha(Path(path)) == digest for path, digest in source["input_hashes"].items()))
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        return verify_saved(root, run_id)
    except Exception as exc:
        manifest.record_validation("threshold_failure", False, str(exc))
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str, *, rehash_inputs: bool = True) -> dict:
    root = _root(root)
    evidence, data = _paths(root, run_id)
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    spec_path = evidence / "frozen_plan.json"
    lock_path = evidence / "locked_policy.json"
    locked = _read_json(lock_path)
    if (manifest.status != "complete" or not manifest.validations
            or not all(x["passed"] for x in manifest.validations)
            or summary["status"] != "complete" or summary["run_id"] != run_id
            or summary["version"] != VERSION or metadata["status"] != "complete"
            or summary["frozen_plan_sha256"] != _sha(spec_path)
            or manifest.config["frozen_plan_sha256"] != _sha(spec_path)
            or summary["locked_policy_sha256"] != _sha(lock_path)
            or metadata["locked_policy_sha256"] != _sha(lock_path)
            or summary["sensitivity_sha256"] != _sha(evidence / "sensitivity.json")
            or metadata["sensitivity_sha256"] != _sha(evidence / "sensitivity.json")
            or metadata["summary_sha256"] != _sha(evidence / "summary.json")
            or metadata["report_sha256"] != _sha(evidence / "report.md")
            or any(metadata["plot_hashes"][name] != _sha(evidence / name)
                   for name in ("reference_curve.svg", "sensitivity.svg"))
            or locked["status"] != "locked" or locked["run_id"] != run_id
            or locked["frozen_plan_sha256"] != _sha(spec_path)
            or locked["selection_result"] != summary["reference"]
            or locked["candidate"] != SELECTED
            or summary["test_predictor_files_opened"] != 0
            or summary["test_predictions_computed"] != 0 or summary["model_refit"] is not False
            or len(summary["outputs"]) != 4 or len(manifest.outputs) != 4):
        raise ValueError("Threshold lock, summary, manifest or evidence is inconsistent")
    plan = _read_json(spec_path)
    selected = {"split": plan["split_run"],
                "development_primary": plan["development_runs"][0],
                "development_repeat": plan["development_runs"][1],
                "cost_primary": plan["cost_runs"][0], "cost_repeat": plan["cost_runs"][1]}
    if (set(selected) != set(DEFAULT_RUNS)
            or manifest.config.get("source_runs", selected) != selected
            or ((root / "runs/source_vintage_bundle.json").is_file()
                and any(selected[key] == default for key, default in DEFAULT_RUNS.items()))):
        raise ValueError("Saved threshold upstream lineage is inconsistent")
    if (plan["selected_candidate"] != SELECTED or plan["input_hashes"] != summary["input_hashes"]
            or plan["oof_rows"] != summary["oof_rows"]
            or plan["oof_positive_rows"] != summary["positive_rows"]
            or plan["oof_distinct_events"] != summary["distinct_events"]
            or summary["policy_count"] != summary["distinct_score_thresholds"] + 2):
        raise ValueError("Threshold frozen plan and summary disagree")
    saved_tables = {}
    for item in summary["outputs"]:
        path = Path(item["path"])
        if (not path.resolve().is_relative_to(data.resolve()) or _sha(path) != item["sha256"]
                or metadata["output_hashes"].get(item["name"]) != item["sha256"]
                or not any(m["path"] == item["path"] and m["sha256"] == item["sha256"]
                           for m in manifest.outputs)):
            raise ValueError(f"Threshold curve path or SHA-256 differs: {path}")
        with pq.ParquetFile(path) as file:
            if (file.schema_arrow != CURVE_SCHEMA or file.metadata.num_rows != item["rows"]
                    or item["rows"] != summary["policy_count"]
                    or any(file.metadata.row_group(g).column(c).compression != "ZSTD"
                           for g in range(file.metadata.num_row_groups)
                           for c in range(file.metadata.row_group(g).num_columns))):
                raise ValueError(f"Threshold curve schema, count or codec differs: {path}")
            saved_tables[item["name"]] = file.read()
    if set(saved_tables) != {"overall", "fold_1", "fold_2", "fold_3"}:
        raise ValueError("Threshold curve is missing an expected fold")
    overall = saved_tables["overall"]
    n_policies = summary["policy_count"]
    indices = overall["POLICY_INDEX"].to_pylist()
    kinds = overall["POLICY_KIND"].to_pylist()
    thresholds = overall["THRESHOLD"].to_pylist()
    if (indices != list(range(n_policies)) or kinds[0] != "never_alarm"
            or kinds[-1] != "always_alarm" or any(kind != "score_ge" for kind in kinds[1:-1])
            or thresholds[0] is not None or thresholds[-1] is not None
            or any(not a > b for a, b in zip(thresholds[1:-2], thresholds[2:-1]))):
        raise ValueError("Saved threshold policies are not complete and monotonic")
    curve = {"thresholds": np.asarray([np.nan if x is None else x for x in thresholds]),
             "kinds": kinds, "fold_counts": [], "n": summary["oof_rows"]}
    plan_medians = plan["training_only_fold_medians_nominal_thousands_usd"]
    for i, name in enumerate(("fold_1", "fold_2", "fold_3")):
        table = saved_tables[name]
        if (table["POLICY_INDEX"].to_pylist() != indices
                or table["POLICY_KIND"].to_pylist() != kinds
                or table["THRESHOLD"].to_pylist() != thresholds):
            raise ValueError(f"Fold policy alignment differs: {name}")
        item = {key: np.asarray(table[key].to_numpy(), dtype=np.int64)
                for key in ("TP", "FP", "TN", "FN")}
        item = {"tp": item["TP"], "fp": item["FP"], "tn": item["TN"],
                "fn": item["FN"], "n": int(table["N"][0].as_py()),
                "median": _positive_median(plan_medians[i], name)}
        if (not np.all(item["tp"] + item["fn"] == plan["folds"][i]["validation_positive_rows"])
                or not np.all(item["fp"] + item["tn"] == item["n"] -
                                   plan["folds"][i]["validation_positive_rows"])):
            raise ValueError(f"Fold confusion totals differ: {name}")
        curve["fold_counts"].append(item)
    for name, values in _overall_counts(curve).items():
        if not np.array_equal(values, np.asarray(overall[name.upper()].to_numpy(), dtype=np.int64)):
            raise ValueError(f"Overall and fold curve counts differ: {name}")
    if (not np.all(np.asarray(overall["N"].to_numpy()) == summary["oof_rows"])
            or summary["oof_rows"] != sum(item["n"] for item in curve["fold_counts"])):
        raise ValueError("Overall curve denominators differ")
    rebuilt_reference, rebuilt_sensitivity = sensitivity(curve, _read_json(
        root / "runs/task_5_4_cost" / selected["cost_primary"] / "scenario.json"))
    _, _, normalized = select_policy(curve, REFERENCE_ALPHA, REFERENCE_BETA)
    rebuilt_normalized = _metrics(curve, _select_index(curve, normalized), REFERENCE_ALPHA, REFERENCE_BETA)
    if (rebuilt_reference != summary["reference"] or rebuilt_sensitivity != summary["sensitivity"]
            or ("normalized_diagnostic_optimum" in summary and
                rebuilt_normalized != summary["normalized_diagnostic_optimum"])
            or rebuilt_reference != locked["selection_result"]):
        raise ValueError("Saved threshold optimum or sensitivity differs from complete curve")
    for name, table in saved_tables.items():
        for metric in ("TP", "FP", "TN", "FN", "ALARMS"):
            if np.any(np.asarray(table[metric].to_numpy()) < 0):
                raise ValueError(f"Negative confusion count in {name}")
        alarms = np.asarray(table["ALARMS"].to_numpy())
        if not np.array_equal(alarms, np.asarray(table["TP"].to_numpy()) +
                              np.asarray(table["FP"].to_numpy())):
            raise ValueError(f"Alarm counts differ in {name}")
        false_cost = np.asarray(table["FALSE_ALARM_COST_THOUSANDS"].to_numpy())
        miss_cost = np.asarray(table["MISSED_COST_THOUSANDS"].to_numpy())
        objective = np.asarray(table["OBJECTIVE_THOUSANDS_PER_ROW"].to_numpy())
        denominator = np.asarray(table["N"].to_numpy())
        if not np.allclose((false_cost + miss_cost) / denominator, objective,
                           rtol=0, atol=1e-12):
            raise ValueError(f"Saved cost components do not reconcile in {name}")
        if name == "overall":
            expected_false = sum(item["median"] * REFERENCE_BETA * item["fp"]
                                 for item in curve["fold_counts"])
            expected_miss = sum(item["median"] * REFERENCE_ALPHA * item["fn"]
                                for item in curve["fold_counts"])
        else:
            item = curve["fold_counts"][int(name[-1]) - 1]
            expected_false = item["median"] * REFERENCE_BETA * item["fp"]
            expected_miss = item["median"] * REFERENCE_ALPHA * item["fn"]
        if (not np.allclose(false_cost, expected_false, rtol=0, atol=1e-8)
                or not np.allclose(miss_cost, expected_miss, rtol=0, atol=1e-8)):
            raise ValueError(f"Saved fold-scaled COST differs in {name}")
        tp = np.asarray(table["TP"].to_numpy())
        fp = np.asarray(table["FP"].to_numpy())
        tn = np.asarray(table["TN"].to_numpy())
        fn = np.asarray(table["FN"].to_numpy())
        expected_metrics = {
            "ALARM_RATE": alarms / denominator,
            "PRECISION": _ratio(tp, alarms),
            "RECALL": _ratio(tp, tp + fn),
            "FALSE_POSITIVE_RATE": _ratio(fp, fp + tn),
            "NORMALIZED_OBJECTIVE": (REFERENCE_ALPHA * fn + REFERENCE_BETA * fp) / denominator
                if name != "overall" else
                (REFERENCE_ALPHA * _overall_counts(curve)["fn"] +
                 REFERENCE_BETA * _overall_counts(curve)["fp"]) / denominator,
        }
        for metric, values in expected_metrics.items():
            stored = np.asarray(table[metric].to_numpy(zero_copy_only=False), dtype=np.float64)
            if not np.allclose(stored, values, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(f"Saved {metric} differs in {name}")
    if rehash_inputs and any(_sha(Path(path)) != digest
                             for path, digest in summary["input_hashes"].items()):
        raise ValueError("Accepted upstream threshold input checksum changed")
    return {"status": "complete", "run_id": run_id,
            "reference": summary["reference"], "policy_count": summary["policy_count"],
            "oof_rows": summary["oof_rows"], "distinct_events": summary["distinct_events"],
            "evidence": str(evidence)}


def compare_runs(root: Path, first: str, second: str) -> dict:
    root = _root(root)
    verify_saved(root, first)
    verify_saved(root, second)
    ea, _ = _paths(root, first)
    eb, _ = _paths(root, second)
    a, b = _read_json(ea / "summary.json"), _read_json(eb / "summary.json")
    pa, pb = _read_json(ea / "frozen_plan.json"), _read_json(eb / "frozen_plan.json")
    la, lb = _read_json(ea / "locked_policy.json"), _read_json(eb / "locked_policy.json")
    if pa != pb or a["input_hashes"] != b["input_hashes"]:
        raise ValueError("Independent frozen plans or inputs differ")
    if (la["policy"] != lb["policy"] or la["reference"] != lb["reference"]
            or la["configuration"] != lb["configuration"]
            or a["policy_count"] != b["policy_count"]
            or a["positive_rows"] != b["positive_rows"]
            or a["distinct_events"] != b["distinct_events"]):
        raise ValueError("Independent threshold policy or counts differ")
    for left, right in [(a["reference"], b["reference"])] + [
        (x[part], y[part]) for x, y in zip(a["sensitivity"], b["sensitivity"])
        for part in ("optimal", "reference_policy", "always_alarm", "never_alarm")]:
        if left.keys() != right.keys():
            raise ValueError("Independent scenario fields differ")
        for key in left:
            x, y = left[key], right[key]
            if isinstance(x, float) and isinstance(y, float):
                if not math.isclose(x, y, abs_tol=COMPARE_ATOL, rel_tol=0):
                    raise ValueError(f"Independent threshold metric differs: {key}")
            elif x != y:
                raise ValueError(f"Independent threshold metric differs: {key}")
    if [x["sha256"] for x in a["outputs"]] != [x["sha256"] for x in b["outputs"]]:
        raise ValueError("Independent threshold curves differ in bytes")
    return {"status": "matching", "runs": [first, second],
            "selected_policy": la["policy"], "policy_count": a["policy_count"],
            "metric_tolerance": COMPARE_ATOL, "curve_hashes_equal": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a development-only FDIC COST alarm threshold")
    parser.add_argument("mode", choices=("plan", "execute", "verify", "show", "compare"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compare-run-id")
    for key in DEFAULT_RUNS:
        parser.add_argument("--" + key.replace("_", "-"))
    args = parser.parse_args()
    if args.mode == "plan":
        selected = {key: getattr(args, key) for key in DEFAULT_RUNS}
        if any(selected.values()) and not all(selected.values()):
            parser.error("All threshold upstream run IDs are required when selecting a replay chain")
        result = plan(args.project_root, args.run_id, selected if all(selected.values()) else None)
    elif args.mode == "execute":
        result = execute(args.project_root, args.run_id)
    elif args.mode == "verify":
        result = verify_saved(args.project_root, args.run_id)
    elif args.mode == "show":
        result = verify_saved(args.project_root, args.run_id, rehash_inputs=False)
    else:
        if not args.compare_run_id:
            parser.error("--compare-run-id is required for compare")
        result = compare_runs(args.project_root, args.run_id, args.compare_run_id)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
