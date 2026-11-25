"""One-use locked final evaluation for the accepted chronological split.

The project-level registry survives process death. A process lock serializes
transitions. Once test access is recorded, only a persisted score file can be
used to resume reporting; an absent score requires investigation, not a new
run ID or an automatic second scoring pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
import math
import msvcrt
import os
from pathlib import Path
import platform
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from bft65b.config import load_config
from bft65b.features.preprocessing import PREDICTOR_COLUMNS, PREPROCESSING_VERSION
from bft65b.manifest import RunManifest, dependency_versions
from bft65b.modelling.development import candidate_pipeline, _development_frame
from bft65b.modelling.cost_audit import _write_text_atomic
from bft65b.modelling.split import _write_json_atomic, verify_saved as verify_split
from bft65b.modelling.threshold import verify_saved as verify_threshold
from bft65b.panel.retrospective import _root, _run_id, _quarter
from bft65b.quality.audit import _read_json, _sha

VERSION = "locked_final_evaluation_v1"
NAMESPACE = "task_5_6_final"
SPLIT_RUN = "split20261120b"
DEV_RUN = "dev20261121a"
THRESHOLD_RUN = "threshold20261122c"
MODEL_RUN = "featuretable20261120a"
THREADS = 2
SCORE_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("Q", pa.date32(), nullable=False),
    pa.field("Y", pa.int8(), nullable=False),
    pa.field("SCORE", pa.float64(), nullable=False),
    pa.field("ALARM", pa.bool_(), nullable=False),
], metadata={b"bft65b.final_evaluation": VERSION.encode()})


def _paths(root: Path, run_id: str) -> tuple[Path, Path, Path, Path]:
    base = root / "runs" / NAMESPACE
    return (base / run_id, root / "data_processed" / NAMESPACE / run_id,
            base / "test_use_registry.json", base / ".registry.lock")


@contextmanager
def _ownership(lock_path: Path):
    """Nonblocking Windows byte lock, released by Windows on process death."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if lock_path.stat().st_size == 0:
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another final-evaluation process owns the project registry") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _registry(path: Path) -> dict:
    return _read_json(path) if path.exists() else {"version": VERSION, "designs": {}}


def _transition(path: Path, design: str, run_id: str, state: str, **extra: object) -> dict:
    registry = _registry(path)
    item = registry["designs"].get(design)
    if item is None:
        if state != "planned":
            raise ValueError("Final-evaluation design is not registered")
        item = {"run_id": run_id, "history": []}
        registry["designs"][design] = item
    elif item["run_id"] != run_id:
        raise RuntimeError(f"Locked design already belongs to {item['run_id']}; new run ID cannot reset test use")
    item.update(extra)
    item["state"] = state
    item["history"].append({"state": state, "at_utc": datetime.now(timezone.utc).isoformat()})
    _write_json_atomic(path, registry)
    return item


def _accepted(root: Path) -> dict:
    verify_split(root, SPLIT_RUN)
    verify_threshold(root, THRESHOLD_RUN)
    lock_path = root / "runs/task_5_5_threshold" / THRESHOLD_RUN / "locked_policy.json"
    lock = _read_json(lock_path)
    split_path = root / "runs/task_5_1_split" / SPLIT_RUN / "summary.json"
    split = _read_json(split_path)
    development_path = root / "runs/task_5_2_development" / DEV_RUN / "frozen_plan.json"
    development = _read_json(development_path)
    model_summary_path = root / "runs/task_4_3_feature_table" / MODEL_RUN / "summary.json"
    model = _read_json(model_summary_path)
    bounds = split["bounds"]
    if (lock["status"] != "locked" or lock["run_id"] != THRESHOLD_RUN
            or lock["configuration"] not in development["candidates"]
            or lock["policy"]["kind"] != "score_ge"
            or lock["policy"]["score_comparison"] != "SCORE >= threshold"
            or bounds["gap_quarters"] < 4
            or bounds["development_target_right_end"] >= bounds["test_start"]
            or bounds["test_end"] != "2025-06-30"):
        raise ValueError("Accepted lock, candidate or chronological split differs")
    outputs = {item["quarter"]: item for item in model["outputs"] if item["dataset"] == "model_table"}
    dev_paths = development["development_model_paths"]
    test_items = [outputs[q] for q in sorted(outputs)
                  if bounds["test_start"].replace("-", "") <= q <= bounds["test_end"].replace("-", "")]
    if (len(test_items) != split["periods"]["test"]["prediction_quarters"]
            or len(dev_paths) != split["periods"]["development"]["prediction_quarters"]):
        raise ValueError("Frozen development or test path inventory differs")
    for item in test_items:
        if _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Accepted test file hash changed: {item['path']}")
    sources = {str(path): _sha(path) for path in (lock_path, split_path, development_path, model_summary_path)}
    for item in split["outputs"]:
        if _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Accepted split output hash changed: {item['path']}")
        sources[item["path"]] = item["sha256"]
    for path in (root / "runs/task_5_1_split" / SPLIT_RUN / "run_manifest.json",
                 root / "runs/task_4_3_feature_table" / MODEL_RUN / "run_manifest.json",
                 root / "runs/task_5_5_threshold" / THRESHOLD_RUN / "run_manifest.json"):
        sources[str(path)] = _sha(path)
    for path in dev_paths:
        sources[path] = development["input_hashes"][path]
    for item in test_items:
        sources[item["path"]] = item["sha256"]
    return {"lock": lock, "split": split, "development": development,
            "test_items": test_items, "sources": sources,
            "design": _sha(lock_path) + ":" + _sha(split_path)}


def _plan_document(root: Path, run_id: str, accepted: dict, data: Path) -> dict:
    lock, split = accepted["lock"], accepted["split"]
    return {
        "version": VERSION, "run_id": run_id, "design_key": accepted["design"],
        "source_hashes": accepted["sources"], "bounds": split["bounds"],
        "configuration": lock["configuration"], "predictor_columns": list(PREDICTOR_COLUMNS),
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": lock["configuration"]["params"]["random_state"], "thread_limit": THREADS,
        "fit": "Fresh accepted preprocessing and estimator; development only; no early stopping or calibration",
        "test": "Score once with saved fitted pipeline; persist all test scores before reporting; never rescore after persistence",
        "threshold_policy": lock["policy"], "reference": lock["reference"],
        "cost_scale_nominal_thousands_usd": lock["full_development_representative_cost_nominal_thousands_usd"],
        "metrics": ["sklearn average_precision_score", "sklearn roc_auc_score", "sklearn brier_score_loss",
                    "TP", "FP", "TN", "FN", "precision", "recall", "specificity", "false_positive_rate", "alarm_rate"],
        "event_detection": "A distinct represented failure source ID is detected if any eligible test-positive quarter alarms",
        "uncertainty": "No confidence interval; repeated bank-quarters within banks/events are dependent",
        "diagnostics": ["fixed always_alarm", "fixed never_alarm"],
        "score_path": str(data / "test_scores.parquet"), "model_path": str(data / "fitted_pipeline.joblib"),
        "recovery": "Project-level registry prevents new-run reuse. After test access, resume reports only from persisted scores; absent scores block automatic retry.",
        "test_predictors_opened": False,
    }


def plan(root: Path, run_id: str) -> dict:
    root = _root(root)
    _run_id(run_id)
    evidence, data, registry_path, lock_path = _paths(root, run_id)
    with _ownership(lock_path):
        accepted = _accepted(root)
        existing = _registry(registry_path)["designs"].get(accepted["design"])
        if existing or evidence.exists() or data.exists():
            raise FileExistsError(f"Final-evaluation design or run already exists: {existing or run_id}")
        evidence.mkdir(parents=True)
        spec = _plan_document(root, run_id, accepted, data)
        plan_hash = _write_json_atomic(evidence / "frozen_plan.json", spec)
        manifest = RunManifest.new(run_id, load_config(root / "config/financials_full_v2.json"))
        manifest.config.update({"final_evaluation_version": VERSION, "frozen_plan_sha256": plan_hash})
        for path, digest in sorted(spec["source_hashes"].items()):
            manifest.record_input(path, digest)
        manifest.stages["preparation"] = "complete"
        manifest.write(evidence / "run_manifest.json")
        _transition(registry_path, accepted["design"], run_id, "planned", plan_sha256=plan_hash)
    return {"status": "planned", "run_id": run_id, "plan_sha256": plan_hash,
            "test_predictors_opened": False}


def _registered(root: Path, run_id: str) -> tuple[dict, dict, Path, Path, Path, str]:
    evidence, data, registry_path, _ = _paths(root, run_id)
    spec = _read_json(evidence / "frozen_plan.json")
    item = _registry(registry_path)["designs"].get(spec["design_key"])
    if (not item or item["run_id"] != run_id or item["plan_sha256"] != _sha(evidence / "frozen_plan.json")
            or spec["version"] != VERSION or spec["predictor_columns"] != list(PREDICTOR_COLUMNS)):
        raise ValueError("Frozen final plan or project-level registry is inconsistent")
    return spec, item, evidence, data, registry_path, spec["design_key"]


def _ensure_source_hashes(spec: dict) -> None:
    for path, digest in spec["source_hashes"].items():
        if _sha(Path(path)) != digest:
            raise ValueError(f"Accepted source hash changed: {path}")


def _model_probe(pipeline, frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".model.", suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        joblib.dump(pipeline, temporary)
        loaded = joblib.load(temporary)
        probe = frame.iloc[:min(128, len(frame))][list(PREDICTOR_COLUMNS)]
        left = pipeline.predict_proba(probe)[:, 1]
        right = loaded.predict_proba(probe)[:, 1]
        if not np.array_equal(left, right):
            raise ValueError("Fitted model serialization changed development probe scores")
        if path.exists():
            raise FileExistsError("Fitted model already exists")
        os.replace(temporary, path)
        return _sha(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(root: Path, run_id: str) -> dict:
    root = _root(root)
    _run_id(run_id)
    _, _, registry_path, lock_path = _paths(root, run_id)
    with _ownership(lock_path):
        spec, item, evidence, data, _, design = _registered(root, run_id)
        if item["state"] != "planned":
            raise RuntimeError(f"Cannot refit final model from registry state {item['state']}")
        _ensure_source_hashes(spec)
        _transition(registry_path, design, run_id, "fitting")
        manifest = RunManifest.read(evidence / "run_manifest.json")
        manifest.set_status("running")
        manifest.stages["model_fit"] = "running"
        manifest.write(evidence / "run_manifest.json")
        try:
            dev_spec = _read_json(root / "runs/task_5_2_development" / DEV_RUN / "frozen_plan.json")
            start = time.perf_counter()
            frame, allowed = _development_frame(root, dev_spec)
            bounds = spec["bounds"]
            split = _read_json(root / "runs/task_5_1_split" / SPLIT_RUN / "summary.json")
            period = split["periods"]["development"]
            if (len(frame) != period["model_rows"] or int(frame["Y"].sum()) != period["positive_rows"]
                    or len(allowed) != len(frame)
                    or min(frame["Q"]) < date.fromisoformat(bounds["development_start"])
                    or max(frame["Q"]) > date.fromisoformat(bounds["development_end"])):
                raise ValueError("Full-development rows differ from accepted split")
            pipeline = candidate_pipeline(spec["configuration"])
            with threadpool_limits(limits=spec["thread_limit"]):
                pipeline.fit(frame[list(PREDICTOR_COLUMNS)], frame["Y"].astype(int))
                model_hash = _model_probe(pipeline, frame, data / "fitted_pipeline.joblib")
            elapsed = time.perf_counter() - start
            preprocess = pipeline.named_steps["preprocessing"]
            imputer = preprocess.named_steps["imputer"]
            scaler = preprocess.named_steps["scaler"]
            binding = {"run_id": run_id, "plan_sha256": _sha(evidence / "frozen_plan.json"),
                       "model_sha256": model_hash, "fit_rows": len(frame),
                       "positive_rows": int(frame["Y"].sum()), "fit_seconds_including_read": elapsed,
                       "feature_names": list(PREDICTOR_COLUMNS), "imputer_statistics": imputer.statistics_.tolist(),
                       "scaler_mean": scaler.mean_.tolist(), "scaler_var": scaler.var_.tolist(),
                       "scaler_scale": scaler.scale_.tolist(), "python": platform.python_version(),
                       "dependencies": dependency_versions()}
            binding_hash = _write_json_atomic(evidence / "model_binding.json", binding)
            manifest.record_output(path=str(data / "fitted_pipeline.joblib"), dataset="final_model",
                rows=len(frame), sha256=model_hash, compression="joblib", schema_version=VERSION,
                partition=None, source_total=len(frame), source_query=None)
            manifest.record_validation("full_development_only_fit_and_probe_roundtrip", True)
            manifest.stages["model_fit"] = "complete"
            manifest.write(evidence / "run_manifest.json")
            _transition(registry_path, design, run_id, "model_fitted", model_sha256=model_hash,
                        binding_sha256=binding_hash)
            return {"status": "model_fitted", "run_id": run_id, "rows": len(frame),
                    "positive_rows": int(frame["Y"].sum()), "model_sha256": model_hash,
                    "fit_seconds_including_read": elapsed}
        except Exception:
            manifest.stages["model_fit"] = "failed"
            manifest.set_status("failed")
            manifest.write(evidence / "run_manifest.json")
            _transition(registry_path, design, run_id, "fit_failed")
            raise


def _expected_test(root: Path) -> tuple[dict[tuple[int, date], int], dict[tuple[int, date], str]]:
    summary = _read_json(root / "runs/task_5_1_split" / SPLIT_RUN / "summary.json")
    outputs = {item["dataset"]: item for item in summary["outputs"]}
    assignments = pq.ParquetFile(outputs["assignments"]["path"]).read(columns=["CERT", "Q", "PERIOD", "Y"]).to_pylist()
    links = pq.ParquetFile(outputs["positive_linkage"]["path"]).read(columns=["CERT", "Q", "SOURCE_ID", "PERIOD"]).to_pylist()
    test_assignments = [row for row in assignments if row["PERIOD"] == "test"]
    expected = {(row["CERT"], row["Q"]): row["Y"] for row in test_assignments}
    test_links = [row for row in links if row["PERIOD"] == "test"]
    event = {(row["CERT"], row["Q"]): row["SOURCE_ID"] for row in test_links}
    period = summary["periods"]["test"]
    if (len(expected) != len(test_assignments) or len(expected) != period["model_rows"]
            or sum(expected.values()) != period["positive_rows"]
            or len(test_links) != len(event) or len(event) != period["positive_rows"]
            or len(set(event.values())) != period["distinct_qualifying_events"]
            or set(event) != {k for k, y in expected.items() if y == 1}):
        raise ValueError("Accepted test assignment or positive event linkage differs")
    return expected, event


def _test_frame(root: Path, expected: dict, spec: dict) -> pd.DataFrame:
    summary = _read_json(root / "runs/task_4_3_feature_table" / MODEL_RUN / "summary.json")
    bounds = spec["bounds"]
    items = [item for item in summary["outputs"] if item["dataset"] == "model_table"
             and bounds["test_start"].replace("-", "") <= item["quarter"] <= bounds["test_end"].replace("-", "")]
    if len(items) != _read_json(root / "runs/task_5_1_split" / SPLIT_RUN / "summary.json")["periods"]["test"]["prediction_quarters"]:
        raise ValueError("Test quarter path inventory differs")
    frames = []
    for item in sorted(items, key=lambda value: value["quarter"]):
        if _sha(Path(item["path"])) != spec["source_hashes"][item["path"]]:
            raise ValueError(f"Test predictor hash changed: {item['path']}")
        frame = pq.ParquetFile(item["path"]).read(columns=["CERT", "Q", "Y", *PREDICTOR_COLUMNS]).to_pandas()
        frame["Q"] = frame["Q"].map(lambda value: pd.Timestamp(value).date())
        if any(q.strftime("%Y%m%d") != item["quarter"] for q in frame["Q"]):
            raise ValueError(f"Test partition Q mismatch: {item['path']}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values(["Q", "CERT"], kind="stable").reset_index(drop=True)
    keys = list(zip((int(x) for x in result["CERT"]), result["Q"]))
    if (len(keys) != len(expected) or len(set(keys)) != len(keys)
            or {key: int(y) for key, y in zip(keys, result["Y"])} != expected):
        raise ValueError("Test predictor keys or labels differ from accepted split")
    return result


def _scores(frame: pd.DataFrame, pipeline, threshold: float) -> pa.Table:
    values = np.asarray(pipeline.predict_proba(frame[list(PREDICTOR_COLUMNS)])[:, 1], dtype=np.float64)
    if (len(values) != len(frame) or not np.all(np.isfinite(values))
            or np.any(values < 0) or np.any(values > 1)
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Final scores or locked threshold are invalid")
    rows = [{"CERT": int(cert), "Q": q, "Y": int(y), "SCORE": float(score),
             "ALARM": bool(score >= threshold)}
            for cert, q, y, score in zip(frame["CERT"], frame["Q"], frame["Y"], values)]
    return pa.Table.from_pylist(rows, schema=SCORE_SCHEMA)


def _write_scores(path: Path, table: pa.Table, binding_path: Path,
                  model_sha256: str, plan_sha256: str, start: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".scores.", suffix=".parquet", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        pq.write_table(table, temporary, compression="zstd", compression_level=3)
        if pq.read_table(temporary).equals(table) is False:
            raise ValueError("Temporary score read-back differs")
        if path.exists():
            raise FileExistsError("Final scores already exist; do not overwrite")
        digest = _sha(temporary)
        # Bind bytes before promotion so a crash immediately after promotion
        # leaves enough evidence for score-only recovery without rescoring.
        _write_json_atomic(binding_path, {"score_sha256": digest,
            "model_sha256": model_sha256, "plan_sha256": plan_sha256,
            "score_seconds_through_temp_readback": time.perf_counter() - start})
        os.replace(temporary, path)
        if _sha(path) != digest:
            raise ValueError("Promoted score checksum differs")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _validate_scores(table: pa.Table, expected: dict, threshold: float) -> None:
    if table.schema != SCORE_SCHEMA or table.num_rows != len(expected):
        raise ValueError("Saved final score schema or row count differs")
    rows = table.to_pylist()
    keys = [(row["CERT"], row["Q"]) for row in rows]
    if (len(keys) != len(set(keys)) or keys != sorted(keys, key=lambda x: (x[1], x[0]))
            or {key: row["Y"] for key, row in zip(keys, rows)} != expected):
        raise ValueError("Saved final score keys or labels differ")
    if any(not math.isfinite(row["SCORE"]) or not 0 <= row["SCORE"] <= 1
           or row["ALARM"] != (row["SCORE"] >= threshold) for row in rows):
        raise ValueError("Saved final score or alarm differs from locked policy")


def _metrics(table: pa.Table, event: dict, lock: dict) -> dict:
    rows = table.to_pylist()
    y = np.asarray([row["Y"] for row in rows], dtype=np.int8)
    score = np.asarray([row["SCORE"] for row in rows], dtype=np.float64)
    alarm = np.asarray([row["ALARM"] for row in rows], dtype=bool)
    tp = int(np.count_nonzero(alarm & (y == 1)))
    fp = int(np.count_nonzero(alarm & (y == 0)))
    fn = int(np.count_nonzero(~alarm & (y == 1)))
    tn = int(np.count_nonzero(~alarm & (y == 0)))
    n = len(rows)
    if tp + fp + tn + fn != n or len(np.unique(y)) != 2:
        raise ValueError("Final confusion or label classes differ")
    m = float(lock["full_development_representative_cost_nominal_thousands_usd"])
    a, b = float(lock["reference"]["alpha"]), float(lock["reference"]["beta"])
    detected = {event[(row["CERT"], row["Q"])] for row in rows if row["Y"] and row["ALARM"]}
    total_events = set(event.values())
    return {"rows": n, "positive_rows": int(np.count_nonzero(y)), "negative_rows": int(np.count_nonzero(y == 0)),
            "prevalence": float(np.mean(y)), "distinct_represented_events": len(total_events),
            "detected_events": len(detected), "missed_events": len(total_events - detected),
            "average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)), "brier": float(brier_score_loss(y, score)),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn, "alarms": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp), "false_positive_rate": fp / (tn + fp),
            "alarm_rate": (tp + fp) / n,
            "cost_scale_nominal_thousands_usd": m, "alpha_assumed": a, "beta_assumed": b,
            "false_alarm_surrogate_nominal_thousands_usd": m * b * fp,
            "missed_positive_surrogate_nominal_thousands_usd": m * a * fn,
            "reference_surrogate_per_row_nominal_thousands_usd": m * (b * fp + a * fn) / n,
            "always_alarm_surrogate_per_row_nominal_thousands_usd": m * b * int(np.count_nonzero(y == 0)) / n,
            "never_alarm_surrogate_per_row_nominal_thousands_usd": m * a * int(np.count_nonzero(y)) / n}


def _report(summary: dict) -> str:
    r = summary["result"]
    return "\n".join([
        "# Locked final predictive test evaluation", "",
        f"Run: `{summary['run_id']}`. Model and threshold were frozen before the first test predictor read.",
        "The prior global test-outcome inspection was for split feasibility only.", "",
        "| Scope | Value |", "|---|---:|",
        f"| Test bank-quarters | {r['rows']:,} |",
        f"| Positive bank-quarters | {r['positive_rows']:,} |",
        f"| Distinct represented failures | {r['distinct_represented_events']:,} |",
        f"| Detected distinct failures | {r['detected_events']:,} |",
        f"| Average precision (sklearn stepwise) | {r['average_precision']:.9f} |",
        f"| ROC-AUC | {r['roc_auc']:.9f} |",
        f"| Brier score | {r['brier']:.9f} |",
        f"| TP / FP / TN / FN | {r['TP']} / {r['FP']} / {r['TN']} / {r['FN']} |",
        f"| Alarm rate | {r['alarm_rate']:.6%} |",
        f"| Precision / recall | {r['precision']:.6f} / {r['recall']:.6f} |",
        f"| Reference surrogate per bank-quarter (nominal USD thousands) | {r['reference_surrogate_per_row_nominal_thousands_usd']:.9f} |",
        f"| Always / never alarm surrogates per bank-quarter | {r['always_alarm_surrogate_per_row_nominal_thousands_usd']:.9f} / {r['never_alarm_surrogate_per_row_nominal_thousands_usd']:.9f} |",
        "", "![Fixed-policy confusion counts](confusion.svg)", "",
        "The monetary figures are assumed row-level loss surrogates using the locked full-development FDIC COST scale; they are not literal event losses. Scores are uncalibrated. Repeated bank-quarters share banks and failures, so no independent-row confidence interval is asserted. The 46-quarter test horizon may include distribution shift. Development OOF served both model and threshold selection; these test results were not used to revise either choice.", "",
    ])


def _confusion_svg(result: dict) -> str:
    cells = [("TP", result["TP"], "#2e7d32"), ("FP", result["FP"], "#ef6c00"),
             ("TN", result["TN"], "#1565c0"), ("FN", result["FN"], "#c62828")]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="230" viewBox="0 0 640 230">',
             '<rect width="640" height="230" fill="white"/>',
             '<text x="28" y="32" font-size="20">Fixed-threshold test confusion counts</text>']
    for i, (label, count, color) in enumerate(cells):
        x = 28 + i * 152
        parts += [f'<rect x="{x}" y="70" width="136" height="92" fill="{color}"/>',
                  f'<text x="{x + 12}" y="105" fill="white" font-size="18">{label}</text>',
                  f'<text x="{x + 12}" y="140" fill="white" font-size="20">{count:,}</text>']
    parts.append('</svg>')
    return "".join(parts)


def evaluate(root: Path, run_id: str) -> dict:
    root = _root(root)
    _run_id(run_id)
    _, _, registry_path, lock_path = _paths(root, run_id)
    with _ownership(lock_path):
        spec, item, evidence, data, _, design = _registered(root, run_id)
        score_path = data / "test_scores.parquet"
        binding = _read_json(evidence / "model_binding.json")
        if (binding["plan_sha256"] != _sha(evidence / "frozen_plan.json")
                or binding["model_sha256"] != _sha(data / "fitted_pipeline.joblib")
                or item.get("model_sha256") != binding["model_sha256"]):
            raise ValueError("Fitted model is not bound to frozen plan and registry")
        if item["state"] == "complete":
            return verify_saved(root, run_id)
        if item["state"] not in ("model_fitted", "test_access", "scores_persisted"):
            raise RuntimeError(f"Cannot evaluate from registry state {item['state']}")
        manifest = RunManifest.read(evidence / "run_manifest.json")
        if item["state"] == "model_fitted":
            _ensure_source_hashes(spec)
            _transition(registry_path, design, run_id, "test_access")
            manifest.stages["test_access"] = "running"
            manifest.write(evidence / "run_manifest.json")
            start = time.perf_counter()
            try:
                expected, event = _expected_test(root)
                frame = _test_frame(root, expected, spec)
                pipeline = joblib.load(data / "fitted_pipeline.joblib")
                with threadpool_limits(limits=spec["thread_limit"]):
                    table = _scores(frame, pipeline, float(spec["threshold_policy"]["threshold"]))
                _validate_scores(table, expected, float(spec["threshold_policy"]["threshold"]))
                digest = _write_scores(score_path, table, evidence / "score_binding.json",
                                       binding["model_sha256"], binding["plan_sha256"], start)
                seconds = time.perf_counter() - start
                _write_json_atomic(evidence / "score_binding.json", {"score_sha256": digest,
                    "model_sha256": binding["model_sha256"], "plan_sha256": binding["plan_sha256"],
                    "score_seconds_including_read_and_write": seconds})
                _transition(registry_path, design, run_id, "scores_persisted", score_sha256=digest)
                manifest.record_output(path=str(score_path), dataset="final_test_scores", rows=table.num_rows,
                    sha256=digest, compression="zstd", schema_version=VERSION, partition=None,
                    source_total=table.num_rows, source_query=None)
                manifest.stages["test_access"] = "complete"
                manifest.write(evidence / "run_manifest.json")
            except Exception:
                manifest.stages["test_access"] = "failed"
                manifest.set_status("failed")
                manifest.write(evidence / "run_manifest.json")
                raise
        elif not score_path.exists():
            raise RuntimeError("Test access was recorded but no durable scores exist; automatic rescore is forbidden")
        # Reporting and recovery below use only saved scores, never test predictors.
        score_binding = _read_json(evidence / "score_binding.json")
        if (score_binding["score_sha256"] != _sha(score_path)
                or score_binding["model_sha256"] != binding["model_sha256"]
                or score_binding["plan_sha256"] != binding["plan_sha256"]):
            raise ValueError("Persisted final scores differ from score binding")
        expected, event = _expected_test(root)
        table = pq.read_table(score_path)
        _validate_scores(table, expected, float(spec["threshold_policy"]["threshold"]))
        result = _metrics(table, event, _read_json(root / "runs/task_5_5_threshold" / THRESHOLD_RUN / "locked_policy.json"))
        summary = {"status": "complete", "version": VERSION, "run_id": run_id,
                   "plan_sha256": _sha(evidence / "frozen_plan.json"), "model_sha256": binding["model_sha256"],
                   "score_sha256": score_binding["score_sha256"], "result": result,
                   "fit_seconds_including_read": binding["fit_seconds_including_read"],
                   "score_seconds_including_read_and_write": score_binding.get(
                       "score_seconds_including_read_and_write", score_binding.get("score_seconds_through_temp_readback")),
                   "test_use": "one locked predictive use; earlier test outcome counts inspected only for split feasibility"}
        summary_hash = _write_json_atomic(evidence / "summary.json", summary)
        report_hash = _write_text_atomic(evidence / "report.md", _report(summary))
        plot_hash = _write_text_atomic(evidence / "confusion.svg", _confusion_svg(result))
        manifest = RunManifest.read(evidence / "run_manifest.json")
        manifest.record_validation("one_use_registry_and_bound_model", True)
        manifest.record_validation("saved_test_score_readback_key_label_alarm", True)
        manifest.record_validation("saved_score_metrics_and_cost_reconciled", True)
        manifest.stages["reporting"] = "complete"
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        _write_json_atomic(evidence / "metadata.json", {"status": "complete", "summary_sha256": summary_hash,
            "report_sha256": report_hash, "plot_sha256": plot_hash})
        _transition(registry_path, design, run_id, "complete", summary_sha256=summary_hash)
        return verify_saved(root, run_id)


def verify_saved(root: Path, run_id: str) -> dict:
    root = _root(root)
    _run_id(run_id)
    spec, item, evidence, data, _, _ = _registered(root, run_id)
    if item["state"] != "complete":
        raise ValueError(f"Final test registry state is {item['state']}")
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    binding = _read_json(evidence / "model_binding.json")
    score_binding = _read_json(evidence / "score_binding.json")
    plan_sha256 = _sha(evidence / "frozen_plan.json")
    model_path = data / "fitted_pipeline.joblib"
    model_sha256 = _sha(model_path)
    scores = data / "test_scores.parquet"
    score_sha256 = _sha(scores)
    outputs = {output["dataset"]: output for output in manifest.outputs}
    inputs = {source["path"]: source["sha256"] for source in manifest.inputs}
    if (manifest.status != "complete" or not manifest.validations
            or not all(row["passed"] for row in manifest.validations)
            or summary.get("status") != "complete" or summary.get("version") != VERSION
            or summary.get("run_id") != run_id or metadata.get("status") != "complete"
            or metadata["summary_sha256"] != _sha(evidence / "summary.json")
            or metadata["report_sha256"] != _sha(evidence / "report.md")
            or metadata["plot_sha256"] != _sha(evidence / "confusion.svg")
            or item["summary_sha256"] != metadata["summary_sha256"]
            or summary.get("plan_sha256") != plan_sha256
            or binding.get("plan_sha256") != plan_sha256
            or score_binding.get("plan_sha256") != plan_sha256
            or item.get("plan_sha256") != plan_sha256
            or manifest.config.get("frozen_plan_sha256") != plan_sha256
            or summary.get("model_sha256") != binding.get("model_sha256")
            or binding.get("model_sha256") != model_sha256
            or score_binding.get("model_sha256") != model_sha256
            or item.get("model_sha256") != model_sha256
            or item.get("binding_sha256") != _sha(evidence / "model_binding.json")
            or summary.get("score_sha256") != score_sha256
            or score_binding.get("score_sha256") != score_sha256
            or item.get("score_sha256") != score_sha256
            or len(manifest.outputs) != 2
            or set(outputs) != {"final_model", "final_test_scores"}
            or outputs.get("final_model", {}).get("path") != str(model_path)
            or outputs.get("final_model", {}).get("sha256") != model_sha256
            or outputs.get("final_model", {}).get("rows") != binding.get("fit_rows")
            or outputs.get("final_test_scores", {}).get("path") != str(scores)
            or outputs.get("final_test_scores", {}).get("sha256") != score_sha256
            or outputs.get("final_test_scores", {}).get("rows") != summary.get("result", {}).get("rows")
            or len(inputs) != len(manifest.inputs)
            or inputs != spec["source_hashes"]):
        raise ValueError("Final saved manifest, summary, registry or artifact hash differs")
    if sum(row["state"] == "test_access" for row in item["history"]) != 1:
        raise ValueError("Final design has more than one recorded test-access transition")
    _ensure_source_hashes(spec)
    expected, event = _expected_test(root)
    with pq.ParquetFile(scores) as file:
        if any(file.metadata.row_group(g).column(c).compression != "ZSTD"
               for g in range(file.metadata.num_row_groups)
               for c in range(file.metadata.row_group(g).num_columns)):
            raise ValueError("Final score codec differs")
        table = file.read()
    _validate_scores(table, expected, float(spec["threshold_policy"]["threshold"]))
    locked = _read_json(root / "runs/task_5_5_threshold" / THRESHOLD_RUN / "locked_policy.json")
    computed = _metrics(table, event, locked)
    for key, value in computed.items():
        previous = summary["result"][key]
        if not (math.isclose(value, previous, rel_tol=0, abs_tol=1e-12)
                if isinstance(value, float) else value == previous):
            raise ValueError(f"Saved final metric differs: {key}")
    return {"status": "complete", "run_id": run_id, "result": computed,
            "model_sha256": summary["model_sha256"], "score_sha256": summary["score_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="One-use locked final BFT65B test evaluation")
    parser.add_argument("mode", choices=("plan", "fit", "evaluate", "verify", "show"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        result = plan(args.project_root, args.run_id)
    elif args.mode == "fit":
        result = fit(args.project_root, args.run_id)
    elif args.mode == "evaluate":
        result = evaluate(args.project_root, args.run_id)
    else:
        result = verify_saved(args.project_root, args.run_id)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
