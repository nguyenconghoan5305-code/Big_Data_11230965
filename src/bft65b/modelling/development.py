"""Development-only forward validation for the frozen BFT65B split.

The plan command freezes folds and candidates before any predictor values are
loaded. The execute command opens only development model-table partitions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from bft65b.config import load_config
from bft65b.features.preprocessing import PREDICTOR_COLUMNS, make_preprocessing_pipeline
from bft65b.features.leakage_audit import verify_saved as verify_leakage
from bft65b.features.production import verify_saved as verify_labels
from bft65b.features.table import verify_saved as verify_table
from bft65b.manifest import RunManifest
from bft65b.modelling.split import (
    NAMESPACE as SPLIT_NAMESPACE, MODEL_RUN, VERSION as SPLIT_VERSION,
    _write_json_atomic, quarter_index, verify_saved as verify_split,
)
from bft65b.panel.build import _write_verified
from bft65b.panel.retrospective import _root, _run_id, _quarter
from bft65b.quality.audit import _read_json, _sha


VERSION = "development_forward_validation_v1"
NAMESPACE = "task_5_2_development"
SPLIT_RUN = "split20261120b"
FOLD_COUNT = 3
VALIDATION_QUARTERS = 16
INTERVENING_QUARTERS = 4
SEED = 42
THREADS = 2
AP_TIE_TOLERANCE = 1e-12
CANDIDATES = (
    {"id": "logistic_baseline", "estimator": "LogisticRegression",
     "params": {"C": 1.0, "solver": "liblinear", "max_iter": 250,
                "class_weight": None, "random_state": SEED}},
    {"id": "bounded_hist_gradient_boosting", "estimator": "HistGradientBoostingClassifier",
     "params": {"learning_rate": 0.05, "max_iter": 40, "max_leaf_nodes": 15,
                "max_depth": 3, "l2_regularization": 1.0,
                "early_stopping": False, "random_state": SEED}},
)
OOF_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("Q", pa.date32(), nullable=False),
    pa.field("Y", pa.int8(), nullable=False),
    pa.field("FOLD_ID", pa.string(), nullable=False),
    pa.field("CANDIDATE_ID", pa.string(), nullable=False),
    pa.field("SCORE", pa.float64(), nullable=False),
], metadata={b"bft65b.development": VERSION.encode()})


def _day(value: str) -> date:
    return date.fromisoformat(value) if "-" in value else _quarter(value)


def _period_indices(fold: dict) -> tuple[int, int, int]:
    train_end = quarter_index(_day(fold["train_end"]))
    val_start = quarter_index(_day(fold["validation_start"]))
    val_end = quarter_index(_day(fold["validation_end"]))
    if val_start - train_end - 1 != INTERVENING_QUARTERS or val_end < val_start:
        raise ValueError("Fold lacks four complete intervening prediction quarters")
    if _day(fold["training_target_right_end"]) >= _day(fold["validation_start"]):
        raise ValueError("Training target window reaches the validation start")
    return train_end, val_start, val_end


def build_folds(quarter_rows: list[dict], positive_links: list[dict]) -> list[dict]:
    """Use the last 48 development quarters as three contiguous 16-quarter blocks."""
    development = [row for row in quarter_rows if row["period"] == "development"]
    dates = [_day(row["quarter"]) for row in development]
    if (not development or len(dates) < FOLD_COUNT * VALIDATION_QUARTERS +
            INTERVENING_QUARTERS + 1 or dates != sorted(set(dates))
            or any(quarter_index(b) - quarter_index(a) != 1
                   for a, b in zip(dates, dates[1:]))):
        raise ValueError("Development quarter inventory cannot support frozen folds")
    base = len(dates) - FOLD_COUNT * VALIDATION_QUARTERS
    links = [row for row in positive_links if row["PERIOD"] == "development"]
    folds = []
    prior_val_end = None
    for i in range(FOLD_COUNT):
        first = base + i * VALIDATION_QUARTERS
        last = first + VALIDATION_QUARTERS - 1
        train_last = first - INTERVENING_QUARTERS - 1
        train_end, val_start, val_end = dates[train_last], dates[first], dates[last]
        gap_start, gap_end = dates[train_last + 1], dates[first - 1]
        if prior_val_end is not None and quarter_index(val_start) != quarter_index(prior_val_end) + 1:
            raise ValueError("Validation blocks must be contiguous and non-overlapping")
        prior_val_end = val_end
        train_rows = development[:train_last + 1]
        val_rows = development[first:last + 1]
        train_positive = sum(row["positive_rows"] for row in train_rows)
        train_negative = sum(row["negative_rows"] for row in train_rows)
        val_positive = sum(row["positive_rows"] for row in val_rows)
        val_negative = sum(row["negative_rows"] for row in val_rows)
        train_ids = {row["SOURCE_ID"] for row in links if row["Q"] <= train_end}
        val_links = [row for row in links if val_start <= row["Q"] <= val_end]
        val_ids = {row["SOURCE_ID"] for row in val_links}
        if (not train_positive or not train_negative or not val_positive or not val_negative
                or not train_ids or not val_ids or train_ids & val_ids):
            raise ValueError(f"Fold {i + 1} has a single class or shared failure event")
        fold = {"id": f"fold_{i + 1}", "train_start": str(dates[0]),
                "train_end": str(train_end), "gap_start": str(gap_start),
                "gap_end": str(gap_end), "validation_start": str(val_start),
                "validation_end": str(val_end),
                "training_target_right_end": str(date(train_end.year + 1, train_end.month, train_end.day)),
                "training_rows": train_positive + train_negative,
                "training_positive_rows": train_positive,
                "training_negative_rows": train_negative,
                "training_distinct_events": len(train_ids),
                "validation_rows": val_positive + val_negative,
                "validation_positive_rows": val_positive,
                "validation_negative_rows": val_negative,
                "validation_distinct_positive_banks": len({row["CERT"] for row in val_links}),
                "validation_distinct_events": len(val_ids),
                "validation_prevalence": val_positive / (val_positive + val_negative)}
        _period_indices(fold)
        folds.append(fold)
    return folds


def _paths(root: Path, run_id: str) -> tuple[Path, Path]:
    _run_id(run_id)
    return (root / "runs" / NAMESPACE / run_id,
            root / "data_processed" / NAMESPACE / run_id)


def _source_metadata(root: Path, split_run: str = SPLIT_RUN) -> tuple[dict, dict, list[dict], dict[str, str]]:
    if (root / "runs/source_vintage_bundle.json").is_file() and split_run == SPLIT_RUN:
        raise ValueError("Portable development requires an explicit split run")
    verify_split(root, split_run, rehash_inputs=True)
    split_root = root / "runs" / SPLIT_NAMESPACE / split_run
    split = _read_json(split_root / "summary.json")
    selected = split.get("source_runs", {})
    model_run = selected.get("model_table", MODEL_RUN)
    label_run = selected.get("labels", "labels20261120a")
    leakage_run = selected.get("leakage_audit", "p44audit20261120a")
    verify_table(root, model_run, check_upstream=True)
    verify_labels(root, label_run)
    verify_leakage(root, leakage_run, rehash_inputs=True)
    if split["version"] != SPLIT_VERSION or split["bounds"]["development_end"] != "2012-12-31":
        raise ValueError("Accepted split version or development end differs")
    model_root = root / "runs/task_4_3_feature_table" / model_run
    model = _read_json(model_root / "summary.json")
    model_files = [row for row in model["outputs"] if row["dataset"] == "model_table"
                   and row["quarter"] <= "20121231"]
    if len(model_files) != 116:
        raise ValueError("Expected 116 development model partitions")
    link_path = next(Path(item["path"]) for item in split["outputs"]
                     if item["dataset"] == "positive_linkage")
    with pq.ParquetFile(link_path) as file:
        links = file.read(columns=["CERT", "Q", "SOURCE_ID", "PERIOD"]).to_pylist()
    files = {str(Path(row["path"])): row["sha256"] for row in model_files}
    for path in (split_root / "summary.json", split_root / "run_manifest.json",
                 split_root / "metadata.json", link_path,
                 root / "runs/task_4_4_leakage_audit" / leakage_run / "summary.json",
                 model_root / "summary.json"):
        files[str(path)] = _sha(path)
    if (root / "runs/source_vintage_bundle.json").is_file() and any(
            not Path(path).resolve().is_relative_to(root) for path in files):
        raise ValueError("Portable development input escaped selected root")
    return split, model, links, files


def plan(root: Path, run_id: str, *, split_run: str = SPLIT_RUN) -> dict:
    """Freeze the fold/candidate contract without reading any predictor value."""
    root = _root(root)
    evidence, data = _paths(root, run_id)
    if evidence.exists() or data.exists():
        raise FileExistsError(f"Development run already exists: {run_id}")
    split, model, links, inputs = _source_metadata(root, split_run)
    model_run = split["source_runs"]["model_table"]
    folds = build_folds(split["by_quarter"], links)
    if (sum(row["model_rows"] for row in split["by_quarter"]
            if row["period"] == "development") != model["overall"]["model_positive"] -
            sum(row["positive_rows"] for row in split["by_quarter"]
                if row["period"] != "development") +
            model["overall"]["model_negative"] -
            sum(row["negative_rows"] for row in split["by_quarter"]
                if row["period"] != "development")):
        raise ValueError("Development row reconciliation differs")
    first_val = _day(folds[0]["validation_start"])
    unscored = [row for row in split["by_quarter"] if row["period"] == "development"
                and _day(row["quarter"]) < first_val]
    spec = {"status": "frozen", "version": VERSION, "run_id": run_id,
            "split_run": split_run, "model_run": model_run,
            "source_runs": split["source_runs"],
            "predictor_columns": list(PREDICTOR_COLUMNS), "seed": SEED,
            "thread_limit": THREADS, "folds": folds, "candidates": list(CANDIDATES),
            "selection": {"primary": "mean_defined_fold_average_precision",
                "tie_tolerance": AP_TIE_TOLERANCE,
                "tie_break": "higher minimum fold AP, then lexicographically smaller candidate ID",
                "undefined_fold_policy": "record null metric and fail candidate-selection gate; never drop a fold"},
            "metrics": {"average_precision": "sklearn.metrics.average_precision_score; non-interpolated stepwise AP, not trapezoidal PR-AUC",
                "roc_auc": "sklearn.metrics.roc_auc_score when both classes exist",
                "brier": "sklearn.metrics.brier_score_loss on uncalibrated candidate scores"},
            "unscored_initial_development": {"first_q": split["bounds"]["development_start"],
                "last_q": str(_day(unscored[-1]["quarter"])),
                "quarters": len(unscored),
                "rows": sum(row["model_rows"] for row in unscored)},
            "development_model_paths": [row["path"] for row in model["outputs"]
                if row["dataset"] == "model_table" and row["quarter"] <= "20121231"],
            "input_hashes": inputs,
            "test_predictor_files_opened": 0,
            "created_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest = RunManifest.new(run_id, load_config(root / "config/financials_full_v2.json"))
    manifest.config["development_policy_version"] = VERSION
    manifest.config["split_run"] = split_run
    manifest.config["source_runs"] = split["source_runs"]
    for path, digest in sorted(inputs.items()):
        manifest.record_input(path, digest)
    digest = _write_json_atomic(evidence / "frozen_plan.json", spec)
    manifest.config["frozen_plan_sha256"] = digest
    manifest.write(evidence / "run_manifest.json")
    return {"status": "frozen", "run_id": run_id, "plan_sha256": digest,
            "folds": folds, "candidates": [item["id"] for item in CANDIDATES],
            "unscored_initial_development": spec["unscored_initial_development"]}


def _validated_plan(root: Path, run_id: str) -> tuple[dict, RunManifest, Path, Path]:
    evidence, data = _paths(root, run_id)
    manifest = RunManifest.read(evidence / "run_manifest.json")
    spec_path = evidence / "frozen_plan.json"
    spec = _read_json(spec_path)
    if (manifest.run_id != run_id or manifest.status != "pending"
            or manifest.config.get("frozen_plan_sha256") != _sha(spec_path)
            or spec.get("status") != "frozen" or spec.get("version") != VERSION
            or spec.get("run_id") != run_id
            or spec.get("predictor_columns") != list(PREDICTOR_COLUMNS)
            or spec.get("candidates") != list(CANDIDATES)
            or spec.get("test_predictor_files_opened") != 0
            or manifest.config.get("split_run", spec.get("split_run")) != spec.get("split_run")
            or manifest.config.get("source_runs", spec.get("source_runs")) != spec.get("source_runs")
            or len(spec.get("folds", [])) != FOLD_COUNT
            or {item["path"]: item["sha256"] for item in manifest.inputs}
               != spec["input_hashes"]):
        raise ValueError("Frozen development plan or manifest differs")
    for fold in spec["folds"]:
        _period_indices(fold)
    if any(_sha(Path(path)) != digest for path, digest in spec["input_hashes"].items()):
        raise ValueError("Frozen development source hash differs")
    if (root / "runs/source_vintage_bundle.json").is_file():
        if spec.get("split_run") == SPLIT_RUN or any(
                not Path(path).resolve().is_relative_to(root)
                for path in spec["input_hashes"]):
            raise ValueError("Portable development plan references historical inputs")
    if data.exists():
        raise FileExistsError("Development output namespace already exists")
    return spec, manifest, evidence, data


def _development_frame(root: Path, spec: dict) -> tuple[pd.DataFrame, set[tuple[int, date]]]:
    """Open only frozen development model paths; enforce split membership."""
    split = _read_json(root / "runs" / SPLIT_NAMESPACE / spec["split_run"] / "summary.json")
    assignment_path = next(item["path"] for item in split["outputs"]
                           if item["dataset"] == "assignments")
    assignments = pq.ParquetFile(assignment_path).read(
        columns=["CERT", "Q", "PERIOD"]).to_pylist()
    dev_by_q: dict[date, set[int]] = defaultdict(set)
    for row in assignments:
        if row["PERIOD"] == "development":
            dev_by_q[row["Q"]].add(row["CERT"])
    paths = spec["development_model_paths"]
    if len(paths) != 116 or any("model_table" not in path for path in paths):
        raise ValueError("Frozen development path inventory differs")
    tables = []
    for path_text in paths:
        path = Path(path_text)
        if _sha(path) != spec["input_hashes"][path_text]:
            raise ValueError(f"Development partition checksum differs: {path}")
        qtext = path.parent.name.removeprefix("Q=")
        q = _quarter(qtext)
        if q > date(2012, 12, 31) or path.name != "part-000.parquet":
            raise ValueError(f"Nondevelopment model path in frozen plan: {path}")
        with pq.ParquetFile(path) as file:
            rows = file.read(columns=["CERT", "Q", "Y", *PREDICTOR_COLUMNS]).to_pandas()
        if (len(rows) != len(dev_by_q[q]) or len(set(rows["CERT"])) != len(rows)
                or set(int(value) for value in rows["CERT"]) != dev_by_q[q]
                or any(pd.Timestamp(value).date() != q for value in rows["Q"])):
            raise ValueError(f"Development predictor keys differ from accepted split: {q}")
        rows["Q"] = q
        tables.append(rows)
    frame = pd.concat(tables, ignore_index=True)
    frame = frame.sort_values(["Q", "CERT"], kind="stable").reset_index(drop=True)
    allowed = {(int(row.CERT), row.Q) for row in frame[["CERT", "Q"]].itertuples(index=False)}
    if len(allowed) != len(frame) or sum(map(len, dev_by_q.values())) != len(frame):
        raise ValueError("Development predictor rows are not unique or complete")
    return frame, allowed


def guard_development_rows(frame: pd.DataFrame, allowed: set[tuple[int, date]],
                           *, start: date, end: date) -> None:
    """Reject gap/test keys and keys outside the fold's chronological role."""
    if not {"CERT", "Q", "Y", *PREDICTOR_COLUMNS}.issubset(frame.columns):
        raise ValueError("Fold input lacks required model columns")
    keys = list(zip((int(x) for x in frame["CERT"]), frame["Q"]))
    if (not len(frame) or len(set(keys)) != len(keys)
            or any(q < start or q > end or (cert, q) not in allowed for cert, q in keys)
            or set(frame["Y"].unique()) - {0, 1}):
        raise ValueError("Fold input contains duplicate, gap/test, out-of-window or invalid-label keys")


def candidate_pipeline(candidate: dict) -> Pipeline:
    if candidate not in CANDIDATES:
        raise ValueError("Candidate is outside the frozen search list")
    if candidate["estimator"] == "LogisticRegression":
        estimator = LogisticRegression(**candidate["params"])
    elif candidate["estimator"] == "HistGradientBoostingClassifier":
        estimator = HistGradientBoostingClassifier(**candidate["params"])
    else:
        raise ValueError("Unknown frozen estimator")
    return Pipeline([("preprocessing", make_preprocessing_pipeline()),
                     ("estimator", estimator)])


def fold_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if (len(y) != len(score) or not len(y) or set(np.unique(y)) - {0, 1}
            or not np.all(np.isfinite(score)) or np.any(score < 0) or np.any(score > 1)):
        raise ValueError("Validation scores or labels are invalid")
    positives = int(y.sum())
    negatives = len(y) - positives
    defined = bool(positives and negatives)
    return {"rows": len(y), "positive_rows": positives, "negative_rows": negatives,
            "prevalence": positives / len(y),
            "average_precision": float(average_precision_score(y, score)) if defined else None,
            "roc_auc": float(roc_auc_score(y, score)) if defined else None,
            "brier": float(brier_score_loss(y, score)),
            "undefined_reason": None if defined else "single_class_validation"}


def validate_oof_rows(rows: list[dict], *, expected_rows: int,
                      candidate_id: str) -> None:
    keys = [(row["CERT"], row["Q"]) for row in rows]
    if (len(rows) != expected_rows or len(keys) != len(set(keys))
            or any(row["CANDIDATE_ID"] != candidate_id
                   or row["Y"] not in (0, 1)
                   or not math.isfinite(row["SCORE"])
                   or not 0 <= row["SCORE"] <= 1 for row in rows)):
        raise ValueError("Candidate OOF keys, labels or scores are invalid")


def fit_predict_fold(train: pd.DataFrame, validation: pd.DataFrame,
                     allowed: set[tuple[int, date]], fold: dict,
                     candidate: dict) -> tuple[Pipeline, np.ndarray, dict]:
    """Fresh complete pipeline; only training rows are passed to fit."""
    _period_indices(fold)
    train_start, train_end = _day(fold["train_start"]), _day(fold["train_end"])
    val_start, val_end = _day(fold["validation_start"]), _day(fold["validation_end"])
    guard_development_rows(train, allowed, start=train_start, end=train_end)
    guard_development_rows(validation, allowed, start=val_start, end=val_end)
    if set(train["Y"].unique()) != {0, 1}:
        raise ValueError("Training fold must contain both classes")
    pipeline = candidate_pipeline(candidate)
    x_train = train.loc[:, list(PREDICTOR_COLUMNS)]
    x_validation = validation.loc[:, list(PREDICTOR_COLUMNS)]
    y_train = train["Y"].to_numpy(dtype=np.int8)
    t0 = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        score = pipeline.predict_proba(x_validation)[:, 1]
    prediction_seconds = time.perf_counter() - t0
    metrics = fold_metrics(validation["Y"].to_numpy(dtype=np.int8), score)
    metrics.update({"fold_id": fold["id"], "candidate_id": candidate["id"],
                    "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds})
    return pipeline, score, metrics


def select_candidate(fold_results: list[dict]) -> tuple[str, list[dict]]:
    """Macro mean of defined fold AP; minimum-fold AP and ID break ties."""
    rankings = []
    for candidate in CANDIDATES:
        rows = [row for row in fold_results if row["candidate_id"] == candidate["id"]]
        if len(rows) != FOLD_COUNT or len({row["fold_id"] for row in rows}) != FOLD_COUNT:
            raise ValueError("Candidate lacks exactly one result per frozen fold")
        values = [row["average_precision"] for row in rows]
        if any(value is None for value in values):
            raise ValueError("Undefined fold average precision blocks selection")
        rankings.append({"candidate_id": candidate["id"],
                         "mean_fold_average_precision": float(np.mean(values)),
                         "minimum_fold_average_precision": min(values)})
    best = rankings[0]
    for row in rankings[1:]:
        difference = row["mean_fold_average_precision"] - best["mean_fold_average_precision"]
        if (difference > AP_TIE_TOLERANCE or
                (abs(difference) <= AP_TIE_TOLERANCE and
                 (row["minimum_fold_average_precision"] > best["minimum_fold_average_precision"] + AP_TIE_TOLERANCE or
                  (abs(row["minimum_fold_average_precision"] - best["minimum_fold_average_precision"]) <= AP_TIE_TOLERANCE and
                   row["candidate_id"] < best["candidate_id"])))):
            best = row
    return best["candidate_id"], rankings


def execute(root: Path, run_id: str) -> dict:
    """Fit frozen candidates only on expanding development training folds."""
    root = _root(root)
    spec, manifest, evidence, data = _validated_plan(root, run_id)
    split = _read_json(root / "runs" / SPLIT_NAMESPACE / spec["split_run"] / "summary.json")
    link_path = next(item["path"] for item in split["outputs"]
                     if item["dataset"] == "positive_linkage")
    links = pq.ParquetFile(link_path).read(columns=["CERT", "Q", "SOURCE_ID", "PERIOD"]).to_pylist()
    if build_folds(split["by_quarter"], links) != spec["folds"]:
        raise ValueError("Frozen fold counts or dates no longer match accepted split")
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    started = time.perf_counter()
    try:
        frame, allowed = _development_frame(root, spec)
        if len(frame) != split["periods"]["development"]["model_rows"]:
            raise ValueError("Development-only predictor row count differs")
        results: list[dict] = []
        pooled = []
        outputs = []
        covered_quarters = set()
        for candidate in CANDIDATES:
            predictions = []
            for fold in spec["folds"]:
                train_end = _day(fold["train_end"])
                val_start, val_end = _day(fold["validation_start"]), _day(fold["validation_end"])
                train = frame.loc[frame["Q"] <= train_end]
                validation = frame.loc[(frame["Q"] >= val_start) & (frame["Q"] <= val_end)]
                if (len(train) != fold["training_rows"] or len(validation) != fold["validation_rows"]
                        or int(train["Y"].sum()) != fold["training_positive_rows"]
                        or int(validation["Y"].sum()) != fold["validation_positive_rows"]):
                    raise ValueError(f"Fold counts differ from frozen plan: {fold['id']}")
                pipeline, scores, metrics = fit_predict_fold(
                    train, validation, allowed, fold, candidate)
                if not np.array_equal(pipeline.named_steps["preprocessing"].get_feature_names_out(),
                                      np.asarray(PREDICTOR_COLUMNS, dtype=object)):
                    raise ValueError("Fitted preprocessing feature order changed")
                metrics["distinct_positive_banks"] = fold["validation_distinct_positive_banks"]
                metrics["distinct_qualifying_events"] = fold["validation_distinct_events"]
                results.append(metrics)
                covered_quarters.update(q for q in validation["Q"])
                for cert, q, y, score in zip(validation["CERT"], validation["Q"],
                                             validation["Y"], scores):
                    predictions.append({"CERT": int(cert), "Q": q, "Y": int(y),
                                        "FOLD_ID": fold["id"],
                                        "CANDIDATE_ID": candidate["id"],
                                        "SCORE": float(score)})
            validate_oof_rows(predictions,
                expected_rows=sum(fold["validation_rows"] for fold in spec["folds"]),
                candidate_id=candidate["id"])
            table = pa.Table.from_pylist(predictions, schema=OOF_SCHEMA)
            path = data / f"{candidate['id']}_oof.parquet"
            digest = _write_verified(path, table)
            item = {"candidate_id": candidate["id"], "path": str(path),
                    "rows": table.num_rows, "sha256": digest}
            outputs.append(item)
            manifest.record_output(path=str(path), dataset="development_oof",
                rows=table.num_rows, sha256=digest, compression="ZSTD",
                schema_version=VERSION, partition=candidate["id"],
                source_total=table.num_rows, source_query=None)
            pooled_metrics = fold_metrics(
                np.asarray([row["Y"] for row in predictions], dtype=np.int8),
                np.asarray([row["SCORE"] for row in predictions], dtype=np.float64))
            pooled_metrics.update({"candidate_id": candidate["id"],
                "distinct_positive_banks": len({row["CERT"] for row in predictions if row["Y"]}),
                "distinct_qualifying_events": len({row["SOURCE_ID"] for row in links
                    if row["PERIOD"] == "development" and row["Q"] in covered_quarters}),
                "aggregation": "pooled observation-level OOF; distinct events are a separate count"})
            pooled.append(pooled_metrics)
        selected, rankings = select_candidate(results)
        if any(_sha(Path(path)) != digest for path, digest in spec["input_hashes"].items()):
            raise ValueError("Accepted development source changed during fitting")
        summary = {"status": "complete", "version": VERSION, "run_id": run_id,
                   "split_run": spec["split_run"], "model_run": spec["model_run"],
                   "frozen_plan_sha256": manifest.config["frozen_plan_sha256"],
                   "folds": spec["folds"], "candidates": spec["candidates"],
                   "fold_results": results, "pooled_results": pooled,
                   "selection_ranking": rankings, "selected_candidate": selected,
                   "selected_oof_path": next(item["path"] for item in outputs
                                             if item["candidate_id"] == selected),
                   "outputs": outputs, "development_rows_loaded": len(frame),
                   "validation_quarters_covered": len(covered_quarters),
                   "unscored_initial_development": spec["unscored_initial_development"],
                   "test_predictor_files_opened": 0,
                   "test_predictions_computed": 0,
                   "model_refit_on_full_development": False,
                   "threshold_selected": False,
                   "environment": {"os": platform.platform(),
                       "python": platform.python_version(),
                       "logical_processors": os.cpu_count(),
                       "thread_limit": THREADS,
                       "sklearn": __import__("sklearn").__version__,
                       "pandas": pd.__version__, "pyarrow": pa.__version__},
                   "wall_seconds": time.perf_counter() - started,
                   "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_json_atomic(evidence / "summary.json", summary)
        _write_json_atomic(evidence / "metadata.json", {
            "status": "complete", "summary_sha256": _sha(evidence / "summary.json"),
            "frozen_plan_sha256": manifest.config["frozen_plan_sha256"],
            "output_hashes": {row["candidate_id"]: row["sha256"] for row in outputs}})
        manifest.record_validation("frozen_folds_and_candidates", True,
            "Three chronological expanding folds and two candidate configs fixed before fitting")
        manifest.record_validation("development_only_training_and_scoring", True,
            f"{len(frame)} development rows loaded; no gap/test predictor file opened")
        manifest.record_validation("out_of_fold_keys_scores_and_metrics", True,
            f"{len(covered_quarters)} validation quarters; unique finite bounded scores")
        manifest.record_validation("atomic_output_and_upstream_hashes", True,
            "Both OOF Parquet files read back; accepted source hashes unchanged")
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        verify_saved(root, run_id, rehash_inputs=True)
        return {"status": "complete", "run_id": run_id,
                "selected_candidate": selected, "fold_results": results,
                "pooled_results": pooled, "evidence": str(evidence)}
    except Exception as exc:
        manifest.record_validation("development_failure", False, str(exc))
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise


def verify_saved(root: Path, run_id: str, *, rehash_inputs: bool = True) -> dict:
    root = _root(root)
    evidence, data = _paths(root, run_id)
    manifest = RunManifest.read(evidence / "run_manifest.json")
    spec_path = evidence / "frozen_plan.json"
    spec = _read_json(spec_path)
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if (manifest.status != "complete" or not manifest.validations
            or any(not item["passed"] for item in manifest.validations)
            or manifest.config.get("frozen_plan_sha256") != _sha(spec_path)
            or summary.get("status") != "complete" or summary.get("version") != VERSION
            or summary.get("run_id") != run_id or summary.get("folds") != spec["folds"]
            or summary.get("candidates") != spec["candidates"]
            or summary.get("frozen_plan_sha256") != _sha(spec_path)
            or metadata.get("summary_sha256") != _sha(evidence / "summary.json")
            or metadata.get("status") != "complete"
            or summary.get("test_predictor_files_opened") != 0
            or summary.get("test_predictions_computed") != 0
            or summary.get("threshold_selected") is not False
            or len(summary.get("outputs", [])) != len(CANDIDATES)
            or len(manifest.outputs) != len(CANDIDATES)):
        raise ValueError("Development manifest, frozen plan or summary is incomplete")
    if select_candidate(summary["fold_results"])[0] != summary["selected_candidate"]:
        raise ValueError("Saved development selection differs from frozen rule")
    expected_rows = sum(fold["validation_rows"] for fold in spec["folds"])
    split = _read_json(root / "runs" / SPLIT_NAMESPACE / spec["split_run"] / "summary.json")
    assignment_path = next(item["path"] for item in split["outputs"]
                           if item["dataset"] == "assignments")
    with pq.ParquetFile(assignment_path) as file:
        assignments = file.read(columns=["CERT", "Q", "PERIOD"]).to_pylist()
    validation_ranges = [(_day(fold["validation_start"]), _day(fold["validation_end"]))
                         for fold in spec["folds"]]
    expected_validation_keys = {(row["CERT"], row["Q"]) for row in assignments
        if row["PERIOD"] == "development" and any(
            start <= row["Q"] <= end for start, end in validation_ranges)}
    if len(expected_validation_keys) != expected_rows:
        raise ValueError("Frozen validation keys do not reconcile to accepted split")
    expected_keys = None
    for item in summary["outputs"]:
        path = Path(item["path"])
        if (not path.resolve().is_relative_to(data.resolve())
                or _sha(path) != item["sha256"]
                or metadata["output_hashes"].get(item["candidate_id"]) != item["sha256"]
                or not any(out["path"] == str(path) and out["sha256"] == item["sha256"]
                           for out in manifest.outputs)):
            raise ValueError(f"OOF path or checksum differs: {path}")
        with pq.ParquetFile(path) as file:
            if (file.schema_arrow != OOF_SCHEMA or file.metadata.num_rows != expected_rows
                    or any(file.metadata.row_group(g).column(c).compression != "ZSTD"
                           for g in range(file.metadata.num_row_groups)
                           for c in range(file.metadata.row_group(g).num_columns))):
                raise ValueError(f"OOF schema, codec or rows differ: {path}")
            rows = file.read().to_pylist()
        keys = {(row["CERT"], row["Q"]) for row in rows}
        if (len(rows) != expected_rows or len(keys) != len(rows)
                or any(row["CANDIDATE_ID"] != item["candidate_id"]
                       or not math.isfinite(row["SCORE"])
                       or not 0 <= row["SCORE"] <= 1 for row in rows)):
            raise ValueError("OOF duplicate key, candidate or score differs")
        fold_lookup = {fold["id"]: fold for fold in spec["folds"]}
        if any(row["FOLD_ID"] not in fold_lookup
               or not _day(fold_lookup[row["FOLD_ID"]]["validation_start"]) <= row["Q"] <=
                      _day(fold_lookup[row["FOLD_ID"]]["validation_end"])
               for row in rows):
            raise ValueError("OOF row lies outside its frozen validation block")
        if keys != expected_validation_keys:
            raise ValueError("OOF keys do not cover the frozen validation subset")
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("Candidates have different OOF key coverage")
    if rehash_inputs and any(_sha(Path(path)) != digest
                             for path, digest in spec["input_hashes"].items()):
        raise ValueError("Accepted development input hash changed")
    if (root / "runs/source_vintage_bundle.json").is_file():
        if (spec["split_run"] == SPLIT_RUN
                or any(not Path(path).resolve().is_relative_to(root)
                       for path in spec["input_hashes"]) or
                summary.get("split_run") != spec["split_run"] or
                summary.get("model_run") != spec["model_run"]):
            raise ValueError("Portable development saved lineage differs")
    return {"status": "complete", "run_id": run_id,
            "selected_candidate": summary["selected_candidate"],
            "folds": len(spec["folds"]), "oof_rows_per_candidate": expected_rows}


def compare_runs(root: Path, run_id: str, other_run: str, *, tolerance: float = 1e-10) -> dict:
    root = _root(root)
    first = verify_saved(root, run_id)
    second = verify_saved(root, other_run)
    first_summary = _read_json(root / "runs" / NAMESPACE / run_id / "summary.json")
    second_summary = _read_json(root / "runs" / NAMESPACE / other_run / "summary.json")
    first_spec = _read_json(root / "runs" / NAMESPACE / run_id / "frozen_plan.json")
    second_spec = _read_json(root / "runs" / NAMESPACE / other_run / "frozen_plan.json")
    equal_policy = all(first_spec[key] == second_spec[key] for key in
                       ("folds", "candidates", "predictor_columns", "seed", "selection", "input_hashes"))
    if not equal_policy or first["selected_candidate"] != second["selected_candidate"]:
        raise ValueError("Independent development runs differ in policy or selection")
    maximum_difference = 0.0
    rows_compared = 0
    for a, b in zip(first_summary["outputs"], second_summary["outputs"]):
        left = pq.ParquetFile(a["path"]).read().to_pylist()
        right = pq.ParquetFile(b["path"]).read().to_pylist()
        if len(left) != len(right):
            raise ValueError("Independent OOF row counts differ")
        for lrow, rrow in zip(left, right):
            if any(lrow[key] != rrow[key] for key in
                   ("CERT", "Q", "Y", "FOLD_ID", "CANDIDATE_ID")):
                raise ValueError("Independent OOF keys or labels differ")
            maximum_difference = max(maximum_difference, abs(lrow["SCORE"] - rrow["SCORE"]))
        rows_compared += len(left)
    if maximum_difference > tolerance:
        raise ValueError(f"Independent OOF score difference exceeds {tolerance}: {maximum_difference}")
    result = {"status": "complete", "first_run": run_id, "repeat_run": other_run,
              "selected_candidate": first["selected_candidate"],
              "scores_compared": rows_compared, "max_abs_score_difference": maximum_difference,
              "absolute_tolerance": tolerance, "frozen_policy_equal": equal_policy}
    repeat_path = root / "runs" / NAMESPACE / other_run / "repeat_comparison.json"
    if repeat_path.exists():
        raise FileExistsError("Repeat comparison evidence already exists")
    _write_json_atomic(repeat_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only forward validation")
    parser.add_argument("mode", choices=("plan", "execute", "verify", "show", "compare"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compare-run-id")
    parser.add_argument("--split-run", default=SPLIT_RUN)
    args = parser.parse_args()
    if args.mode == "plan":
        result = plan(args.project_root, args.run_id, split_run=args.split_run)
    elif args.mode == "execute":
        result = execute(args.project_root, args.run_id)
    elif args.mode == "compare":
        if not args.compare_run_id:
            parser.error("compare requires --compare-run-id")
        result = compare_runs(args.project_root, args.run_id, args.compare_run_id)
    elif args.mode == "verify":
        result = verify_saved(args.project_root, args.run_id)
    else:
        result = verify_saved(args.project_root, args.run_id, rehash_inputs=False)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
