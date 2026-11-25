"""Fixed, development-only Spark MLlib demonstration for accepted fold 3."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bft65b.config import load_config
from bft65b.features.preprocessing import PREDICTOR_COLUMNS
from bft65b.manifest import RunManifest
from bft65b.modelling.development import (
    _day, _period_indices, verify_saved as verify_development,
)
from bft65b.modelling.split import _write_json_atomic, verify_saved as verify_split
from bft65b.features.table import verify_saved as verify_table
from bft65b.features.leakage_audit import verify_saved as verify_leakage
from bft65b.panel.build import _write_verified
from bft65b.panel.retrospective import _root, _run_id
from bft65b.quality.audit import _read_json, _sha


VERSION = "spark_mllib_fold3_v1"
NAMESPACE = "task_5_3_spark_mllib"
FOLD_ID = "fold_3"
MODEL_ID = "spark_logistic_demo"
SELECTED_CANDIDATE = "bounded_hist_gradient_boosting"
SPLIT_RUN = "split20261120b"
TABLE_RUN = "featuretable20261120a"
DEVELOPMENT_RUNS = ("dev20261121a", "dev20261121b")
LEAKAGE_RUN = "p44audit20261120a"
DEFAULT_RUNS = {"split": SPLIT_RUN, "table": TABLE_RUN,
                "development_primary": DEVELOPMENT_RUNS[0],
                "development_repeat": DEVELOPMENT_RUNS[1],
                "leakage": LEAKAGE_RUN}
MEDIAN_RELATIVE_ERROR = 0.001
SCORE_TOLERANCE = 1e-8
SPARK_SETTINGS = {
    "master": "local[2]", "spark.sql.shuffle.partitions": "8",
    "spark.sql.adaptive.enabled": "false", "spark.ui.enabled": "false",
    "spark.driver.memory": "4g", "spark.sql.session.timeZone": "UTC",
}
LOGISTIC_SETTINGS = {"maxIter": 20, "regParam": 0.1,
                     "elasticNetParam": 0.0, "tol": 1e-6,
                     "standardization": False, "fitIntercept": True}
SCORE_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64(), nullable=False),
    pa.field("Q", pa.date32(), nullable=False),
    pa.field("Y", pa.int8(), nullable=False),
    pa.field("RUN_ID", pa.string(), nullable=False),
    pa.field("FOLD_ID", pa.string(), nullable=False),
    pa.field("MODEL_ID", pa.string(), nullable=False),
    pa.field("SCORE", pa.float64(), nullable=False),
], metadata={b"bft65b.spark_demo": VERSION.encode()})


def paths(root: Path, run_id: str) -> tuple[Path, Path]:
    _run_id(run_id)
    root = _root(root)
    return (root / "runs" / NAMESPACE / run_id,
            root / "data_processed" / NAMESPACE / run_id)


def _quarter_from_path(path: str) -> date:
    part = Path(path).parent.name
    if not part.startswith("Q=") or Path(path).name != "part-000.parquet":
        raise ValueError(f"Unexpected model partition path: {path}")
    return _day(part[2:])


def select_fold_paths(plan: dict, fold: dict) -> tuple[list[str], list[str]]:
    """Select only exact frozen training and validation quarter files."""
    _period_indices(fold)
    if fold["id"] != FOLD_ID:
        raise ValueError("Spark demonstration requires accepted fold_3")
    train_start, train_end = _day(fold["train_start"]), _day(fold["train_end"])
    val_start, val_end = _day(fold["validation_start"]), _day(fold["validation_end"])
    train, validation = [], []
    for path in plan["development_model_paths"]:
        q = _quarter_from_path(path)
        if train_start <= q <= train_end:
            train.append(path)
        elif val_start <= q <= val_end:
            validation.append(path)
    if (len(train) != 96 or len(validation) != 16
            or len({_quarter_from_path(p) for p in train + validation}) != 112
            or any(p not in plan["input_hashes"] for p in train + validation)):
        raise ValueError("Frozen fold model-partition coverage differs")
    return train, validation


def accepted_scope(root: Path, selected: dict[str, str] | None = None) -> dict:
    """Validate immutable accepted inputs; do not open test predictors."""
    root = _root(root)
    selected = dict(DEFAULT_RUNS if selected is None else selected)
    if set(selected) != set(DEFAULT_RUNS) or any(not value for value in selected.values()):
        raise ValueError("Spark demo upstream selection is incomplete")
    portable = (root / "runs/source_vintage_bundle.json").is_file()
    if portable and any(selected[key] == default for key, default in DEFAULT_RUNS.items()):
        raise ValueError("Portable Spark demo requires explicit upstream runs")
    verify_split(root, selected["split"], rehash_inputs=True)
    verify_table(root, selected["table"], check_upstream=True)
    verify_leakage(root, selected["leakage"], rehash_inputs=True)
    for run in (selected["development_primary"], selected["development_repeat"]):
        checked = verify_development(root, run, rehash_inputs=True)
        if checked["selected_candidate"] != SELECTED_CANDIDATE:
            raise ValueError(f"Accepted selected candidate differs: {run}")
    first = _read_json(root / "runs" / "task_5_2_development" /
                       selected["development_primary"] / "frozen_plan.json")
    second = _read_json(root / "runs" / "task_5_2_development" /
                        selected["development_repeat"] / "frozen_plan.json")
    if (first.get("split_run") != selected["split"] or
            second.get("split_run") != selected["split"] or
            first.get("model_run") != selected["table"] or
            second.get("model_run") != selected["table"]):
        raise ValueError("Spark demo split/development/table lineage differs")
    for key in ("folds", "predictor_columns", "candidates", "seed", "input_hashes"):
        if first[key] != second[key]:
            raise ValueError(f"Accepted independent development plans differ: {key}")
    repeat = _read_json(root / "runs" / "task_5_2_development" /
                        selected["development_repeat"] / "repeat_comparison.json")
    if (repeat["status"] != "complete" or not repeat["frozen_policy_equal"]
            or repeat["max_abs_score_difference"] > repeat["absolute_tolerance"]):
        raise ValueError("Accepted development repeat is incomplete")
    if tuple(first["predictor_columns"]) != tuple(PREDICTOR_COLUMNS):
        raise ValueError("Approved seven-predictor allowlist differs")
    fold = next(row for row in first["folds"] if row["id"] == FOLD_ID)
    train, validation = select_fold_paths(first, fold)
    split = _read_json(root / "runs" / "task_5_1_split" / selected["split"] / "summary.json")
    if portable and (split.get("source_runs", {}).get("model_table") != selected["table"] or
                     split.get("source_runs", {}).get("leakage_audit") != selected["leakage"]):
        raise ValueError("Portable Spark demo selected upstream lineage differs")
    assignment = next(row["path"] for row in split["outputs"]
                      if row["dataset"] == "assignments")
    inputs = {path: first["input_hashes"][path] for path in train + validation}
    for path in (assignment,
                 root / "runs" / "task_5_2_development" / selected["development_primary"] / "frozen_plan.json",
                 root / "runs" / "task_5_2_development" / selected["development_primary"] / "summary.json",
                 root / "runs" / "task_5_2_development" / selected["development_repeat"] / "summary.json",
                 root / "runs" / "task_5_2_development" / selected["development_repeat"] / "repeat_comparison.json"):
        inputs[str(path)] = _sha(Path(path))
    if portable and any(not Path(path).resolve().is_relative_to(root) for path in inputs):
        raise ValueError("Portable Spark demo input escaped selected root")
    return {"fold": fold, "training_paths": train, "validation_paths": validation,
            "assignment_path": assignment, "input_hashes": inputs,
            "predictors": list(PREDICTOR_COLUMNS), "version": VERSION,
            "source_runs": selected}


def _runtime_hashes(root: Path) -> dict[str, str]:
    files = {"winutils.exe": "a0ca6e358357c41ef56ebdb02c38e4a4d55da7ca7a13001678bb2ef7d644adea",
             "hadoop.dll": "d3dd64afdc85f2a7eb5345abf2ecaa744b0a157de40859313337d47f81ee1c7b"}
    actual = {}
    for name, expected in files.items():
        path = root / "tmp" / "hadoop-3.3.5" / "bin" / name
        digest = _sha(path)
        if digest.lower() != expected:
            raise ValueError(f"Verified Spark native file differs: {path}")
        actual[str(path)] = digest
    return actual


def _spark(root: Path, data: Path, run_id: str):
    from pyspark.sql import SparkSession
    worker = Path(os.environ.get("PYSPARK_PYTHON", ""))
    hadoop = Path(os.environ.get("HADOOP_HOME", ""))
    if (not worker.is_file() or not worker.is_absolute()
            or hadoop.resolve() != (root / "tmp/hadoop-3.3.5").resolve()):
        raise RuntimeError("Use the documented absolute PYSPARK_PYTHON and HADOOP_HOME")
    builder = SparkSession.builder.master(SPARK_SETTINGS["master"]).appName(
        f"bft65b-{run_id}")
    for key, value in SPARK_SETTINGS.items():
        if key != "master":
            builder = builder.config(key, value)
    builder = builder.config("spark.local.dir", str(data / "spark_local"))
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    if session.version != "3.5.7":
        session.stop()
        raise RuntimeError("Verified Spark version differs")
    return session


def _role_frame(spark, files: list[str]):
    return spark.read.parquet(*files).select("CERT", "Q", "Y", *PREDICTOR_COLUMNS)


def validate_role(actual, expected, fold: dict, role: str) -> dict:
    """Compare complete (CERT,Q,Y) relations, not aggregate counts alone."""
    from pyspark.sql import functions as F
    low = _day(fold["train_start"] if role == "training" else fold["validation_start"])
    high = _day(fold["train_end"] if role == "training" else fold["validation_end"])
    triple = actual.select("CERT", "Q", "Y")
    reference = expected.select("CERT", "Q", "Y")
    count = triple.count()
    positives = triple.agg(F.sum("Y").alias("n")).first()["n"]
    if (actual.where((F.col("Q") < F.lit(low)) | (F.col("Q") > F.lit(high))).limit(1).count()
            or actual.where(F.col("CERT").isNull() | F.col("Q").isNull() |
                            ~F.col("Y").isin(0, 1)).limit(1).count()
            or triple.groupBy("CERT", "Q").count().where(F.col("count") != 1).limit(1).count()
            or count != fold[f"{role}_rows"]
            or positives != fold[f"{role}_positive_rows"]
            or triple.exceptAll(reference).limit(1).count()
            or reference.exceptAll(triple).limit(1).count()):
        raise ValueError(f"{role} keys, labels, classes or dates differ from accepted fold")
    return {"rows": count, "positive_rows": int(positives),
            "negative_rows": count - int(positives)}


def numeric_features(frame):
    """Cast approved decimal/double inputs; reject nonfinite values."""
    from pyspark.sql import functions as F
    values = frame.select("CERT", "Q", F.col("Y").cast("double").alias("label"),
                          *(F.col(name).cast("double").alias(name)
                            for name in PREDICTOR_COLUMNS))
    bad = None
    for name in PREDICTOR_COLUMNS:
        invalid = F.isnan(F.col(name)) | (F.abs(F.col(name)) == F.lit(float("inf")))
        bad = invalid if bad is None else bad | invalid
    if values.where(bad).limit(1).count():
        raise ValueError("Nonfinite approved predictor after decimal-to-double conversion")
    return values


def fit_medians(training, *, relative_error: float = MEDIAN_RELATIVE_ERROR) -> dict[str, float]:
    """Fit approximate Spark medians on training only; all-null columns use zero."""
    if not 0 <= relative_error <= 0.01:
        raise ValueError("Invalid frozen median relative error")
    quantiles = training.stat.approxQuantile(list(PREDICTOR_COLUMNS), [0.5], relative_error)
    medians = {name: (float(values[0]) if values else 0.0)
               for name, values in zip(PREDICTOR_COLUMNS, quantiles)}
    if len(medians) != len(PREDICTOR_COLUMNS) or not all(
            math.isfinite(value) for value in medians.values()):
        raise ValueError("Invalid fitted training median")
    return medians


def apply_medians(frame, medians: dict[str, float]):
    from pyspark.sql import functions as F
    if list(medians) != list(PREDICTOR_COLUMNS):
        raise ValueError("Fitted median order differs from predictor allowlist")
    return frame.select("CERT", "Q", "label", *(
        F.coalesce(F.col(name), F.lit(medians[name])).alias(f"imputed_{i}")
        for i, name in enumerate(PREDICTOR_COLUMNS)))


def model_pipeline():
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    inputs = [f"imputed_{i}" for i in range(len(PREDICTOR_COLUMNS))]
    return Pipeline(stages=[
        VectorAssembler(inputCols=inputs, outputCol="unscaled_features", handleInvalid="error"),
        StandardScaler(inputCol="unscaled_features", outputCol="features",
                       withMean=True, withStd=True),
        LogisticRegression(featuresCol="features", labelCol="label",
                           probabilityCol="probability", **LOGISTIC_SETTINGS),
    ])


def validate_scores(rows: list[dict], fold: dict, run_id: str) -> None:
    keys = [(row["CERT"], row["Q"]) for row in rows]
    if (len(rows) != fold["validation_rows"] or len(keys) != len(set(keys))
            or sum(row["Y"] for row in rows) != fold["validation_positive_rows"]
            or any(row["RUN_ID"] != run_id or row["FOLD_ID"] != FOLD_ID
                   or row["MODEL_ID"] != MODEL_ID or row["Y"] not in (0, 1)
                   or not math.isfinite(row["SCORE"]) or not 0 <= row["SCORE"] <= 1
                   or not _day(fold["validation_start"]) <= row["Q"] <=
                          _day(fold["validation_end"]) for row in rows)):
        raise ValueError("Spark validation score coverage, identity or probability differs")


def score_metrics(rows: list[dict]) -> dict:
    labels = np.asarray([row["Y"] for row in rows], dtype=np.int8)
    scores = np.asarray([row["SCORE"] for row in rows], dtype=np.float64)
    if not len(rows) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Spark validation metrics require both classes")
    return {"average_precision": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "brier": float(brier_score_loss(labels, scores)),
            "rows": len(rows), "positive_rows": int(labels.sum()),
            "negative_rows": int(len(rows) - labels.sum()),
            "prevalence": float(labels.mean())}


def read_score_file(path: Path, expected_sha256: str, fold: dict,
                    run_id: str) -> list[dict]:
    """Independently validate a promoted score file before trusting metrics."""
    if _sha(path) != expected_sha256:
        raise ValueError("Spark score file checksum differs")
    with pq.ParquetFile(path) as file:
        if (file.schema_arrow != SCORE_SCHEMA
                or file.metadata.num_rows != fold["validation_rows"]
                or any(file.metadata.row_group(g).column(c).compression != "ZSTD"
                       for g in range(file.metadata.num_row_groups)
                       for c in range(file.metadata.row_group(g).num_columns))):
            raise ValueError("Spark score Parquet schema, rows or codec differs")
        rows = file.read().to_pylist()
    validate_scores(rows, fold, run_id)
    return rows


def run(root: Path, run_id: str, *, selected: dict[str, str] | None = None) -> dict:
    root = _root(root)
    evidence, data = paths(root, run_id)
    if evidence.exists() or data.exists():
        raise FileExistsError(f"Spark demo run already exists: {run_id}")
    scope = accepted_scope(root, selected)
    native_hashes = _runtime_hashes(root)
    inputs = {**scope["input_hashes"], **native_hashes}
    manifest = RunManifest.new(run_id, load_config(root / "config/financials_full_v2.json"))
    manifest.config.update({"spark_demo_version": VERSION, "fold": scope["fold"],
                            "predictors": list(PREDICTOR_COLUMNS),
                            "median_relative_error": MEDIAN_RELATIVE_ERROR,
                            "spark_settings": SPARK_SETTINGS,
                            "logistic_settings": LOGISTIC_SETTINGS,
                            "selected_candidate_unchanged": SELECTED_CANDIDATE})
    manifest.config["source_runs"] = scope["source_runs"]
    for path, digest in sorted(inputs.items()):
        manifest.record_input(path, digest)
    manifest.set_status("running")
    manifest.write(evidence / "run_manifest.json")
    spark = None
    started = time.perf_counter()
    try:
        spark = _spark(root, data, run_id)
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql import functions as F
        assignment = spark.read.parquet(scope["assignment_path"]).select(
            "CERT", "Q", "Y", "PERIOD")
        fold = scope["fold"]
        expected_train = assignment.where((F.col("PERIOD") == "development") &
            F.col("Q").between(F.lit(_day(fold["train_start"])),
                                F.lit(_day(fold["train_end"]))))
        expected_val = assignment.where((F.col("PERIOD") == "development") &
            F.col("Q").between(F.lit(_day(fold["validation_start"])),
                                F.lit(_day(fold["validation_end"]))))
        train = _role_frame(spark, scope["training_paths"])
        validation = _role_frame(spark, scope["validation_paths"])
        train_counts = validate_role(train, expected_train, fold, "training")
        val_counts = validate_role(validation, expected_val, fold, "validation")
        prepared_train = numeric_features(train)
        prepared_val = numeric_features(validation)
        median_started = time.perf_counter()
        medians = fit_medians(prepared_train)
        median_seconds = time.perf_counter() - median_started
        fit_started = time.perf_counter()
        fitted = model_pipeline().fit(apply_medians(prepared_train, medians))
        fit_seconds = time.perf_counter() - fit_started
        if len(fitted.stages[1].std) != len(PREDICTOR_COLUMNS):
            raise ValueError("Spark fitted feature vector dimension differs")
        scoring_started = time.perf_counter()
        prediction = fitted.transform(apply_medians(prepared_val, medians))
        selected = prediction.select("CERT", "Q", F.col("label").cast("byte").alias("Y"),
            vector_to_array("probability")[1].alias("SCORE"))
        plan_before = selected._jdf.queryExecution().executedPlan().toString()
        collected = selected.collect()  # Full Spark inference is materialized inside the timer.
        scoring_seconds = time.perf_counter() - scoring_started
        plan_after = selected._jdf.queryExecution().executedPlan().toString()
        rows = sorted(({"CERT": int(item["CERT"]), "Q": item["Q"], "Y": int(item["Y"]),
                        "RUN_ID": run_id, "FOLD_ID": FOLD_ID, "MODEL_ID": MODEL_ID,
                        "SCORE": float(item["SCORE"])} for item in collected),
                      key=lambda item: (item["Q"], item["CERT"]))
        validate_scores(rows, fold, run_id)
        metrics = score_metrics(rows)
        _write_json_atomic(evidence / "plan_before.json", {"physical_plan": plan_before})
        _write_json_atomic(evidence / "plan_after.json", {"physical_plan": plan_after})
        output = data / "validation_scores.parquet"
        digest = _write_verified(output, pa.Table.from_pylist(rows, schema=SCORE_SCHEMA))
        manifest.record_output(path=str(output), dataset="spark_validation_scores",
            rows=len(rows), sha256=digest, compression="ZSTD", schema_version=VERSION,
            partition=FOLD_ID, source_total=len(rows), source_query=None)
        if any(_sha(Path(path)) != digest for path, digest in inputs.items()):
            raise ValueError("Accepted Spark demo input changed during execution")
        jvm = spark._jvm
        runtime = {"os": platform.platform(), "python": platform.python_version(),
            "spark": spark.version,
            "hadoop": jvm.org.apache.hadoop.util.VersionInfo.getVersion(),
            "java": jvm.java.lang.System.getProperty("java.version"),
            "native_hadoop_loaded": bool(jvm.org.apache.hadoop.util.NativeCodeLoader.isNativeCodeLoaded()),
            "master": spark.sparkContext.master,
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "spark_settings": {key: spark.conf.get(key) for key in SPARK_SETTINGS if key != "master"},
            "input_partitions": {"training_files": len(scope["training_paths"]),
                                 "validation_files": len(scope["validation_paths"]),
                                 "training_spark_partitions": train.rdd.getNumPartitions(),
                                 "validation_spark_partitions": validation.rdd.getNumPartitions()},
            "native_file_hashes": native_hashes}
        summary = {"status": "complete", "version": VERSION, "run_id": run_id,
            "source_runs": scope["source_runs"],
            "fold": fold, "selected_candidate_unchanged": SELECTED_CANDIDATE,
            "model_id": MODEL_ID, "predictors": list(PREDICTOR_COLUMNS),
            "training": train_counts, "validation": val_counts,
            "training_medians": medians,
            "training_scaler_mean": list(fitted.stages[1].mean),
            "training_scaler_std": list(fitted.stages[1].std),
            "logistic_settings": LOGISTIC_SETTINGS,
            "median_relative_error": MEDIAN_RELATIVE_ERROR,
            "metrics": metrics, "timings_seconds": {
                "median_fit": median_seconds, "spark_pipeline_fit": fit_seconds,
                "spark_transform_collect": scoring_seconds,
                "total_before_session_stop": time.perf_counter() - started},
            "runtime": runtime, "output": {"path": str(output), "sha256": digest,
                                                "rows": len(rows)},
            "input_hashes": inputs, "test_predictor_files_opened": 0,
            "test_predictions_computed": 0, "threshold_selected": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_json_atomic(evidence / "summary.json", summary)
        _write_json_atomic(evidence / "metadata.json", {"status": "complete",
            "summary_sha256": _sha(evidence / "summary.json"), "output_sha256": digest})
        manifest.record_validation("accepted_fold_keys_and_classes", True,
            "Training and validation (CERT,Q,Y) relations equal frozen split assignments")
        manifest.record_validation("spark_train_only_preprocessing_and_inference", True,
            "Training medians and Spark MLlib pipeline fit on training only; validation fully collected")
        manifest.record_validation("saved_scores_and_source_integrity", True,
            "ZSTD read-back and source hashes checked")
        manifest.set_status("complete")
        manifest.write(evidence / "run_manifest.json")
        verify_saved(root, run_id)
        return {"status": "complete", "run_id": run_id,
                "training": train_counts, "validation": val_counts,
                "metrics": metrics, "timings_seconds": summary["timings_seconds"]}
    except Exception as exc:
        manifest.record_validation("spark_demo_failure", False, str(exc))
        manifest.set_status("failed")
        manifest.write(evidence / "run_manifest.json")
        raise
    finally:
        if spark is not None:
            spark.stop()


def verify_saved(root: Path, run_id: str, *, rehash_inputs: bool = True) -> dict:
    evidence, data = paths(root, run_id)
    manifest = RunManifest.read(evidence / "run_manifest.json")
    summary = _read_json(evidence / "summary.json")
    metadata = _read_json(evidence / "metadata.json")
    if (manifest.status != "complete" or not manifest.validations
            or any(not check["passed"] for check in manifest.validations)
            or summary["status"] != "complete" or summary["version"] != VERSION
            or summary["run_id"] != run_id or summary["fold"]["id"] != FOLD_ID
            or summary["selected_candidate_unchanged"] != SELECTED_CANDIDATE
            or summary["model_id"] != MODEL_ID
            or summary["predictors"] != list(PREDICTOR_COLUMNS)
            or metadata["summary_sha256"] != _sha(evidence / "summary.json")
            or metadata["status"] != "complete"
            or summary["test_predictor_files_opened"] != 0
            or summary["test_predictions_computed"] != 0
            or summary["threshold_selected"] is not False):
        raise ValueError("Spark demonstration manifest or summary is incomplete")
    item = summary["output"]
    output = Path(item["path"])
    if (output.parent.resolve() != data.resolve()
            or item["sha256"] != metadata["output_sha256"]
            or _sha(output) != item["sha256"]
            or len(manifest.outputs) != 1
            or manifest.outputs[0]["sha256"] != item["sha256"]):
        raise ValueError("Spark demonstration output path or checksum differs")
    rows = read_score_file(output, item["sha256"], summary["fold"], run_id)
    if any(abs(score_metrics(rows)[key] - summary["metrics"][key]) > 1e-12
           for key in ("average_precision", "roc_auc", "brier")):
        raise ValueError("Saved Spark metric differs from validation scores")
    selected = summary.get("source_runs", DEFAULT_RUNS)
    if (set(selected) != set(DEFAULT_RUNS) or
            manifest.config.get("source_runs", selected) != selected):
        raise ValueError("Spark demonstration saved upstream lineage differs")
    if (root / "runs/source_vintage_bundle.json").is_file() and any(
            selected[key] == default for key, default in DEFAULT_RUNS.items()):
        raise ValueError("Portable Spark demonstration references historical defaults")
    split = _read_json(root / "runs/task_5_1_split" / selected["split"] / "summary.json")
    assignment_path = next(item["path"] for item in split["outputs"]
                           if item["dataset"] == "assignments")
    assignment = pq.read_table(assignment_path,
        columns=["CERT", "Q", "Y", "PERIOD"])
    q = assignment["Q"]
    fold = summary["fold"]
    mask = pc.and_(pc.equal(assignment["PERIOD"], "development"), pc.and_(
        pc.greater_equal(q, pa.scalar(_day(fold["validation_start"]))),
        pc.less_equal(q, pa.scalar(_day(fold["validation_end"])))))
    expected = {(item["CERT"], item["Q"], item["Y"])
                for item in assignment.filter(mask).to_pylist()}
    if expected != {(item["CERT"], item["Q"], item["Y"]) for item in rows}:
        raise ValueError("Saved Spark scores do not cover frozen validation keys")
    if rehash_inputs and any(_sha(Path(path)) != digest
                             for path, digest in summary["input_hashes"].items()):
        raise ValueError("Spark demo accepted source hash differs")
    return {"status": "complete", "run_id": run_id,
            "validation_rows": len(rows), "metrics": summary["metrics"]}


def compare_runs(root: Path, first_id: str, second_id: str) -> dict:
    verify_saved(root, first_id)
    verify_saved(root, second_id)
    root = _root(root)
    a = _read_json(root / "runs" / NAMESPACE / first_id / "summary.json")
    b = _read_json(root / "runs" / NAMESPACE / second_id / "summary.json")
    if any(a[key] != b[key] for key in ("fold", "predictors", "training",
                                      "validation", "training_medians",
                                      "training_scaler_mean", "training_scaler_std",
                                      "logistic_settings", "input_hashes")):
        raise ValueError("Independent Spark demo policy or fitted training statistics differ")
    left = pq.read_table(a["output"]["path"]).to_pylist()
    right = pq.read_table(b["output"]["path"]).to_pylist()
    if len(left) != len(right):
        raise ValueError("Spark repeat score row counts differ")
    maximum = 0.0
    for x, y in zip(left, right):
        if any(x[key] != y[key] for key in ("CERT", "Q", "Y", "FOLD_ID", "MODEL_ID")):
            raise ValueError("Spark repeat score keys or labels differ")
        maximum = max(maximum, abs(x["SCORE"] - y["SCORE"]))
    if maximum > SCORE_TOLERANCE:
        raise ValueError(f"Spark repeat score difference {maximum} exceeds {SCORE_TOLERANCE}")
    result = {"status": "complete", "first_run": first_id, "repeat_run": second_id,
              "scores_compared": len(left), "max_abs_score_difference": maximum,
              "absolute_tolerance": SCORE_TOLERANCE,
              "training_statistics_equal": True}
    target = root / "runs" / NAMESPACE / second_id / "repeat_comparison.json"
    if target.exists():
        raise FileExistsError("Spark repeat evidence already exists")
    _write_json_atomic(target, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Accepted fold-3 Spark MLlib demonstration")
    parser.add_argument("mode", choices=("run", "verify", "compare", "show"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compare-run-id")
    for key in DEFAULT_RUNS:
        parser.add_argument("--" + key.replace("_", "-"))
    args = parser.parse_args()
    if args.mode == "run":
        selected = {key: getattr(args, key) for key in DEFAULT_RUNS}
        if any(selected.values()) and not all(selected.values()):
            parser.error("All Spark demo upstream run IDs are required for replay")
        result = run(args.project_root, args.run_id,
                     selected=selected if any(selected.values()) else None)
    elif args.mode == "verify":
        result = verify_saved(args.project_root, args.run_id)
    elif args.mode == "compare":
        if not args.compare_run_id:
            parser.error("compare requires --compare-run-id")
        result = compare_runs(args.project_root, args.run_id, args.compare_run_id)
    else:
        result = _read_json(paths(args.project_root, args.run_id)[0] / "summary.json")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
