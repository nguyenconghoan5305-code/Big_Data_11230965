"""Synthetic, bounded acceptance cases for the development-only MLlib demo."""
from datetime import date
from pathlib import Path
import os

import numpy as np
import pyarrow as pa
import pytest
from pyspark.sql.types import DateType, DoubleType, IntegerType, LongType, StructField, StructType
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bft65b.features.preprocessing import PREDICTOR_COLUMNS
from bft65b.modelling.spark_demo import (
    FOLD_ID, MODEL_ID, SCORE_SCHEMA, _spark, apply_medians, fit_medians,
    model_pipeline, numeric_features, read_score_file, score_metrics,
    select_fold_paths, validate_role, validate_scores,
)
from bft65b.panel.build import _write_verified
from bft65b.modelling import spark_demo


pytestmark = pytest.mark.skipif(
    not os.environ.get("HADOOP_HOME") or not os.environ.get("PYSPARK_PYTHON"),
    reason="Task 3.0 Spark runtime variables are not configured")


def test_portable_spark_demo_rejects_historical_fallback_and_partial_selection(tmp_path):
    (tmp_path / "final_report_bft65b.pdf").write_bytes(b"synthetic")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit upstream"):
        spark_demo.accepted_scope(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        spark_demo.accepted_scope(tmp_path, {"split": "portable-split"})


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    session = _spark(root, tmp_path_factory.mktemp("p53_spark"), "synthetic")
    yield session
    session.stop()


def _fold():
    return {"id": FOLD_ID, "train_start": "1984-03-31", "train_end": "2007-12-31",
            "gap_start": "2008-03-31", "gap_end": "2008-12-31",
            "training_target_right_end": "2008-12-31",
            "validation_start": "2009-03-31", "validation_end": "2012-12-31",
            "training_rows": 6, "training_positive_rows": 3,
            "validation_rows": 2, "validation_positive_rows": 1}


def _frame(spark, role="training"):
    q = date(2007, 12, 31) if role == "training" else date(2009, 3, 31)
    n = 6 if role == "training" else 2
    rows = []
    for i in range(n):
        vals = [float(i + j) for j in range(len(PREDICTOR_COLUMNS))]
        vals[2] = 2.0  # Constant feature.
        vals[5] = None if role == "training" else float(i)  # All-missing training feature.
        rows.append((i + 1 + (0 if role == "training" else 100), q, i % 2, *vals))
    schema = StructType([
        StructField("CERT", LongType(), False), StructField("Q", DateType(), False),
        StructField("Y", IntegerType(), False),
        *(StructField(name, DoubleType(), True) for name in PREDICTOR_COLUMNS),
    ])
    return spark.createDataFrame(rows, schema)


def test_frozen_paths_exclude_inner_gap_outer_gap_and_test():
    import calendar
    quarters = [date(year, month, calendar.monthrange(year, month)[1])
                for year in range(1984, 2013) for month in (3, 6, 9, 12)]
    paths = [f"C:/synthetic/Q={q:%Y%m%d}/part-000.parquet" for q in quarters]
    plan = {"development_model_paths": paths,
            "input_hashes": {path: "synthetic" for path in paths}}
    train, validation = select_fold_paths(plan, _fold())
    assert len(train) == 96 and len(validation) == 16
    assert all("Q=2008" not in path for path in train + validation)
    assert all("Q=2013" not in path and "Q=2014" not in path
               for path in train + validation)
    plan["development_model_paths"].remove(validation[0])
    with pytest.raises(ValueError, match="coverage"):
        select_fold_paths(plan, _fold())


def test_exact_role_keys_labels_and_rejected_gap_test_duplicates(spark):
    from pyspark.sql import functions as F
    fold = _fold()
    train, validation = _frame(spark), _frame(spark, "validation")
    assert validate_role(train, train, fold, "training") == {
        "rows": 6, "positive_rows": 3, "negative_rows": 3}
    assert validate_role(validation, validation, fold, "validation")["rows"] == 2
    gap = train.limit(1).withColumn("Q", F.lit(date(2008, 3, 31)))
    with pytest.raises(ValueError, match="training keys"):
        validate_role(train.unionByName(gap), train, fold, "training")
    test = validation.limit(1).withColumn("Q", F.lit(date(2014, 3, 31)))
    with pytest.raises(ValueError, match="validation keys"):
        validate_role(validation.unionByName(test), validation, fold, "validation")
    with pytest.raises(ValueError, match="training keys"):
        validate_role(train.unionByName(train.limit(1)), train, fold, "training")
    with pytest.raises(ValueError, match="validation keys"):
        validate_role(validation.limit(1), validation, fold, "validation")


def test_training_only_spark_medians_pipeline_and_negative_control(spark):
    from pyspark.sql import functions as F
    train, validation = _frame(spark), _frame(spark, "validation")
    trained = numeric_features(train)
    heldout = numeric_features(validation)
    medians = fit_medians(trained)
    assert list(medians) == list(PREDICTOR_COLUMNS)
    assert medians["asset_growth_qoq_pct"] == 0.0
    first = model_pipeline().fit(apply_medians(trained, medians))
    changed = heldout.withColumn("asset_size", F.lit(1e12)).withColumn("label", 1-F.col("label"))
    second = model_pipeline().fit(apply_medians(trained, fit_medians(trained)))
    assert len(first.stages[1].std) == len(PREDICTOR_COLUMNS) == 7
    np.testing.assert_array_equal(list(first.stages[1].mean), list(second.stages[1].mean))
    np.testing.assert_array_equal(list(first.stages[2].coefficients),
                                  list(second.stages[2].coefficients))
    assert changed.count() == validation.count()
    original_scores = [tuple(row.probability) for row in
        first.transform(apply_medians(heldout, medians)).select("probability").collect()]
    changed_scores = [tuple(row.probability) for row in
        first.transform(apply_medians(changed, medians)).select("probability").collect()]
    assert original_scores != changed_scores
    contaminated = fit_medians(trained.unionByName(changed))
    assert contaminated["asset_size"] != medians["asset_size"]
    assert "CERT" not in first.stages[0].getInputCols()
    assert "Q" not in first.stages[0].getInputCols()
    assert "label" not in first.stages[0].getInputCols()
    assert first.stages[0].getInputCols() == [f"imputed_{i}" for i in range(7)]


def test_nonfinite_numeric_input_is_rejected(spark):
    from pyspark.sql import functions as F
    bad = _frame(spark).withColumn("asset_size", F.lit(float("inf")))
    with pytest.raises(ValueError, match="Nonfinite"):
        numeric_features(bad)


def test_saved_score_schema_keys_checksum_and_metric_definition(tmp_path):
    fold = dict(_fold(), validation_start="2009-03-31", validation_end="2009-03-31")
    rows = [{"CERT": cert, "Q": date(2009, 3, 31), "Y": label,
             "RUN_ID": "synthetic", "FOLD_ID": FOLD_ID, "MODEL_ID": MODEL_ID,
             "SCORE": score}
            for cert, label, score in ((1, 0, 0.2), (2, 1, 0.8))]
    assert score_metrics(rows)["average_precision"] == average_precision_score([0, 1], [0.2, 0.8])
    assert score_metrics(rows)["roc_auc"] == roc_auc_score([0, 1], [0.2, 0.8])
    assert score_metrics(rows)["brier"] == brier_score_loss([0, 1], [0.2, 0.8])
    path = tmp_path / "scores.parquet"
    digest = _write_verified(path, pa.Table.from_pylist(rows, schema=SCORE_SCHEMA))
    assert read_score_file(path, digest, fold, "synthetic") == rows
    with pytest.raises(ValueError, match="coverage"):
        validate_scores([rows[0], rows[0]], fold, "synthetic")
    with pytest.raises(ValueError, match="probability"):
        validate_scores([rows[0], dict(rows[1], SCORE=float("nan"))], fold, "synthetic")
    with pytest.raises(ValueError, match="checksum"):
        read_score_file(path, "0" * 64, fold, "synthetic")
    path.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="checksum"):
        read_score_file(path, digest, fold, "synthetic")
