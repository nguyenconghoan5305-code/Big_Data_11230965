"""Synthetic native-Spark parity for the frozen P3 workload."""

from datetime import date
import os

import polars as pl
import pytest

from bft65b.benchmarks.panel import SCHEMAS
from bft65b.processing.polars_panel import lazy_queries, materialize as polars_materialize
from bft65b.processing.spark_panel import (
    materialize as spark_materialize, queries, runtime_settings, start_spark,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("HADOOP_HOME") or not os.environ.get("PYSPARK_PYTHON"),
    reason="Task 3.0 verified Spark runtime environment is not configured")
QUARTERS = ("20220930", "20221231", "20230331")


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    session = start_spark(tmp_path_factory.mktemp("spark_p33") / "local")
    yield session
    session.stop()


def _files(tmp_path, *, duplicate=False, invalid_absent=False):
    rows = {
        "20220930": [
            {"CERT": 1, "REPDTE": date(2022, 9, 30), "ID": "a", "P3ASSET": None,
             "SOURCE_MISSING_FIELDS": [], "DQ_FLAGS": []}],
        "20221231": [
            {"CERT": 2, "REPDTE": date(2022, 12, 31), "ID": "b",
             "P3ASSET": 8.0 if invalid_absent else None,
             "SOURCE_MISSING_FIELDS": ["P3ASSET"], "DQ_FLAGS": []}],
        "20230331": [
            {"CERT": 1, "REPDTE": date(2023, 3, 31), "ID": "c",
             "P3ASSET": 1.23456789012345, "SOURCE_MISSING_FIELDS": [],
             "DQ_FLAGS": ["financial_cert_absent_from_institutions:CERT"]},
            {"CERT": 2, "REPDTE": date(2023, 3, 31), "ID": "d",
             "P3ASSET": 0.0, "SOURCE_MISSING_FIELDS": [], "DQ_FLAGS": []}],
    }
    if duplicate:
        rows["20230331"].append(rows["20230331"][0].copy())
    paths = []
    for quarter, values in rows.items():
        path = tmp_path / f"REPDTE={quarter}" / "part-000.parquet"
        path.parent.mkdir(parents=True)
        pl.DataFrame(values, schema_overrides={
            "CERT": pl.Int64, "REPDTE": pl.Date, "ID": pl.String,
            "P3ASSET": pl.Float64, "SOURCE_MISSING_FIELDS": pl.List(pl.String),
            "DQ_FLAGS": pl.List(pl.String),
        }).write_parquet(path)
        paths.append(path)
    return paths


def test_spark_matches_all_three_polars_relations_and_saves_actual_plans(spark, tmp_path):
    paths = _files(tmp_path / "source")
    expected = polars_materialize(lazy_queries(paths), QUARTERS)
    actual = spark_materialize(queries(spark, paths), QUARTERS, SCHEMAS, tmp_path / "plans")
    for name, table in actual.items():
        assert pl.from_arrow(table).equals(expected[name], null_equal=True), name
        assert (tmp_path / "plans" / f"{name}_before.json").exists()
        assert (tmp_path / "plans" / f"{name}_after.json").exists()
    assert actual["gaps"]["MISSING_INTERVENING_QUARTERS"].to_pylist() == [1]
    assert actual["quality"]["P3ASSET_SOURCE_ABSENT_ROWS"].to_pylist() == [0, 1, 0]
    assert actual["quality"]["P3ASSET_EXPLICIT_NULL_ROWS"].to_pylist() == [1, 0, 0]
    assert actual["quality"]["UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS"].to_pylist() == [0, 0, 1]
    settings = runtime_settings(spark)
    assert settings["spark"] == "3.5.7"
    assert settings["default_parallelism_effective"] == 4


def test_spark_rejects_duplicate_bank_quarter(spark, tmp_path):
    with pytest.raises(ValueError, match="key uniqueness"):
        spark_materialize(queries(spark, _files(tmp_path, duplicate=True)), QUARTERS, SCHEMAS)


def test_spark_rejects_nonnull_source_absent_value(spark, tmp_path):
    with pytest.raises(ValueError, match="source-absent and explicit-null"):
        spark_materialize(queries(spark, _files(tmp_path, invalid_absent=True)),
                          QUARTERS, SCHEMAS)
