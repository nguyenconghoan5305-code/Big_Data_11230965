"""PySpark implementation of the frozen P3 observed-panel workload.

The Spark DataFrame queries use native SQL expressions. Python/Arrow is used
only at the final full-materialization sink shared with the other engines.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from bft65b.processing.polars_panel import (
    GAP_COLUMNS, INPUT_COLUMNS, QUALITY_COLUMNS, UNMATCHED_FLAG,
)


SPARK_MASTER = "local[4]"
SHUFFLE_PARTITIONS = 16
ADAPTIVE_ENABLED = "true"


def start_spark(scratch: Path) -> SparkSession:
    """Create the verified local runtime, with per-run ignored scratch."""
    worker = os.environ.get("PYSPARK_PYTHON")
    if not worker or not Path(worker).is_file():
        raise RuntimeError("PYSPARK_PYTHON must identify the verified local interpreter")
    if not os.environ.get("HADOOP_HOME"):
        raise RuntimeError("HADOOP_HOME must identify the verified Windows native runtime")
    scratch.mkdir(parents=True, exist_ok=False)
    spark = (SparkSession.builder.master(SPARK_MASTER)
             .appName("bft65b-p3-observed-panel")
             .config("spark.ui.enabled", "false")
             .config("spark.local.dir", str(scratch))
             .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
             .config("spark.sql.adaptive.enabled", ADAPTIVE_ENABLED)
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def queries(spark: SparkSession, paths: Sequence[Path]) -> dict[str, DataFrame]:
    """Build the three relations without collecting or deduplicating rows."""
    if not paths or len({str(path) for path in paths}) != len(paths):
        raise ValueError("Input file list is empty or duplicated")
    base = spark.read.parquet(*[str(path) for path in paths]).select(*INPUT_COLUMNS)
    observed = base.orderBy("REPDTE", "CERT")

    preceding = Window.partitionBy("CERT").orderBy("REPDTE")
    ordered = base.select("CERT", "REPDTE").withColumn(
        "PREVIOUS_REPDTE", F.lag("REPDTE").over(preceding))
    current_index = F.year("REPDTE") * F.lit(4) + F.quarter("REPDTE") - F.lit(1)
    previous_index = (F.year("PREVIOUS_REPDTE") * F.lit(4)
                      + F.quarter("PREVIOUS_REPDTE") - F.lit(1))
    gaps = (ordered.withColumn("MISSING_INTERVENING_QUARTERS",
                               current_index - previous_index - F.lit(1))
            .where(F.col("MISSING_INTERVENING_QUARTERS") > 0)
            .select("CERT", "PREVIOUS_REPDTE",
                    F.col("REPDTE").alias("NEXT_REPDTE"),
                    "MISSING_INTERVENING_QUARTERS")
            .orderBy("CERT", "NEXT_REPDTE"))

    absent = F.coalesce(F.array_contains("SOURCE_MISSING_FIELDS", "P3ASSET"), F.lit(False))
    unmatched = F.coalesce(F.array_contains("DQ_FLAGS", UNMATCHED_FLAG), F.lit(False))
    null = F.col("P3ASSET").isNull()
    count_if = lambda condition: F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0)))
    quality = (base.groupBy("REPDTE").agg(
        F.count(F.lit(1)).alias("ROW_COUNT"),
        F.countDistinct("CERT").alias("DISTINCT_KEY_COUNT"),
        count_if(null).alias("P3ASSET_NULL_ROWS"),
        count_if(absent).alias("P3ASSET_SOURCE_ABSENT_ROWS"),
        count_if(null & ~absent).alias("P3ASSET_EXPLICIT_NULL_ROWS"),
        count_if(unmatched).alias("UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS"),
    ).select(*QUALITY_COLUMNS).orderBy("REPDTE"))
    return {"observed": observed, "gaps": gaps, "quality": quality}


def _plan(frame: DataFrame) -> dict[str, str | bool | None]:
    physical = frame._jdf.queryExecution().executedPlan()
    is_final = None
    try:
        is_final = bool(physical.isFinalPlan())
    except Exception:
        pass
    formatted = io.StringIO()
    with redirect_stdout(formatted):
        frame.explain(mode="formatted")
    return {"physical": physical.toString(), "formatted": formatted.getvalue(),
            "adaptive_is_final": is_final}


def materialize(frames: Mapping[str, DataFrame], quarters: Sequence[str],
                schemas: Mapping[str, pa.Schema],
                plan_dir: Path | None = None) -> dict[str, pa.Table]:
    """Evaluate every selected value, then safely convert to frozen Arrow types."""
    if plan_dir is not None:
        plan_dir.mkdir(parents=True, exist_ok=True)
    expected = set(quarters)
    tables: dict[str, pa.Table] = {}
    for name in ("observed", "gaps", "quality"):
        frame = frames[name]
        if plan_dir is not None:
            before = _plan(frame)
            (plan_dir / f"{name}_before.json").write_text(
                json.dumps(before, indent=2), encoding="utf-8")
        pdf = frame.toPandas()  # Complete Arrow-backed collection, not a prunable count().
        if list(pdf.columns) != schemas[name].names:
            raise ValueError(f"Spark {name} column order differs from the frozen workload")
        table = pa.Table.from_pandas(pdf, schema=schemas[name], preserve_index=False, safe=True)
        if table.num_rows != len(pdf) or not table.schema.equals(schemas[name]):
            raise ValueError(f"Spark {name} Arrow conversion differs")
        tables[name] = table
        if plan_dir is not None:
            after = _plan(frame)
            (plan_dir / f"{name}_after.json").write_text(
                json.dumps(after, indent=2), encoding="utf-8")
    observed, gaps, quality = (tables[name] for name in ("observed", "gaps", "quality"))
    if (observed.num_rows != sum(quality["ROW_COUNT"].to_pylist())
            or quality.num_rows != len(expected)
            or {d.strftime("%Y%m%d") for d in quality["REPDTE"].to_pylist()} != expected
            or any(a != b for a, b in zip(quality["ROW_COUNT"].to_pylist(),
                                          quality["DISTINCT_KEY_COUNT"].to_pylist()))
            or any(x <= 0 for x in gaps["MISSING_INTERVENING_QUARTERS"].to_pylist())):
        raise ValueError("Spark quarter inventory, key uniqueness, or gap invariant differs")
    if any(total != absent + explicit for total, absent, explicit in zip(
            quality["P3ASSET_NULL_ROWS"].to_pylist(),
            quality["P3ASSET_SOURCE_ABSENT_ROWS"].to_pylist(),
            quality["P3ASSET_EXPLICIT_NULL_ROWS"].to_pylist())):
        raise ValueError("Spark source-absent and explicit-null counts do not reconcile")
    return tables


def runtime_settings(spark: SparkSession) -> dict[str, object]:
    jvm = spark._jvm
    config = spark.sparkContext.getConf()
    keys = ("spark.master", "spark.default.parallelism", "spark.sql.shuffle.partitions",
            "spark.sql.adaptive.enabled", "spark.sql.execution.arrow.pyspark.enabled",
            "spark.local.dir", "spark.ui.enabled")
    return {"spark": spark.version,
            "hadoop": jvm.org.apache.hadoop.util.VersionInfo.getVersion(),
            "java": jvm.java.lang.System.getProperty("java.version"),
            "native_loaded": bool(jvm.org.apache.hadoop.util.NativeCodeLoader.isNativeCodeLoaded()),
            "native_library": jvm.org.apache.hadoop.util.NativeCodeLoader.getLibraryName(),
            "default_parallelism_effective": spark.sparkContext.defaultParallelism,
            "configuration": {key: spark.conf.get(key, config.get(key, None)) for key in keys},
            "PYSPARK_PYTHON": os.environ.get("PYSPARK_PYTHON"),
            "HADOOP_HOME": os.environ.get("HADOOP_HOME")}
