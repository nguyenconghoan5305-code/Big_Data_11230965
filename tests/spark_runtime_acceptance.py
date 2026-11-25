"""Manual, synthetic Task 3.0 Spark acceptance. Run with documented environment."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import pyarrow
import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, DecimalType, LongType

from bft65b.acquisition.storage import ParquetStore
from bft65b.config import ProjectConfig
from bft65b.manifest import RunManifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scratch = Path(tempfile.mkdtemp(prefix="spark_acceptance_", dir=root / "tmp"))
    worker = os.environ.get("PYSPARK_PYTHON")
    if not worker or not Path(worker).is_file():
        raise RuntimeError("PYSPARK_PYTHON must be an absolute existing interpreter path")
    spark = (SparkSession.builder.master("local[2]")
             .appName("bft65b-task-3-0-acceptance")
             .config("spark.ui.enabled", "false")
             .config("spark.local.dir", str(scratch))
             .getOrCreate())
    try:
        spark.sparkContext.setLogLevel("WARN")
        jvm = spark._jvm
        environment = {
            "os": platform.platform(), "python": sys.version.split()[0],
            "java": jvm.java.lang.System.getProperty("java.version"),
            "java_arch": jvm.java.lang.System.getProperty("os.arch"),
            "spark": spark.version,
            "pyspark": pyspark.__version__, "pyarrow": pyarrow.__version__,
            "hadoop": jvm.org.apache.hadoop.util.VersionInfo.getVersion(),
            "hadoop_native_loaded": jvm.org.apache.hadoop.util.NativeCodeLoader.isNativeCodeLoaded(),
            "hadoop_native_library": jvm.org.apache.hadoop.util.NativeCodeLoader.getLibraryName(),
            "JAVA_HOME": os.environ.get("JAVA_HOME"),
            "HADOOP_HOME": os.environ.get("HADOOP_HOME"),
            "SPARK_HOME": os.environ.get("SPARK_HOME"),
            "PYSPARK_PYTHON": worker,
            "PYSPARK_DRIVER_PYTHON": os.environ.get("PYSPARK_DRIVER_PYTHON"),
            "scratch": str(scratch),
        }
        print("ENVIRONMENT " + json.dumps(environment, sort_keys=True), flush=True)

        base = spark.createDataFrame([(1,), (2,), (3,)], ["value"])
        doubled = [row.doubled for row in base.selectExpr("value * 2 AS doubled").collect()]
        assert doubled == [2, 4, 6]
        print("DATAFRAME_TRANSFORMATION_PASS " + str(doubled), flush=True)

        config = ProjectConfig()
        store = ParquetStore(scratch / "data_raw", config)
        manifest = RunManifest.new("spark-synthetic-acceptance", config)
        source_rows = [
            {"CERT": 14, "REPDTE": "20230331", "STNAME": "MA", "ASSET": "100.2500",
             "DEP": None, "EQ": 10, "NETINC": 1, "ROA": 1.5,
             "NCLNLS": None, "LNATRES": 2, "RBC1AAJ": 12.5},
            {"CERT": "035", "REPDTE": "20230331", "STNAME": "CA", "ASSET": 200,
             "DEP": "150.5000", "EQ": 20, "NETINC": -1, "ROA": -0.5,
             "NCLNLS": 3, "LNATRES": None, "RBC1AAJ": None},
        ]
        written = store.write(
            "financials", source_rows, quarter="20230331", source_total=2,
            manifest=manifest, manifest_path=scratch / "manifest.json",
            source_query="synthetic:task-3-0",
        )
        store.validate_file(written.path, "financials", quarter="20230331",
                            expected_rows=2, expected_sha256=written.sha256)
        frame = spark.read.parquet(str(written.path))
        assert isinstance(frame.schema["CERT"].dataType, LongType)
        assert isinstance(frame.schema["REPDTE"].dataType, DateType)
        assert frame.schema["ASSET"].dataType == DecimalType(20, 4)
        assert frame.schema["DEP"].dataType == DecimalType(20, 4)
        values = [row.asDict() for row in frame.select(
            "CERT", "REPDTE", "ASSET", "DEP", "ROA").orderBy("CERT").collect()]
        assert values == [
            {"CERT": 14, "REPDTE": date(2023, 3, 31), "ASSET": Decimal("100.2500"),
             "DEP": None, "ROA": 1.5},
            {"CERT": 35, "REPDTE": date(2023, 3, 31), "ASSET": Decimal("200.0000"),
             "DEP": Decimal("150.5000"), "ROA": -0.5},
        ]
        print("PYARROW_TO_SPARK_PARITY_PASS " + repr(values), flush=True)

        target = scratch / "spark_roundtrip.parquet"
        frame.select("CERT", "REPDTE", "ASSET", "DEP", "ROA").write.parquet(str(target))
        reread = spark.read.parquet(str(target))
        second = [row.asDict() for row in reread.orderBy("CERT").collect()]
        assert second == values
        assert reread.schema == frame.select("CERT", "REPDTE", "ASSET", "DEP", "ROA").schema
        print("SPARK_PARQUET_ROUNDTRIP_PASS " + repr(second), flush=True)

        training = spark.createDataFrame([
            (0.0, Vectors.dense([0.0, 0.0])),
            (0.0, Vectors.dense([0.2, 0.1])),
            (0.0, Vectors.dense([0.1, 0.2])),
            (1.0, Vectors.dense([0.8, 0.9])),
            (1.0, Vectors.dense([1.0, 1.0])),
            (1.0, Vectors.dense([0.9, 0.8])),
        ], ["label", "features"])
        model = LogisticRegression(maxIter=5, regParam=0.1).fit(training)
        predictions = model.transform(training).select("label", "prediction").collect()
        assert len(predictions) == 6
        assert {row.prediction for row in predictions}.issubset({0.0, 1.0})
        print("MLLIB_FIT_TRANSFORM_PASS " + repr(predictions), flush=True)
        print("TASK_3_0_ACCEPTANCE_PASS", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
