"""Shared fresh-process benchmark for the frozen observed-panel workload.

The same sink and timing boundary cover pandas, Polars, and local PySpark.
Parent-side source verification and output parity are untimed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from bft65b.benchmarks.process_memory import sample_tree_rss
from bft65b.processing.polars_panel import (
    INPUT_COLUMNS, WORKLOAD_ID, _accepted_result, compare, lazy_queries, materialize,
)
from bft65b.quality.audit import _read_json, _sha, _write_json


REFERENCE_RUN = "polars20261115a"
NAMESPACE = "task_3_2_panel_benchmark"
ENGINES = ("pandas", "polars")
THREE_ENGINES = ("pandas", "polars", "spark")
SAMPLE_SECONDS = 0.05
THREADS = 4
THREAD_ENV = ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SCHEMAS = {
    "observed": pa.schema([
        ("CERT", pa.int64()), ("REPDTE", pa.date32()), ("ID", pa.large_string()),
        ("P3ASSET", pa.float64()),
        ("SOURCE_MISSING_FIELDS", pa.large_list(pa.large_string())),
        ("DQ_FLAGS", pa.large_list(pa.large_string())),
    ]),
    "gaps": pa.schema([
        ("CERT", pa.int64()), ("PREVIOUS_REPDTE", pa.date32()),
        ("NEXT_REPDTE", pa.date32()), ("MISSING_INTERVENING_QUARTERS", pa.int32()),
    ]),
    "quality": pa.schema([
        ("REPDTE", pa.date32()), ("ROW_COUNT", pa.uint32()),
        ("DISTINCT_KEY_COUNT", pa.uint32()), ("P3ASSET_NULL_ROWS", pa.uint32()),
        ("P3ASSET_SOURCE_ABSENT_ROWS", pa.uint32()),
        ("P3ASSET_EXPLICIT_NULL_ROWS", pa.uint32()),
        ("UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS", pa.uint32()),
    ]),
}


def paths_for(root: Path, run_id: str) -> tuple[Path, Path]:
    if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
        raise ValueError("Invalid benchmark run ID")
    return (root / "runs" / NAMESPACE / run_id,
            root / "data_processed" / NAMESPACE / run_id)


def input_contract(root: Path, reference_run: str = REFERENCE_RUN) -> dict[str, Any]:
    """Independently verify accepted P3.1 manifest, inventory and source bytes."""
    if not reference_run or not all(c.isalnum() or c in '_-' for c in reference_run):
        raise ValueError("Invalid benchmark reference run ID")
    if (root / "runs/source_vintage_bundle.json").is_file() and reference_run == REFERENCE_RUN:
        raise ValueError("Portable benchmark requires an explicit portable reference")
    _accepted_result(root, reference_run)
    contract_path = root / "runs" / "task_3_1_polars" / reference_run / "contract.json"
    contract = _read_json(contract_path)
    files = contract.get("input_files", [])
    if (contract.get("status") != "frozen_after_parity"
            or contract.get("workload_id") != WORKLOAD_ID
            or contract.get("input_columns") != list(INPUT_COLUMNS)
            or len(files) != 170
            or len({row["quarter"] for row in files}) != 170
            or sum(row["rows"] for row in files) != 1682616):
        raise ValueError("Frozen P3.1 workload contract differs")
    for name, schema in SCHEMAS.items():
        reference = root / "data_processed" / "task_3_1_polars" / reference_run / f"{name}.parquet"
        if (_sha(reference) != contract["expected_outputs"][name]["sha256"]
                or not pq.ParquetFile(reference).schema_arrow.equals(schema)):
            raise ValueError(f"Frozen {name} reference/schema differs")
    for row in files:
        path = Path(row["path"])
        if (not path.resolve().is_relative_to(root.resolve())
                or path.name != "part-000.parquet" or path.parent.name != f"REPDTE={row['quarter']}"
                or _sha(path) != row["sha256"]
                or pq.ParquetFile(path).metadata.num_rows != row["rows"]):
            raise ValueError(f"Frozen input differs: {row['quarter']}")
    return contract


def worker(engine: str, contract_path: Path, output: Path) -> None:
    """All real transformations, full materialization, and three writes."""
    pa.set_cpu_count(THREADS)
    contract = _read_json(contract_path)
    paths = [Path(row["path"]) for row in contract["input_files"]]
    quarters = [row["quarter"] for row in contract["input_files"]]
    spark = None
    try:
        if engine == "pandas":
            from bft65b.processing.pandas_panel import to_arrow, transform
            tables = to_arrow(transform(paths), SCHEMAS)
        elif engine == "polars":
            tables = {name: frame.to_arrow() for name, frame in
                      materialize(lazy_queries(paths), quarters).items()}
        elif engine == "spark":
            from bft65b.processing.spark_panel import (
                materialize as spark_materialize, queries as spark_queries,
                runtime_settings, start_spark,
            )
            spark = start_spark(output.parent / f"{output.name}_spark_local")
            plan_dir = os.environ.get("BFT65B_SPARK_PLAN_DIR")
            if plan_dir:
                target = Path(plan_dir)
                target.mkdir(parents=True, exist_ok=False)
                (target / "settings.json").write_text(
                    json.dumps(runtime_settings(spark), indent=2), encoding="utf-8")
            tables = spark_materialize(spark_queries(spark, paths), quarters, SCHEMAS,
                                       Path(plan_dir) if plan_dir else None)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
        output.mkdir(parents=True, exist_ok=False)
        for name in ("observed", "gaps", "quality"):
            table = tables[name]
            if not table.schema.equals(SCHEMAS[name]) or table.num_rows != contract["expected_outputs"][name]["rows"]:
                raise ValueError(f"{engine} {name} output schema/row count differs")
            pq.write_table(table, output / f"{name}.parquet", compression="zstd",
                           compression_level=3, row_group_size=65536,
                           use_dictionary=True, write_statistics=True, version="2.6")
    finally:
        if spark is not None:
            spark.stop()


def run_order(engines: tuple[str, ...] = ENGINES, repetitions: int = 5) -> list[tuple[int, str]]:
    """Deterministic round-robin engine rotation, independent of results."""
    if repetitions < 5 or not engines or len(set(engines)) != len(engines):
        raise ValueError("At least five repetitions and unique engines are required")
    return [(round_index + 1, engines[(offset + round_index) % len(engines)])
            for round_index in range(repetitions) for offset in range(len(engines))]


def summarize(records: list[dict[str, Any]], engines: tuple[str, ...] = ENGINES,
              repetitions: int = 5) -> dict[str, Any]:
    result = {}
    for engine in engines:
        selected = [row for row in records if row["engine"] == engine and row["kind"] == "measured"]
        if len(selected) != repetitions or any(row["status"] != "pass" for row in selected):
            raise ValueError(f"Incomplete measured runs for {engine}")
        durations = [row["duration_ns"] for row in selected]
        peaks = [row["peak_tree_rss_bytes"] for row in selected]
        result[engine] = {
            "duration_ns": durations, "peak_tree_rss_bytes": peaks,
            "median_duration_ns": statistics.median(durations),
            "min_duration_ns": min(durations), "max_duration_ns": max(durations),
            "median_peak_tree_rss_bytes": statistics.median(peaks),
            "min_peak_tree_rss_bytes": min(peaks), "max_peak_tree_rss_bytes": max(peaks),
        }
    if "pandas" in result:
        baseline = result["pandas"]
        for row in result.values():
            row["speedup_vs_pandas_median"] = baseline["median_duration_ns"] / row["median_duration_ns"]
            row["memory_ratio_vs_pandas_median"] = row["median_peak_tree_rss_bytes"] / baseline["median_peak_tree_rss_bytes"]
    return result


def compare_diagnostic(root: Path, candidate_dir: Path, reference_run: str = REFERENCE_RUN) -> dict[str, Any]:
    """Exact parity, with the first differing relation/row/field on failure."""
    try:
        return compare(root, reference_run, candidate_dir)
    except ValueError as exc:
        reference_dir = root / "data_processed" / "task_3_1_polars" / reference_run
        for name in ("observed", "gaps", "quality"):
            left_path = reference_dir / f"{name}.parquet"
            right_path = candidate_dir / f"{name}.parquet"
            if not right_path.exists():
                raise ValueError(f"{name}: candidate file missing: {right_path}") from exc
            left = pl.read_parquet(left_path, hive_partitioning=False)
            right = pl.read_parquet(right_path, hive_partitioning=False)
            if left.schema != right.schema or left.columns != right.columns:
                raise ValueError(f"{name}: schema differs: expected={left.schema}, actual={right.schema}") from exc
            if len(left) != len(right):
                raise ValueError(f"{name}: row count differs: expected={len(left)}, actual={len(right)}") from exc
            if left.equals(right, null_equal=True):
                continue
            for index, (a, b) in enumerate(zip(left.iter_rows(), right.iter_rows())):
                if a != b:
                    column = next(col for col, expected, actual in zip(left.columns, a, b)
                                  if expected != actual)
                    expected = repr(a[left.columns.index(column)])[:160]
                    actual = repr(b[left.columns.index(column)])[:160]
                    raise ValueError(f"{name}: first mismatch row={index} field={column} "
                                     f"expected={expected} actual={actual}") from exc
        raise ValueError(f"Exact comparator failed without a row mismatch: {exc}") from exc


def _run_process(root: Path, engine: str, contract_path: Path, output: Path,
                 log_dir: Path, kind: str, round_number: int,
                 plan_dir: Path | None = None) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: str(THREADS) for key in THREAD_ENV})
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if engine == "spark":
        native = root / "tmp" / "hadoop-3.3.5"
        env["HADOOP_HOME"] = str(native)
        env["PATH"] = str(native / "bin") + os.pathsep + env["PATH"]
        env["PYSPARK_PYTHON"] = sys.executable
        env["PYSPARK_DRIVER_PYTHON"] = sys.executable
        if plan_dir is not None:
            env["BFT65B_SPARK_PLAN_DIR"] = str(plan_dir)
    command = [sys.executable, "-m", "bft65b.benchmarks.panel", "_worker", engine,
               str(contract_path), str(output)]
    stdout_path, stderr_path = log_dir / "stdout.txt", log_dir / "stderr.txt"
    peak = samples = max_processes = unreadable = 0
    stop = threading.Event()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        start = time.perf_counter_ns()
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)

        def sample() -> None:
            nonlocal peak, samples, max_processes, unreadable
            while not stop.is_set():
                try:
                    rss, processes, missed = sample_tree_rss(process.pid)
                    peak = max(peak, rss)
                    max_processes = max(max_processes, processes)
                    unreadable += missed
                    samples += 1
                except OSError:
                    unreadable += 1
                stop.wait(SAMPLE_SECONDS)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        code = process.wait()
        elapsed = time.perf_counter_ns() - start
        stop.set()
        sampler.join()
    row = {"engine": engine, "kind": kind, "round": round_number,
           "command": command, "status": "pass" if code == 0 else "failed",
           "exit_code": code, "duration_ns": elapsed,
           "peak_tree_rss_bytes": peak, "memory_samples": samples,
           "max_processes_seen": max_processes, "unreadable_process_samples": unreadable,
           "stdout": str(stdout_path), "stderr": str(stderr_path), "output_dir": str(output)}
    if code != 0 or peak <= 0 or samples <= 0:
        row["status"] = "failed"
        raise RuntimeError(f"{engine} {kind} round {round_number} failed: {row}")
    return row


def _environment(root: Path, contract: dict[str, Any],
                 engines: tuple[str, ...] = ENGINES) -> dict[str, Any]:
    versions = {}
    for name in ("pandas", "polars", "pyarrow", "numpy", "pyspark"):
        versions[name] = importlib.metadata.version(name)
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    ram = None
    if sys.platform == "win32":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        value = MemoryStatus()
        value.length = ctypes.sizeof(value)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
        ram = value.total_physical
    elif sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = int(line.split()[1]) * 1024
                break
    return {"os": platform.platform(), "python": platform.python_version(),
            "executable": sys.executable, "versions": versions, "cpu": cpu,
            "logical_processors": os.cpu_count(), "ram_bytes": ram,
            "free_disk_bytes_before": shutil.disk_usage(root).free,
            "input_files": len(contract["input_files"]),
            "input_rows": sum(row["rows"] for row in contract["input_files"]),
            "input_stored_bytes": sum(Path(row["path"]).stat().st_size for row in contract["input_files"]),
            "threads_per_worker_configured": THREADS, "thread_environment": {key: str(THREADS) for key in THREAD_ENV},
            "pyarrow_cpu_count_per_worker": THREADS,
            "memory": {"metric": "sampled sum of process-tree resident working sets",
                       "unit": "bytes", "interval_ms": int(SAMPLE_SECONDS * 1000),
                       "limitation": "sampling can miss short-lived peaks and processes"},
            "timer": "time.perf_counter_ns monotonic; subprocess startup through completed output writes",
            "cache": "OS file cache not flushed or controlled; repeated warm-cache conditions, not cold-cache",
            "process_isolation": "new interpreter process for each warm-up and measured run",
            "writer": {"format": "Parquet", "compression": "zstd", "level": 3,
                       "row_group_size": 65536, "dictionary": True, "statistics": True, "version": "2.6"},
            "ordering": {"warmup_engines": list(engines),
                         "measured": [{"round": number, "engine": engine}
                                      for number, engine in run_order(engines)],
                         "randomness": "none"}}


def execute(root: Path, run_id: str,
            engines: tuple[str, ...] = ENGINES, *, reference_run: str = REFERENCE_RUN) -> dict[str, Any]:
    if engines not in (ENGINES, THREE_ENGINES):
        raise ValueError("Select the accepted two- or three-engine protocol")
    evidence, data = paths_for(root, run_id)
    if evidence.exists() or data.exists():
        raise FileExistsError("Benchmark run ID already exists; evidence is immutable")
    contract = input_contract(root, reference_run)
    if "spark" in engines:
        native = root / "tmp" / "hadoop-3.3.5" / "bin"
        required_hashes = {
            "winutils.exe": "a0ca6e358357c41ef56ebdb02c38e4a4d55da7ca7a13001678bb2ef7d644adea",
            "hadoop.dll": "d3dd64afdc85f2a7eb5345abf2ecaa744b0a157de40859313337d47f81ee1c7b",
        }
        if any(_sha(native / name) != digest for name, digest in required_hashes.items()):
            raise ValueError("Spark native runtime hash differs from Task 3.0")
    contract_path = root / "runs" / "task_3_1_polars" / reference_run / "contract.json"
    evidence.mkdir(parents=True)
    data.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    status = "failed"
    try:
        parity = {}
        for engine in engines:
            output = data / "preflight" / engine
            row = _run_process(root, engine, contract_path, output,
                               evidence / "logs" / "preflight" / engine, "preflight", 0,
                               evidence / "plans" if engine == "spark" else None)
            parity[engine] = compare_diagnostic(root, output, reference_run)
            records.append(row)
        _write_json(evidence / "parity.json", parity)
        environment = _environment(root, contract, engines)
        environment["engines"] = list(engines)
        if "spark" in engines:
            environment["spark_runtime"] = _read_json(evidence / "plans" / "settings.json")
        _write_json(evidence / "environment.json", environment)
        for engine in engines:
            row = _run_process(root, engine, contract_path,
                               data / "warmup" / engine,
                               evidence / "logs" / "warmup" / engine, "warmup", 0)
            row["parity"] = compare_diagnostic(root, Path(row["output_dir"]), reference_run)
            records.append(row)
            _write_json(evidence / "records.json", {"status": "running", "records": records})
        for index, (round_number, engine) in enumerate(run_order(engines)):
            row = _run_process(root, engine, contract_path,
                               data / "measured" / f"{index+1:02d}_{engine}",
                               evidence / "logs" / "measured" / f"{index+1:02d}_{engine}",
                               "measured", round_number)
            row["parity"] = compare_diagnostic(root, Path(row["output_dir"]), reference_run)
            records.append(row)
            _write_json(evidence / "records.json", {"status": "running", "records": records})
        result = summarize(records, engines)
        input_contract(root, reference_run)  # Ensure the accepted source and references remained unchanged.
        summary = {"status": "complete", "run_id": run_id,
                   "reference_run": reference_run, "workload_id": WORKLOAD_ID,
                   "parity": parity, "measured_runs": len(engines) * 5,
                   "by_engine": result, "environment": str(evidence / "environment.json"),
                   "records": str(evidence / "records.json"),
                   "input_contract_sha256": _sha(contract_path),
                   "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_json(evidence / "summary.json", summary)
        status = "complete"
        return summary
    except Exception as exc:
        _write_json(evidence / "failure.json", {"status": "failed", "run_id": run_id,
                    "reason": f"{type(exc).__name__}: {exc}", "records": records})
        raise
    finally:
        _write_json(evidence / "records.json", {"status": status, "records": records})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P3.2 shared observed-panel benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "show"):
        part = sub.add_parser(name)
        part.add_argument("--project-root", type=Path, default=Path.cwd())
        part.add_argument("--run-id", required=True)
        if name == "run":
            part.add_argument("--reference-run", default=REFERENCE_RUN)
            part.add_argument("--include-spark", action="store_true",
                              help="Rerun pandas and Polars with Spark under the shared protocol")
    internal = sub.add_parser("_worker", help=argparse.SUPPRESS)
    internal.add_argument("engine", choices=THREE_ENGINES)
    internal.add_argument("contract", type=Path)
    internal.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        worker(args.engine, args.contract, args.output)
    elif args.command == "run":
        selected = THREE_ENGINES if args.include_spark else ENGINES
        print(json.dumps(execute(args.project_root.resolve(), args.run_id, selected, reference_run=args.reference_run), indent=2))
    else:
        evidence, _ = paths_for(args.project_root.resolve(), args.run_id)
        summary = _read_json(evidence / "summary.json")
        records = _read_json(evidence / "records.json")
        if summary.get("status") != "complete" or records.get("status") != "complete":
            raise ValueError("Benchmark evidence is not complete")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
