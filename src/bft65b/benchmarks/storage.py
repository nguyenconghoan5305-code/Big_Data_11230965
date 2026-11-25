"""Task 1.6 storage experiment on the accepted, immutable FDIC snapshot."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal
from statistics import median
from urllib.parse import parse_qs, urlsplit
from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from bft65b.acquisition.storage import ParquetStore, EXTENDED_FINANCIAL_BASE_FIELDS
from bft65b.config import EXTENDED_FINANCIAL_FIELDS, load_config


class BenchmarkError(RuntimeError):
    """A benchmark acceptance gate failed."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise BenchmarkError(f"Expected object: {path}")
    return result


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".benchmark-", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def quarter_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[4:6])
    result = []
    while len(result) <= 170:
        result.append(f"{year:04d}{month:02d}{ {3:31,6:30,9:30,12:31}[month]:02d}")
        if result[-1] == last:
            return result
        month += 3
        if month == 15:
            year, month = year + 1, 3
    raise BenchmarkError("Quarter range exceeds assignment")


def schema(fields: list[str]) -> pa.Schema:
    base = pa.schema(EXTENDED_FINANCIAL_BASE_FIELDS)
    return pa.schema([base.field(name) for name in fields])


def locations(root: Path, spec: dict) -> tuple[Path, Path]:
    data = (root / spec["storage_root"]).resolve()
    evidence = (root / spec["evidence_root"]).resolve()
    if not data.is_relative_to((root / "data_processed").resolve()) or not evidence.is_relative_to((root / "runs").resolve()):
        raise BenchmarkError("Benchmark outputs must be isolated in ignored data_processed and runs")
    return data, evidence


def source_records(root: Path, spec: dict) -> list[dict]:
    manifest, summary = read_json(root / spec["source_manifest"]), read_json(root / spec["source_summary"])
    quarters = quarter_range(spec["full"]["first_quarter"], spec["full"]["last_quarter"])
    if (manifest.get("status") != "complete" or summary.get("status") != "complete"
            or manifest.get("run_id") != summary.get("run_id")
            or len(quarters) != 170 or summary.get("planned_quarters") != quarters
            or len(manifest.get("outputs", [])) != 172 or len(manifest.get("sources", [])) != 172
            or len(summary.get("quarters", [])) != 170
            or tuple(manifest["config"].get("financial_fields", [])) != EXTENDED_FINANCIAL_FIELDS):
        raise BenchmarkError("Accepted 172-job manifest, summary, or 26-field contract differs")
    if (Counter(x.get("dataset") for x in manifest["outputs"]) !=
            Counter({"financials": 170, "failures": 1, "institutions": 1})
            or Counter(x.get("endpoint") for x in manifest["sources"]) !=
            Counter({"financials": 170, "failures": 1, "institutions": 1})
            or Counter(x.get("endpoint") for x in summary.get("small_tables", [])) !=
            Counter({"failures": 1, "institutions": 1})
            or any(not x.get("passed") for x in manifest.get("validations", []))):
        raise BenchmarkError("Accepted source has incomplete endpoint or validation evidence")
    outputs = {x["partition"].split("=")[1]: x for x in manifest["outputs"] if x["dataset"] == "financials"}
    sources = {parse_qs(urlsplit(x["query"]).query)["filters"][0].split(":", 1)[1]: x
               for x in manifest["sources"] if x["endpoint"] == "financials"}
    summaries = {x["quarter"]: x for x in summary["quarters"]}
    if set(outputs) != set(quarters) or set(sources) != set(quarters) or set(summaries) != set(quarters):
        raise BenchmarkError("Missing or duplicate Financials quarter")
    config = load_config(root / "config/financials_full_v2.json")
    store = ParquetStore(root / config.data_root, config)
    if (root / "runs/source_vintage_bundle.json").is_file():
        from bft65b.manifest import RunManifest
        from bft65b.quality.source_vintage import relocated_view

        relocated, _ = relocated_view(
            root, RunManifest.read(root / spec["source_manifest"]), summary, store
        )
        relocated_outputs = {
            item["partition"].split("=")[1]: item
            for item in relocated.outputs if item["dataset"] == "financials"
        }
    else:
        relocated_outputs = outputs
    result = []
    for quarter in quarters:
        item, source, reported = outputs[quarter], sources[quarter], summaries[quarter]
        path = Path(relocated_outputs[quarter]["path"]).resolve()
        if (path != store.target_path("financials", quarter).resolve()
                or item.get("schema_version") != "2" or item.get("compression") != "ZSTD"
                or item.get("rows") != source.get("api_total")
                or item.get("rows") != source.get("downloaded_rows")
                or item.get("rows") != item.get("source_total")
                or item.get("source_query") != source.get("query")
                or item.get("rows") != reported.get("api_total")
                or item.get("sha256") != reported.get("sha256")
                or item["path"] != reported.get("parquet_path")):
            raise BenchmarkError(f"Accepted source metadata differs: {quarter}")
        check = store.validate_file(path, "financials", quarter=quarter,
                                    expected_rows=item["rows"], expected_sha256=item["sha256"])
        result.append({"quarter": quarter, "path": str(path), "rows": check.rows,
                       "bytes": path.stat().st_size, "sha256": check.sha256})
    if sum(x["rows"] for x in result) != spec["full"]["rows"]:
        raise BenchmarkError("Accepted Financials total differs")
    return result


def source_table(item: dict, fields: list[str]) -> pa.Table:
    with pq.ParquetFile(item["path"]) as parquet:
        result = parquet.read(columns=fields, use_threads=False)
    result = result.select(fields).replace_schema_metadata(None)
    if not result.schema.equals(schema(fields), check_metadata=False):
        raise BenchmarkError(f"Source projected schema differs: {item['quarter']}")
    return result


def preflight(root: Path, spec: dict) -> dict:
    _, evidence = locations(root, spec)
    if (evidence / "source_inventory.json").exists():
        raise BenchmarkError(f"Preflight evidence already exists; use a new benchmark namespace: {evidence}")
    if len(spec["logical_fields"]) != 26 or tuple(spec["logical_fields"]) != EXTENDED_FINANCIAL_FIELDS:
        raise BenchmarkError("Benchmark must use the exact verified 26 fields")
    items = source_records(root, spec)
    select = spec["selective"]
    chosen = [x for x in items if select["first_quarter"] <= x["quarter"] <= select["last_quarter"]]
    if len(chosen) != 40 or chosen[0]["quarter"] != "20100331" or chosen[-1]["quarter"] != "20191231":
        raise BenchmarkError("Selective range must contain exactly 40 quarters")
    if len(select["fields"]) != 8 or not set(select["fields"]).issubset(spec["logical_fields"]):
        raise BenchmarkError("Invalid selective fields")
    for item in items:
        table = source_table(item, spec["logical_fields"])
        if pc.any(pc.equal(table["STNAME"], spec["csv"]["null_sentinel"])).as_py():
            raise BenchmarkError("CSV null sentinel collides with a source string")
        for name in ("ROA", "RBC1AAJ", "P3ASSET", "P9ASSET", "NAASSET", "NIMY", "RBCRWAJ"):
            if pc.any(pc.invert(pc.fill_null(pc.is_finite(table[name]), True))).as_py():
                raise BenchmarkError(f"Nonfinite float: {item['quarter']} {name}")
    inventory = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "spec_sha256": digest(root / spec.get("_spec_path", "config/storage_benchmark_v1.json")),
                 "manifest_sha256": digest(root / spec["source_manifest"]),
                 "summary_sha256": digest(root / spec["source_summary"]),
                 "selective_expected_rows": sum(x["rows"] for x in chosen), "sources": items}
    save_json(evidence / "source_inventory.json", inventory)
    frozen = {**spec, "selective_expected_rows": inventory["selective_expected_rows"],
              "logical_schema": [{"name": x.name, "type": str(x.type), "nullable": x.nullable}
                                 for x in schema(spec["logical_fields"])],
              "source_inventory_sha256": digest(evidence / "source_inventory.json")}
    save_json(evidence / "frozen_spec.json", frozen)
    return {"quarters": len(items), "rows": sum(x["rows"] for x in items),
            "selective_rows": frozen["selective_expected_rows"], "inventory": str(evidence / "source_inventory.json")}


def loaded(root: Path, spec: dict) -> tuple[dict, dict, Path, Path]:
    data, evidence = locations(root, spec)
    inventory, frozen = read_json(evidence / "source_inventory.json"), read_json(evidence / "frozen_spec.json")
    if (inventory.get("status") != "complete"
            or frozen.get("source_inventory_sha256") != digest(evidence / "source_inventory.json")
            or inventory.get("spec_sha256") != digest(root / spec.get("_spec_path", "config/storage_benchmark_v1.json"))
            or inventory.get("manifest_sha256") != digest(root / spec["source_manifest"])
            or inventory.get("summary_sha256") != digest(root / spec["source_summary"])):
        raise BenchmarkError("Frozen source or specification changed")
    return inventory, frozen, data, evidence


def unchanged(root: Path, spec: dict, inventory: dict) -> list[dict]:
    current = source_records(root, spec)
    if current != inventory["sources"]:
        raise BenchmarkError("Accepted raw input inventory changed")
    return current


def target(data: Path, fmt: str, quarter: str) -> Path:
    return data / fmt / f"REPDTE={quarter}" / f"part-000.{'csv' if fmt == 'csv' else 'parquet'}"


def write_csv(table: pa.Table, path: Path, spec: dict) -> None:
    null = spec["csv"]["null_sentinel"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(table.column_names)
        for batch in table.to_batches(max_chunksize=8192):
            columns = [col.to_pylist() for col in batch.columns]
            for values in zip(*columns):
                writer.writerow([null if v is None else repr(v) if isinstance(v, float) else
                                 str(v) if isinstance(v, Decimal) else
                                 v.isoformat() if hasattr(v, "isoformat") else v for v in values])


def read_file(path: Path, fmt: str, fields: list[str], spec: dict) -> pa.Table:
    if fmt == "csv":
        table = pacsv.read_csv(path,
            read_options=pacsv.ReadOptions(use_threads=False, block_size=1 << 20),
            parse_options=pacsv.ParseOptions(delimiter=",", quote_char='"', newlines_in_values=False),
            convert_options=pacsv.ConvertOptions(column_types=schema(spec["logical_fields"]),
                include_columns=fields, include_missing_columns=False,
                null_values=[spec["csv"]["null_sentinel"]], strings_can_be_null=True,
                quoted_strings_can_be_null=True, check_utf8=True, auto_dict_encode=False))
    else:
        with pq.ParquetFile(path) as parquet:
            table = parquet.read(columns=fields, use_threads=False)
    table = table.select(fields).replace_schema_metadata(None)
    expected = schema(fields)
    if [field.type for field in table.schema] != [field.type for field in expected]:
        raise BenchmarkError(f"Read schema differs: {path}")
    # Arrow CSV parsing has no non-nullable field declaration; enforce the
    # frozen field nullability after verifying physical logical types.
    return table.cast(expected, safe=True)


def check_codec(path: Path, fmt: str) -> None:
    if fmt == "csv":
        return
    with pq.ParquetFile(path) as parquet:
        codecs = {parquet.metadata.row_group(i).column(j).compression
                  for i in range(parquet.metadata.num_row_groups)
                  for j in range(parquet.metadata.row_group(i).num_columns)}
    if codecs != {fmt.upper()}:
        raise BenchmarkError(f"Codec mismatch: {path} {codecs}")


def check_parity(expected: pa.Table, actual: pa.Table, quarter: str) -> None:
    if (expected.num_rows != actual.num_rows or expected.column_names != actual.column_names
            or not expected.schema.equals(actual.schema, check_metadata=False)):
        raise BenchmarkError(f"Count, order, or type mismatch: {quarter}")
    if not expected.equals(actual, check_metadata=False):
        differing = [name for name in expected.column_names if not expected[name].equals(actual[name])]
        raise BenchmarkError(f"Row-level value mismatch: {quarter} {differing}")
    keys = list(zip(actual["CERT"].to_pylist(), actual["REPDTE"].to_pylist()))
    if len(keys) != len(set(keys)) or any(day.strftime("%Y%m%d") != quarter for _, day in keys):
        raise BenchmarkError(f"Duplicate or wrong-quarter key: {quarter}")
    if any(expected[name].null_count != actual[name].null_count for name in expected.column_names):
        raise BenchmarkError(f"Null-count mismatch: {quarter}")


def prepare(root: Path, spec: dict) -> dict:
    inventory, _, data, evidence = loaded(root, spec)
    items = unchanged(root, spec, inventory)
    files = []
    for item in items:
        expected = source_table(item, spec["logical_fields"])
        quarter = item["quarter"]
        for fmt in spec["formats"]:
            path = target(data, fmt, quarter)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                check_parity(expected, read_file(path, fmt, spec["logical_fields"], spec), quarter)
                check_codec(path, fmt)
            else:
                with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".part-000.", suffix=".tmp", delete=False) as stream:
                    temp = Path(stream.name)
                try:
                    if fmt == "csv":
                        write_csv(expected, temp, spec)
                    else:
                        opts = spec["parquet"]
                        pq.write_table(expected, temp, compression=fmt,
                            compression_level=opts["zstd_level"] if fmt == "zstd" else None,
                            row_group_size=opts["row_group_size"], version=opts["version"],
                            use_dictionary=opts["use_dictionary"], write_statistics=opts["write_statistics"],
                            data_page_version=opts["data_page_version"], data_page_size=opts["data_page_size"],
                            write_page_index=opts["write_page_index"],
                            use_byte_stream_split=opts["use_byte_stream_split"])
                    check_parity(expected, read_file(temp, fmt, spec["logical_fields"], spec), quarter)
                    check_codec(temp, fmt)
                    os.replace(temp, path)
                finally:
                    temp.unlink(missing_ok=True)
            files.append({"format": fmt, "quarter": quarter, "path": str(path),
                          "rows": expected.num_rows, "bytes": path.stat().st_size, "sha256": digest(path)})
    save_json(evidence / "representations.json", {"status": "complete", "files": files,
        "inventory_sha256": digest(evidence / "source_inventory.json"),
        "spec_sha256": digest(evidence / "frozen_spec.json")})
    return {"files": len(files), "bytes_by_format": {fmt: sum(x["bytes"] for x in files if x["format"] == fmt)
                                                  for fmt in spec["formats"]}}


def representations(root: Path, spec: dict) -> tuple[dict, dict, list[dict], Path, Path, dict]:
    inventory, frozen, data, evidence = loaded(root, spec)
    reps = read_json(evidence / "representations.json")
    items = unchanged(root, spec, inventory)
    expected = {(fmt, x["quarter"]): target(data, fmt, x["quarter"])
                for x in items for fmt in spec["formats"]}
    if (reps.get("status") != "complete" or len(reps.get("files", [])) != 510
            or reps.get("inventory_sha256") != digest(evidence / "source_inventory.json")
            or reps.get("spec_sha256") != digest(evidence / "frozen_spec.json")
            or {(x["format"], x["quarter"]): Path(x["path"]) for x in reps["files"]} != expected):
        raise BenchmarkError("Benchmark representation inventory incomplete")
    for file in reps["files"]:
        path = Path(file["path"])
        if not path.is_file() or path.stat().st_size != file["bytes"] or digest(path) != file["sha256"]:
            raise BenchmarkError(f"Missing/corrupt benchmark file: {path}")
        check_codec(path, file["format"])
    return inventory, frozen, items, data, evidence, reps


def parity(root: Path, spec: dict) -> dict:
    _, frozen, items, data, evidence, _ = representations(root, spec)
    results = []
    for workload in ("full", "selective"):
        fields = spec["logical_fields"] if workload == "full" else spec["selective"]["fields"]
        chosen = [x for x in items if workload == "full" or spec["selective"]["first_quarter"] <= x["quarter"] <= spec["selective"]["last_quarter"]]
        for fmt in spec["formats"]:
            count = 0
            for item in chosen:
                expected = source_table(item, fields)
                actual = read_file(target(data, fmt, item["quarter"]), fmt, fields, spec)
                check_parity(expected, actual, item["quarter"])
                count += actual.num_rows
            if count != (spec["full"]["rows"] if workload == "full" else frozen["selective_expected_rows"]):
                raise BenchmarkError(f"Parity row count differs: {fmt}/{workload}")
            results.append({"format": fmt, "workload": workload, "quarters": len(chosen),
                            "rows": count, "columns": len(fields), "exact_row_values": True,
                            "null_counts_equal": True, "keys_unique": True, "max_float_difference": 0.0})
    result = {"status": "complete", "checks": results,
              "representations_sha256": digest(evidence / "representations.json")}
    save_json(evidence / "parity.json", result)
    return result


def peak_rss() -> int:
    if os.name != "nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_current = ctypes.windll.kernel32.GetCurrentProcess
    get_current.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
    get_memory.restype = ctypes.c_int
    if not get_memory(get_current(), ctypes.byref(counters), counters.cb):
        raise BenchmarkError("GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def worker(root: Path, spec: dict, fmt: str, workload: str) -> dict:
    inventory, frozen, data, _ = loaded(root, spec)
    fields = spec["logical_fields"] if workload == "full" else spec["selective"]["fields"]
    selected = [x for x in inventory["sources"] if workload == "full" or
                spec["selective"]["first_quarter"] <= x["quarter"] <= spec["selective"]["last_quarter"]]
    gc.collect()
    start = time.perf_counter_ns()
    tables = [read_file(target(data, fmt, x["quarter"]), fmt, fields, spec) for x in selected]
    table = pa.concat_tables(tables)
    duration = time.perf_counter_ns() - start
    expected = spec["full"]["rows"] if workload == "full" else frozen["selective_expected_rows"]
    if table.num_rows != expected or table.num_columns != len(fields):
        raise BenchmarkError("Timed read did not materialize expected result")
    return {"format": fmt, "workload": workload, "files": len(selected),
            "rows": table.num_rows, "columns": table.num_columns,
            "duration_ns": duration, "peak_rss_bytes": peak_rss()}


def rotation(spec: dict) -> list[tuple[str, str, int, bool]]:
    formats = spec["formats"]
    order = []
    for workload in ("full", "selective"):
        order.extend((fmt, workload, 0, True) for fmt in formats)
        for rep in range(spec["benchmark"]["repetitions_per_combination"]):
            order.extend((fmt, workload, rep + 1, False)
                         for fmt in formats[rep % 3:] + formats[:rep % 3])
    return order


def metrics(records: list[dict], reps: dict, spec: dict) -> list[dict]:
    sizes = {fmt: sum(x["bytes"] for x in reps["files"] if x["format"] == fmt) for fmt in spec["formats"]}
    result = []
    for workload in ("full", "selective"):
        baseline = [x["duration_ns"] for x in records if x["format"] == "csv" and x["workload"] == workload]
        if len(baseline) != 5:
            raise BenchmarkError("Incomplete CSV baseline")
        for fmt in spec["formats"]:
            group = [x for x in records if x["format"] == fmt and x["workload"] == workload]
            if len(group) != 5:
                raise BenchmarkError("Incomplete measured repetitions")
            values = [x["duration_ns"] for x in group]
            result.append({"format": fmt, "workload": workload, "files_stored": 170,
                "files_read": group[0]["files"], "rows_read": group[0]["rows"],
                "columns_read": group[0]["columns"], "stored_bytes": sizes[fmt],
                "size_relative_to_csv_percent": 100 * sizes[fmt] / sizes["csv"],
                "durations_seconds": [x / 1e9 for x in values],
                "median_seconds": median(values) / 1e9, "min_seconds": min(values) / 1e9,
                "max_seconds": max(values) / 1e9,
                "speedup_relative_to_csv": median(baseline) / median(values)})
    return result


def environment(root: Path) -> dict:
    import importlib.metadata as metadata
    disk = shutil.disk_usage(root)
    result = {"os": platform.platform(), "cpu": platform.processor(),
              "logical_processors": os.cpu_count(), "python": platform.python_version(),
              "disk_device": str(root.drive), "disk_total_bytes": disk.total,
              "disk_free_bytes": disk.free, "pyarrow_cpu_count": pa.cpu_count(),
              "libraries": {name: metadata.version(name) for name in
                  ("pyarrow", "pandas", "polars", "pyspark", "numpy", "scikit-learn", "requests")}}
    if os.name == "nt":
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        memory = Memory()
        memory.dwLength = ctypes.sizeof(memory)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        result["ram_bytes"] = int(memory.ullTotalPhys)
    return result


def run(root: Path, spec: dict) -> dict:
    inventory, _, _, _, evidence, reps = representations(root, spec)
    existing = evidence / "timings.json"
    if existing.exists():
        raise BenchmarkError(f"Timing attempt already exists; preserve its evidence: {existing}")
    checked = read_json(evidence / "parity.json")
    if (checked.get("status") != "complete" or len(checked.get("checks", [])) != 6
            or checked.get("representations_sha256") != digest(evidence / "representations.json")):
        raise BenchmarkError("All six parity combinations must pass before timing")
    unchanged(root, spec, inventory)
    order = rotation(spec)
    evidence_record = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(root), "frozen_spec_sha256": digest(evidence / "frozen_spec.json"),
        "parity_sha256": digest(evidence / "parity.json"),
        "execution_order": [{"format": f, "workload": w, "repetition": r, "warmup": u} for f,w,r,u in order],
        "warmups": [], "records": [], "failures": []}
    save_json(evidence / "timings.json", evidence_record)
    for fmt, workload, rep, warm in order:
        command = [sys.executable, "-m", "bft65b.benchmarks.storage", "--project-root", str(root),
                   "--spec", spec.get("_spec_path", "config/storage_benchmark_v1.json"),
                   "worker", "--format", fmt, "--workload", workload]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        process = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        if process.returncode:
            evidence_record["failures"].append({"format": fmt, "workload": workload,
                "repetition": rep, "warmup": warm, "returncode": process.returncode,
                "stderr": process.stderr[-8000:]})
            evidence_record["status"] = "failed"
            save_json(evidence / "timings.json", evidence_record)
            raise BenchmarkError(f"Worker failed: {fmt}/{workload}/{rep}: {process.stderr[-500:]}")
        item = json.loads(process.stdout)
        item.update({"repetition": rep, "warmup": warm})
        evidence_record["warmups" if warm else "records"].append(item)
        save_json(evidence / "timings.json", evidence_record)
    if len(evidence_record["warmups"]) != 6 or len(evidence_record["records"]) != 30:
        raise BenchmarkError("Incomplete warm-up or measured timing record")
    evidence_record["summary"] = metrics(evidence_record["records"], reps, spec)
    try:
        representations(root, spec)
        unchanged(root, spec, inventory)
        save_json(evidence / "source_postcheck.json", {"status": "complete", "files": 170,
            "rows": spec["full"]["rows"], "hashes_unchanged": True,
            "source_inventory_sha256": digest(evidence / "source_inventory.json")})
    except Exception as exc:
        evidence_record["failures"].append({"phase": "post_measurement_integrity", "error": str(exc)})
        evidence_record["status"] = "failed"
        save_json(evidence / "timings.json", evidence_record)
        raise
    evidence_record["status"] = "complete"
    evidence_record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(evidence / "timings.json", evidence_record)
    return {"measured": 30, "warmups": 6, "summary": evidence_record["summary"]}


def verify(root: Path, spec: dict) -> dict:
    inventory, _, _, _, evidence, reps = representations(root, spec)
    parity_record = read_json(evidence / "parity.json")
    timing = read_json(evidence / "timings.json")
    postcheck = read_json(evidence / "source_postcheck.json")
    if (parity_record.get("status") != "complete" or len(parity_record.get("checks", [])) != 6
            or parity_record.get("representations_sha256") != digest(evidence / "representations.json")
            or timing.get("status") != "complete" or timing.get("failures")
            or timing.get("parity_sha256") != digest(evidence / "parity.json")
            or timing.get("frozen_spec_sha256") != digest(evidence / "frozen_spec.json")
            or len(timing.get("warmups", [])) != 6 or len(timing.get("records", [])) != 30
            or timing.get("summary") != metrics(timing["records"], reps, spec)
            or postcheck.get("status") != "complete" or not postcheck.get("hashes_unchanged")
            or postcheck.get("source_inventory_sha256") != digest(evidence / "source_inventory.json")
            or any(item.get("duration_ns", 0) <= 0 or item.get("peak_rss_bytes", 0) <= 0
                   for item in timing["records"])):
        raise BenchmarkError("Stored parity, timing, or immutable-source evidence is incomplete")
    unchanged(root, spec, inventory)
    return {"status": "PASS", "source_files": 170, "benchmark_files": 510,
            "parity_combinations": 6, "warmups": 6, "measured_reads": 30,
            "source_hashes_unchanged": True, "summary": timing["summary"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Task 1.6 immutable-source storage benchmark")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--spec", type=Path, default=Path("config/storage_benchmark_v1.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "prepare", "parity", "run", "verify", "show"):
        commands.add_parser(command)
    sub = commands.add_parser("worker", help=argparse.SUPPRESS)
    sub.add_argument("--format", choices=("csv", "snappy", "zstd"), required=True)
    sub.add_argument("--workload", choices=("full", "selective"), required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    spec = read_json(root / args.spec)
    spec["_spec_path"] = str(args.spec)
    if args.command == "preflight":
        result = preflight(root, spec)
    elif args.command == "prepare":
        result = prepare(root, spec)
    elif args.command == "parity":
        result = parity(root, spec)
    elif args.command == "run":
        result = run(root, spec)
    elif args.command == "verify":
        result = verify(root, spec)
    elif args.command == "worker":
        result = worker(root, spec, args.format, args.workload)
    else:
        _, evidence = locations(root, spec)
        result = read_json(evidence / "timings.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
