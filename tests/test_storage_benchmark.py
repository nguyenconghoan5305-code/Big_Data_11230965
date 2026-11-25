"""Synthetic acceptance checks for Task 1.6; never mutate accepted FDIC files."""

from datetime import date
from decimal import Decimal
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.benchmarks import storage as b
from bft65b.config import EXTENDED_FINANCIAL_FIELDS


@pytest.fixture
def spec():
    return json.loads((Path(__file__).parents[1] / "config/storage_benchmark_v1.json").read_text())


@pytest.fixture
def table():
    values = {name: None for name in EXTENDED_FINANCIAL_FIELDS}
    values.update(CERT=123, REPDTE=date(2023, 3, 31), STNAME="Example, \"Bank\"",
                  ASSET=Decimal("1234.5000"), DEP=Decimal("0.0000"),
                  ROA=0.10000000000000002, NIMY=2.125, RBCRWAJ=None)
    return pa.Table.from_pylist([values], schema=b.schema(list(EXTENDED_FINANCIAL_FIELDS)))


def test_exact_logical_schema():
    result = b.schema(list(EXTENDED_FINANCIAL_FIELDS))
    assert len(result) == 26
    assert result.names == list(EXTENDED_FINANCIAL_FIELDS)
    assert str(result.field("ASSET").type) == "decimal128(20, 4)"
    assert str(result.field("REPDTE").type) == "date32[day]"


def test_exact_quarter_generation():
    quarters = b.quarter_range("19840331", "20260630")
    assert len(quarters) == 170 and len(set(quarters)) == 170
    assert quarters[0] == "19840331" and quarters[-1] == "20260630"


def test_isolated_output_roots(tmp_path, spec):
    data, evidence = b.locations(tmp_path, spec)
    assert data.is_relative_to(tmp_path / "data_processed")
    assert evidence.is_relative_to(tmp_path / "runs")
    spec["storage_root"] = "data_raw/unsafe"
    with pytest.raises(b.BenchmarkError):
        b.locations(tmp_path, spec)


def test_csv_date_decimal_float_null_roundtrip(tmp_path, table, spec):
    path = tmp_path / "roundtrip.csv"
    b.write_csv(table, path, spec)
    raw = path.read_bytes()
    assert b"\r\n" not in raw and b"__BFT65B_NULL__" in raw
    observed = b.read_file(path, "csv", list(EXTENDED_FINANCIAL_FIELDS), spec)
    b.check_parity(table, observed, "20230331")
    assert observed["DEP"][0].as_py() == Decimal("0.0000")
    assert observed["RBCRWAJ"][0].as_py() is None
    assert observed["ROA"][0].as_py() == 0.10000000000000002


@pytest.mark.parametrize("fmt", ["snappy", "zstd"])
def test_parquet_codecs_and_exact_parity(tmp_path, table, spec, fmt):
    path = tmp_path / f"data-{fmt}.parquet"
    pq.write_table(table, path, compression=fmt,
                   compression_level=3 if fmt == "zstd" else None)
    b.check_codec(path, fmt)
    b.check_parity(table, b.read_file(path, fmt, list(EXTENDED_FINANCIAL_FIELDS), spec), "20230331")
    with pytest.raises(b.BenchmarkError):
        b.check_codec(path, "snappy" if fmt == "zstd" else "zstd")


def test_column_order_and_row_value_mismatch(table):
    with pytest.raises(b.BenchmarkError, match="order"):
        b.check_parity(table, table.select(list(reversed(table.column_names))), "20230331")
    changed = table.set_column(table.schema.get_field_index("CERT"), table.schema.field("CERT"), pa.array([124], type=pa.int64()))
    with pytest.raises(b.BenchmarkError, match="Row-level"):
        b.check_parity(table, changed, "20230331")


def test_duplicate_and_wrong_quarter_rejected(table):
    with pytest.raises(b.BenchmarkError, match="Duplicate"):
        b.check_parity(pa.concat_tables([table, table]), pa.concat_tables([table, table]), "20230331")
    with pytest.raises(b.BenchmarkError, match="wrong-quarter"):
        b.check_parity(table, table, "20230630")
    with pytest.raises(b.BenchmarkError, match="Count"):
        b.check_parity(pa.concat_tables([table, table]), table, "20230331")


def test_null_vs_zero_rejected(table):
    changed = table.set_column(table.schema.get_field_index("RBCRWAJ"), "RBCRWAJ", pa.array([0.0]))
    with pytest.raises(b.BenchmarkError):
        b.check_parity(table, changed, "20230331")


def test_truncated_benchmark_file_rejected(tmp_path, table, spec):
    path = tmp_path / "broken.parquet"
    pq.write_table(table, path, compression="zstd")
    path.write_bytes(path.read_bytes()[:30])
    with pytest.raises((pa.ArrowException, OSError)):
        b.read_file(path, "zstd", list(EXTENDED_FINANCIAL_FIELDS), spec)


def test_hash_immutable_source_detection(tmp_path, monkeypatch, spec):
    source = tmp_path / "source.bin"
    source.write_bytes(b"healthy")
    original = [{"path": str(source), "sha256": b.digest(source)}]
    monkeypatch.setattr(b, "source_records", lambda root, config: [{"path": str(source), "sha256": b.digest(source)}])
    assert b.unchanged(tmp_path, spec, {"sources": original}) == original
    source.write_bytes(b"changed")
    with pytest.raises(b.BenchmarkError, match="changed"):
        b.unchanged(tmp_path, spec, {"sources": original})


def test_manifest_rejects_incomplete(tmp_path, spec):
    path = tmp_path / spec["source_manifest"]
    b.save_json(path, {"status": "failed"})
    b.save_json(tmp_path / spec["source_summary"], {"status": "complete"})
    with pytest.raises(b.BenchmarkError):
        b.source_records(tmp_path, spec)


def test_completed_preflight_evidence_is_write_once(tmp_path, spec):
    _, evidence = b.locations(tmp_path, spec)
    b.save_json(evidence / "source_inventory.json", {"status": "complete"})
    with pytest.raises(b.BenchmarkError, match="already exists"):
        b.preflight(tmp_path, spec)


def test_rotation_has_six_warmups_and_30_measures(spec):
    order = b.rotation(spec)
    assert len(order) == 36
    assert sum(x[3] for x in order) == 6
    assert len([x for x in order if not x[3]]) == 30
    assert order == b.rotation(spec)
    for workload in ("full", "selective"):
        for fmt in spec["formats"]:
            assert len([x for x in order if x[0] == fmt and x[1] == workload and not x[3]]) == 5


def test_size_speedup_formulas(spec):
    files = [{"format": fmt, "bytes": size} for fmt, size in (("csv", 100), ("snappy", 50), ("zstd", 25))]
    records = [{"format": fmt, "workload": workload, "duration_ns": duration,
                "files": 170 if workload == "full" else 40, "rows": 100, "columns": 26}
               for workload in ("full", "selective")
               for fmt, duration in (("csv", 20), ("snappy", 10), ("zstd", 5)) for _ in range(5)]
    summary = b.metrics(records, {"files": files}, spec)
    assert len(summary) == 6
    assert summary[1]["size_relative_to_csv_percent"] == 50
    assert summary[1]["speedup_relative_to_csv"] == 2
    assert summary[2]["speedup_relative_to_csv"] == 4
    with pytest.raises(b.BenchmarkError):
        b.metrics(records[:-1], {"files": files}, spec)


def test_synthetic_full_and_selective_read_parity(tmp_path, table, spec, monkeypatch):
    data = tmp_path / "data_processed/storage_benchmark_v1"
    evidence = tmp_path / "runs/task_1_6_storage"
    rows = []
    for quarter, cert, day in (("20230331", 123, date(2023, 3, 31)),
                               ("20230630", 456, date(2023, 6, 30))):
        record = table.to_pylist()[0]
        record.update(CERT=cert, REPDTE=day)
        current = pa.Table.from_pylist([record], schema=table.schema)
        source = tmp_path / f"source_{quarter}.parquet"
        pq.write_table(current, source, compression="zstd")
        rows.append({"quarter": quarter, "path": str(source), "rows": 1})
        for fmt in spec["formats"]:
            path = b.target(data, fmt, quarter)
            path.parent.mkdir(parents=True)
            if fmt == "csv":
                b.write_csv(current, path, spec)
            else:
                pq.write_table(current, path, compression=fmt)
    spec["full"]["rows"] = 2
    spec["selective"].update(first_quarter="20230331", last_quarter="20230630")
    inventory, frozen = {"sources": rows}, {"selective_expected_rows": 2}
    b.save_json(evidence / "representations.json", {"status": "synthetic"})
    monkeypatch.setattr(b, "loaded", lambda root, config: (inventory, frozen, data, evidence))
    monkeypatch.setattr(b, "representations", lambda root, config: (inventory, frozen, rows, data, evidence, {}))
    result = b.parity(tmp_path, spec)
    assert len(result["checks"]) == 6
    assert all(item["exact_row_values"] and item["rows"] == 2 for item in result["checks"])
    for workload, cols in (("full", 26), ("selective", 8)):
        for fmt in spec["formats"]:
            observed = b.worker(tmp_path, spec, fmt, workload)
            assert observed["rows"] == 2 and observed["columns"] == cols
