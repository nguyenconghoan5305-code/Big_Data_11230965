"""Real PyArrow round trips on synthetic FDIC-shaped records."""

from datetime import date
from decimal import Decimal
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bft65b.acquisition import storage
from bft65b.acquisition.storage import (
    ParquetStore, StorageConflictError, StorageValidationError, arrow_schema,
)
from bft65b.config import ProjectConfig
from bft65b.manifest import RunManifest


def context(tmp_path, config=None):
    config = config or ProjectConfig()
    store = ParquetStore(tmp_path / "data_raw", config)
    manifest = RunManifest.new("synthetic-storage", config)
    manifest_path = tmp_path / "runs" / "manifest.json"
    return store, manifest, manifest_path


def financial(cert=14, **changes):
    row = {
        "CERT": cert, "REPDTE": "20230331", "STNAME": "MASSACHUSETTS",
        "ASSET": 287069000, "DEP": "230488000.2500", "EQ": 26373000,
        "NETINC": -12.5, "ROA": 0.842265, "NCLNLS": None,
        "LNATRES": 115000, "RBC1AAJ": 6.76, "ID": f"{cert}_20230331",
    }
    row.update(changes)
    return row


def write(store, manifest, manifest_path, dataset, rows, *, quarter=None):
    return store.write(
        dataset, rows, source_total=len(rows), quarter=quarter,
        manifest=manifest, manifest_path=manifest_path,
        source_query=f"synthetic:{dataset}",
    )


def test_financial_round_trip_schema_nulls_codec_partition_and_manifest(tmp_path):
    store, manifest, manifest_path = context(tmp_path)
    row = financial()
    del row["LNATRES"]  # Absent is distinct from present JSON null in NCLNLS.
    result = write(store, manifest, manifest_path, "financials", [row], quarter="20230331")
    assert result.path == tmp_path / "data_raw/fdic/financials/REPDTE=20230331/part-000.parquet"
    assert result.path.exists() and not result.skipped
    parquet = pq.ParquetFile(result.path)
    assert parquet.schema_arrow.equals(arrow_schema("financials", "20230331"), check_metadata=True)
    assert {parquet.metadata.row_group(0).column(i).compression
            for i in range(parquet.metadata.row_group(0).num_columns)} == {"ZSTD"}
    saved = parquet.read().to_pylist()[0]
    assert saved["CERT"] == 14 and saved["REPDTE"] == date(2023, 3, 31)
    assert saved["ASSET"] == Decimal("287069000.0000")
    assert saved["DEP"] == Decimal("230488000.2500")
    assert saved["NETINC"] == Decimal("-12.5000")
    assert saved["NCLNLS"] is None and saved["LNATRES"] is None
    assert saved["SOURCE_MISSING_FIELDS"] == ["LNATRES"]
    assert saved["ID"] == "14_20230331"
    assert result.sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()
    loaded = RunManifest.read(manifest_path)
    assert loaded.outputs[0]["sha256"] == result.sha256
    assert loaded.outputs[0]["source_total"] == 1
    assert loaded.outputs[0]["compression"] == "ZSTD"
    assert loaded.outputs[0]["partition"] == "REPDTE=20230331"
    assert loaded.stages["storage:financials:REPDTE=20230331"] == "complete"
    assert store.can_skip("financials", manifest_path, quarter="20230331", source_total=1, source_query="synthetic:financials")


def test_failures_round_trip_date_formats_and_nullable_cost(tmp_path):
    store, manifest, path = context(tmp_path)
    rows = [
        {"CERT": "40", "FAILDATE": "12/31/1980", "COST": 957.43212345, "ID": "001"},
        {"CERT": None, "FAILDATE": "1985-08-16", "COST": None, "ID": "002"},
        {"CERT": 76, "FAILDATE": "19870422", "ID": "003"},
    ]
    result = write(store, manifest, path, "failures", rows)
    saved = pq.ParquetFile(result.path).read().to_pylist()
    assert [row["FAILDATE"] for row in saved] == [
        date(1980, 12, 31), date(1985, 8, 16), date(1987, 4, 22)
    ]
    assert saved[0]["COST"] == 957.43212345
    assert arrow_schema("failures").field("COST").type == pa.float64()
    assert saved[1]["COST"] is None and saved[1]["SOURCE_MISSING_FIELDS"] == []
    assert saved[1]["CERT"] is None and saved[1]["ID"] == "002"
    assert saved[2]["COST"] is None and saved[2]["SOURCE_MISSING_FIELDS"] == ["COST"]
    assert pq.ParquetFile(result.path).schema_arrow.equals(
        arrow_schema("failures"), check_metadata=True
    )
    assert arrow_schema("failures").metadata[b"bft65b.schema_version"] == b"2"
    assert arrow_schema("financials", "20230331").metadata[b"bft65b.schema_version"] == b"1"


def test_institutions_round_trip_preserves_code_text_and_status(tmp_path):
    store, manifest, path = context(tmp_path)
    rows = [
        {"CERT": "004", "STNAME": "Connecticut", "ACTIVE": 0, "CHANGEC1": 223},
        {"CERT": 6, "STNAME": None, "ACTIVE": None, "CHANGEC1": "00223"},
    ]
    result = write(store, manifest, path, "institutions", rows)
    saved = pq.ParquetFile(result.path).read().to_pylist()
    assert [row["CERT"] for row in saved] == [4, 6]
    assert saved[0]["CHANGEC1"] == "223"
    assert saved[1]["CHANGEC1"] == "00223"
    assert saved[1]["STNAME"] is None and saved[1]["ACTIVE"] is None


@pytest.mark.parametrize("rows,quarter,match", [
    ([financial(14), financial("014")], "20230331", "Duplicate"),
    ([financial(REPDTE="20230630")], "20230331", "differs from partition"),
    ([financial(EXTRA=123)], "20230331", "unmodelled"),
    ([financial(ASSET="NaN")], "20230331", "decimal"),
    ([financial(DEP="1.23456")], "20230331", "precision"),
    ([financial(CERT=True)], "20230331", "CERT"),
])
def test_invalid_financial_input_never_promotes(tmp_path, rows, quarter, match):
    store, manifest, path = context(tmp_path)
    with pytest.raises(StorageValidationError, match=match):
        write(store, manifest, path, "financials", rows, quarter=quarter)
    assert not store.target_path("financials", quarter).exists()
    assert manifest.outputs == [] and not path.exists()


def test_duplicate_failure_source_ids_are_rejected(tmp_path):
    store, manifest, path = context(tmp_path)
    rows = [
        {"ID": "same", "CERT": 40, "FAILDATE": "1/2/1984"},
        {"ID": "same", "CERT": "040", "FAILDATE": "1984-01-02"},
    ]
    with pytest.raises(StorageValidationError, match="Duplicate"):
        write(store, manifest, path, "failures", rows)


def test_source_total_mismatch_and_na_cost_fail_closed(tmp_path):
    store, manifest, path = context(tmp_path)
    with pytest.raises(StorageValidationError, match="API-reported total"):
        store.write("financials", [financial()], quarter="20230331",
                    source_total=2, manifest=manifest, manifest_path=path, source_query="synthetic:financials")
    with pytest.raises(StorageValidationError, match="numeric"):
        write(store, manifest, path, "failures",
              [{"ID": "one", "CERT": 40, "FAILDATE": "1/2/1984", "COST": "N/A"}])
    assert manifest.outputs == []


@pytest.mark.parametrize("source_id", [None, "", "   ", 17])
def test_missing_or_invalid_failure_source_id_never_promotes(tmp_path, source_id):
    store, manifest, path = context(tmp_path)
    row = {"CERT": None, "FAILDATE": "5/28/1934"}
    if source_id is not None:
        row["ID"] = source_id
    with pytest.raises(StorageValidationError, match="ID"):
        write(store, manifest, path, "failures", [row])
    assert not store.target_path("failures").exists()


def test_failure_cert_absent_is_distinct_from_null_and_repeated_cert_is_retained(tmp_path):
    store, manifest, path = context(tmp_path)
    rows = [
        {"ID": "1", "FAILDATE": "5/28/1934"},
        {"ID": "2", "CERT": None, "FAILDATE": "5/29/1934"},
        {"ID": "3", "CERT": 40, "FAILDATE": "1/2/1984"},
        {"ID": "4", "CERT": 40, "FAILDATE": "1/2/1984"},
    ]
    result = write(store, manifest, path, "failures", rows)
    saved = pq.ParquetFile(result.path).read().to_pylist()
    assert len(saved) == 4
    assert saved[0]["CERT"] is None and "CERT" in saved[0]["SOURCE_MISSING_FIELDS"]
    assert saved[1]["CERT"] is None and "CERT" not in saved[1]["SOURCE_MISSING_FIELDS"]
    assert saved[2]["CERT"] == saved[3]["CERT"] == 40
    assert RunManifest.read(path).outputs[-1]["schema_version"] == "2"


def test_verified_file_is_skipped_but_different_healthy_data_conflicts(tmp_path):
    store, manifest, path = context(tmp_path)
    first = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    again = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert again.skipped and again.sha256 == first.sha256
    assert len(manifest.outputs) == 1
    changed = financial(ASSET=123)
    with pytest.raises(StorageConflictError, match="Healthy"):
        write(store, manifest, path, "financials", [changed], quarter="20230331")
    assert hashlib.sha256(first.path.read_bytes()).hexdigest() == first.sha256


def test_missing_manifest_entry_is_reconciled_only_if_content_matches(tmp_path):
    store, manifest, path = context(tmp_path)
    first = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    stale = RunManifest.new("stale", store.config)
    assert not store.can_skip("financials", tmp_path / "runs/stale.json", quarter="20230331", source_total=1, source_query="synthetic:financials")
    result = write(store, stale, tmp_path / "runs/stale.json",
                   "financials", [financial()], quarter="20230331")
    assert result.reconciled and not result.skipped
    assert result.sha256 == first.sha256
    assert len(stale.outputs) == 1


def test_corrupt_and_truncated_files_are_detected_then_replaced_with_backup(tmp_path):
    store, manifest, path = context(tmp_path)
    first = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    first.path.write_bytes(first.path.read_bytes()[:40])
    assert not store.can_skip("financials", path, quarter="20230331", source_total=1, source_query="synthetic:financials")
    with pytest.raises(StorageValidationError, match="Unreadable|truncated"):
        store.validate_file(first.path, "financials", quarter="20230331")
    repaired = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert not repaired.skipped and repaired.path.exists()
    assert store.can_skip("financials", path, quarter="20230331", source_total=1, source_query="synthetic:financials")
    assert list(first.path.parent.glob("part-000.parquet.corrupt.*"))
    assert len(manifest.outputs) == 2


def test_corrupt_temporary_file_cannot_replace_existing_file(tmp_path, monkeypatch):
    store, manifest, path = context(tmp_path)
    target = store.target_path("financials", "20230331")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old incomplete bytes")
    original = target.read_bytes()

    def bad_write(_table, destination, **_kwargs):
        destination.write_bytes(b"not parquet")

    monkeypatch.setattr(storage.pq, "write_table", bad_write)
    with pytest.raises(StorageValidationError, match="Unreadable|truncated"):
        write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert target.read_bytes() == original
    assert not list(target.parent.glob(".part-000.parquet.*.tmp"))
    assert not path.exists() and manifest.outputs == []


def test_atomic_promotion_receives_validated_temp_in_same_directory(tmp_path, monkeypatch):
    store, manifest, path = context(tmp_path)
    target = store.target_path("financials", "20230331")
    original_replace = os.replace
    promotions = []

    def checked_replace(source, destination):
        if os.fspath(destination) == os.fspath(target):
            assert os.path.dirname(source) == os.fspath(target.parent)
            assert pq.ParquetFile(source).metadata.num_rows == 1
            promotions.append((source, destination))
        return original_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", checked_replace)
    write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert len(promotions) == 1
    assert not list(target.parent.glob(".part-000.parquet.*.tmp"))


def test_wrong_schema_or_checksum_is_rejected(tmp_path):
    store, manifest, path = context(tmp_path)
    target = store.target_path("financials", "20230331")
    target.parent.mkdir(parents=True)
    pq.write_table(pa.table({"CERT": [14]}), target)
    with pytest.raises(StorageValidationError, match="schema"):
        store.validate_file(target, "financials", quarter="20230331")
    target.unlink()
    result = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    with pytest.raises(StorageValidationError, match="SHA-256"):
        store.validate_file(result.path, "financials", quarter="20230331",
                            expected_sha256="0" * 64)


def test_snappy_is_configurable_and_checked(tmp_path):
    store, manifest, path = context(tmp_path, ProjectConfig(parquet_compression="snappy"))
    result = write(store, manifest, path, "institutions", [{"CERT": 4}])
    assert store.validate_file(result.path, "institutions").compression == "SNAPPY"


def test_config_rejects_unsupported_codec():
    with pytest.raises(ValueError, match="compression"):
        ProjectConfig(parquet_compression="gzip")

def test_only_persisted_complete_manifest_can_skip(tmp_path, monkeypatch):
    store, manifest, path = context(tmp_path)
    target = store.target_path("financials", "20230331")
    original_write = RunManifest.write

    def fail_manifest_write(_self, _path):
        raise OSError("simulated manifest disk failure")

    monkeypatch.setattr(RunManifest, "write", fail_manifest_write)
    with pytest.raises(OSError, match="manifest disk failure"):
        write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert target.exists()
    assert not store.can_skip("financials", path, quarter="20230331", source_total=1, source_query="synthetic:financials")
    monkeypatch.setattr(RunManifest, "write", original_write)
    fresh = RunManifest.new("retry", store.config)
    result = write(store, fresh, path, "financials", [financial()], quarter="20230331")
    assert result.reconciled and not result.skipped
    assert store.can_skip("financials", path, quarter="20230331", source_total=1, source_query="synthetic:financials")


def test_readable_wrong_codec_is_preserved_as_conflict(tmp_path):
    store, manifest, path = context(tmp_path)
    target = store.target_path("financials", "20230331")
    target.parent.mkdir(parents=True)
    table = storage._typed_table("financials", [financial()], "20230331")
    pq.write_table(table, target, compression="snappy")
    before = target.read_bytes()
    with pytest.raises(StorageConflictError, match="readable"):
        write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert target.read_bytes() == before
    assert manifest.outputs == []

def test_valid_file_with_changed_bytes_is_not_silently_reconciled(tmp_path):
    store, manifest, path = context(tmp_path)
    first = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    table = storage._typed_table("financials", [financial()], "20230331")
    pq.write_table(table, first.path, compression="zstd", compression_level=3, write_statistics=False)
    assert hashlib.sha256(first.path.read_bytes()).hexdigest() != first.sha256
    assert not store.can_skip("financials", path, quarter="20230331", source_total=1, source_query="synthetic:financials")
    with pytest.raises(StorageConflictError, match="checksum"):
        write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert len(manifest.outputs) == 1

def test_out_of_scope_or_mismatched_partition_is_rejected(tmp_path):
    store, manifest, path = context(tmp_path)
    with pytest.raises(ValueError, match="assignment range"):
        write(store, manifest, path, "financials",
              [financial(REPDTE="20260930")], quarter="20260930")
    result = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    wrong = tmp_path / "data_raw/fdic/financials/REPDTE=20230630/part-000.parquet"
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(result.path.read_bytes())
    with pytest.raises(StorageValidationError, match="schema|partition"):
        store.validate_file(wrong, "financials", quarter="20230630")


def test_source_query_mismatch_prevents_skip(tmp_path):
    store, manifest, path = context(tmp_path)
    write(store, manifest, path, "institutions", [{"CERT": 4}])
    assert store.can_skip("institutions", path, source_total=1, source_query="synthetic:institutions")
    assert not store.can_skip("institutions", path, source_total=1, source_query="different-query")

def test_read_back_rejects_duplicate_keys_even_if_file_is_valid_parquet(tmp_path):
    store, _, _ = context(tmp_path)
    target = store.target_path("financials", "20230331")
    target.parent.mkdir(parents=True)
    row = {
        "CERT": 14, "REPDTE": date(2023, 3, 31),
        "SOURCE_MISSING_FIELDS": [],
    }
    table = pa.Table.from_pylist([row, row], schema=arrow_schema("financials", "20230331"))
    pq.write_table(table, target, compression="zstd", compression_level=3)
    with pytest.raises(StorageValidationError, match="duplicate Parquet keys"):
        store.validate_file(target, "financials", quarter="20230331", expected_rows=2)


def test_read_back_rejects_wrong_source_count(tmp_path):
    store, manifest, path = context(tmp_path)
    result = write(store, manifest, path, "institutions", [{"CERT": 4}])
    with pytest.raises(StorageValidationError, match="row count"):
        store.validate_file(result.path, "institutions", expected_rows=2)


@pytest.mark.parametrize("change", [
    "failed_status", "pending_status", "failed_validation", "output_incomplete",
    "wrong_query", "wrong_total", "wrong_rows", "wrong_schema", "wrong_codec",
    "wrong_checksum", "wrong_partition", "wrong_config", "malformed_outputs",
    "missing_stage", "missing_validation",
])
def test_skip_rejects_inconsistent_persisted_manifest(tmp_path, change):
    store, manifest, path = context(tmp_path)
    result = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    original = result.path.read_bytes()
    assert store.can_skip("financials", path, quarter="20230331", source_total=1,
                          source_query="synthetic:financials")
    data = json.loads(path.read_text(encoding="utf-8"))
    if change == "failed_status":
        data["status"] = "failed"
    elif change == "pending_status":
        data["status"] = "pending"
    elif change == "failed_validation":
        data["validations"].append({"passed": False})
    elif change == "output_incomplete":
        data["outputs"][0]["status"] = "running"
    elif change == "wrong_query":
        data["outputs"][0]["source_query"] = "wrong"
    elif change == "wrong_total":
        data["outputs"][0]["source_total"] = 2
    elif change == "wrong_rows":
        data["outputs"][0]["rows"] = 2
    elif change == "wrong_schema":
        data["outputs"][0]["schema_version"] = "0"
    elif change == "wrong_codec":
        data["outputs"][0]["compression"] = "SNAPPY"
    elif change == "wrong_checksum":
        data["outputs"][0]["sha256"] = "0" * 64
    elif change == "wrong_partition":
        data["outputs"][0]["partition"] = "REPDTE=20230630"
    elif change == "wrong_config":
        data["config"]["seed"] = 999
    elif change == "missing_stage":
        data["stages"].pop("storage:financials:REPDTE=20230331")
    elif change == "missing_validation":
        data["validations"].clear()
    else:
        data["outputs"] = "invalid"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert not store.can_skip("financials", path, quarter="20230331", source_total=1,
                              source_query="synthetic:financials")
    assert result.path.read_bytes() == original


def test_missing_or_unreadable_manifest_cannot_skip_but_identical_raw_is_reconciled(tmp_path):
    store, manifest, path = context(tmp_path)
    result = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    before, mtime = result.path.read_bytes(), result.path.stat().st_mtime_ns
    path.write_text("{invalid-json", encoding="utf-8")
    assert not store.can_skip("financials", path, quarter="20230331", source_total=1,
                              source_query="synthetic:financials")
    path.unlink()
    assert not store.can_skip("financials", path, quarter="20230331", source_total=1,
                              source_query="synthetic:financials")
    new = RunManifest.new("recovered", store.config)
    repaired = write(store, new, path, "financials", [financial()], quarter="20230331")
    assert repaired.reconciled and not repaired.skipped and repaired.sha256 == result.sha256
    assert result.path.read_bytes() == before and result.path.stat().st_mtime_ns == mtime


def test_skip_requires_explicit_source_total_and_query(tmp_path):
    store, manifest, path = context(tmp_path)
    write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert not store.can_skip("financials", path, quarter="20230331")
    assert not store.can_skip("financials", path, quarter="20230331", source_total=1)
    assert not store.can_skip("financials", path, quarter="20230331",
                              source_total=1, source_query="")


@pytest.mark.parametrize("damage", ["truncate", "garbage"])
def test_failure_v2_corruption_is_rebuilt_without_losing_null_cert(tmp_path, damage):
    store, manifest, path = context(tmp_path)
    rows = [
        {"ID": "1", "CERT": None, "FAILDATE": "5/28/1934", "COST": None},
        {"ID": "2", "CERT": 40, "FAILDATE": "1/2/1984", "COST": 3.1234567},
    ]
    first = write(store, manifest, path, "failures", rows)
    first.path.write_bytes(first.path.read_bytes()[:40] if damage == "truncate" else b"corrupt")
    assert not store.can_skip("failures", path, source_total=2, source_query="synthetic:failures")
    with pytest.raises(StorageValidationError, match="Unreadable|truncated"):
        store.validate_file(first.path, "failures", expected_rows=2)
    recovered = write(store, manifest, path, "failures", rows)
    assert not recovered.skipped
    assert store.validate_file(first.path, "failures", expected_rows=2,
                               expected_sha256=recovered.sha256).rows == 2
    assert pq.ParquetFile(first.path).read(columns=["ID", "CERT"]).to_pylist()[0] == {"ID": "1", "CERT": None}
    assert list(first.path.parent.glob("failures.parquet.corrupt.*"))


def test_failure_v2_manifest_write_failure_after_promotion_recovers_without_rewrite(tmp_path, monkeypatch):
    store, manifest, path = context(tmp_path)
    rows = [{"ID": "one", "CERT": None, "FAILDATE": "5/28/1934", "COST": None}]
    original_write = RunManifest.write

    def unavailable(_self, _path):
        raise OSError("manifest write interrupted")

    monkeypatch.setattr(RunManifest, "write", unavailable)
    with pytest.raises(OSError, match="manifest write interrupted"):
        write(store, manifest, path, "failures", rows)
    target = store.target_path("failures")
    verified = store.validate_file(target, "failures", expected_rows=1)
    before, mtime = target.read_bytes(), target.stat().st_mtime_ns
    assert not store.can_skip("failures", path, source_total=1, source_query="synthetic:failures")
    monkeypatch.setattr(RunManifest, "write", original_write)
    fresh = RunManifest.new("retry", store.config)
    result = write(store, fresh, path, "failures", rows)
    assert result.reconciled and not result.skipped and result.sha256 == verified.sha256
    assert target.read_bytes() == before and target.stat().st_mtime_ns == mtime
    assert store.can_skip("failures", path, source_total=1, source_query="synthetic:failures")


def test_partial_temporary_file_is_never_a_completed_partition(tmp_path):
    store, manifest, path = context(tmp_path)
    target = store.target_path("financials", "20230331")
    target.parent.mkdir(parents=True)
    stray = target.parent / ".part-000.parquet.interrupted.tmp"
    stray.write_bytes(b"partial parquet bytes")
    assert not store.can_skip("financials", path, quarter="20230331", source_total=1,
                              source_query="synthetic:financials")
    result = write(store, manifest, path, "financials", [financial()], quarter="20230331")
    assert result.path == target and result.path.read_bytes() != stray.read_bytes()
    assert store.validate_file(target, "financials", quarter="20230331", expected_rows=1).rows == 1
