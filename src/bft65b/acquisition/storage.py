"""Typed, validated, atomic Parquet storage for FDIC API records."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.config import (
    BASE_FINANCIAL_FIELDS, EXTENDED_FINANCIAL_FIELDS, ProjectConfig,
)
from bft65b.manifest import RunManifest

SCHEMA_VERSIONS = {"financials": "1", "failures": "2", "institutions": "1"}
EXTENDED_FINANCIAL_SCHEMA_VERSION = "2"
AMOUNT = pa.decimal128(20, 4)
AMOUNT_FIELDS = frozenset({
    "ASSET", "DEP", "EQ", "NETINC", "NCLNLS", "LNATRES", "CHBAL", "SC",
    "BRO", "DEPDOM", "EINTEXP", "LNLSGR", "LNRE", "LNCI", "NTLNLS", "ELNATR",
})
RATIO_FIELDS = frozenset({"ROA", "RBC1AAJ", "NIMY", "RBCRWAJ"})
UNRESOLVED_UNIT_NUMERIC_FIELDS = frozenset({"P3ASSET", "P9ASSET", "NAASSET"})
SOURCE_FIELDS = {
    "financials": ("CERT", "REPDTE", "STNAME", "ASSET", "DEP", "EQ", "NETINC",
                   "ROA", "NCLNLS", "LNATRES", "RBC1AAJ"),
    "failures": ("CERT", "FAILDATE", "COST"),
    "institutions": ("CERT", "STNAME", "ACTIVE", "CHANGEC1"),
}
KEY_FIELDS = {
    "financials": ("CERT", "REPDTE"),
    "failures": ("ID",),
    "institutions": ("CERT",),
}
REQUIRED_FIELDS = {
    "financials": ("CERT", "REPDTE"),
    "failures": ("ID", "FAILDATE"),
    "institutions": ("CERT",),
}
BASE_FIELDS = {
    "financials": [
        pa.field("CERT", pa.int64(), nullable=False),
        pa.field("REPDTE", pa.date32(), nullable=False),
        pa.field("STNAME", pa.string()),
        pa.field("ASSET", AMOUNT),
        pa.field("DEP", AMOUNT),
        pa.field("EQ", AMOUNT),
        pa.field("NETINC", AMOUNT),
        pa.field("ROA", pa.float64()),
        pa.field("NCLNLS", AMOUNT),
        pa.field("LNATRES", AMOUNT),
        pa.field("RBC1AAJ", pa.float64()),
    ],
    "failures": [
        pa.field("CERT", pa.int64()),
        pa.field("FAILDATE", pa.date32(), nullable=False),
        pa.field("COST", pa.float64()),
    ],
    "institutions": [
        pa.field("CERT", pa.int64(), nullable=False),
        pa.field("STNAME", pa.string()),
        pa.field("ACTIVE", pa.int8()),
        pa.field("CHANGEC1", pa.string()),
    ],
}
EXTENDED_FINANCIAL_BASE_FIELDS = BASE_FIELDS["financials"] + [
    pa.field("CHBAL", AMOUNT), pa.field("SC", AMOUNT),
    pa.field("BRO", AMOUNT), pa.field("DEPDOM", AMOUNT),
    pa.field("EINTEXP", AMOUNT), pa.field("LNLSGR", AMOUNT),
    pa.field("LNRE", AMOUNT), pa.field("LNCI", AMOUNT),
    pa.field("P3ASSET", pa.float64()), pa.field("P9ASSET", pa.float64()),
    pa.field("NAASSET", pa.float64()), pa.field("NTLNLS", AMOUNT),
    pa.field("ELNATR", AMOUNT), pa.field("NIMY", pa.float64()),
    pa.field("RBCRWAJ", pa.float64()),
]


class StorageValidationError(ValueError):
    """Input or on-disk data failed a storage acceptance check."""


class CorruptParquetError(StorageValidationError):
    """Parquet bytes cannot be opened or read."""


class StorageConflictError(StorageValidationError):
    """A healthy immutable artifact disagrees with new source rows."""


class FullAcquisitionBusyError(RuntimeError):
    """Another process owns the shared full-acquisition raw namespace."""


@dataclass(frozen=True)
class FileCheck:
    path: Path
    rows: int
    sha256: str
    compression: str


@dataclass(frozen=True)
class WriteResult:
    path: Path
    rows: int
    sha256: str
    skipped: bool
    reconciled: bool


def arrow_schema(
    dataset: str, quarter: str | None = None, *,
    financial_fields: tuple[str, ...] = BASE_FINANCIAL_FIELDS,
) -> pa.Schema:
    """Return the versioned Arrow schema, including provenance columns."""
    if dataset not in BASE_FIELDS:
        raise ValueError(f"Unknown FDIC dataset: {dataset}")
    if dataset == "financials":
        _quarter(quarter)
        if financial_fields not in {BASE_FINANCIAL_FIELDS, EXTENDED_FINANCIAL_FIELDS}:
            raise ValueError("Unknown Financials field set")
    elif quarter is not None:
        raise ValueError("Only Financials have a quarter partition")
    extended = dataset == "financials" and financial_fields == EXTENDED_FINANCIAL_FIELDS
    metadata = {
        b"bft65b.schema_version": (EXTENDED_FINANCIAL_SCHEMA_VERSION if extended else SCHEMA_VERSIONS[dataset]).encode(),
        b"bft65b.dataset": dataset.encode(),
        b"bft65b.partition": (f"REPDTE={quarter}" if quarter else "none").encode(),
        b"bft65b.amount_unit": b"thousands_of_usd_provisional",
        b"bft65b.ratio_unit": b"percent",
    }
    if extended:
        metadata[b"bft65b.unverified_numeric_units"] = b"P3ASSET,P9ASSET,NAASSET"
    return pa.schema(
        (EXTENDED_FINANCIAL_BASE_FIELDS if extended else BASE_FIELDS[dataset])
        + [
            pa.field("ID", pa.string(), nullable=(dataset != "failures")),
            pa.field("SOURCE_MISSING_FIELDS", pa.list_(pa.field("element", pa.string())), nullable=False),
        ],
        metadata=metadata,
    )


def _quarter(value: str | None) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise ValueError("Quarter must be YYYYMMDD")
    try:
        result = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError("Invalid quarter date") from exc
    if (result.month, result.day) not in {(3, 31), (6, 30), (9, 30), (12, 31)}:
        raise ValueError("Not a calendar quarter-end")
    return result


def _parse_date(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        formats = ("%Y%m%d",) if field == "REPDTE" else ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d")
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                pass
        if field == "FAILDATE":
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except ValueError:
                pass
    raise StorageValidationError(f"Invalid {field} date representation: {value!r}")


def _cert(value: Any) -> int:
    if isinstance(value, bool):
        raise StorageValidationError("CERT must be an integer identifier")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"\d+", value):
        return int(value)
    raise StorageValidationError("CERT must be an integer identifier")


def _amount(value: Any, field: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise StorageValidationError(f"{field} must be numeric")
    try:
        number = Decimal(str(value))
        if not number.is_finite():
            raise StorageValidationError(f"{field} is not a finite decimal")
        fixed = number.quantize(Decimal("0.0001"))
    except (InvalidOperation, ValueError) as exc:
        raise StorageValidationError(f"{field} is not a finite decimal") from exc
    if fixed != number:
        raise StorageValidationError(f"{field} exceeds Decimal(20,4) precision")
    return fixed


def _ratio(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise StorageValidationError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StorageValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise StorageValidationError(f"{field} must be finite")
    return number


def _unresolved_numeric(value: Any, field: str) -> float | None:
    """Store unknown-scale source numbers only when float64 keeps their decimal value."""
    number = _ratio(value, field)
    if number is None:
        return None
    if isinstance(value, int) and abs(value) > 2**53:
        raise StorageValidationError(f"{field} exceeds exact float64 integer precision")
    if isinstance(value, str):
        try:
            if Decimal(value) != Decimal(str(number)):
                raise StorageValidationError(f"{field} loses source numeric precision")
        except InvalidOperation as exc:
            raise StorageValidationError(f"{field} must be numeric") from exc
    return number


def _convert(field: str, value: Any) -> Any:
    if field == "CERT":
        return _cert(value)
    if field in {"REPDTE", "FAILDATE"}:
        return _parse_date(value, field)
    if field in AMOUNT_FIELDS:
        return _amount(value, field)
    if field in UNRESOLVED_UNIT_NUMERIC_FIELDS:
        return _unresolved_numeric(value, field)
    if field in RATIO_FIELDS or field == "COST":
        return _ratio(value, field)
    if value is None:
        return None
    if field == "ACTIVE":
        if isinstance(value, bool) or str(value) not in {"0", "1"}:
            raise StorageValidationError("ACTIVE must be 0, 1, or null")
        return int(value)
    if field in {"STNAME", "CHANGEC1", "ID"}:
        if field in {"CHANGEC1", "ID"} and isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if not isinstance(value, str):
            raise StorageValidationError(f"{field} must be text or null")
        return value
    raise StorageValidationError(f"Unexpected field: {field}")


def _typed_table(
    dataset: str, rows: Iterable[Mapping[str, Any]], quarter: str | None,
    financial_fields: tuple[str, ...] = BASE_FINANCIAL_FIELDS,
) -> pa.Table:
    schema = arrow_schema(dataset, quarter, financial_fields=financial_fields)
    source_fields = financial_fields if dataset == "financials" else SOURCE_FIELDS[dataset]
    allowed = set(source_fields) | {"ID"}
    records = []
    seen: set[tuple[Any, ...]] = set()
    for position, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise StorageValidationError(f"Row {position} is not a mapping")
        unexpected = set(source) - allowed
        if unexpected:
            raise StorageValidationError(f"Row {position} has unmodelled fields: {sorted(unexpected)}")
        missing_keys = [key for key in REQUIRED_FIELDS[dataset] if key not in source or source[key] is None]
        if missing_keys:
            raise StorageValidationError(f"Row {position} is missing keys: {missing_keys}")
        if dataset == "failures" and (
            not isinstance(source["ID"], str) or not source["ID"].strip()
        ):
            raise StorageValidationError("Failures source ID must be a nonempty string")
        record = {
            field: (None if dataset == "failures" and field == "CERT" and source.get(field) is None
                    else _convert(field, source.get(field)))
            for field in source_fields
        }
        record["ID"] = _convert("ID", source.get("ID"))
        record["SOURCE_MISSING_FIELDS"] = [
            field for field in source_fields
            if field not in source and field not in REQUIRED_FIELDS[dataset]
        ]
        if dataset == "financials" and record["REPDTE"] != _quarter(quarter):
            raise StorageValidationError(f"Row {position} REPDTE differs from partition")
        key = tuple(record[field] for field in KEY_FIELDS[dataset])
        if key in seen:
            raise StorageValidationError(f"Duplicate {dataset} key: {key}")
        seen.add(key)
        records.append(record)
    if not records:
        raise StorageValidationError("Refuse to mark an empty source query complete")
    records.sort(key=lambda row: tuple(row[field] for field in KEY_FIELDS[dataset]))
    try:
        return pa.Table.from_pylist(records, schema=schema)
    except (pa.ArrowException, OverflowError, ValueError) as exc:
        raise StorageValidationError("Source values do not fit the explicit Arrow schema") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ParquetStore:
    """One validated file per dataset/quarter; manifest commits occur last."""

    def __init__(self, raw_root: Path, config: ProjectConfig | None = None) -> None:
        self.raw_root = Path(raw_root)
        self.config = config or ProjectConfig()
        self._full_namespace_owned = False
        if not self.config.fields_verified:
            raise ValueError("Configured FDIC fields must be verified")
        for dataset, fields in (
            ("financials", self.config.financial_fields),
            ("failures", self.config.failure_fields),
            ("institutions", self.config.institution_fields),
        ):
            expected = (
                self.config.financial_fields if dataset == "financials"
                else SOURCE_FIELDS[dataset]
            )
            if tuple(fields) != expected or (
                dataset == "financials" and expected not in {BASE_FINANCIAL_FIELDS, EXTENDED_FINANCIAL_FIELDS}
            ):
                raise ValueError(f"Storage schema does not match configured {dataset} fields")

    @contextmanager
    def full_namespace_ownership(self) -> Iterator[None]:
        """Hold an OS lock through all full-run writes and the final commit.

        The persistent lock file must not be removed: a process killed without
        cleanup releases its Windows byte-range lock automatically.
        """
        if Path(self.config.data_root).parts != ("data_raw", "task_1_5_full"):
            raise ValueError("Full namespace ownership requires the dedicated raw root")
        lock_path = self.raw_root.parent / ".task_1_5_full.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise FullAcquisitionBusyError(
                    f"Full acquisition raw namespace is owned by another process; "
                    f"lock={lock_path}. No attempt artifacts were changed."
                ) from exc
            self._full_namespace_owned = True
            try:
                yield
            finally:
                self._full_namespace_owned = False
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def schema_version(self, dataset: str) -> str:
        if dataset == "financials" and self.config.financial_fields == EXTENDED_FINANCIAL_FIELDS:
            return EXTENDED_FINANCIAL_SCHEMA_VERSION
        return SCHEMA_VERSIONS[dataset]

    def schema(self, dataset: str, quarter: str | None = None) -> pa.Schema:
        return arrow_schema(dataset, quarter, financial_fields=self.config.financial_fields)

    def target_path(self, dataset: str, quarter: str | None = None) -> Path:
        arrow_schema(dataset, quarter)
        if dataset == "financials":
            day = _quarter(quarter).isoformat()
            if not self.config.financial_start <= day <= self.config.financial_end:
                raise ValueError("Financials quarter is outside the assignment range")
        namespace = (
            "fdic_extended_v2"
            if self.config.financial_fields == EXTENDED_FINANCIAL_FIELDS else "fdic"
        )
        root = self.raw_root / namespace
        if dataset == "financials":
            return root / "financials" / f"REPDTE={quarter}" / "part-000.parquet"
        return root / f"{dataset}.parquet"

    def validate_file(
        self,
        path: Path,
        dataset: str,
        *,
        quarter: str | None = None,
        expected_rows: int | None = None,
        expected_sha256: str | None = None,
        expected_table: pa.Table | None = None,
    ) -> FileCheck:
        path = Path(path)
        if path != self.target_path(dataset, quarter) and not (
            path.parent == self.target_path(dataset, quarter).parent
            and path.name.startswith(".part-000.parquet.")
        ) and not (
            path.parent == self.target_path(dataset, quarter).parent
            and path.name.startswith(f".{dataset}.parquet.")
        ):
            raise StorageValidationError("File path does not match the expected partition")
        try:
            with path.open("rb") as handle, pq.ParquetFile(handle) as parquet:
                actual = parquet.read()
                schema = self.schema(dataset, quarter)
                if not actual.schema.equals(schema, check_metadata=True):
                    raise StorageValidationError("Parquet Arrow schema or partition metadata differs")
                if expected_rows is not None and actual.num_rows != expected_rows:
                    raise StorageValidationError("Parquet row count differs from source")
                if actual.num_rows == 0:
                    raise StorageValidationError("Empty Parquet file")
                keys = list(zip(*(actual[field].to_pylist() for field in KEY_FIELDS[dataset])))
                if any(None in key for key in keys) or len(set(keys)) != len(keys):
                    raise StorageValidationError("Missing or duplicate Parquet keys")
                if dataset == "failures" and any(
                    value is None for value in actual["FAILDATE"].to_pylist()
                ):
                    raise StorageValidationError("Missing Failures event date")
                if dataset == "failures" and any(
                    not isinstance(value, str) or not value.strip()
                    for value in actual["ID"].to_pylist()
                ):
                    raise StorageValidationError("Missing or invalid Failures source ID")
                if dataset == "financials" and any(day != _quarter(quarter) for _, day in keys):
                    raise StorageValidationError("Parquet REPDTE differs from partition")
                if expected_table is not None and not actual.equals(expected_table, check_metadata=True):
                    raise StorageValidationError("Parquet read-back values differ from source")
                codec = self.config.parquet_compression.upper()
                observed = {
                    parquet.metadata.row_group(group).column(column).compression
                    for group in range(parquet.metadata.num_row_groups)
                    for column in range(parquet.metadata.row_group(group).num_columns)
                }
                if observed != {codec}:
                    raise StorageValidationError(f"Parquet compression differs: {observed}")
        except (OSError, pa.ArrowException) as exc:
            raise CorruptParquetError(f"Unreadable or truncated Parquet: {path}") from exc
        checksum = _sha256(path)
        if expected_sha256 is not None and checksum != expected_sha256:
            raise StorageValidationError("Parquet SHA-256 differs from manifest")
        return FileCheck(path, actual.num_rows, checksum, codec)

    def can_skip(
        self, dataset: str, manifest_path: Path, *, quarter: str | None = None,
        source_total: int | None = None, source_query: str | None = None,
    ) -> bool:
        """Trust only a persisted complete output record plus a healthy matching file."""
        target = self.target_path(dataset, quarter)
        if (not target.exists() or not Path(manifest_path).exists()
                or type(source_total) is not int or source_total < 1
                or not isinstance(source_query, str) or not source_query.strip()):
            return False
        try:
            manifest = RunManifest.read(manifest_path)
        except (OSError, ValueError, TypeError, KeyError):
            return False
        if json.dumps(manifest.config, sort_keys=True) != json.dumps(self.config.to_dict(), sort_keys=True):
            return False
        if (manifest.status not in {"running", "complete"}
                or not isinstance(manifest.outputs, list)
                or not isinstance(manifest.validations, list)
                or not isinstance(manifest.stages, dict)
                or any(not isinstance(item, dict) or not item.get("passed", False)
                       for item in manifest.validations)):
            return False
        stage = f"storage:{dataset}:{f'REPDTE={quarter}' if quarter else 'all'}"
        validation = f"storage:{dataset}:{quarter or 'all'}"
        if (manifest.stages.get(stage) != "complete" or
                not any(item.get("name") == validation and item.get("passed")
                        for item in manifest.validations)):
            return False
        matches = [
            item for item in manifest.outputs
            if isinstance(item, dict) and item.get("path") == str(target)
            and item.get("dataset") == dataset
        ]
        if not matches or matches[-1].get("status") != "complete":
            return False
        record = matches[-1]
        if type(record.get("rows")) is not int or type(record.get("source_total")) is not int:
            return False
        if record.get("source_total") != source_total or record.get("rows") != source_total:
            return False
        if record.get("source_query") != source_query:
            return False
        if record.get("schema_version") != self.schema_version(dataset):
            return False
        if record.get("partition") != (f"REPDTE={quarter}" if quarter else None):
            return False
        if record.get("compression") != self.config.parquet_compression.upper():
            return False
        try:
            self.validate_file(
                target, dataset, quarter=quarter,
                expected_rows=record["rows"], expected_sha256=record["sha256"],
            )
        except (StorageValidationError, KeyError):
            return False
        return True
    def write(
        self,
        dataset: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_total: int,
        manifest: RunManifest,
        manifest_path: Path,
        quarter: str | None = None,
        source_query: str | None = None,
    ) -> WriteResult:
        if (Path(self.config.data_root).parts == ("data_raw", "task_1_5_full")
                and not self._full_namespace_owned):
            raise StorageValidationError(
                "Full-acquisition raw writes require exclusive namespace ownership"
            )
        if json.dumps(manifest.config, sort_keys=True) != json.dumps(self.config.to_dict(), sort_keys=True):
            raise StorageValidationError("Run manifest configuration differs from storage configuration")
        if isinstance(source_total, bool) or not isinstance(source_total, int) or source_total < 1:
            raise StorageValidationError("source_total must be a positive API count")
        if not isinstance(source_query, str) or not source_query.strip():
            raise StorageValidationError("source_query must document the source request")
        table = _typed_table(dataset, rows, quarter, self.config.financial_fields)
        if table.num_rows != source_total:
            raise StorageValidationError("Source rows do not match API-reported total")
        target = self.target_path(dataset, quarter)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing_valid = False
        if target.exists():
            try:
                existing = self.validate_file(target, dataset, quarter=quarter)
                existing_valid = True
                with pq.ParquetFile(target) as parquet:
                    previous = parquet.read()
                if not previous.equals(table, check_metadata=True):
                    raise StorageConflictError("Healthy raw Parquet differs from new source rows")
                if Path(manifest_path).exists():
                    try:
                        persisted = RunManifest.read(manifest_path)
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        raise StorageConflictError("Existing manifest is unreadable") from exc
                    completed = [
                        item for item in persisted.outputs
                        if item.get("path") == str(target) and item.get("status") == "complete"
                    ]
                    if completed and completed[-1].get("sha256") != existing.sha256:
                        raise StorageConflictError("Existing file checksum differs from completed manifest")
                if self.can_skip(dataset, manifest_path, quarter=quarter, source_total=source_total, source_query=source_query):
                    return WriteResult(target, existing.rows, existing.sha256, True, False)
                check = existing
            except StorageConflictError:
                raise
            except CorruptParquetError:
                existing_valid = False
            except StorageValidationError as exc:
                raise StorageConflictError("Existing readable Parquet failed validation") from exc
        if not existing_valid:
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            os.close(fd)
            temporary = Path(temp_name)
            try:
                pq.write_table(
                    table, temporary,
                    compression=self.config.parquet_compression,
                    compression_level=(
                        self.config.parquet_zstd_level
                        if self.config.parquet_compression == "zstd" else None
                    ),
                )
                check = self.validate_file(
                    temporary, dataset, quarter=quarter,
                    expected_rows=source_total, expected_table=table,
                )
                if target.exists():
                    backup = target.with_name(target.name + f".corrupt.{_sha256(target)[:12]}")
                    if not backup.exists():
                        shutil.copy2(target, backup)
                os.replace(temporary, target)
                check = self.validate_file(
                    target, dataset, quarter=quarter,
                    expected_rows=source_total, expected_sha256=check.sha256,
                    expected_table=table,
                )
            finally:
                temporary.unlink(missing_ok=True)
        manifest.record_output(
            path=str(target), dataset=dataset, rows=check.rows,
            sha256=check.sha256, compression=check.compression,
            schema_version=self.schema_version(dataset), partition=f"REPDTE={quarter}" if quarter else None,
            source_total=source_total, source_query=source_query,
        )
        manifest.record_validation(
            f"storage:{dataset}:{quarter or 'all'}", True,
            f"rows={check.rows}; sha256={check.sha256}; compression={check.compression}",
        )
        manifest.write(manifest_path)
        return WriteResult(target, check.rows, check.sha256, False, existing_valid)
