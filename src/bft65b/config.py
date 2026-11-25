"""Validated, serializable configuration for future pipeline stages."""

from dataclasses import asdict, dataclass, field
from datetime import date
import json
import os
from pathlib import Path


BASE_FINANCIAL_FIELDS = (
    "CERT", "REPDTE", "STNAME", "ASSET", "DEP", "EQ", "NETINC", "ROA",
    "NCLNLS", "LNATRES", "RBC1AAJ",
)
EXTENDED_FINANCIAL_FIELDS = BASE_FINANCIAL_FIELDS + (
    "CHBAL", "SC", "BRO", "DEPDOM", "EINTEXP", "LNLSGR", "LNRE", "LNCI",
    "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ELNATR", "NIMY", "RBCRWAJ",
)
VERIFIED_FINANCIAL_FIELD_SETS = frozenset((BASE_FINANCIAL_FIELDS, EXTENDED_FINANCIAL_FIELDS))
VERIFIED_FAILURE_FIELDS = ("CERT", "FAILDATE", "COST")
VERIFIED_INSTITUTION_FIELDS = ("CERT", "STNAME", "ACTIVE", "CHANGEC1")


def _quarter_end(value: str) -> date:
    day = date.fromisoformat(value)
    if (day.month, day.day) not in {(3, 31), (6, 30), (9, 30), (12, 31)}:
        raise ValueError(f"Not a calendar quarter-end: {value}")
    return day


@dataclass(frozen=True)
class ProjectConfig:
    """Defaults reflect the assignment; configured codes were checked in FDIC dictionaries."""

    financial_start: str = "1984-03-31"
    financial_end: str = "2026-06-30"
    modelling_end: str = "2025-06-30"
    pilot_start: str = "2023-03-31"
    pilot_end: str = "2023-12-31"
    data_root: str = "data_raw"
    processed_root: str = "data_processed"
    run_root: str = "runs"
    seed: int = 65
    api_page_size: int = 10_000
    availability_lag_days: int = 120
    benchmark_repeats: int = 5
    parquet_compression: str = "zstd"
    parquet_zstd_level: int = 3
    financial_fields: tuple[str, ...] = BASE_FINANCIAL_FIELDS
    failure_fields: tuple[str, ...] = VERIFIED_FAILURE_FIELDS
    institution_fields: tuple[str, ...] = VERIFIED_INSTITUTION_FIELDS
    fields_verified: bool = True

    def __post_init__(self) -> None:
        start, end, model_end = map(_quarter_end, (self.financial_start, self.financial_end, self.modelling_end))
        pilot_start, pilot_end = map(_quarter_end, (self.pilot_start, self.pilot_end))
        if not (start <= pilot_start <= pilot_end <= end and start <= model_end <= end):
            raise ValueError("Date ranges must be ordered and within the Financials range")
        if self.seed < 0 or self.api_page_size < 1 or self.availability_lag_days < 0 or self.benchmark_repeats < 1:
            raise ValueError("Execution settings must be non-negative and counts positive")
        if self.parquet_compression not in {"zstd", "snappy"} or not 1 <= self.parquet_zstd_level <= 22:
            raise ValueError("Unsupported Parquet compression settings")
        if not isinstance(self.fields_verified, bool) or not all(
            isinstance(names, tuple) and all(isinstance(field, str) for field in names)
            for names in (self.financial_fields, self.failure_fields, self.institution_fields)
        ):
            raise ValueError("Field verification and field lists need explicit Boolean/tuple types")
        for names, required in ((self.financial_fields, {"CERT", "REPDTE"}), (self.failure_fields, {"CERT", "FAILDATE"}), (self.institution_fields, {"CERT"})):
            if not required.issubset(names) or len(names) != len(set(names)):
                raise ValueError("Field lists need required keys and no duplicates")
        if self.fields_verified and (
            self.financial_fields not in VERIFIED_FINANCIAL_FIELD_SETS
            or self.failure_fields != VERIFIED_FAILURE_FIELDS
            or self.institution_fields != VERIFIED_INSTITUTION_FIELDS
        ):
            raise ValueError("fields_verified=True requires an exact dictionary-verified field set")

    def to_dict(self) -> dict:
        return asdict(self)

    def paths(self, base: Path) -> dict[str, Path]:
        """Resolve relative data paths against the project root or an external override."""
        external = os.getenv("BFT65B_DATA_ROOT")
        raw = Path(external) if external else Path(self.data_root)
        if not raw.is_absolute():
            raw = base / raw
        processed = Path(self.processed_root)
        runs = Path(self.run_root)
        return {
            "raw": raw.resolve(),
            "processed": (processed if processed.is_absolute() else base / processed).resolve(),
            "runs": (runs if runs.is_absolute() else base / runs).resolve(),
        }


def load_config(path: Path | None = None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object")
    allowed = set(ProjectConfig.__dataclass_fields__)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for key in ("financial_fields", "failure_fields", "institution_fields"):
        if key in payload:
            payload[key] = tuple(payload[key])
    return ProjectConfig(**payload)
