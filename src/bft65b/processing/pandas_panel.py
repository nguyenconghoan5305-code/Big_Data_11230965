"""Pandas implementation of the frozen P3 observed-panel workload.

The selected P3.1 contract has no decimal-valued column. P3ASSET is Float64;
Arrow schemas are applied explicitly on output so pandas NaN round-trips as
source null, never as an invented zero. No input row is deduplicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bft65b.processing.polars_panel import (
    GAP_COLUMNS, INPUT_COLUMNS, QUALITY_COLUMNS, UNMATCHED_FLAG,
)


def _flag_contains(value: object, token: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("Flag/provenance value is not a list")
    return token in value


def transform(paths: Sequence[Path]) -> dict[str, pd.DataFrame]:
    """Read all projected values and fully materialize all three relations."""
    if not paths or len({str(path) for path in paths}) != len(paths):
        raise ValueError("Input file list is empty or duplicated")
    pieces = []
    for path in paths:
        with pq.ParquetFile(path) as file:
            table = file.read(columns=list(INPUT_COLUMNS), use_threads=True)
        pieces.append(table.to_pandas(date_as_object=True))
    observed = pd.concat(pieces, ignore_index=True)
    observed = observed.loc[:, list(INPUT_COLUMNS)].sort_values(
        ["REPDTE", "CERT"], kind="mergesort", ignore_index=True)
    if (observed["CERT"].isna().any() or observed["REPDTE"].isna().any()
            or observed.duplicated(["CERT", "REPDTE"]).any()):
        raise ValueError("Missing or duplicate bank-quarter key")
    absent = observed["SOURCE_MISSING_FIELDS"].map(
        lambda value: _flag_contains(value, "P3ASSET"))
    null = observed["P3ASSET"].isna()
    if (absent & ~null).any():
        raise ValueError("Source-absent P3ASSET has a non-null value")
    unmatched = observed["DQ_FLAGS"].map(
        lambda value: _flag_contains(value, UNMATCHED_FLAG))

    dates = observed.loc[:, ["CERT", "REPDTE"]].sort_values(
        ["CERT", "REPDTE"], kind="mergesort", ignore_index=True)
    previous = dates.groupby("CERT", sort=False)["REPDTE"].shift(1)
    current_dt = pd.to_datetime(dates["REPDTE"])
    previous_dt = pd.to_datetime(previous)
    current_quarter = current_dt.dt.year * 4 + current_dt.dt.month // 3 - 1
    previous_quarter = previous_dt.dt.year * 4 + previous_dt.dt.month // 3 - 1
    missing_quarters = current_quarter - previous_quarter - 1
    has_gap = missing_quarters.gt(0)
    gaps = pd.DataFrame({
        "CERT": dates.loc[has_gap, "CERT"].to_numpy(dtype=np.int64),
        "PREVIOUS_REPDTE": previous.loc[has_gap].to_list(),
        "NEXT_REPDTE": dates.loc[has_gap, "REPDTE"].to_list(),
        "MISSING_INTERVENING_QUARTERS": missing_quarters.loc[has_gap].to_numpy(dtype=np.int32),
    }, columns=list(GAP_COLUMNS))
    gaps = gaps.sort_values(["CERT", "NEXT_REPDTE"], kind="mergesort", ignore_index=True)

    measures = pd.DataFrame({"REPDTE": observed["REPDTE"],
        "CERT": observed["CERT"], "P3ASSET_NULL_ROWS": null.astype(np.uint32),
        "P3ASSET_SOURCE_ABSENT_ROWS": absent.astype(np.uint32),
        "P3ASSET_EXPLICIT_NULL_ROWS": (null & ~absent).astype(np.uint32),
        "UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS": unmatched.astype(np.uint32)})
    quality = (measures.groupby("REPDTE", sort=True, dropna=False)
        .agg(ROW_COUNT=("CERT", "size"), DISTINCT_KEY_COUNT=("CERT", "nunique"),
             P3ASSET_NULL_ROWS=("P3ASSET_NULL_ROWS", "sum"),
             P3ASSET_SOURCE_ABSENT_ROWS=("P3ASSET_SOURCE_ABSENT_ROWS", "sum"),
             P3ASSET_EXPLICIT_NULL_ROWS=("P3ASSET_EXPLICIT_NULL_ROWS", "sum"),
             UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS=("UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS", "sum"))
        .reset_index().loc[:, list(QUALITY_COLUMNS)])
    for field in QUALITY_COLUMNS[1:]:
        quality[field] = quality[field].astype(np.uint32)
    if (quality["ROW_COUNT"] != quality["DISTINCT_KEY_COUNT"]).any():
        raise ValueError("Duplicate bank-quarter key in quality relation")
    return {"observed": observed, "gaps": gaps, "quality": quality}


def to_arrow(outputs: Mapping[str, pd.DataFrame],
             schemas: Mapping[str, pa.Schema]) -> dict[str, pa.Table]:
    """Preserve frozen Arrow types without lossy or implicit numeric casts."""
    result = {}
    for name in ("observed", "gaps", "quality"):
        frame = outputs[name]
        schema = schemas[name]
        if list(frame.columns) != schema.names:
            raise ValueError(f"{name} output column order differs from frozen schema")
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
        if table.num_rows != len(frame) or not table.schema.equals(schema, check_metadata=False):
            raise ValueError(f"{name} Arrow conversion differs")
        result[name] = table
    return result
