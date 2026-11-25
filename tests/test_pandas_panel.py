"""Pandas baseline parity on a complete small frozen-workload fixture."""

from datetime import date

import polars as pl
import pytest

from bft65b.processing.pandas_panel import to_arrow, transform
from bft65b.processing.polars_panel import lazy_queries, materialize


QUARTERS = ("20220930", "20221231", "20230331")


def _paths(tmp_path, *, duplicate=False, invalid_absent=False):
    records = {
        "20220930": [
            {"CERT": 1, "REPDTE": date(2022, 9, 30), "ID": "one", "P3ASSET": None,
             "SOURCE_MISSING_FIELDS": [], "DQ_FLAGS": ["source_explicit_null:P3ASSET"]},
        ],
        "20221231": [
            {"CERT": 2, "REPDTE": date(2022, 12, 31), "ID": "two",
             "P3ASSET": 3.0 if invalid_absent else None,
             "SOURCE_MISSING_FIELDS": ["P3ASSET"], "DQ_FLAGS": ["source_absent:P3ASSET"]},
        ],
        "20230331": [
            {"CERT": 1, "REPDTE": date(2023, 3, 31), "ID": "three",
             "P3ASSET": 1.23456789012345, "SOURCE_MISSING_FIELDS": [],
             "DQ_FLAGS": ["financial_cert_absent_from_institutions:CERT"]},
            {"CERT": 2, "REPDTE": date(2023, 3, 31), "ID": "four", "P3ASSET": 0.0,
             "SOURCE_MISSING_FIELDS": [], "DQ_FLAGS": []},
        ],
    }
    if duplicate:
        records["20230331"].append(records["20230331"][0].copy())
    paths = []
    for quarter, rows in records.items():
        path = tmp_path / f"REPDTE={quarter}" / "part-000.parquet"
        path.parent.mkdir(parents=True)
        pl.DataFrame(rows, schema_overrides={
            "CERT": pl.Int64, "REPDTE": pl.Date, "ID": pl.String,
            "P3ASSET": pl.Float64, "SOURCE_MISSING_FIELDS": pl.List(pl.String),
            "DQ_FLAGS": pl.List(pl.String),
        }).write_parquet(path)
        paths.append(path)
    return paths


def test_pandas_matches_all_three_polars_relations_exactly(tmp_path):
    paths = _paths(tmp_path)
    expected = materialize(lazy_queries(paths), QUARTERS)
    schemas = {name: frame.to_arrow().schema for name, frame in expected.items()}
    outputs = to_arrow(transform(paths), schemas)
    for name, table in outputs.items():
        assert pl.from_arrow(table).equals(expected[name], null_equal=True), name
    assert outputs["observed"]["P3ASSET"].null_count == 2
    assert outputs["quality"]["P3ASSET_SOURCE_ABSENT_ROWS"].to_pylist() == [0, 1, 0]
    assert outputs["quality"]["P3ASSET_EXPLICIT_NULL_ROWS"].to_pylist() == [1, 0, 0]
    assert outputs["quality"]["UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS"].to_pylist() == [0, 0, 1]
    assert outputs["gaps"]["MISSING_INTERVENING_QUARTERS"].to_pylist() == [1]


def test_pandas_rejects_duplicate_key(tmp_path):
    with pytest.raises(ValueError, match="duplicate bank-quarter"):
        transform(_paths(tmp_path, duplicate=True))


def test_pandas_rejects_nonnull_source_absent_value(tmp_path):
    with pytest.raises(ValueError, match="Source-absent"):
        transform(_paths(tmp_path, invalid_absent=True))
