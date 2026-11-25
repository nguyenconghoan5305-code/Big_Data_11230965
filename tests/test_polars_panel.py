"""Synthetic P3.1 workload checks with exact expected counts and values."""

from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from bft65b.processing.polars_panel import (
    INPUT_COLUMNS, _check_parity, compare, lazy_queries, materialize,
)
from bft65b.quality.audit import _sha


QUARTERS = ("20220930", "20221231", "20230331", "20230630")


def _row(cert: int, day: date, identity: str, p3: float | None,
         missing: list[str] | None = None, flags: list[str] | None = None) -> dict:
    return {"CERT": cert, "REPDTE": day, "ID": identity, "P3ASSET": p3,
            "SOURCE_MISSING_FIELDS": missing or [], "DQ_FLAGS": flags or []}


def _source_rows() -> dict[str, list[dict]]:
    return {
        "20220930": [_row(1, date(2022, 9, 30), "a", None,
                          flags=["source_explicit_null:P3ASSET"])],
        "20221231": [_row(2, date(2022, 12, 31), "b", None,
                          missing=["P3ASSET"], flags=["source_absent:P3ASSET"])],
        "20230331": [_row(1, date(2023, 3, 31), "c", 1.5,
                          flags=["financial_cert_absent_from_institutions:CERT"]),
                     _row(2, date(2023, 3, 31), "d", 2.5)],
        "20230630": [_row(1, date(2023, 6, 30), "e", 3.0)],
    }


def _files(tmp_path: Path, rows: dict[str, list[dict]]) -> list[Path]:
    paths = []
    for quarter, values in rows.items():
        path = tmp_path / f"REPDTE={quarter}" / "part-000.parquet"
        path.parent.mkdir(parents=True)
        df = pl.DataFrame(values, schema_overrides={"CERT": pl.Int64, "REPDTE": pl.Date,
            "ID": pl.String, "P3ASSET": pl.Float64,
            "SOURCE_MISSING_FIELDS": pl.List(pl.String), "DQ_FLAGS": pl.List(pl.String)})
        df.write_parquet(path)
        paths.append(path)
    return paths


def _audit(rows: dict[str, list[dict]]) -> dict:
    periods = []
    for quarter, values in rows.items():
        absent = sum("P3ASSET" in r["SOURCE_MISSING_FIELDS"] for r in values)
        explicit = sum(r["P3ASSET"] is None and "P3ASSET" not in r["SOURCE_MISSING_FIELDS"]
                       for r in values)
        unmatched = sum("financial_cert_absent_from_institutions:CERT" in r["DQ_FLAGS"]
                        for r in values)
        periods.append({"dataset": "financials", "period": quarter,
                        "rows": len(values), "distinct_cert_count": len({r["CERT"] for r in values}),
                        "field_counts": {"P3ASSET": {"absent": absent, "explicit_null": explicit}},
                        "institution_snapshot_join": {"unmatched_records": unmatched}})
    return {"periods": periods}


def test_lazy_workload_matches_exact_gap_quality_and_independent_reference(tmp_path):
    rows = _source_rows()
    source = _files(tmp_path / "source", rows)
    queries = lazy_queries(source)
    assert all("Parquet SCAN" in query.explain(optimized=True) for query in queries.values())
    result = materialize(queries, QUARTERS)
    assert result["observed"].columns == list(INPUT_COLUMNS)
    assert len(result["observed"]) == 5
    assert result["gaps"].to_dicts() == [{
        "CERT": 1, "PREVIOUS_REPDTE": date(2022, 9, 30),
        "NEXT_REPDTE": date(2023, 3, 31), "MISSING_INTERVENING_QUARTERS": 1}]
    by_q = {row["REPDTE"]: row for row in result["quality"].to_dicts()}
    assert by_q[date(2022, 9, 30)]["P3ASSET_NULL_ROWS"] == 1
    assert by_q[date(2022, 9, 30)]["P3ASSET_EXPLICIT_NULL_ROWS"] == 1
    assert by_q[date(2022, 12, 31)]["P3ASSET_SOURCE_ABSENT_ROWS"] == 1
    assert by_q[date(2022, 12, 31)]["P3ASSET_EXPLICIT_NULL_ROWS"] == 0
    assert by_q[date(2023, 3, 31)]["UNMATCHED_INSTITUTIONS_SNAPSHOT_ROWS"] == 1
    references = _files(tmp_path / "reference", rows)
    path_by_q = {q: path for q, path in zip(QUARTERS, references, strict=True)}
    gaps_path = tmp_path / "reference_gaps.parquet"
    pq.write_table(result["gaps"].to_arrow(), gaps_path)
    parity = _check_parity(result, path_by_q, gaps_path, _audit(rows))
    assert parity["projected_rows_exact"] == 5
    assert parity["gap_events_exact"] == parity["gap_distinct_banks"] == 1
    assert parity["missing_intervening_quarters"] == 1
    assert len(parity["by_quarter"]) == 4


def test_duplicate_bank_quarter_fails_instead_of_deduplicating(tmp_path):
    rows = _source_rows()
    rows["20230331"].append(dict(rows["20230331"][0]))
    with pytest.raises(ValueError, match="duplicate bank-quarter"):
        materialize(lazy_queries(_files(tmp_path, rows)), QUARTERS)


def test_absent_source_field_cannot_have_non_null_value(tmp_path):
    rows = _source_rows()
    rows["20221231"][0]["P3ASSET"] = 9.0
    with pytest.raises(ValueError, match="Source absence and explicit null"):
        materialize(lazy_queries(_files(tmp_path, rows)), QUARTERS)


def test_full_row_parity_rejects_projected_value_change(tmp_path):
    rows = _source_rows()
    result = materialize(lazy_queries(_files(tmp_path / "source", rows)), QUARTERS)
    changed = _source_rows()
    changed["20230331"][0]["ID"] = "different"
    references = _files(tmp_path / "reference", changed)
    gap = tmp_path / "gaps.parquet"
    pq.write_table(result["gaps"].to_arrow(), gap)
    with pytest.raises(ValueError, match="keys or values differ"):
        _check_parity(result, dict(zip(QUARTERS, references, strict=True)), gap, _audit(rows))


def test_omitted_quarter_fails_scope_gate(tmp_path):
    paths = _files(tmp_path, _source_rows())
    with pytest.raises(ValueError, match="Quarter inventory"):
        materialize(lazy_queries(paths[:-1]), QUARTERS)


def test_frozen_comparator_requires_exact_rows_order_and_reference_hash(tmp_path, monkeypatch):
    frames = materialize(lazy_queries(_files(tmp_path / "source", _source_rows())), QUARTERS)
    reference_dir = tmp_path / "data_processed" / "task_3_1_polars" / "fixture"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir(parents=True)
    candidate_dir.mkdir()
    artifacts = {}
    for name, frame in frames.items():
        path = reference_dir / f"{name}.parquet"
        frame.write_parquet(path)
        frame.write_parquet(candidate_dir / f"{name}.parquet")
        artifacts[name] = {"path": str(path), "sha256": _sha(path)}
    import bft65b.processing.polars_panel as module
    monkeypatch.setattr(module, "_accepted_result", lambda root, run_id: {
        "artifacts": artifacts, "overall": {"observed_bank_quarters": 5}})
    assert compare(tmp_path, "fixture", candidate_dir)["rows_by_relation"] == {
        "observed": 5, "gaps": 1, "quality": 4}
    frames["observed"].reverse().write_parquet(candidate_dir / "observed.parquet")
    with pytest.raises(ValueError, match="Candidate output differs"):
        compare(tmp_path, "fixture", candidate_dir)
    frames["observed"].write_parquet(candidate_dir / "observed.parquet")
    frames["quality"].reverse().write_parquet(reference_dir / "quality.parquet")
    with pytest.raises(ValueError, match="reference checksum/path differs"):
        compare(tmp_path, "fixture", candidate_dir)
