"""Exact keyed comparison contracts for the P2.4 revision repair."""

from datetime import date

import pyarrow as pa
import pytest

from bft65b.panel.asof import PREDICTOR_FIELDS
from bft65b.panel.compare_revision import COMPARISON_FIELDS, compare_tables


def _row(cert: int, *, asset: float = 1.0, disposition: str = "eligible") -> dict:
    row = {"CERT": cert, "Q": date(2023, 6, 30),
           "ELIGIBILITY_DISPOSITION": disposition,
           "SELECTED_REPDTE": date(2022, 12, 31),
           "AVAILABILITY_DATE": date(2023, 4, 30),
           "REPORT_AGE_DAYS": 181,
           **{name: None for name in PREDICTOR_FIELDS}}
    row["ASSET"] = asset
    return row


def test_comparison_is_keyed_and_exact_across_order():
    before = pa.Table.from_pylist([_row(1), _row(2, asset=2.0)])
    after = pa.Table.from_pylist([_row(2, asset=2.0), _row(1)])
    result = compare_tables(before, after)
    assert result["shared_keys"] == 2
    assert result["previous_only_keys"] == result["current_only_keys"] == 0
    assert set(result["field_differences"]) == set(COMPARISON_FIELDS)
    assert all(value == 0 for value in result["field_differences"].values())


def test_comparison_counts_membership_disposition_and_predictor_changes():
    before = pa.Table.from_pylist([_row(1), _row(2, asset=2.0)])
    after = pa.Table.from_pylist([_row(1, asset=3.0,
                                        disposition="unresolved_identity"), _row(3)])
    result = compare_tables(before, after)
    assert result["shared_keys"] == 1
    assert result["previous_only_keys"] == result["current_only_keys"] == 1
    assert result["field_differences"]["ASSET"] == 1
    assert result["field_differences"]["ELIGIBILITY_DISPOSITION"] == 1
    assert sum(result["field_differences"].values()) == 2


def test_comparison_rejects_duplicate_prediction_keys():
    duplicated = pa.Table.from_pylist([_row(1), _row(1)])
    with pytest.raises(ValueError, match="duplicate CERT/q"):
        compare_tables(duplicated, pa.Table.from_pylist([_row(1)]))
