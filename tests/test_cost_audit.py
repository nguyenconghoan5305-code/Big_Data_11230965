"""Synthetic COST-contract tests; no test-period COST enters fitted scenarios."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal
import json

import pytest

from bft65b.modelling.cost_audit import (
    DEFAULT_RUNS, _preflight, _write_text_atomic, cost_status, fold_training_costs, reconcile_costs, representative_training_cost,
    scenario_spec, thousand_usd_to_usd, validate_scenarios,
)


def _link(source_id: str, cert: int, q: date, faildate: date, period: str = "development") -> dict:
    return {"SOURCE_ID": source_id, "CERT": cert, "Q": q, "FAILDATE": faildate,
            "PERIOD": period}


def _source(source_id: str, cert: int, faildate: date, cost, *, absent: bool = False) -> dict:
    return {"ID": source_id, "CERT": cert, "FAILDATE": faildate, "COST": cost,
            "SOURCE_MISSING_FIELDS": ["COST"] if absent else []}


def _repair(source: dict) -> dict:
    return {**source, "TRANSACTION_DISPOSITION": "verified_failure",
            "BANK_LINK_DISPOSITION": "chronological_candidate", "SUPPLEMENT_MATCH": "matched"}


FOLDS = [{"id": "fold_1", "validation_start": "2012-03-31", "validation_end": "2012-12-31"}]


def test_repeated_positive_rows_count_one_event_and_final_window() -> None:
    fail = date(2013, 3, 1)  # After final development Q, inside its outcome window.
    source = _source("A", 12, fail, 25.5)
    links = [_link("A", 12, date(2012, 6, 30), fail),
             _link("A", 12, date(2012, 12, 31), fail),
             _link("T", 90, date(2014, 3, 31), date(2014, 6, 1), "test")]
    rows, measured = reconcile_costs(links, [_repair(source)], [source], FOLDS)
    assert len(rows) == 1
    assert rows[0]["DEVELOPMENT_POSITIVE_ROWS"] == 2
    assert rows[0]["OOF_POSITIVE_ROWS"] == 2
    assert measured["development"]["distinct_events"] == 1
    assert measured["positive_bank_quarter_rows"] == 2
    assert measured["development_events_with_faildate_after_dev_q_end"] == 1
    assert measured["test_only_distinct_events"] == 1
    assert measured["development"]["nominal_thousands_usd"]["sum"] == 25.5


def test_identity_conflicts_duplicates_and_unmatched_are_rejected() -> None:
    fail = date(2012, 6, 1)
    source = _source("A", 12, fail, 4.0)
    links = [_link("A", 12, date(2012, 3, 31), fail)]
    with pytest.raises(ValueError, match="Duplicate FDIC source ID"):
        reconcile_costs(links, [_repair(source)] * 2, [source], FOLDS)
    with pytest.raises(ValueError, match="Unmatched development source ID"):
        reconcile_costs(links, [_repair(source)], [], FOLDS)
    changed = _repair(source)
    changed["COST"] = 5.0
    with pytest.raises(ValueError, match="Conflicting source identity"):
        reconcile_costs(links, [changed], [source], FOLDS)
    with pytest.raises(ValueError, match="Duplicate or invalid positive linkage key"):
        reconcile_costs(links * 2, [_repair(source)], [source], FOLDS)
    wrong = deepcopy(links)
    wrong[0]["Q"] = date(2013, 6, 30)
    with pytest.raises(ValueError, match="four-quarter window"):
        reconcile_costs(wrong, [_repair(source)], [source], FOLDS)


def test_cost_categories_units_and_missingness_are_explicit() -> None:
    assert cost_status(None) == "missing"
    assert cost_status(None, absent=True) == "source_absent"
    assert cost_status(0) == "zero"
    assert cost_status(-1) == "negative"
    assert cost_status(2.5) == "positive"
    for invalid in ("N/A", float("nan"), float("inf"), True):
        assert cost_status(invalid) == "invalid_or_special"
    assert thousand_usd_to_usd(Decimal("1.2345")) == Decimal("1234.5000")
    with pytest.raises(ValueError, match="Nonfinite"):
        thousand_usd_to_usd(float("nan"))


def test_synthetic_event_coverage_counts_include_missing_zero_negative_special() -> None:
    fail = date(2012, 6, 1)
    values = [("missing", None, False), ("absent", None, True),
              ("zero", 0.0, False), ("negative", -2.0, False),
              ("positive", 8.0, False), ("special", "N/A", False)]
    links = [_link(source_id, i + 1, date(2012, 3, 31), fail)
             for i, (source_id, _, _) in enumerate(values)]
    sources = [_source(source_id, i + 1, fail, value, absent=absent)
               for i, (source_id, value, absent) in enumerate(values)]
    rows, measured = reconcile_costs(links, [_repair(x) for x in sources], sources, FOLDS)
    assert len(rows) == 6
    assert measured["development"]["cost_status_events"] == {
        "invalid_or_special": 1, "missing": 1, "negative": 1,
        "positive": 1, "source_absent": 1, "zero": 1}
    assert measured["development"]["usable_nonnegative_events"] == 2
    assert measured["development"]["usable_coverage"] == pytest.approx(2 / 6)
    assert measured["development"]["nominal_thousands_usd"]["median"] == 4.0


def test_fold_local_estimate_ignores_heldout_costs_and_rejects_missing() -> None:
    rows = [
        {"SOURCE_ID": "train_a", "COST": 0.0, "SOURCE_FIELD_ABSENT": False},
        {"SOURCE_ID": "train_b", "COST": 10.0, "SOURCE_FIELD_ABSENT": False},
        {"SOURCE_ID": "train_c", "COST": None, "SOURCE_FIELD_ABSENT": False},
        {"SOURCE_ID": "train_d", "COST": -5.0, "SOURCE_FIELD_ABSENT": False},
        {"SOURCE_ID": "val", "COST": 100.0, "SOURCE_FIELD_ABSENT": False},
    ]
    training = {"train_a", "train_b", "train_c", "train_d"}
    assert representative_training_cost(training, rows) == 5.0
    rows[-1]["COST"] = 1e15
    assert representative_training_cost(training, rows) == 5.0
    rows[1]["COST"] = 20.0
    assert representative_training_cost(training, rows) == 10.0
    with pytest.raises(ValueError, match="Duplicate"):
        representative_training_cost(training, rows + [rows[0]])
    with pytest.raises(ValueError, match="no usable"):
        representative_training_cost({"train_c", "train_d"}, rows)


def test_fold_schedule_uses_training_events_not_validation_cost() -> None:
    fold = {"id": "fold_1", "train_start": "2010-03-31", "train_end": "2010-12-31",
            "validation_start": "2012-03-31", "validation_end": "2012-12-31",
            "training_positive_rows": 2, "training_distinct_events": 1}
    links = [_link("train", 1, date(2010, 3, 31), date(2010, 6, 1)),
             _link("train", 1, date(2010, 6, 30), date(2010, 9, 1)),
             _link("validation", 2, date(2012, 3, 31), date(2012, 6, 1))]
    costs = [{"SOURCE_ID": "train", "COST": 7.0, "COST_STATUS": "positive",
              "SOURCE_FIELD_ABSENT": False},
             {"SOURCE_ID": "validation", "COST": 1e9, "COST_STATUS": "positive",
              "SOURCE_FIELD_ABSENT": False}]
    assert fold_training_costs(links, costs, [fold])[0]["training_only_median_nominal_thousands_usd"] == 7
    costs[1]["COST"] = 1e12
    assert fold_training_costs(links, costs, [fold])[0]["training_only_median_nominal_thousands_usd"] == 7
    fold["training_positive_rows"] = 3
    with pytest.raises(ValueError, match="Frozen fold"):
        fold_training_costs(links, costs, [fold])


def test_scenario_is_deterministic_and_not_a_threshold() -> None:
    first = scenario_spec()
    validate_scenarios(first)
    assert json.dumps(first, sort_keys=True) == json.dumps(scenario_spec(), sort_keys=True)
    assert first["threshold_selected"] is False
    first["threshold_selected"] = True
    with pytest.raises(ValueError, match="threshold-dependent"):
        validate_scenarios(first)


def test_report_text_is_promoted_atomically(tmp_path) -> None:
    path = tmp_path / "report.md"
    digest = _write_text_atomic(path, "first\nsecond\n")
    assert path.read_bytes() == b"first\nsecond\n"
    assert len(digest) == 64
    assert not list(tmp_path.glob(".report.md.*.tmp"))


def test_portable_cost_preflight_rejects_implicit_historical_fallback(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="historical default"):
        _preflight(tmp_path, DEFAULT_RUNS.copy())
    with pytest.raises(ValueError, match="All COST upstream"):
        _preflight(tmp_path, {"split": "portable"})


def test_portable_cost_preflight_rejects_missing_selected_inputs(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "final_report_bft65b.pdf").write_bytes(b"synthetic assignment fixture")
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}", encoding="utf-8")
    selected = {key: "portable_" + key for key in DEFAULT_RUNS}
    with pytest.raises(FileNotFoundError):
        _preflight(tmp_path, selected)
