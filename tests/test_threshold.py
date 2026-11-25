"""Synthetic complete-curve, boundary, scaling and temporal OOF guards."""
from __future__ import annotations

from datetime import date
import itertools
import math

import numpy as np
import pytest

from bft65b.modelling.cost_audit import representative_training_cost, scenario_spec
from bft65b.modelling.threshold import (
    DEFAULT_RUNS, SELECTED, _preflight, _curve_table, _metrics, _positive_median, build_curve,
    select_policy, sensitivity, validate_oof_contract,
)


def test_portable_threshold_requires_explicit_existing_upstreams(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "final_report_bft65b.pdf").write_bytes(b"synthetic assignment fixture")
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="historical default"):
        _preflight(tmp_path)
    with pytest.raises(ValueError, match="All threshold upstream"):
        _preflight(tmp_path, {"split": "portable"})
    with pytest.raises(FileNotFoundError):
        _preflight(tmp_path, {key: "portable_" + key for key in DEFAULT_RUNS})


def test_ties_enter_together_and_equality_alarms() -> None:
    curve = build_curve(np.array([0.7, 0.7, 0.2]), np.array([1, 0, 0]),
                        np.array([0, 0, 0]), [10.0])
    assert curve["distinct_scores"] == 2
    assert curve["kinds"] == ["never_alarm", "score_ge", "score_ge", "always_alarm"]
    assert list(curve["fold_counts"][0]["tp"]) == [0, 1, 1, 1]
    assert list(curve["fold_counts"][0]["fp"]) == [0, 1, 2, 2]
    assert curve["thresholds"][1] == 0.7
    assert _metrics(curve, 0, .1, .01)["precision"] is None
    assert _metrics(curve, 0, .1, .01)["FN"] == 1
    assert _metrics(curve, 3, .1, .01)["alarms"] == 3
    table = _curve_table(curve, None, .1, .01)
    assert table.schema.names[0:3] == ["POLICY_INDEX", "POLICY_KIND", "THRESHOLD"]
    assert table["THRESHOLD"].to_pylist() == [None, .7, .2, None]
    assert table["PRECISION"].to_pylist()[0] is None


def test_complete_empirical_curve_matches_brute_force_and_cost_components() -> None:
    scores = np.array([.1, .4, .4, .8, .9])
    labels = np.array([0, 1, 0, 0, 1])
    folds = np.array([0, 0, 0, 1, 1])
    medians = [2.0, 7.0]
    curve = build_curve(scores, labels, folds, medians)
    alpha, beta = .1, .01
    best, objective, normalized = select_policy(curve, alpha, beta)
    brute = []
    for kind, threshold in zip(curve["kinds"], curve["thresholds"]):
        alarm = np.ones(len(scores), dtype=bool) if kind == "always_alarm" else (
            np.zeros(len(scores), dtype=bool) if kind == "never_alarm" else scores >= threshold)
        amount = 0.0
        for f, median in enumerate(medians):
            fn = int(np.sum((folds == f) & (labels == 1) & ~alarm))
            fp = int(np.sum((folds == f) & (labels == 0) & alarm))
            amount += median * (alpha * fn + beta * fp)
        brute.append(amount / len(scores))
    assert objective.tolist() == pytest.approx(brute)
    assert objective[best] == min(objective)
    row = _metrics(curve, best, alpha, beta)
    assert row["TP"] + row["FN"] == 2
    assert row["FP"] + row["TN"] == 3
    assert row["alarms"] == row["TP"] + row["FP"]
    assert normalized[best] == pytest.approx((alpha * row["FN"] + beta * row["FP"]) / 5)
    overall = _curve_table(curve, None, alpha, beta)
    first = _curve_table(curve, 0, alpha, beta)
    second = _curve_table(curve, 1, alpha, beta)
    for name in ("TP", "FP", "TN", "FN", "FALSE_ALARM_COST_THOUSANDS", "MISSED_COST_THOUSANDS"):
        assert overall[name][best].as_py() == pytest.approx(
            first[name][best].as_py() + second[name][best].as_py())


def test_exact_tie_break_prefers_fewer_fp_then_higher_threshold() -> None:
    curve = build_curve([.8, .5, .2], [1, 0, 0], [0, 0, 0], [10])
    # No missed-positive penalty: never-alarm and the .8 threshold have zero FP.
    chosen, _, _ = select_policy(curve, 0, 1)
    assert curve["kinds"][chosen] == "never_alarm"
    # A zero-FP finite threshold wins when the never policy misses a positive.
    chosen, _, _ = select_policy(curve, 1, 1)
    assert curve["thresholds"][chosen] == .8
    assert _metrics(curve, chosen, 1, 1)["FP"] == 0


def test_within_fold_median_cancels_but_cross_fold_scale_can_change_global_choice() -> None:
    one = build_curve([.9, .2], [1, 0], [0, 0], [1])
    hundred = build_curve([.9, .2], [1, 0], [0, 0], [100])
    assert select_policy(one, 1, 1)[0] == select_policy(hundred, 1, 1)[0]
    # Two folds trade a missed positive against a false alarm. The raw global
    # objective weights the second fold 100x, whereas the normalized one does not.
    scores = [.6, .6, .6, .4, .4, .6]
    labels = [1, 1, 1, 0, 1, 0]
    folds = [0, 0, 0, 0, 1, 1]
    equal = build_curve(scores, labels, folds, [1, 1])
    scaled = build_curve(scores, labels, folds, [1, 100])
    equal_best, _, _ = select_policy(equal, 1, 1)
    scaled_best, _, normalized = select_policy(scaled, 1, 1)
    assert equal["thresholds"][equal_best] == .6
    assert scaled["thresholds"][scaled_best] == .4
    assert normalized[scaled_best] == pytest.approx(2 / 6)
    assert scaled["thresholds"][_metrics(scaled, equal_best, 1, 1)["policy_index"]] == .6


@pytest.mark.parametrize("median", [None, 0, -1, float("nan"), float("inf"), "N/A"])
def test_invalid_fold_medians_fail_closed(median) -> None:
    with pytest.raises(ValueError, match="median"):
        _positive_median(median, "fold_1")
    with pytest.raises(ValueError, match="median"):
        build_curve([.5], [1], [0], [median])


def test_oof_contract_rejects_duplicate_missing_test_gap_and_wrong_labels() -> None:
    q = date(2001, 3, 31)
    key = (12, q)
    row = {"CERT": 12, "Q": q, "Y": 1, "FOLD_ID": "fold_1",
           "CANDIDATE_ID": SELECTED, "SCORE": .75}
    accepted = validate_oof_contract([row], {key: 1}, {key: "source-1"},
                                     {q: "fold_1"}, ["fold_1"])
    assert accepted["distinct_positive_events"] == 1
    assert accepted["positive_rows"] == 1
    for bad_rows, expected, links in [
        ([row, row], {key: 1}, {key: "source-1"}),
        ([], {key: 1}, {key: "source-1"}),
        ([row], {key: 0}, {key: "source-1"}),
        ([row], {}, {key: "source-1"}),
        ([{**row, "Q": date(2014, 3, 31)}], {key: 1}, {key: "source-1"}),
    ]:
        with pytest.raises(ValueError):
            validate_oof_contract(bad_rows, expected, links, {q: "fold_1"}, ["fold_1"])


def test_oof_positive_count_does_not_overflow_int8() -> None:
    q = date(2001, 3, 31)
    rows = [{"CERT": cert, "Q": q, "Y": 1, "FOLD_ID": "fold_1",
             "CANDIDATE_ID": SELECTED, "SCORE": .5} for cert in range(300)]
    expected = {(row["CERT"], q): 1 for row in rows}
    links = {key: f"event-{key[0]}" for key in expected}
    checked = validate_oof_contract(rows, expected, links, {q: "fold_1"}, ["fold_1"])
    assert checked["positive_rows"] == 300
    assert checked["distinct_positive_events"] == 300


def test_fold_local_cost_estimator_ignores_heldout_event_value() -> None:
    rows = [{"SOURCE_ID": "train", "COST": 10.0, "SOURCE_FIELD_ABSENT": False},
            {"SOURCE_ID": "heldout", "COST": 100.0, "SOURCE_FIELD_ABSENT": False}]
    before = representative_training_cost({"train"}, rows)
    rows[1]["COST"] = 1e15
    assert representative_training_cost({"train"}, rows) == before


def test_sensitivity_keeps_reference_policy_locked() -> None:
    curve = build_curve([.9, .8, .4, .2], [1, 0, 1, 0], [0, 0, 0, 0], [10])
    scenario = scenario_spec()
    reference, rows = sensitivity(curve, scenario)
    assert len(rows) == 9
    assert all(row["reference_policy"]["policy_index"] == reference["policy_index"] for row in rows)
    assert any(row["alpha"] == .1 and row["beta"] == .01 and
               row["optimal"]["policy_index"] == reference["policy_index"] for row in rows)
    assert all(row["always_alarm"]["policy_kind"] == "always_alarm" and
               row["never_alarm"]["policy_kind"] == "never_alarm" for row in rows)
    equal_ratio = {(row["alpha"], row["beta"]): row["optimal"]["policy_index"] for row in rows}
    assert equal_ratio[(.05, .01)] == equal_ratio[(.25, .05)]
