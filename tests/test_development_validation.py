"""Synthetic guards for development-only forward model validation."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bft65b.features.preprocessing import PREDICTOR_COLUMNS
from bft65b.modelling.development import (
    CANDIDATES, _period_indices, build_folds, candidate_pipeline,
    fit_predict_fold, fold_metrics, guard_development_rows,
    select_candidate, validate_oof_rows,
)
from bft65b.modelling import development


def _quarters():
    return [date(y, m, {3: 31, 6: 30, 9: 30, 12: 31}[m])
            for y in range(1984, 2013) for m in (3, 6, 9, 12)]


def _schedule_inputs():
    dates = _quarters()
    summary = [{"quarter": q.strftime("%Y%m%d"), "period": "development",
                "model_rows": 11, "positive_rows": 1, "negative_rows": 10}
               for q in dates]
    links = [{"CERT": i + 1, "Q": q, "SOURCE_ID": f"event-{i}",
              "PERIOD": "development"} for i, q in enumerate(dates)]
    return summary, links


def _model_rows():
    train_dates = [date(2010, 3, 31), date(2010, 6, 30)]
    val_dates = [date(2011, 9, 30), date(2011, 12, 31)]
    rows = []
    for i in range(12):
        q = train_dates[i % 2] if i < 8 else val_dates[i % 2]
        row = {name: float(i + j) for j, name in enumerate(PREDICTOR_COLUMNS)}
        row.update({"CERT": i + 1, "Q": q, "Y": i % 2,
                    "FAILDATE": date(2020, 1, 1), "SOURCE_ID": "audit-only"})
        row["asset_growth_qoq_pct"] = None if i < 8 else float(i)
        row["pca_leverage_reported_pct"] = 1.0  # constant training feature
        rows.append(row)
    train = pd.DataFrame(rows[:8])
    validation = pd.DataFrame(rows[8:])
    allowed = {(row["CERT"], row["Q"]) for row in rows}
    fold = {"id": "synthetic", "train_start": "2010-03-31",
            "train_end": "2010-06-30", "gap_start": "2010-09-30",
            "gap_end": "2011-06-30", "validation_start": "2011-09-30",
            "validation_end": "2011-12-31", "training_target_right_end": "2011-06-30"}
    return train, validation, allowed, fold


def test_frozen_expanding_folds_have_exact_gap_event_isolation_and_coverage():
    summary, links = _schedule_inputs()
    folds = build_folds(summary, links)
    assert len(folds) == 3
    assert [(f["train_end"], f["validation_start"], f["validation_end"])
            for f in folds] == [
        ("1999-12-31", "2001-03-31", "2004-12-31"),
        ("2003-12-31", "2005-03-31", "2008-12-31"),
        ("2007-12-31", "2009-03-31", "2012-12-31")]
    assert all(f["validation_distinct_events"] == 16 for f in folds)
    assert all(f["validation_positive_rows"] == 16 for f in folds)
    assert all(_period_indices(f)[1] - _period_indices(f)[0] - 1 == 4 for f in folds)
    assert folds[0]["validation_end"] < folds[1]["validation_start"]


def test_insufficient_gap_and_shared_event_fail():
    summary, links = _schedule_inputs()
    folds = build_folds(summary, links)
    bad = dict(folds[0], train_end="2000-03-31")
    with pytest.raises(ValueError, match="four complete"):
        _period_indices(bad)
    bad = dict(folds[0], training_target_right_end="2001-03-31")
    with pytest.raises(ValueError, match="target window"):
        _period_indices(bad)
    links[-1] = dict(links[-1], SOURCE_ID=links[0]["SOURCE_ID"])
    with pytest.raises(ValueError, match="shared failure event"):
        build_folds(summary, links)


def test_portable_development_plan_cannot_fall_back_to_historical_split(tmp_path):
    (tmp_path / "final_report_bft65b.pdf").write_bytes(b"synthetic")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/source_vintage_bundle.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit split run"):
        development.plan(tmp_path, "portable-new-plan")
    assert not (tmp_path / "runs/task_5_2_development/portable-new-plan").exists()


def test_single_class_fold_fails_without_dropping_it():
    summary, links = _schedule_inputs()
    for row in summary[-16:]:
        row["positive_rows"] = 0
        row["model_rows"] = row["negative_rows"]
    with pytest.raises(ValueError, match="single class"):
        build_folds(summary, links)
    metric = fold_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert metric["average_precision"] is None
    assert metric["roc_auc"] is None
    assert metric["undefined_reason"] == "single_class_validation"


def test_gap_and_test_keys_rejected_at_fit_and_scoring_entry_points():
    train, validation, allowed, fold = _model_rows()
    gap = train.iloc[:1].copy()
    gap.loc[:, "Q"] = date(2010, 9, 30)
    with pytest.raises(ValueError, match="gap/test"):
        fit_predict_fold(pd.concat([train, gap]), validation, allowed, fold, CANDIDATES[0])
    test = validation.iloc[:1].copy()
    test.loc[:, "Q"] = date(2014, 3, 31)
    with pytest.raises(ValueError, match="gap/test"):
        fit_predict_fold(train, pd.concat([validation, test]), allowed, fold, CANDIDATES[0])


def test_heldout_perturbation_does_not_change_fit_and_training_changes_do():
    train, validation, allowed, fold = _model_rows()
    fitted, scores, _ = fit_predict_fold(train, validation, allowed, fold, CANDIDATES[0])
    extreme = validation.copy()
    extreme["asset_size"] = 1e12
    extreme["asset_growth_qoq_pct"] = -1e9
    extreme["Y"] = 1 - extreme["Y"]
    rerun, new_scores, _ = fit_predict_fold(train, extreme, allowed, fold, CANDIDATES[0])
    a = fitted.named_steps["preprocessing"]
    b = rerun.named_steps["preprocessing"]
    np.testing.assert_array_equal(a.named_steps["imputer"].statistics_,
                                  b.named_steps["imputer"].statistics_)
    np.testing.assert_array_equal(a.named_steps["scaler"].mean_,
                                  b.named_steps["scaler"].mean_)
    np.testing.assert_array_equal(fitted.named_steps["estimator"].coef_,
                                  rerun.named_steps["estimator"].coef_)
    assert not np.array_equal(scores, new_scores)
    changed_train = train.copy()
    changed_train.loc[0, "asset_size"] = 1e8
    changed, _, _ = fit_predict_fold(changed_train, validation, allowed, fold, CANDIDATES[0])
    assert not np.array_equal(a.named_steps["scaler"].mean_,
                              changed.named_steps["preprocessing"].named_steps["scaler"].mean_)
    assert not np.array_equal(fitted.named_steps["estimator"].coef_,
                              changed.named_steps["estimator"].coef_)


def test_all_missing_column_stable_allowlist_and_deterministic_scores():
    train, validation, allowed, fold = _model_rows()
    first, scores, metrics = fit_predict_fold(train, validation, allowed, fold, CANDIDATES[0])
    second, repeat, _ = fit_predict_fold(train, validation, allowed, fold, CANDIDATES[0])
    np.testing.assert_array_equal(scores, repeat)
    assert first.named_steps["preprocessing"].get_feature_names_out().tolist() == list(PREDICTOR_COLUMNS)
    assert first.named_steps["preprocessing"].transform(train[list(PREDICTOR_COLUMNS)]).shape[1] == 7
    assert first.named_steps["preprocessing"].named_steps["imputer"].statistics_[-2] == 0
    assert "CERT" not in PREDICTOR_COLUMNS and "Q" not in PREDICTOR_COLUMNS
    assert "Y" not in PREDICTOR_COLUMNS and "FAILDATE" not in PREDICTOR_COLUMNS
    assert metrics["average_precision"] is not None
    assert all(0 <= score <= 1 for score in scores)
    assert candidate_pipeline(CANDIDATES[1]).named_steps["estimator"].early_stopping is False


def test_deliberate_combined_period_fit_is_detected_by_role_guard():
    train, validation, allowed, fold = _model_rows()
    combined = pd.DataFrame([*train.to_dict("records"), *validation.to_dict("records")])
    with pytest.raises(ValueError, match="gap/test"):
        guard_development_rows(combined, allowed,
                               start=date(2010, 3, 31), end=date(2010, 6, 30))
    clean, _, _ = fit_predict_fold(train, validation, allowed, fold, CANDIDATES[0])
    contaminated = candidate_pipeline(CANDIDATES[0]).fit(
        combined[list(PREDICTOR_COLUMNS)], combined["Y"])
    assert not np.array_equal(
        clean.named_steps["preprocessing"].named_steps["scaler"].mean_,
        contaminated.named_steps["preprocessing"].named_steps["scaler"].mean_)


def test_oof_duplicate_invalid_score_and_selection_tie_break():
    row = {"CERT": 1, "Q": date(2011, 9, 30), "Y": 1, "FOLD_ID": "f1",
           "CANDIDATE_ID": CANDIDATES[0]["id"], "SCORE": 0.8}
    validate_oof_rows([row], expected_rows=1, candidate_id=CANDIDATES[0]["id"])
    with pytest.raises(ValueError, match="OOF"):
        validate_oof_rows([row, row], expected_rows=2, candidate_id=CANDIDATES[0]["id"])
    with pytest.raises(ValueError, match="OOF"):
        validate_oof_rows([dict(row, SCORE=float("nan"))], expected_rows=1,
                          candidate_id=CANDIDATES[0]["id"])
    results = [{"candidate_id": candidate["id"], "fold_id": f"fold_{i}",
                "average_precision": 0.2} for candidate in CANDIDATES for i in range(1, 4)]
    selected, ranking = select_candidate(results)
    assert selected == "bounded_hist_gradient_boosting"  # lexical tie break
    assert len(ranking) == 2
    results[0]["average_precision"] = None
    with pytest.raises(ValueError, match="Undefined"):
        select_candidate(results)
