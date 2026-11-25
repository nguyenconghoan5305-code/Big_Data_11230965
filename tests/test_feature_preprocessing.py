"""Training-only fit and stable numeric predictor contract."""
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from bft65b.features.contract import FEATURE_NAMES
from bft65b.features.preprocessing import (
    NumericFeatureSelector, PREDICTOR_COLUMNS, make_preprocessing_pipeline,
)


def frame(asset, growth):
    rows = []
    for a, g in zip(asset, growth):
        row = {name: None for name in FEATURE_NAMES}
        row.update(asset_size=Decimal(a), asset_growth_qoq_pct=g,
                   roa_reported_pct=1.0, pca_leverage_reported_pct=8.0,
                   equity_to_assets_pct=Decimal("10.000000"),
                   deposits_to_assets_pct=Decimal("80.000000"))
        row.update(CERT=11, Q="2022-12-31", Y=0, FAILDATE="future",
                   MEMBERSHIP="eligible", SOURCE_ID_DIAGNOSTIC="audit")
        rows.append(row)
    return pd.DataFrame(rows)


def test_forward_fold_fit_ignores_heldout_extremes_and_metadata():
    train = frame(["100.0000", "200.0000"], [None, None])
    holdout = frame(["1000000.0000"], [Decimal("25.000000")])
    model = make_preprocessing_pipeline()
    assert not hasattr(model.named_steps["imputer"], "statistics_")
    transformed_train = model.fit_transform(train)
    stats = model.named_steps["imputer"].statistics_.copy()
    means = model.named_steps["scaler"].mean_.copy()
    first = model.transform(holdout)
    altered = holdout.copy()
    altered["asset_size"] = Decimal("999999999.0000")
    altered["FAILDATE"] = "different_future"
    second = model.transform(altered)
    np.testing.assert_array_equal(stats, model.named_steps["imputer"].statistics_)
    np.testing.assert_array_equal(means, model.named_steps["scaler"].mean_)
    assert transformed_train.shape == (2, 7)
    assert first.shape == second.shape == (1, 7)
    assert first[0, 0] != second[0, 0]
    assert first[0, 5] == second[0, 5]
    assert list(model.get_feature_names_out()) == list(PREDICTOR_COLUMNS)
    assert model.named_steps["imputer"].statistics_[5] == 0.0
    assert not set(model.get_feature_names_out()) & {
        "CERT", "Q", "Y", "FAILDATE", "MEMBERSHIP", "SOURCE_ID_DIAGNOSTIC"}


def test_different_forward_training_fold_learns_its_own_statistics():
    early = make_preprocessing_pipeline().fit(frame(["100", "200"], [None, None]))
    later = make_preprocessing_pipeline().fit(frame(["1000", "2000"], [Decimal("2"), Decimal("4")]))
    assert early.named_steps["scaler"].mean_[0] == 150.0
    assert later.named_steps["scaler"].mean_[0] == 1500.0
    assert early.named_steps["imputer"].statistics_[5] == 0.0
    assert later.named_steps["imputer"].statistics_[5] == 3.0


def test_invalid_or_missing_predictors_fail_without_using_outcome():
    selector = NumericFeatureSelector().fit(frame(["1"], [None]))
    with_nan = frame(["1"], [None])
    with_nan["roa_reported_pct"] = np.nan
    assert np.isnan(selector.transform(with_nan)[0, 3])
    with pytest.raises(ValueError, match="missing"):
        selector.transform(frame(["1"], [None]).drop(columns=[FEATURE_NAMES[0]]))
    bad = frame(["1"], [None])
    bad["ROA_reported_pct"] = 1.0
    bad["roa_reported_pct"] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        selector.transform(bad)
    bad["roa_reported_pct"] = "1.0"
    with pytest.raises(TypeError, match="Unsupported"):
        selector.transform(bad)
