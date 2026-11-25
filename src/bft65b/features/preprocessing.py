"""Unfitted, training-fold-only preprocessing for the seven approved features."""
from __future__ import annotations

from decimal import Decimal
import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bft65b.features.contract import FEATURE_NAMES


PREPROCESSING_VERSION = "numeric_median_scale_v1"
PREDICTOR_COLUMNS = FEATURE_NAMES


class NumericFeatureSelector(TransformerMixin, BaseEstimator):
    """Select only approved predictors and convert exact values to float64.

    Conversion is deterministic and learns nothing. Decimal source/derived
    precision is retained in the saved Arrow feature table; sklearn's numeric
    estimators consume float64. Nulls become NaN for training-fold imputation.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "NumericFeatureSelector":
        self._check_columns(X)
        self.n_features_in_ = len(PREDICTOR_COLUMNS)
        self.feature_names_in_ = np.asarray(PREDICTOR_COLUMNS, dtype=object)
        return self

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Preprocessing requires a pandas DataFrame")
        missing = [name for name in PREDICTOR_COLUMNS if name not in X.columns]
        if missing:
            raise ValueError(f"Required predictor columns are missing: {missing}")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        result = np.empty((len(X), len(PREDICTOR_COLUMNS)), dtype=np.float64)
        for column, name in enumerate(PREDICTOR_COLUMNS):
            for row, value in enumerate(X[name]):
                if (value is None or value is pd.NA or
                        (isinstance(value, (float, np.floating)) and math.isnan(float(value)))):
                    result[row, column] = np.nan
                elif type(value) in (Decimal, float, int, np.float64, np.int64):
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"Nonfinite predictor {name} at row {row}")
                    result[row, column] = number
                else:
                    raise TypeError(f"Unsupported predictor {name} type at row {row}: {type(value).__name__}")
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(PREDICTOR_COLUMNS, dtype=object)


def make_preprocessing_pipeline() -> Pipeline:
    """Return an unfitted pipeline; callers must fit on a training fold only."""
    return Pipeline([
        ("predictors", NumericFeatureSelector()),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
