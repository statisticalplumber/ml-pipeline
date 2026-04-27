"""Data Preprocessing Module.

Handles missing values, outlier removal, categorical encoding, and numerical scaling.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers using the IQR method."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.bounds_: dict[int, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OutlierRemover":
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.bounds_ = {}
        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.multiplier * IQR
            upper = Q3 + self.multiplier * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            mask = (X_clean[col] >= lower) & (X_clean[col] <= upper)
            n_removed = (~mask).sum()
            if n_removed > 0:
                logger.info(f"Removed {n_removed} outliers from column '{col}'")
        return X_clean


class Preprocessor(BaseEstimator, TransformerMixin):
    """Full preprocessing pipeline for tabular data.

    Parameters
    ----------
    missing_strategy : str
        Imputation strategy for missing values ('mean', 'median', 'most_frequent').
    categorical_encoding : str
        Encoding method for categorical columns ('onehot', 'ordinal', 'label').
    numerical_scaling : str
        Scaling method for numerical columns ('standard', 'minmax', 'robust').
    outlier_iqr_multiplier : float
        IQR multiplier for outlier detection (1.5 = standard).
    """

    def __init__(
        self,
        missing_strategy: str = "median",
        categorical_encoding: str = "onehot",
        numerical_scaling: str = "standard",
        outlier_iqr_multiplier: float = 1.5,
    ):
        self.missing_strategy = missing_strategy
        self.categorical_encoding = categorical_encoding
        self.numerical_scaling = numerical_scaling
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.preprocessor_: ColumnTransformer | None = None
        self.outlier_remover_: OutlierRemover | None = None
        self.feature_names_after_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Preprocessor":
        """Fit the preprocessor on training data."""
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.info(f"Numeric columns ({len(self.numeric_columns_)}): {self.numeric_columns_}")
        logger.info(f"Categorical columns ({len(self.categorical_columns_)}): {self.categorical_columns_}")

        # Outlier removal (optional)
        self.outlier_remover_ = OutlierRemover(multiplier=self.outlier_iqr_multiplier)
        X_imputed = self._impute(X)
        X_cleaned = self.outlier_remover_.fit_transform(X_imputed)

        # Build sklearn preprocessing pipeline
        transformers = []

        if self.numeric_columns_:
            scaling_map = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
            }
            scaler = scaling_map.get(self.numerical_scaling, StandardScaler())
            transformers.append(
                ("num", scaler, self.numeric_columns_)
            )

        if self.categorical_columns_:
            encoders = {
                "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                "label": FunctionTransformer(lambda x: x.astype(int)),
            }
            encoder = encoders.get(self.categorical_encoding, OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            transformers.append(
                ("cat", encoder, self.categorical_columns_)
            )

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )
        self.preprocessor_.fit(X_cleaned)

        # Store feature names after transformation
        self.feature_names_after_ = self._get_feature_names()
        logger.info(f"Output features: {len(self.feature_names_after_)}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline."""
        X_imputed = self._impute(X)
        X_cleaned = self.outlier_remover_.transform(X_imputed)
        return self.preprocessor_.transform(X_cleaned)

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform (only works if preprocessor was created)."""
        if self.preprocessor_ is None:
            raise RuntimeError("Preprocessor not fitted yet.")
        return self.preprocessor_.inverse_transform(X)

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using sklearn SimpleImputer."""
        from sklearn.impute import SimpleImputer as SKImputer

        X_imp = X.copy()

        # Impute numeric columns
        if self.numeric_columns_:
            strat_map = {
                "mean": "mean",
                "median": "median",
                "most_frequent": "most_frequent",
                "drop": "most_frequent",  # fallback
            }
            strategy = strat_map.get(self.missing_strategy, "median")
            num_imputer = SKImputer(strategy=strategy)
            X_imp[self.numeric_columns_] = num_imputer.fit_transform(X_imp[self.numeric_columns_])

        # Impute categorical columns
        if self.categorical_columns_:
            cat_imputer = SKImputer(strategy="most_frequent")
            X_imp[self.categorical_columns_] = cat_imputer.fit_transform(X_imp[self.categorical_columns_])

        return X_imp

    def _get_feature_names(self) -> list[str]:
        """Get feature names after transformation."""
        if self.preprocessor_ is None:
            return []
        try:
            return self.preprocessor_.get_feature_names_out().tolist()
        except Exception:
            return [f"feature_{i}" for i in range(self.transform(pd.DataFrame()).shape[1])]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "missing_strategy": self.missing_strategy,
            "categorical_encoding": self.categorical_encoding,
            "numerical_scaling": self.numerical_scaling,
            "outlier_iqr_multiplier": self.outlier_iqr_multiplier,
        }

    def set_params(self, **params: Any) -> "Preprocessor":
        for key, value in params.items():
            setattr(self, key, value)
        return self
