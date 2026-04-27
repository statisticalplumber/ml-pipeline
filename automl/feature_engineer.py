"""Feature Engineering Module.

Handles feature generation, transformation, and selection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
)
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Automatic feature engineering and selection.

    Parameters
    ----------
    polynomial_features : bool
        Whether to create polynomial features.
    polynomial_degree : int
        Degree of polynomial features (2 or 3).
    interaction_features : bool
        Whether to create interaction (cross) features.
    feature_selection_method : str
        Method for feature selection ('mutual_info', 'variance', 'select_k').
    n_features_to_select : int
        Number of features to keep (-1 = all).
    k_for_select_k : int = 10
        K parameter for SelectKBest.
    """

    def __init__(
        self,
        polynomial_features: bool = True,
        polynomial_degree: int = 2,
        interaction_features: bool = False,
        feature_selection_method: str = "mutual_info",
        n_features_to_select: int = -1,
        k_for_select_k: int = 10,
    ):
        self.polynomial_features = polynomial_features
        self.polynomial_degree = polynomial_degree
        self.interaction_features = interaction_features
        self.feature_selection_method = feature_selection_method
        self.n_features_to_select = n_features_to_select
        self.k_for_select_k = k_for_select_k

        self.selected_mask_: np.ndarray | None = None
        self.var_mask_: np.ndarray | None = None
        self.poly_: PolynomialFeatures | None = None
        self.selector_: Any = None
        self.n_features_before_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureEngineer":
        """Fit feature engineering pipeline."""
        self.n_features_before_ = X.shape[1]
        logger.info(f"Input features: {self.n_features_before_}")

        X_eng = X.copy()

        # 1. Polynomial features
        if self.polynomial_features:
            self.poly_ = PolynomialFeatures(
                degree=self.polynomial_degree,
                interaction_only=not self.interaction_features,
                include_bias=False,
            )
            X_eng = self.poly_.fit_transform(X_eng)
            logger.info(f"After polynomial features: {X_eng.shape[1]}")

        # 2. Variance threshold (remove near-constant features)
        var_thresh = VarianceThreshold(threshold=1e-4)
        X_var = var_thresh.fit_transform(X_eng)
        self.var_mask_ = var_thresh.get_support()
        if not np.all(self.var_mask_):
            X_eng = X_var
            logger.info(f"After variance threshold: {X_eng.shape[1]}")

        # 3. Feature selection
        self._select_features(X_eng, y)

        logger.info(f"Final features after selection: {X_eng.shape[1]}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted pipeline."""
        X_eng = X.copy()

        if self.poly_ is not None:
            X_eng = self.poly_.transform(X_eng)

        # Apply variance threshold mask (on polynomial-expanded features)
        if self.var_mask_ is not None:
            X_eng = X_eng[:, self.var_mask_]

        if self.selected_mask_ is not None:
            X_eng = X_eng[:, self.selected_mask_]

        return X_eng

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> None:
        """Select top features based on the chosen method."""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        if self.n_features_to_select == 0:
            self.selected_mask_ = np.zeros(X.shape[1], dtype=bool)
            return

        if self.feature_selection_method == "mutual_info":
            is_classification = len(np.unique(y)) <= 10
            k = min(self.n_features_to_select, X.shape[1]) if self.n_features_to_select > 0 else X.shape[1]
            # Use mutual info scores directly
            if is_classification:
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)
            threshold_idx = len(scores) if self.n_features_to_select < 0 else self.n_features_to_select
            sorted_indices = np.argsort(scores)[::-1]
            self.selected_mask_ = np.zeros(X.shape[1], dtype=bool)
            self.selected_mask_[sorted_indices[:threshold_idx]] = True

        elif self.feature_selection_method == "select_k":
            k = min(self.n_features_to_select, X.shape[1]) if self.n_features_to_select > 0 else X.shape[1]
            self.selector_ = SelectKBest(k=k)
            self.selector_.fit(X, y)
            self.selected_mask_ = self.selector_.get_support()

        elif self.feature_selection_method == "variance":
            vt = VarianceThreshold(threshold=0.01)
            vt.fit(X)
            self.selected_mask_ = vt.get_support()
        else:
            # No selection, keep all
            self.selected_mask_ = np.ones(X.shape[1], dtype=bool)

    def get_feature_importance(self) -> np.ndarray | None:
        """Return feature importance scores if available."""
        if self.selector_ is not None and hasattr(self.selector_, "scores_"):
            return self.selector_.scores_
        return None
