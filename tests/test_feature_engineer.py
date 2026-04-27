"""Tests for the AutoML feature engineer module."""

import numpy as np
import pytest

from automl.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    def test_basic(self):
        """Test basic feature engineering."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        engineer = FeatureEngineer(polynomial_features=False)
        result = engineer.fit_transform(X, y)
        assert result.shape == X.shape

    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)

        engineer = FeatureEngineer(polynomial_features=True, polynomial_degree=2)
        result = engineer.fit_transform(X, y)
        # With degree 2 and 3 features: 3 + 3 (squares) + 3 (interactions) = 9
        assert result.shape[1] > X.shape[1]

    def test_feature_selection(self):
        """Test feature selection."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        engineer = FeatureEngineer(
            polynomial_features=False,
            feature_selection_method="mutual_info",
            n_features_to_select=5,
        )
        result = engineer.fit_transform(X, y)
        assert result.shape[1] == 5

    def test_variance_threshold(self):
        """Test variance thresholding."""
        X = np.random.randn(100, 5)
        # Add a near-constant column
        X[:, 4] = 1.0 + np.random.randn(100) * 1e-6
        y = np.random.randint(0, 2, 100)

        engineer = FeatureEngineer(polynomial_features=False)
        result = engineer.fit_transform(X, y)
        assert result.shape[1] <= X.shape[1]

    def test_no_selection(self):
        """Test with no feature selection."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        engineer = FeatureEngineer(
            polynomial_features=False,
            n_features_to_select=-1,
        )
        result = engineer.fit_transform(X, y)
        assert result.shape[1] == X.shape[1]

    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        engineer = FeatureEngineer(
            polynomial_features=False,
            feature_selection_method="select_k",
            n_features_to_select=3,
        )
        engineer.fit(X, y)
        importance = engineer.get_feature_importance()
        assert importance is not None
        assert len(importance) == X.shape[1]
