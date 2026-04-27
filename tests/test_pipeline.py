"""Tests for the AutoML pipeline module."""

import numpy as np
import pandas as pd
import pytest

from automl.pipeline import AutoMLClassifier, AutoMLRegressor


class TestAutoMLClassifier:
    def test_basic_classification(self):
        """Test basic classification pipeline."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=300, n_features=10, n_informative=5,
            n_classes=2, random_state=42
        )
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]

        automl = AutoMLClassifier(
            time_limit=60,
            n_trials=5,
            cv_folds=3,
            random_seed=42,
        )
        automl.fit(X_train, y_train)
        predictions = automl.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_classification_with_dataframe(self):
        """Test classification with pandas DataFrame input."""
        from sklearn.datasets import load_iris

        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

        automl = AutoMLClassifier(
            time_limit=60,
            n_trials=5,
            cv_folds=3,
            random_seed=42,
        )
        automl.fit(X[:100], y[:100])
        predictions = automl.predict(X[100:120])
        assert len(predictions) == 20

    def test_predict_proba(self):
        """Test probability prediction."""
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=300, n_features=5, random_state=42)
        automl = AutoMLClassifier(time_limit=30, n_trials=3, cv_folds=2, random_seed=42)
        automl.fit(X[:200], y[:200])
        proba = automl.predict_proba(X[200:210])
        assert proba.shape == (10, 2)

    def test_evaluate(self):
        """Test model evaluation."""
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=300, n_features=5, random_state=42)
        automl = AutoMLClassifier(time_limit=30, n_trials=3, cv_folds=2, random_seed=42)
        automl.fit(X[:200], y[:200])
        metrics = automl.evaluate(X[200:], y[200:])
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0

    def test_not_fitted_raises(self):
        """Test that predict raises error before fitting."""
        automl = AutoMLClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            automl.predict(np.random.randn(5, 10))


class TestAutoMLRegressor:
    def test_basic_regression(self):
        """Test basic regression pipeline."""
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=300, n_features=10, random_state=42)
        automl = AutoMLRegressor(
            time_limit=60,
            n_trials=5,
            cv_folds=3,
            random_seed=42,
        )
        automl.fit(X[:200], y[:200])
        predictions = automl.predict(X[200:])
        assert len(predictions) == 100

    def test_regression_with_dataframe(self):
        """Test regression with pandas DataFrame input."""
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

        automl = AutoMLRegressor(
            time_limit=60,
            n_trials=5,
            cv_folds=3,
            random_seed=42,
        )
        automl.fit(X[:300], y[:300])
        predictions = automl.predict(X[300:320])
        assert len(predictions) == 20

    def test_evaluate_regression(self):
        """Test regression evaluation metrics."""
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=300, n_features=5, random_state=42)
        automl = AutoMLRegressor(time_limit=30, n_trials=3, cv_folds=2, random_seed=42)
        automl.fit(X[:200], y[:200])
        metrics = automl.evaluate(X[200:], y[200:])
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics

    def test_not_fitted_raises(self):
        """Test that predict raises error before fitting."""
        automl = AutoMLRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            automl.predict(np.random.randn(5, 10))
