"""Tests for the AutoML model selector module."""

import numpy as np
import pytest

from automl.model_selector import ModelSelector


class TestModelSelector:
    def test_classification_models(self):
        """Test that classification models are available."""
        selector = ModelSelector(problem_type="classification")
        models = selector.get_models()
        assert len(models) > 0
        assert "LogisticRegression" in models

    def test_regression_models(self):
        """Test that regression models are available."""
        selector = ModelSelector(problem_type="regression")
        models = selector.get_models()
        assert len(models) > 0
        assert "LinearRegression" in models

    def test_train_classification(self):
        """Test training classification models."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        selector = ModelSelector(problem_type="classification")
        results = selector.train_all(X, y, X, y, scoring_metric="accuracy")
        assert len(results) > 0
        assert selector.best_result_ is not None
        assert selector.best_result_.cv_score >= 0

    def test_train_regression(self):
        """Test training regression models."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randn(200)

        selector = ModelSelector(problem_type="regression")
        results = selector.train_all(X, y, X, y, scoring_metric="r2")
        assert len(results) > 0
        assert selector.best_result_ is not None

    def test_predict(self):
        """Test prediction with best model."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        selector = ModelSelector(problem_type="classification")
        selector.train_all(X, y, X, y, scoring_metric="accuracy")
        predictions = selector.predict(X[:5])
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)

    def test_results_table(self):
        """Test results table generation."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        selector = ModelSelector(problem_type="classification")
        selector.train_all(X, y, X, y, scoring_metric="accuracy")
        table = selector.get_results_table()
        assert len(table) > 0
        for entry in table:
            assert "model" in entry
            assert "cv_score" in entry

    def test_disabled_models(self):
        """Test with some model types disabled."""
        selector = ModelSelector(
            problem_type="classification",
            include_linear=False,
            include_svm=True,
            include_ensemble=False,
        )
        models = selector.get_models()
        assert "LogisticRegression" not in models
        assert len(models) >= 1  # Should still have some models
