"""Tests for the AutoML tuner module."""

import numpy as np
import pytest

from automl.tuner import HyperparameterTuner
from sklearn.ensemble import RandomForestClassifier


class TestHyperparameterTuner:
    def test_basic_optimization(self):
        """Test basic hyperparameter optimization."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            problem_type="classification",
            n_trials=5,
            time_limit=None,
            cv_folds=3,
            random_seed=42,
        )
        results = tuner.optimize(X, y)
        assert "best_params" in results
        assert "best_score" in results
        assert "study" in results

    def test_best_params_not_empty(self):
        """Test that best params are found."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            problem_type="classification",
            n_trials=5,
            cv_folds=3,
            random_seed=42,
        )
        results = tuner.optimize(X, y)
        assert len(results["best_params"]) > 0

    def test_predict_after_optimize(self):
        """Test prediction after optimization."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            problem_type="classification",
            n_trials=3,
            cv_folds=2,
            random_seed=42,
        )
        tuner.optimize(X, y)
        predictions = tuner.predict(X[:5])
        assert len(predictions) == 5

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fitting."""
        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            problem_type="classification",
            n_trials=3,
            cv_folds=2,
            random_seed=42,
        )
        with pytest.raises(RuntimeError):
            tuner.predict(np.random.randn(5, 10))

    def test_get_results(self):
        """Test retrieving all trial results."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            problem_type="classification",
            n_trials=5,
            cv_folds=2,
            random_seed=42,
        )
        tuner.optimize(X, y)
        results = tuner.get_results()
        assert len(results) == 5

    def test_fit_method(self):
        """Test the fit() convenience method."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        tuner = HyperparameterTuner(
            model_class=RandomForestClassifier,
            problem_type="classification",
            n_trials=3,
            cv_folds=2,
            random_seed=42,
        )
        tuner.fit(X, y)
        predictions = tuner.predict(X[:5])
        assert len(predictions) == 5
