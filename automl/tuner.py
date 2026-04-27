"""Hyperparameter Optimization Module.

Uses Optuna for Bayesian hyperparameter optimization.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Optimize hyperparameters using Optuna.

    Parameters
    ----------
    model_class : type[BaseEstimator]
        The model class to optimize.
    problem_type : str
        'classification' or 'regression'.
    n_trials : int
        Number of optimization trials.
    time_limit : int | None
        Time limit in seconds (None = unlimited).
    cv_folds : int
        Number of cross-validation folds.
    scoring_metric : str
        Scoring metric for optimization.
    sampler : str
        Optuna sampler ('tpe', 'cmaes').
    random_seed : int
        Random seed.
    """

    def __init__(
        self,
        model_class: type[BaseEstimator],
        problem_type: str = "classification",
        n_trials: int = 50,
        time_limit: int | None = 300,
        cv_folds: int = 5,
        scoring_metric: str = "accuracy",
        sampler: str = "tpe",
        random_seed: int = 42,
    ):
        self.model_class = model_class
        self.problem_type = problem_type
        self.n_trials = n_trials
        self.time_limit = time_limit
        self.cv_folds = cv_folds
        self.scoring_metric = scoring_metric
        self.random_seed = random_seed

        # Define the search space for this model type
        self._define_search_space()

        self.study_: optuna.study.Study | None = None
        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = 0.0
        self.best_model_: BaseEstimator | None = None

    def _define_search_space(self) -> None:
        """Define hyperparameter search space based on model class."""
        self.search_space: dict[str, Callable] = {}

        model_name = self.model_class.__name__

        if "RandomForest" in model_name:
            self.search_space = {
                "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 500),
                "max_depth": lambda trial: trial.suggest_int("max_depth", 5, 50, step=5),
                "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": lambda trial: trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": lambda trial: trial.suggest_categorical("bootstrap", [True, False]),
            }

        elif "GradientBoosting" in model_name or "GB" in model_name:
            self.search_space = {
                "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 20),
                "subsample": lambda trial: trial.suggest_float("subsample", 0.5, 1.0),
                "max_features": lambda trial: trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

        elif "DecisionTree" in model_name:
            self.search_space = {
                "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": lambda trial: trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

        elif "LogisticRegression" in model_name or "Linear" in model_name or "Ridge" in model_name or "Lasso" in model_name or "ElasticNet" in model_name:
            self.search_space = {
                "C": lambda trial: trial.suggest_float("C", 0.001, 10.0, log=True),
                "penalty": lambda trial: trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": lambda trial: trial.suggest_categorical("solver", ["liblinear", "saga"]),
            }

        elif "SVM" in model_name or "SVR" in model_name or "SVC" in model_name:
            self.search_space = {
                "C": lambda trial: trial.suggest_float("C", 0.01, 100.0, log=True),
                "kernel": lambda trial: trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
                "gamma": lambda trial: trial.suggest_categorical("gamma", ["scale", "auto"]),
            }

        elif "MLP" in model_name:
            self.search_space = {
                "hidden_layer_sizes": lambda trial: tuple(
                    trial.suggest_int(f"layer_{i}_size", 32, 256) for i in range(trial.suggest_int("n_layers", 1, 3))
                ),
                "alpha": lambda trial: trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": lambda trial: trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "batch_size": lambda trial: trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            }

        else:
            logger.warning(f"No search space defined for {model_name}. Using defaults.")

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for Optuna optimization."""
        # Sample hyperparameters
        params = {}
        for name, suggest_fn in self.search_space.items():
            params[name] = suggest_fn(trial)

        # Create model instance with sampled parameters
        try:
            model = self.model_class(**params, random_state=self.random_seed)
        except TypeError:
            # Some models have different parameter names
            param_keys = set(self.model_class.__init__.__code__.co_varnames)
            filtered_params = {k: v for k, v in params.items() if k in param_keys}
            model = self.model_class(**filtered_params, random_state=self.random_seed)

        # Cross-validate
        cv_scores = cross_val_score(
            model, X, y,
            cv=self.cv_folds, scoring=self.scoring_metric, n_jobs=-1
        )
        return float(np.mean(cv_scores))

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        study_name: str = "automl_study",
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Returns
        -------
        dict
            Contains 'best_params', 'best_score', and 'study' objects.
        """
        start_time = time.time()

        sampler_map = {
            "tpe": TPESampler(seed=self.random_seed),
            "cmaes": optuna.samplers.CmaEsSampler(seed=self.random_seed),
        }
        sampler = sampler_map.get("tpe", TPESampler(seed=self.random_seed))

        # Determine optimization direction
        reverse_metric = self.scoring_metric in ("neg_mean_squared_error", "neg_log_loss", "neg_mean_absolute_error")
        direction = "minimize" if reverse_metric else "maximize"

        self.study_ = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=study_name,
            load_if_exists=True,
        )

        # Run optimization
        def objective_wrapper(trial: optuna.Trial) -> float:
            score = self._objective(trial, X, y)
            if self.time_limit:
                elapsed = time.time() - start_time
                if elapsed > self.time_limit:
                    raise optuna.TrialPruned()
            return score

        try:
            self.study_.optimize(objective_wrapper, n_trials=self.n_trials, show_progress_bar=False)
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user.")

        # Extract best results
        if self.study_.best_trial is not None:
            self.best_score_ = self.study_.best_value
            self.best_params_ = self.study_.best_params
            self.best_model_ = self.model_class(
                **self.best_params_, random_state=self.random_seed
            )
            self.best_model_.fit(X, y)  # Fit on full training data
            logger.info(f"Best score: {self.best_score_:.4f}")
            logger.info(f"Best params: {self.best_params_}")
        else:
            raise RuntimeError("Optimization produced no results.")

        elapsed = time.time() - start_time
        logger.info(f"Optimization completed in {elapsed:.1f}s ({len(self.study_.trials)} trials)")

        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "study": self.study_,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HyperparameterTuner":
        """Convenience method: optimize and fit the best model."""
        self.optimize(X, y)
        if self.best_model_ is not None:
            self.best_model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best model."""
        if self.best_model_ is None:
            raise RuntimeError("Model not fitted yet. Call optimize() or fit() first.")
        return self.best_model_.predict(X)

    def get_results(self) -> list[dict[str, Any]]:
        """Get all trial results."""
        if self.study_ is None:
            return []
        return [
            {
                "trial": t.number,
                "value": t.value,
                "params": dict(t.params),
                "state": str(t.state),
            }
            for t in self.study_.trials
        ]
