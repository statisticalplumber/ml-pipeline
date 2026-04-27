"""Model Selector Module.

Manages multiple ML models, trains them, and selects the best one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Stores results from a single model training run."""
    model_name: str
    model: Any
    best_params: dict[str, Any]
    cv_score: float
    cv_std: float
    is_classifier: bool = True


class ModelSelector:
    """Select and train multiple models.

    Parameters
    ----------
    problem_type : str
        'classification' or 'regression'.
    include_linear : bool
        Include linear models.
    include_tree : bool
        Include tree-based models.
    include_svm : bool
        Include SVM models.
    include_ensemble : bool
        Include ensemble models.
    include_neural_network : bool
        Include neural network models.
    random_seed : int = 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        problem_type: str = "classification",
        include_linear: bool = True,
        include_tree: bool = True,
        include_svm: bool = False,
        include_ensemble: bool = True,
        include_neural_network: bool = False,
        random_seed: int = 42,
    ):
        self.problem_type = problem_type
        self.include_linear = include_linear
        self.include_tree = include_tree
        self.include_svm = include_svm
        self.include_ensemble = include_ensemble
        self.include_neural_network = include_neural_network
        self.random_seed = random_seed

        self.results_: list[ModelResult] = []
        self.best_result_: ModelResult | None = None

    def get_models(self) -> dict[str, Any]:
        """Return a dictionary of model instances to try."""
        models: dict[str, Any] = {}

        if self.problem_type == "classification":
            if self.include_linear:
                models["LogisticRegression"] = LogisticRegression(
                    max_iter=1000, random_state=self.random_seed, C=1.0
                )
            if self.include_tree:
                models["RandomForest"] = RandomForestClassifier(
                    n_estimators=100, random_state=self.random_seed
                )
                models["DecisionTree"] = DecisionTreeClassifier(
                    random_state=self.random_seed
                )
                models["GradientBoosting"] = GradientBoostingClassifier(
                    n_estimators=100, random_state=self.random_seed
                )
            if self.include_svm:
                models["SVM"] = SVC(
                    probability=True, random_state=self.random_seed, C=1.0
                )
            if self.include_ensemble:
                base_models = []
                if "RandomForest" in models:
                    base_models.append(("rf", RandomForestClassifier(n_estimators=50, random_state=self.random_seed)))
                if "GradientBoosting" in models:
                    base_models.append(("gb", GradientBoostingClassifier(n_estimators=50, random_state=self.random_seed)))
                if base_models:
                    models["VotingEnsemble"] = VotingClassifier(
                        estimators=base_models, voting="soft"
                    )
            if self.include_neural_network:
                models["MLP"] = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=self.random_seed,
                    early_stopping=True,
                )

        elif self.problem_type == "regression":
            if self.include_linear:
                models["LinearRegression"] = LinearRegression()
                models["Ridge"] = Ridge(alpha=1.0, random_state=self.random_seed)
                models["Lasso"] = Lasso(alpha=0.1, max_iter=1000, random_state=self.random_seed)
                models["ElasticNet"] = ElasticNet(
                    max_iter=1000, random_state=self.random_seed, alpha=0.1, l1_ratio=0.5
                )
            if self.include_tree:
                models["RandomForest"] = RandomForestRegressor(
                    n_estimators=100, random_state=self.random_seed
                )
                models["DecisionTree"] = DecisionTreeRegressor(
                    random_state=self.random_seed
                )
                models["GradientBoosting"] = GradientBoostingRegressor(
                    n_estimators=100, random_state=self.random_seed
                )
            if self.include_svm:
                models["SVR"] = SVR(C=1.0, kernel="rbf")
            if self.include_ensemble:
                base_models = []
                if "RandomForest" in models:
                    base_models.append(("rf", RandomForestRegressor(n_estimators=50, random_state=self.random_seed)))
                if "GradientBoosting" in models:
                    base_models.append(("gb", GradientBoostingRegressor(n_estimators=50, random_state=self.random_seed)))
                if base_models:
                    models["VotingEnsemble"] = VotingRegressor(estimators=base_models)
            if self.include_neural_network:
                models["MLP"] = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=self.random_seed,
                    early_stopping=True,
                )

        return models

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scoring_metric: str = "accuracy",
    ) -> list[ModelResult]:
        """Train all configured models and collect results."""
        from sklearn.model_selection import cross_val_score

        models = self.get_models()
        self.results_ = []

        is_classifier = self.problem_type == "classification"

        for name, model in models.items():
            logger.info(f"Training: {name}")
            try:
                # Quick cross-validation to estimate performance
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=3, scoring=scoring_metric, n_jobs=-1
                )
                cv_score = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))

                # Also train on full training set for final model
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train)

                result = ModelResult(
                    model_name=name,
                    model=model_clone,
                    best_params=self._get_params(model),
                    cv_score=cv_score,
                    cv_std=cv_std,
                    is_classifier=is_classifier,
                )
                self.results_.append(result)
                logger.info(f"  {name}: CV={cv_score:.4f} ± {cv_std:.4f}")

            except Exception as e:
                logger.warning(f"  {name} failed: {e}")

        if not self.results_:
            raise RuntimeError("All models failed to train. Check your data and configuration.")

        # Sort by score (descending for accuracy/F1, ascending for MSE/MAE)
        reverse = scoring_metric in ("accuracy", "f1", "precision", "recall", "roc_auc")
        self.results_.sort(key=lambda r: r.cv_score, reverse=reverse)
        self.best_result_ = self.results_[0]

        logger.info(f"Best model: {self.best_result_.model_name} (score={self.best_result_.cv_score:.4f})")
        return self.results_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best model."""
        if self.best_result_ is None:
            raise RuntimeError("No models have been trained yet.")
        return self.best_result_.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the best model."""
        if self.best_result_ is None:
            raise RuntimeError("No models have been trained yet.")
        model = self.best_result_.model
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        raise AttributeError(f"Model '{self.best_result_.model_name}' does not support predict_proba")

    def _clone_model(self, model: Any) -> Any:
        """Clone a model with its current parameters."""
        import copy
        params = model.get_params()
        return type(model)(**params)

    def _get_params(self, model: Any) -> dict[str, Any]:
        """Get model parameters as a serializable dict."""
        try:
            params = model.get_params()
            # Convert numpy types to native Python types
            for k, v in params.items():
                if hasattr(v, "item"):
                    params[k] = v.item()
                elif isinstance(v, (np.integer,)):
                    params[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    params[k] = float(v)
            return params
        except Exception:
            return {}

    def get_results_table(self) -> list[dict[str, Any]]:
        """Return results as a list of dictionaries for reporting."""
        return [
            {
                "model": r.model_name,
                "cv_score": round(r.cv_score, 4),
                "cv_std": round(r.cv_std, 4),
                "params": r.best_params,
            }
            for r in self.results_
        ]
