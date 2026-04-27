"""AutoML Pipeline Module.

Main orchestration module that ties together preprocessing, feature engineering,
model selection, hyperparameter tuning, and evaluation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yaml

from automl.evaluator import Evaluator
from automl.explainer import Explainer
from automl.exporter import ModelExporter
from automl.feature_engineer import FeatureEngineer
from automl.model_selector import ModelSelector, ModelResult
from automl.preprocessor import Preprocessor
from automl.tuner import HyperparameterTuner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AutoMLResult:
    """Container for complete AutoML pipeline results."""
    best_model_name: str
    best_model: Any
    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict[str, Any]]
    evaluation_metrics: dict[str, float]
    feature_importance: dict[str, float] | None = None
    classification_report: str | None = None

    def summary(self) -> str:
        """Return a human-readable summary of results."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║          AutoML Pipeline Results          ║",
            "╠══════════════════════════════════════════╣",
            f"║ Best Model    : {self.best_model_name:<26s} ║",
            f"║ Best Score    : {self.best_score:<26.6f} ║",
            "╠──────────────────────────────────────────╣",
            "║ All Models Ranked:                        ║",
        ]
        for i, result in enumerate(self.all_results, 1):
            name = result.get("model", "Unknown")
            score = result.get("cv_score", 0)
            std = result.get("cv_std", 0)
            lines.append(f"║ {i}. {name:<28s} ║")
            lines.append(f"    Score: {score:.4f} ± {std:.4f}")
        lines.append("╚══════════════════════════════════════════╝")
        return "\n".join(lines)


class AutoMLClassifier:
    """Automated Machine Learning for classification problems.

    Parameters
    ----------
    time_limit : int
        Time limit in seconds for optimization (default: 300).
    n_trials : int
        Number of hyperparameter optimization trials (default: 50).
    cv_folds : int
        Number of cross-validation folds (default: 5).
    random_seed : int
        Random seed for reproducibility (default: 42).
    config_path : str, optional
        Path to YAML configuration file.

    Examples
    --------
    >>> from automl import AutoMLClassifier
    >>> automl = AutoMLClassifier(time_limit=300, n_trials=50)
    >>> automl.fit(X_train, y_train)
    >>> predictions = automl.predict(X_test)
    """

    def __init__(
        self,
        time_limit: int = 300,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_seed: int = 42,
        config_path: str | None = None,
    ):
        self.time_limit = time_limit
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_seed = random_seed

        # Load config
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            cfg = config.get("automl", {})
            self.preprocessing_cfg = cfg.get("preprocessing", {})
            self.feature_engineering_cfg = cfg.get("feature_engineering", {})
            self.models_cfg = cfg.get("models", {})
            self.optimization_cfg = cfg.get("optimization", {})
            self.evaluation_cfg = cfg.get("evaluation", {})
        else:
            self.preprocessing_cfg = {}
            self.feature_engineering_cfg = {}
            self.models_cfg = {}
            self.optimization_cfg = {}
            self.evaluation_cfg = {}

        # Initialize components
        self.preprocessor_ = Preprocessor(**self.preprocessing_cfg)
        self.feature_engineer_ = FeatureEngineer(**self.feature_engineering_cfg)
        self.model_selector_ = ModelSelector(
            problem_type="classification",
            **self.models_cfg,
            random_seed=random_seed,
        )
        self.evaluator_ = Evaluator(
            problem_type="classification",
            primary_metric=self.evaluation_cfg.get("primary_metric", "accuracy"),
            additional_metrics=self.evaluation_cfg.get("additional_metrics", []),
        )

        # Pipeline state
        self.is_fitted_: bool = False
        self.result_: AutoMLResult | None = None
        self.explainer_: Explainer | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series | list,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "AutoMLClassifier":
        """Run the complete AutoML pipeline.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features.
        y : np.ndarray, pd.Series, or list
            Training labels.
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features (for model selection).
        y_val : np.ndarray, optional
            Validation labels.

        Returns
        -------
        AutoMLClassifier
            Self for method chaining.
        """
        start_time = time.time()

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, (list, np.ndarray)):
            y = np.asarray(y)

        logger.info("=" * 60)
        logger.info("Starting AutoML Classification Pipeline")
        logger.info(f"Training samples: {X.shape[0]}, Features: {X.shape[1]}")
        logger.info(f"Classes: {np.unique(y)}")
        logger.info("=" * 60)

        # Step 1: Preprocessing
        logger.info("\n[1/5] Preprocessing data...")
        X_train_processed = self.preprocessor_.fit_transform(X, y)
        logger.info(f"Preprocessed shape: {X_train_processed.shape}")

        # Step 2: Feature Engineering
        logger.info("\n[2/5] Feature engineering...")
        X_train_featured = self.feature_engineer_.fit_transform(X_train_processed, y)
        logger.info(f"Featured shape: {X_train_featured.shape}")

        # Handle validation data
        if X_val is not None and isinstance(X_val, np.ndarray):
            X_val = pd.DataFrame(X_val)
        if X_val is not None:
            X_val_processed = self.preprocessor_.transform(X_val)
            X_val_featured = self.feature_engineer_.transform(X_val_processed)
        else:
            # Use cross-validation instead
            X_val_featured = None
            y_val = None

        # Step 3: Model Selection (baseline)
        logger.info("\n[3/5] Training baseline models...")
        scoring = self.evaluator_.primary_metric or "accuracy"
        results = self.model_selector_.train_all(
            X_train_featured, y,
            X_val_featured, y_val,
            scoring_metric=scoring,
        )

        # Step 4: Hyperparameter Tuning on best models
        logger.info("\n[4/5] Hyperparameter optimization...")
        best_model_class = self.model_selector_.best_result_.model.__class__
        tuner = HyperparameterTuner(
            model_class=best_model_class,
            problem_type="classification",
            n_trials=self.n_trials,
            time_limit=self.time_limit // 2,  # Leave half for final training
            cv_folds=self.cv_folds,
            scoring_metric=scoring,
            random_seed=self.random_seed,
        )

        try:
            tuner_results = tuner.optimize(X_train_featured, y)
            best_tuned_model = tuner.best_model_
            best_tuned_params = tuner.best_params_
            best_tuned_score = tuner.best_score_
            logger.info(f"Tuned model: {best_model_class.__name__} (score={best_tuned_score:.4f})")
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {e}. Using baseline.")
            best_tuned_model = self.model_selector_.best_result_.model
            best_tuned_params = self.model_selector_.best_result_.best_params
            best_tuned_score = self.model_selector_.best_result_.cv_score

        # Step 5: Final Evaluation
        logger.info("\n[5/5] Final evaluation...")
        self.is_fitted_ = True
        self.result_ = AutoMLResult(
            best_model_name=best_tuned_model.__class__.__name__,
            best_model=best_tuned_model,
            best_params=best_tuned_params,
            best_score=best_tuned_score,
            all_results=self.model_selector_.get_results_table(),
            evaluation_metrics={},
        )

        elapsed = time.time() - start_time
        logger.info(f"\nPipeline completed in {elapsed:.1f}s")
        logger.info(self.result_.summary())

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_fitted()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_processed = self.preprocessor_.transform(X)
        X_featured = self.feature_engineer_.transform(X_processed)
        return self.result_.best_model.predict(X_featured)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_processed = self.preprocessor_.transform(X)
        X_featured = self.feature_engineer_.transform(X_processed)
        model = self.result_.best_model
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_featured)
        # Fallback: use decision function for linear models
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_featured)
            from scipy.special import expit
            proba_pos = expit(scores) if scores.ndim == 1 else scores
            if scores.ndim == 1:
                return np.vstack([1 - proba_pos, proba_pos]).T
            return scores
        raise AttributeError(f"Model '{model.__class__.__name__}' does not support predict_proba")

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """Evaluate the model on test data."""
        self._check_fitted()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_processed = self.preprocessor_.transform(X)
        X_featured = self.feature_engineer_.transform(X_processed)

        y_pred = self.result_.best_model.predict(X_featured)
        y_pred = np.round(y_pred).astype(int) if self.evaluator_.problem_type == "classification" else y_pred
        try:
            y_proba = self.result_.best_model.predict_proba(X_featured)
        except AttributeError:
            y_proba = None

        metrics = self.evaluator_.evaluate(y, y_pred, y_proba)
        self.result_.evaluation_metrics = metrics

        # Generate classification report
        self.result_.classification_report = self.evaluator_.generate_classification_report(y, y_pred)

        logger.info(self.evaluator_.report(metrics))
        return metrics

    def explain(self, X: pd.DataFrame | np.ndarray, top_n: int = 10) -> dict[str, float]:
        """Generate SHAP-based feature importance explanation."""
        self._check_fitted()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_processed = self.preprocessor_.transform(X)
        X_featured = self.feature_engineer_.transform(X_processed)

        self.explainer_ = Explainer(
            model=self.result_.best_model,
            X_background=X_featured[:min(100, len(X_featured))],
            problem_type="classification",
        )
        self.explainer_.fit()
        importance = self.explainer_.get_feature_importance(X_featured)
        self.result_.feature_importance = importance

        return importance

    def export_model(
        self,
        output_dir: str = "./models",
        model_name: str = "classifier",
        format: str = "joblib",
    ) -> dict[str, str]:
        """Export the trained model."""
        self._check_fitted()

        metadata = {
            "problem_type": "classification",
            "best_model": self.result_.best_model_name,
            "best_params": self.result_.best_params,
            "best_score": self.result_.best_score,
            "evaluation_metrics": self.result_.evaluation_metrics,
            "feature_importance": self.result_.feature_importance,
            "preprocessing_config": self.preprocessing_cfg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        exporter = ModelExporter(self.result_.best_model, metadata=metadata)
        return exporter.export(output_dir=output_dir, model_name=model_name, format=format)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")


class AutoMLRegressor:
    """Automated Machine Learning for regression problems.

    Parameters
    ----------
    time_limit : int
        Time limit in seconds for optimization (default: 300).
    n_trials : int
        Number of hyperparameter optimization trials (default: 50).
    cv_folds : int
        Number of cross-validation folds (default: 5).
    random_seed : int
        Random seed for reproducibility (default: 42).
    config_path : str, optional
        Path to YAML configuration file.

    Examples
    --------
    >>> from automl import AutoMLRegressor
    >>> automl = AutoMLRegressor(time_limit=300, n_trials=50)
    >>> automl.fit(X_train, y_train)
    >>> predictions = automl.predict(X_test)
    """

    def __init__(
        self,
        time_limit: int = 300,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_seed: int = 42,
        config_path: str | None = None,
    ):
        self.time_limit = time_limit
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_seed = random_seed

        # Load config
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            cfg = config.get("automl", {})
            self.preprocessing_cfg = cfg.get("preprocessing", {})
            self.feature_engineering_cfg = cfg.get("feature_engineering", {})
            self.models_cfg = cfg.get("models", {})
            self.optimization_cfg = cfg.get("optimization", {})
            self.evaluation_cfg = cfg.get("evaluation", {})
        else:
            self.preprocessing_cfg = {}
            self.feature_engineering_cfg = {}
            self.models_cfg = {}
            self.optimization_cfg = {}
            self.evaluation_cfg = {}

        # Initialize components
        self.preprocessor_ = Preprocessor(**self.preprocessing_cfg)
        self.feature_engineer_ = FeatureEngineer(**self.feature_engineering_cfg)
        self.model_selector_ = ModelSelector(
            problem_type="regression",
            **self.models_cfg,
            random_seed=random_seed,
        )
        self.evaluator_ = Evaluator(
            problem_type="regression",
            primary_metric=self.evaluation_cfg.get("primary_metric", "r2"),
            additional_metrics=self.evaluation_cfg.get("additional_metrics", []),
        )

        # Pipeline state
        self.is_fitted_: bool = False
        self.result_: AutoMLResult | None = None
        self.explainer_: Explainer | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series | list,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "AutoMLRegressor":
        """Run the complete AutoML pipeline for regression."""
        start_time = time.time()

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, (list, np.ndarray)):
            y = np.asarray(y, dtype=float)

        logger.info("=" * 60)
        logger.info("Starting AutoML Regression Pipeline")
        logger.info(f"Training samples: {X.shape[0]}, Features: {X.shape[1]}")
        logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        logger.info("=" * 60)

        # Step 1: Preprocessing
        logger.info("\n[1/5] Preprocessing data...")
        X_train_processed = self.preprocessor_.fit_transform(X, y)
        logger.info(f"Preprocessed shape: {X_train_processed.shape}")

        # Step 2: Feature Engineering
        logger.info("\n[2/5] Feature engineering...")
        X_train_featured = self.feature_engineer_.fit_transform(X_train_processed, y)
        logger.info(f"Featured shape: {X_train_featured.shape}")

        # Handle validation data
        if X_val is not None and isinstance(X_val, np.ndarray):
            X_val = pd.DataFrame(X_val)
        if X_val is not None:
            X_val_processed = self.preprocessor_.transform(X_val)
            X_val_featured = self.feature_engineer_.transform(X_val_processed)
        else:
            X_val_featured = None
            y_val = None

        # Step 3: Model Selection
        logger.info("\n[3/5] Training baseline models...")
        scoring = self.evaluator_.primary_metric or "r2"
        results = self.model_selector_.train_all(
            X_train_featured, y,
            X_val_featured, y_val,
            scoring_metric=scoring,
        )

        # Step 4: Hyperparameter Tuning
        logger.info("\n[4/5] Hyperparameter optimization...")
        best_model_class = self.model_selector_.best_result_.model.__class__
        tuner = HyperparameterTuner(
            model_class=best_model_class,
            problem_type="regression",
            n_trials=self.n_trials,
            time_limit=self.time_limit // 2,
            cv_folds=self.cv_folds,
            scoring_metric=scoring,
            random_seed=self.random_seed,
        )

        try:
            tuner_results = tuner.optimize(X_train_featured, y)
            best_tuned_model = tuner.best_model_
            best_tuned_params = tuner.best_params_
            best_tuned_score = tuner.best_score_
            logger.info(f"Tuned model: {best_model_class.__name__} (score={best_tuned_score:.4f})")
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed: {e}. Using baseline.")
            best_tuned_model = self.model_selector_.best_result_.model
            best_tuned_params = self.model_selector_.best_result_.best_params
            best_tuned_score = self.model_selector_.best_result_.cv_score

        # Step 5: Final Evaluation
        logger.info("\n[5/5] Final evaluation...")
        self.is_fitted_ = True
        self.result_ = AutoMLResult(
            best_model_name=best_tuned_model.__class__.__name__,
            best_model=best_tuned_model,
            best_params=best_tuned_params,
            best_score=best_tuned_score,
            all_results=self.model_selector_.get_results_table(),
            evaluation_metrics={},
        )

        elapsed = time.time() - start_time
        logger.info(f"\nPipeline completed in {elapsed:.1f}s")
        logger.info(self.result_.summary())

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        self._check_fitted()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_processed = self.preprocessor_.transform(X)
        X_featured = self.feature_engineer_.transform(X_processed)
        return self.result_.best_model.predict(X_featured)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """Evaluate the model on test data."""
        self._check_fitted()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_processed = self.preprocessor_.transform(X)
        X_featured = self.feature_engineer_.transform(X_processed)

        y_pred = self.result_.best_model.predict(X_featured)
        metrics = self.evaluator_.evaluate(y, y_pred)
        self.result_.evaluation_metrics = metrics

        logger.info(self.evaluator_.report(metrics))
        return metrics

    def explain(self, X: pd.DataFrame | np.ndarray, top_n: int = 10) -> dict[str, float]:
        """Generate SHAP-based feature importance explanation."""
        self._check_fitted()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_processed = self.preprocessor_.transform(X)
        X_featured = self.feature_engineer_.transform(X_processed)

        self.explainer_ = Explainer(
            model=self.result_.best_model,
            X_background=X_featured[:min(100, len(X_featured))],
            problem_type="regression",
        )
        self.explainer_.fit()
        importance = self.explainer_.get_feature_importance(X_featured)
        self.result_.feature_importance = importance

        return importance

    def export_model(
        self,
        output_dir: str = "./models",
        model_name: str = "regressor",
        format: str = "joblib",
    ) -> dict[str, str]:
        """Export the trained model."""
        self._check_fitted()

        metadata = {
            "problem_type": "regression",
            "best_model": self.result_.best_model_name,
            "best_params": self.result_.best_params,
            "best_score": self.result_.best_score,
            "evaluation_metrics": self.result_.evaluation_metrics,
            "feature_importance": self.result_.feature_importance,
            "preprocessing_config": self.preprocessing_cfg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        exporter = ModelExporter(self.result_.best_model, metadata=metadata)
        return exporter.export(output_dir=output_dir, model_name=model_name, format=format)

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
