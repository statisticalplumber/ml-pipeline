"""Evaluator Module.

Computes comprehensive evaluation metrics for model assessment.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Compute and report model evaluation metrics.

    Parameters
    ----------
    problem_type : str
        'classification' or 'regression'.
    primary_metric : str
        Primary metric for model selection.
    additional_metrics : list[str]
        Additional metrics to compute and report.
    """

    def __init__(
        self,
        problem_type: str = "classification",
        primary_metric: str = "accuracy",
        additional_metrics: list[str] | None = None,
    ):
        self.problem_type = problem_type
        self.primary_metric = primary_metric
        self.additional_metrics = additional_metrics or []

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute all evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels/values.
        y_proba : np.ndarray, optional
            Predicted probabilities (for classification).

        Returns
        -------
        dict[str, float]
            Dictionary of metric name -> value.
        """
        metrics: dict[str, float] = {}

        if self.problem_type == "classification":
            metrics = self._compute_classification_metrics(y_true, y_pred, y_proba)
        else:
            metrics = self._compute_regression_metrics(y_true, y_pred)

        # Ensure primary metric is first
        if self.primary_metric in metrics:
            primary_val = metrics.pop(self.primary_metric)
            metrics = {self.primary_metric: primary_val} | metrics

        return metrics

    def _compute_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
    ) -> dict[str, float]:
        """Compute classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

        # Multi-class aware metrics
        n_classes = len(np.unique(y_true))
        average = "macro" if n_classes > 2 else "binary"

        metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

        if y_proba is not None and n_classes == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except (ValueError, TypeError):
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

        if y_proba is not None and n_classes > 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            except (ValueError, TypeError):
                metrics["roc_auc"] = None

        # Log loss if probabilities available
        if y_proba is not None:
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
            except Exception:
                metrics["log_loss"] = None

        return metrics

    def _compute_regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Compute regression metrics."""
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }

    def report(self, metrics: dict[str, float]) -> str:
        """Format metrics as a readable report string."""
        lines = ["=" * 50, "Model Evaluation Report", "=" * 50]
        for name, value in metrics.items():
            if value is not None:
                if isinstance(value, float):
                    lines.append(f"  {name:<20s} : {value:.6f}")
                else:
                    lines.append(f"  {name:<20s} : {value}")
            else:
                lines.append(f"  {name:<20s} : N/A")
        lines.append("=" * 50)
        return "\n".join(lines)

    def get_primary_score(self, metrics: dict[str, float]) -> float:
        """Get the primary metric score."""
        score = metrics.get(self.primary_metric)
        if score is None:
            raise KeyError(f"Primary metric '{self.primary_metric}' not found in metrics.")
        return score

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed sklearn classification report."""
        if self.problem_type != "classification":
            raise RuntimeError("Classification report only applicable for classification problems.")
        return classification_report(y_true, y_pred)
