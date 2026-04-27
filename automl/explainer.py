"""Explainability Module.

Provides SHAP-based model interpretability and feature importance analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Explainer:
    """SHAP-based model explainability.

    Parameters
    ----------
    model : Any
        Trained scikit-learn model to explain.
    X_background : np.ndarray, optional
        Background dataset for SHAP computation (use training data).
    problem_type : str
        'classification' or 'regression'.
    """

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray | None = None,
        problem_type: str = "classification",
    ):
        self.model = model
        self.X_background = X_background
        self.problem_type = problem_type
        self.shap_explainer: Any = None
        self.shap_values: np.ndarray | None = None

    def fit(self) -> "Explainer":
        """Initialize SHAP explainer."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is not installed. Install it with: pip install shap"
            )

        # Choose explainer based on model type
        model_name = self.model.__class__.__name__

        if "Tree" in model_name or "Forest" in model_name or "GB" in model_name:
            # TreeExplainer is fast and exact for tree-based models
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif "Linear" in model_name or "Logistic" in model_name or "Ridge" in model_name or "Lasso" in model_name:
            self.shap_explainer = shap.LinearExplainer(
                self.model,
                self.X_background if self.X_background is not None else np.zeros((1, self.model.coef_.shape[0] if hasattr(self.model, 'coef_') else 1)),
            )
        else:
            # KernelExplainer works for any model but is slower
            background = self.X_background if self.X_background is not None else shap.kmeans(self.model.predict(np.zeros((1, 1))), 10) if hasattr(self.model, 'predict') else None
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict,
                background if background is not None else np.zeros((1, 1))
            )

        logger.info(f"SHAP explainer initialized for {model_name}")
        return self

    def compute_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for the given data."""
        if self.shap_explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        self.shap_values = self.shap_explainer.shap_values(X)
        return self.shap_values

    def get_feature_importance(self, X: np.ndarray | None = None) -> dict[str, float]:
        """Get mean absolute SHAP values as feature importance.

        Parameters
        ----------
        X : np.ndarray, optional
            Data to compute SHAP values on. If None, uses background data.

        Returns
        -------
        dict[str, float]
            Feature name -> importance score mapping.
        """
        if self.shap_values is None:
            if X is not None:
                self.compute_values(X)
            else:
                raise RuntimeError("No SHAP values computed. Call compute_values() first.")

        # Mean absolute SHAP value per feature
        if isinstance(self.shap_values, list):
            # Multi-class: take mean across classes
            mean_shap = np.mean([np.abs(sv) for sv in self.shap_values], axis=0)
        else:
            mean_shap = np.abs(self.shap_values)

        importances = np.mean(mean_shap, axis=0)
        if importances.ndim > 1:
            importances = importances.flatten()
        return {f"feature_{i}": float(importances[i]) for i in range(len(importances))}

    def plot_feature_importance(
        self,
        X: np.ndarray | None = None,
        top_n: int = 10,
        save_path: str | None = None,
    ) -> Any:
        """Plot feature importance bar chart.

        Parameters
        ----------
        X : np.ndarray, optional
            Data for SHAP computation.
        top_n : int
            Number of top features to show.
        save_path : str, optional
            Path to save the plot.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if X is not None:
            self.compute_values(X)

        importances = self.get_feature_importance()
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.5)))
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        bars = ax.barh(range(len(features)), values, color="steelblue")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"Top {top_n} Feature Importance (SHAP)")
        ax.invert_yaxis()

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_summary(self, X: np.ndarray | None = None, save_path: str | None = None) -> Any:
        """Plot SHAP summary bee-swarm plot.

        Parameters
        ----------
        X : np.ndarray, optional
            Data for SHAP computation.
        save_path : str, optional
            Path to save the plot.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import shap
        except ImportError:
            raise ImportError("matplotlib and shap are required for summary plot")

        if X is not None:
            self.compute_values(X)

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X if X is not None else self.X_background, show=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved to {save_path}")

        return fig
