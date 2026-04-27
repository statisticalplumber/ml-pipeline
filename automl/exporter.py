"""Exporter Module.

Handles model serialization and export in multiple formats.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export fitted models and metadata to disk.

    Parameters
    ----------
    model : Any
        The trained model to export.
    metadata : dict[str, Any], optional
        Additional metadata (metrics, config, etc.).
    """

    def __init__(self, model: Any, metadata: dict[str, Any] | None = None):
        self.model = model
        self.metadata = metadata or {}
        self.export_dir: str = ""

    def export(
        self,
        output_dir: str = "./models",
        model_name: str = "automl_model",
        format: str = "joblib",
    ) -> dict[str, str]:
        """Export the model to disk.

        Parameters
        ----------
        output_dir : str
            Directory to save the model.
        model_name : str
            Base name for exported files.
        format : str
            Export format: 'joblib', 'pickle', 'onnx'.

        Returns
        -------
        dict[str, str]
            Mapping of file type -> path.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.export_dir = output_dir

        files: dict[str, str] = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}"

        if format == "joblib":
            path = os.path.join(output_dir, f"{filename}.joblib")
            joblib.dump(self.model, path)
            files["model"] = path
            logger.info(f"Model saved to {path}")

        elif format == "pickle":
            path = os.path.join(output_dir, f"{filename}.pkl")
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            files["model"] = path
            logger.info(f"Model saved to {path}")

        elif format == "onnx":
            try:
                import onnx
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                initial_type = [("float_input", FloatTensorType([None, self.model.coef_.shape[1] if hasattr(self.model, 'coef_') else 10]))]
                onnx_model = convert_sklearn(self.model, initial_types=initial_type)
                path = os.path.join(output_dir, f"{filename}.onnx")
                with open(path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                files["model"] = path
                logger.info(f"ONNX model saved to {path}")
            except ImportError:
                logger.warning("onnx and skl2onnx not installed. Skipping ONNX export.")

        # Export metadata
        meta_path = os.path.join(output_dir, f"{filename}_metadata.json")
        serializable_meta = self._make_serializable(self.metadata)
        with open(meta_path, "w") as f:
            json.dump(serializable_meta, f, indent=2, default=str)
        files["metadata"] = meta_path
        logger.info(f"Metadata saved to {meta_path}")

        return files

    def load(self, model_path: str) -> Any:
        """Load a previously exported model.

        Parameters
        ----------
        model_path : str
            Path to the model file.

        Returns
        -------
        Any
            The loaded model.
        """
        if model_path.endswith(".joblib"):
            return joblib.load(model_path)
        elif model_path.endswith(".pkl"):
            import pickle
            with open(model_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
