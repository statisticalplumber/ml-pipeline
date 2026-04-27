"""Tests for the AutoML preprocessor module."""

import numpy as np
import pandas as pd
import pytest

from automl.preprocessor import OutlierRemover, Preprocessor


class TestOutlierRemover:
    def test_no_outliers(self):
        """Test with data that has no outliers."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        remover = OutlierRemover(multiplier=1.5)
        result = remover.fit_transform(df)
        assert result.shape == df.shape

    def test_with_outliers(self):
        """Test with data that has outliers."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": [10, 20, 30, 40, 50]})
        remover = OutlierRemover(multiplier=1.5)
        result = remover.fit_transform(df)
        assert result.shape == df.shape


class TestPreprocessor:
    def test_numeric_only(self):
        """Test preprocessing with only numeric columns."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        preprocessor = Preprocessor(
            missing_strategy="mean",
            numerical_scaling="standard",
        )
        result = preprocessor.fit_transform(df)
        assert result.shape[1] == 2

    def test_mixed_types(self):
        """Test preprocessing with mixed numeric and categorical columns."""
        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "a", "b", "a"],
        })
        preprocessor = Preprocessor(
            missing_strategy="median",
            categorical_encoding="onehot",
            numerical_scaling="standard",
        )
        result = preprocessor.fit_transform(df)
        assert result.shape[1] >= 2  # At least 1 numeric + 1 one-hot

    def test_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": ["x", None, "y", "x", None],
        })
        preprocessor = Preprocessor(missing_strategy="median")
        result = preprocessor.fit_transform(df)
        assert not np.isnan(result).any()

    def test_different_scaling(self):
        """Test different scaling options."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        for scaling in ["standard", "minmax", "robust"]:
            preprocessor = Preprocessor(numerical_scaling=scaling)
            result = preprocessor.fit_transform(df)
            assert result.shape == (5, 1)

    def test_categorical_encoding(self):
        """Test different encoding options."""
        df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b"]})
        for encoding in ["onehot", "ordinal"]:
            preprocessor = Preprocessor(categorical_encoding=encoding)
            result = preprocessor.fit_transform(df)
            assert result.shape[1] >= 1

    def test_get_params(self):
        """Test parameter serialization."""
        preprocessor = Preprocessor(
            missing_strategy="mean",
            categorical_encoding="onehot",
            numerical_scaling="minmax",
            outlier_iqr_multiplier=2.0,
        )
        params = preprocessor.get_params()
        assert params["missing_strategy"] == "mean"
        assert params["categorical_encoding"] == "onehot"
        assert params["numerical_scaling"] == "minmax"
        assert params["outlier_iqr_multiplier"] == 2.0

    def test_set_params(self):
        """Test parameter setting."""
        preprocessor = Preprocessor()
        preprocessor.set_params(missing_strategy="mean", numerical_scaling="robust")
        assert preprocessor.missing_strategy == "mean"
        assert preprocessor.numerical_scaling == "robust"
