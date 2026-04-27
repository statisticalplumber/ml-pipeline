"""
AutoML Regression Example
==========================
Demonstrates the AutoML pipeline on a regression task using the California Housing dataset.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from automl import AutoMLRegressor

# ─── Load Data ───────────────────────────────────────────────────────────────
print("Loading California Housing dataset...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
print(f"Target mean: {y.mean():.2f} ± {y.std():.2f}")

# ─── Split Data ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ─── Run AutoML ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Starting AutoML Regression Pipeline")
print("=" * 60)

automl = AutoMLRegressor(
    time_limit=120,      # 2 minutes for optimization
    n_trials=30,         # 30 hyperparameter trials
    cv_folds=5,          # 5-fold cross-validation
    random_seed=42,
)

automl.fit(X_train, y_train)

# ─── Evaluate ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Evaluating on Test Set")
print("=" * 60)

metrics = automl.evaluate(X_test, y_test)

# ─── Explain ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Feature Importance (SHAP)")
print("=" * 60)

importance = automl.explain(X_test, top_n=5)
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature:<30s} : {score:.6f}")

# ─── Export Model ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Exporting Model")
print("=" * 60)

files = automl.export_model(output_dir="./models", model_name="housing_regressor")
for file_type, path in files.items():
    print(f"  {file_type}: {path}")

print("\nDone!")
