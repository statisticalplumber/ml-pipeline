# AutoML Pipeline Skill

## Overview

Automated machine learning pipeline for tabular data. Handles the full lifecycle: preprocessing → feature engineering → model selection → hyperparameter tuning → evaluation → export.

## Architecture

```
AutoMLClassifier / AutoMLRegressor
    │
    ├── Preprocessor        (missing values, outliers, encoding, scaling)
    ├── FeatureEngineer     (polynomial features, variance filter, mutual info selection)
    ├── ModelSelector       (train multiple models, rank by CV score)
    ├── HyperparameterTuner (Optuna Bayesian optimization)
    ├── Evaluator           (classification/regression metrics)
    ├── Explainer           (SHAP-based feature importance)
    └── ModelExporter       (joblib/pickle/ONNX serialization)
```

## Core Classes

| Class | File | Purpose |
|---|---|---|
| `AutoMLClassifier` | `automl/pipeline.py` | Classification pipeline entry point |
| `AutoMLRegressor` | `automl/pipeline.py` | Regression pipeline entry point |
| `Preprocessor` | `automl/preprocessor.py` | Impute, remove outliers, encode, scale |
| `FeatureEngineer` | `automl/feature_engineer.py` | Polynomial expansion, variance threshold, feature selection |
| `ModelSelector` | `automl/model_selector.py` | Train baseline models (linear, tree, SVM, ensemble, MLP) |
| `HyperparameterTuner` | `automl/tuner.py` | Optuna-based search with per-model search spaces |
| `Evaluator` | `automl/evaluator.py` | Metrics: accuracy, F1, ROC-AUC, R², RMSE, etc. |
| `Explainer` | `automl/explainer.py` | SHAP TreeExplainer / KernelExplainer / LinearExplainer |
| `ModelExporter` | `automl/exporter.py` | Save/load with metadata JSON |

## Pipeline Flow (fit)

1. **Preprocess** — impute missing → remove IQR outliers → encode categoricals → scale numerics
2. **Feature Engineer** — polynomial expansion → variance threshold → mutual info selection
3. **Model Selection** — cross-validate all enabled model categories, pick best by CV score
4. **Tuning** — Optuna optimize on the best model class (half of `time_limit`)
5. **Evaluate** — compute metrics on held-out or full data

## Usage Pattern

```python
from automl import AutoMLClassifier, AutoMLRegressor

# Classification
clf = AutoMLClassifier(time_limit=300, n_trials=50, cv_folds=5)
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
preds = clf.predict(X_test)
metrics = clf.evaluate(X_test, y_test)
importance = clf.explain(X_test)
clf.export_model("./models", "my_classifier")

# Regression — same API with AutoMLRegressor
```

## Configuration

Load via `config_path="configs/default_config.yaml"` or override flags in code:

```python
automl.model_selector_.include_linear = False
automl.model_selector_.include_tree = True   # RandomForest, DecisionTree, GradientBoosting only
automl.model_selector_.include_svm = False
automl.model_selector_.include_ensemble = False
automl.model_selector_.include_neural_network = False
```

Config sections: `preprocessing`, `feature_engineering`, `models`, `optimization`, `evaluation`.

## Model Categories (ModelSelector)

| Flag | Classification Models | Regression Models |
|---|---|---|
| `include_linear` | LogisticRegression | LinearRegression, Ridge, Lasso, ElasticNet |
| `include_tree` | RandomForest, DecisionTree, GradientBoosting | Same + Reg variants |
| `include_svm` | SVC (probability=True) | SVR |
| `include_ensemble` | VotingClassifier (RF+GB) | VotingRegressor |
| `include_neural_network` | MLPClassifier | MLPRegressor |

## Tuner Search Spaces

Each model class has a predefined Optuna search space in `HyperparameterTuner._define_search_space()`:

- **RandomForest** — n_estimators, max_depth, min_samples_split/leaf, max_features, bootstrap
- **GradientBoosting** — n_estimators, learning_rate (log), max_depth, subsample, max_features
- **DecisionTree** — max_depth, min_samples_split/leaf, max_features
- **Linear models** — C (log), penalty (l1/l2), solver
- **SVM** — C (log), kernel, gamma
- **MLP** — hidden_layer_sizes (dynamic depth), alpha (log), learning_rate_init (log), batch_size

## Key Constraints

- Input: `pd.DataFrame` or `np.ndarray` for X, `np.ndarray`/`list` for y
- Output: `np.ndarray` predictions
- No hardcoded paths — all paths are relative/configurable
- SHAP is optional (`pip install shap`)
- ONNX export requires `onnx` + `skl2onnx` (optional)
- Uses `numpy` types internally — `_make_serializable()` converts to native Python for JSON

## Tests

Located in `tests/`: `test_preprocessor.py`, `test_feature_engineer.py`, `test_model_selector.py`, `test_tuner.py`, `test_pipeline.py`.

## Dependencies

Core: `scikit-learn`, `pandas`, `numpy`, `optuna`, `pyyaml`, `joblib`
Optional: `shap`, `matplotlib`, `onnx`, `skl2onnx`
