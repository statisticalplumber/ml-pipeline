# AutoML - Automated Machine Learning Pipeline

A comprehensive, production-ready AutoML framework that automates the entire machine learning workflow: data preprocessing, feature engineering, model selection, hyperparameter optimization, and model evaluation.

## Features

- 🔄 **Automated Data Preprocessing**: Handles missing values, outliers, encoding, scaling
- 📊 **Feature Engineering**: Automatic feature generation, selection, and transformation
- 🤖 **Model Selection**: Tries multiple algorithms (Linear, Tree-based, SVM, Ensemble, Neural Networks)
- ⚡ **Hyperparameter Optimization**: Bayesian optimization with Optuna
- 📈 **Cross-Validation**: Robust k-fold cross-validation with stratification
- 🎯 **Automated Evaluation**: Comprehensive metrics reporting
- 📦 **Model Serialization**: Save/load models in multiple formats
- 🔬 **Explainability**: SHAP-based feature importance analysis
- 🚀 **Pipeline API**: Clean, composable interface

## Quick Start

```python
from automl import AutoMLClassifier, AutoMLRegressor

# Classification
automl = AutoMLClassifier(
    time_limit=300,      # 5 minutes optimization
    n_trials=50,         # 50 hyperparameter trials
    cv_folds=5           # 5-fold CV
)
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)

# Regression
automl = AutoMLRegressor(
    time_limit=300,
    n_trials=50,
    cv_folds=5
)
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)
```

## Installation

```bash
pip install .           # Install in production mode
pip install .[dev]      # Install with dev dependencies
```

## Project Structure

```
ml_pipeline/
├── automl/                  # Main package
│   ├── __init__.py
│   ├── pipeline.py          # Pipeline orchestration
│   ├── preprocessor.py      # Data preprocessing
│   ├── feature_engineer.py  # Feature engineering & selection
│   ├── model_selector.py    # Model training & selection
│   ├── tuner.py             # Hyperparameter optimization (Optuna)
│   ├── evaluator.py         # Evaluation metrics
│   ├── explainer.py         # SHAP explainability
│   └── exporter.py          # Model serialization
├── examples/                # Usage examples
│   ├── classification.ipynb
│   └── regression.ipynb
├── tests/                   # Unit tests
│   ├── test_preprocessor.py
│   ├── test_feature_engineer.py
│   ├── test_model_selector.py
│   └── test_tuner.py
├── configs/                 # Configuration files
│   └── default_config.yaml
├── requirements.txt
└── setup.py
```

## Usage Examples

See the `examples/` directory for detailed notebooks on:
- Binary classification (e.g., Titanic, Iris)
- Multi-class classification (e.g., MNIST, Wine)
- Regression (e.g., House Prices, Boston Housing)

## Configuration

Edit `configs/default_config.yaml` to customize:
- Preprocessing options
- Models to try
- Optimization parameters
- Evaluation metrics

## Selecting a Specific Model

By default, the AutoML pipeline tries **all configured model categories** and picks the best one. If you want to run with a **specific model only**, disable all other categories.

### Via Configuration File

Edit `configs/default_config.yaml` under `[automl.models]`. Each boolean flag controls a group of models:

| Flag | Models Included (Classification) |
|---|---|
| `include_linear` | LogisticRegression |
| `include_tree` | RandomForest, DecisionTree, GradientBoosting |
| `include_svm` | SVM |
| `include_ensemble` | VotingEnsemble |
| `include_neural_network` | MLP |

**Example: Force RandomForest only**

```yaml
[automl.models]
include_linear = false
include_tree = true        # RandomForest lives here
include_svm = false
include_ensemble = false
include_neural_network = false
```

Pass the config path when creating the pipeline:

```python
from automl import AutoMLClassifier

automl = AutoMLClassifier(config_path="configs/default_config.yaml")
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)
```

### Via Code (No Config File)

You can override the flags directly after instantiation:

```python
from automl import AutoMLClassifier

automl = AutoMLClassifier()

# Disable everything except tree-based models (RandomForest, DecisionTree, GradientBoosting)
automl.model_selector_.include_linear = False
automl.model_selector_.include_svm = False
automl.model_selector_.include_ensemble = False
automl.model_selector_.include_neural_network = False

automl.fit(X_train, y_train)
```

### Directly Using ModelSelector (Bypass AutoML)

If you don't want the full AutoML pipeline (preprocessing, feature engineering, tuning), use `ModelSelector` directly:

```python
from automl.model_selector import ModelSelector

selector = ModelSelector(
    problem_type="classification",
    include_linear=False,
    include_tree=True,
    include_svm=False,
    include_ensemble=False,
    include_neural_network=False,
)
results = selector.train_all(X_train, y_train, X_val, y_val)
best_model = selector.best_result_.model   # RandomForest
```

### Model Quick Reference

| Desired Model | Flags to set `true` | All others `false` |
|---|---|---|
| **LogisticRegression** | `include_linear` | linear=true, rest=false |
| **RandomForest** | `include_tree` | tree=true, rest=false |
| **DecisionTree** | `include_tree` | tree=true, rest=false |
| **GradientBoosting** | `include_tree` | tree=true, rest=false |
| **SVM** | `include_svm` | svm=true, rest=false |
| **VotingEnsemble** | `include_ensemble` | ensemble=true, rest=false |
| **MLP (Neural Net)** | `include_neural_network` | nn=true, rest=false |

## License

MIT License
