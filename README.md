# best-model-toolkit

Train, tune, and compare multiple `scikit-learn` / `XGBoost` models for
**regression** or **classification** with a single function call —
`best_model()` splits your data, optionally runs `GridSearchCV` for every
model, evaluates each one on a held-out test set, and renders a full set of
comparison plots so you can see at a glance which model performs best.

> This is a self-written module — the entire codebase was hand-built from
> scratch, not a wrapper around an existing library.

## Installation

```bash
pip install best-model-toolkit
```

## Quick start

```python
from best_model_toolkit import best_model
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

results = best_model(
    features=X,
    target=y,
    mode="regression",        # or "classification"
    grid="Yes",                 # "Yes" to run GridSearchCV, "No" to skip it
    df=df,                       # used for the overview plot (see below)
    hue=None,                     # column name for coloring in classification plots
)
```

`results` is a dictionary keyed by model name, e.g.:

```python
{
    "LinearRegression": {
        "model": <fitted sklearn model>,
        "param": {...},          # the hyperparameter grid that was searched
        "best_param": {...},     # best hyperparameters found (only if grid="Yes")
        "metrics": {"r2": {...}, "rmse": {...}, "mae": {...}},
    },
    ...
}
```

## How to call `best_model()`

### All parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `features` | DataFrame / array | yes | Input features (X). |
| `target` | Series / array | yes | Target values (y). Passing a `pandas.Series` (e.g. `df["yield"]`) lets `pairplot_type="top_corr"` auto-detect the target column by name. |
| `mode` | `"regression"` or `"classification"` | yes | Which set of models to train. |
| `grid` | `"Yes"` or `"No"` | yes | Whether to run `GridSearchCV` for every model. `"No"` trains each model once with default parameters — much faster, no CV plots. |
| `df` | DataFrame | yes | Used only for the overview plot at the start (see `pairplot_type` below). Can be the full dataframe or a subset. |
| `hue` | str or `None` | yes | Column name in `df` to color points by (classification-style plots). Pass `None` for regression. |
| `models` | list of `(model, param_grid)` tuples | no | Extra custom models to add on top of the built-in list. Default `None`. |
| `pairplot_type` | `"heatmap"` / `"top_corr"` / `"full"` / `"none"` | no | What overview plot to show. Default `"heatmap"`. |
| `top_n` | int | no | Number of top-correlated features to include when `pairplot_type="top_corr"`. Default `5`. |

### Regression example

```python
from best_model_toolkit import best_model
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["yield"])
y = df["yield"]

results = best_model(
    features=X,
    target=y,
    mode="regression",
    grid="Yes",
    df=df,
    hue=None,
    pairplot_type="heatmap",   # or "top_corr", "full", "none"
    top_n=5,
)
```

### Classification example

```python
from best_model_toolkit import best_model
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["label"])
y = df["label"]

results = best_model(
    features=X,
    target=y,
    mode="classification",
    grid="Yes",
    df=df,
    hue="label",   # column to color by in the overview/confusion-matrix-style plots
)
```

### Faster run, no hyperparameter search

```python
results = best_model(
    features=X, target=y, mode="regression", grid="No",
    df=df, hue=None, pairplot_type="none",
)
```



1. Splits `features`/`target` into train/test sets (80/20).
2. Shows one **overview plot** of your data (controlled by `pairplot_type`, see below).
3. For every model in its built-in list (see below), and for any extra
   models you pass via `models=`:
   - If `grid="Yes"`: runs `GridSearchCV` over that model's hyperparameter
     grid and plots the cross-validation results.
   - Refits the model with its best (or default) parameters on the training
     set, evaluates it on the test set, and plots the result.
4. Plots a final side-by-side comparison of all models.

## Models included

**Regression** (`mode="regression"`): `LinearRegression`, `RandomForestRegressor`,
`DecisionTreeRegressor`, `Lasso`, `Ridge`, `KNeighborsRegressor`,
`GradientBoostingRegressor`, `AdaBoostRegressor`, `XGBRegressor`.

**Classification** (`mode="classification"`): `DecisionTreeClassifier`,
`RandomForestClassifier`, `SVC`, `LogisticRegression`, `KNeighborsClassifier`,
`GradientBoostingClassifier`, `AdaBoostClassifier`, `XGBClassifier`, `GaussianNB`.

You can add your own models on top of the built-in list:

```python
from sklearn.svm import SVR

results = best_model(
    features=X, target=y, mode="regression", grid="Yes", df=df, hue=None,
    models=[(SVR(), {"C": [0.1, 1, 10]})],
)
```

## Every plot it draws

### 1. Overview plot (once, at the start) — `pairplot_type`

| `pairplot_type` | What it shows |
|---|---|
| `"heatmap"` (default) | A single correlation heatmap of every numeric column in `df` — compact and readable even for many columns. |
| `"top_corr"` | A pairplot (scatter matrix) limited to the `top_n` features most correlated with the target, plus the target itself. |
| `"full"` | The full pairplot over *every* column in `df` — grows to `n_columns x n_columns` subplots, can get very large for wide dataframes. |
| `"none"` | Skips the overview plot entirely. |

```python
best_model(..., pairplot_type="top_corr", top_n=5)
```

### 2. Per-model cross-validation plots (only when `grid="Yes"`)

For each hyperparameter combination tried by `GridSearchCV`:
- **Regression**: R² per fold (`plot_cv_r2`), and MAE/RMSE per fold (`plot_mae_mse`).
- **Classification**: F1, recall, precision, and accuracy per fold (`plot_cv_metrics`).

(For models with no hyperparameters to tune, e.g. plain `LinearRegression` or
`GaussianNB`, this plot has just a single point — that's expected, it means
there was nothing to search over.)

### 3. Per-model test-set plot (always shown)

- **Regression**: a scatter plot of predicted vs. actual values on the test set (`test_model_reg`).
- **Classification**: a normalized confusion matrix on the test set (`test_model_class`).

### 4. Final comparison plots (once, at the end, across all models)

- **Regression**: R² (`plot_r2`), MAE (`plot_mae`), RMSE (`plot_rmse`) — training vs. validation score, one point per model.
- **Classification**: accuracy score (`plot_score`), F1 (`plot_f1_score`), precision (`plot_score_precision`), recall (`plot_score_recall`), and — for models that support `predict_proba` — average precision (`plot_score_ap`) and precision-recall curves (`plot_curve_generic`).

## Using the plotting functions on their own

Every plot above is also available as a standalone function, in case you
just want a specific chart without running the full pipeline:

```python
from best_model_toolkit import plot_corr_heatmap, plot_pair_grid_top_corr

plot_corr_heatmap(df)
plot_pair_grid_top_corr(df, target=y, top_n=5)
```

## Examples

Ready-to-run scripts (using synthetic data, no dataset needed) are in the
[`examples/`](examples) folder:
- `examples/regression_example.py`
- `examples/classification_example.py`
- `examples/standalone_plots_example.py`

```bash
pip install best-model-toolkit
python examples/regression_example.py
```

## Notes

- Scale your features (e.g. with `sklearn.preprocessing.StandardScaler`)
  before calling `best_model()` if you plan to use `Lasso`, `Ridge`, `KNN`,
  `LogisticRegression`, or `SVM` — these are sensitive to feature scale and
  may otherwise throw `ConvergenceWarning`.
- `XGBRegressor`/`XGBClassifier` require the `xgboost` package, which is
  installed automatically as a dependency.

## Package structure

```
best_model_toolkit/
├── pyproject.toml
├── README.md
└── src/
    └── best_model_toolkit/
        ├── __init__.py
        └── core.py
```

## Publishing a new version to PyPI

```bash
# bump the version number in pyproject.toml first
pip install build twine
python -m build
twine upload dist/*
```
