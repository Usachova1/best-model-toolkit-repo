# best-model-toolkit

A collection of functions for quickly training, tuning (`GridSearchCV`) and
comparing several `scikit-learn` / `XGBoost` models for **regression** and
**classification** tasks, with automatic plots of metrics (R², MAE, RMSE,
F1, precision, recall, PR curves, etc).

## Installation

### Locally (before the package is on PyPI)

```bash
git clone https://github.com/YOUR_USERNAME/best-model-toolkit.git
cd best-model-toolkit
pip install -e .
```

### Directly from GitHub

```bash
pip install git+https://github.com/YOUR_USERNAME/best-model-toolkit.git
```

### From PyPI (once published, see below)

```bash
pip install best-model-toolkit
```

## Usage

```python
from best_model_toolkit import best_model
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

results = best_model(
    features=X,
    target=y,
    mode="regression",      # or "classification"
    grid="Yes",              # "Yes"/"No" — whether to run GridSearchCV
    df=df,
    hue=None,                 # column for pairplot hue (classification)
)

# results — dict {"model_name": {"model": ..., "param": ..., "metrics": {...}}}
```

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

## Publishing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```
