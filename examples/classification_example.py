"""
Classification example for best_model_toolkit.

Run this file directly:
    python classification_example.py

It uses synthetic data (sklearn.datasets.make_classification), so it works
out of the box without needing a CSV file.
"""
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import pandas as pd

from best_model_toolkit import best_model

# --- 1. Create some synthetic classification data ----------------------
X_raw, y_raw = make_classification(
    n_samples=300,
    n_features=6,
    n_informative=4,
    n_classes=2,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(X_raw.shape[1])]
df = pd.DataFrame(X_raw, columns=feature_names)
df["label"] = y_raw

X = df.drop(columns=["label"])
y = df["label"]

# --- 2. Scale features (recommended for SVC/KNN/LogisticRegression) ----
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- 3. Train, tune, and compare all classification models -------------
results = best_model(
    features=X_scaled,
    target=y,
    mode="classification",
    grid="Yes",              # set to "No" for a much faster run
    df=df,
    hue="label",              # color the overview plot by class
    pairplot_type="heatmap",  # try "top_corr" or "full" too
)

# --- 4. Inspect the results ---------------------------------------------
for model_name, info in results.items():
    print(model_name, "->", info["metrics"])
