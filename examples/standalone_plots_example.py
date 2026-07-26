"""
Standalone plotting example for best_model_toolkit.

Shows how to use the plotting helpers on their own, without running the
full best_model() training pipeline.

Run this file directly:
    python standalone_plots_example.py
"""
from sklearn.datasets import make_regression
import pandas as pd

from best_model_toolkit import plot_corr_heatmap, plot_pair_grid_top_corr

# --- Create some synthetic data -----------------------------------------
X_raw, y_raw = make_regression(n_samples=200, n_features=8, noise=10.0, random_state=0)
feature_names = [f"feature_{i}" for i in range(X_raw.shape[1])]
df = pd.DataFrame(X_raw, columns=feature_names)
df["target"] = y_raw

# --- Correlation heatmap of every column ---------------------------------
plot_corr_heatmap(df)

# --- Pairplot limited to the top 4 features most correlated with target --
plot_pair_grid_top_corr(df, target="target", top_n=4)
