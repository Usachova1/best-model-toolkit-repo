"""
Plots gallery example for best_model_toolkit.

Runs the full regression pipeline and shows every kind of plot the
package produces:
    1. Overview plot (correlation heatmap)
    2. Per-model test-set scatter plots (predicted vs. actual)
    3. Final side-by-side comparison plots (R2, RMSE, MAE)
    4. A tidy summary table at the end

Run this file directly, or open it in Jupyter / Google Colab and run
cell by cell to see each plot appear one at a time.

    python plots_gallery_example.py
"""
import numpy as np
import pandas as pd

from best_model_toolkit import best_model, plot_corr_heatmap, plot_pair_grid_top_corr

# --- 1. Synthetic "blueberry yield"-style dataset ------------------------
np.random.seed(42)
n = 300
df = pd.DataFrame({
    "clonesize": np.random.uniform(10, 40, n),
    "honeybee": np.random.uniform(0, 1, n),
    "fruitset": np.random.uniform(0.3, 0.7, n),
    "fruitmass": np.random.uniform(0.3, 0.6, n),
    "seeds": np.random.uniform(20, 45, n),
})
df["yield"] = (
    df["fruitset"] * 3000
    + df["seeds"] * 40
    + df["fruitmass"] * 500
    + np.random.normal(0, 150, n)
)

X = df.drop(columns=["yield"])
y = df["yield"]

# --- 2. Standalone overview plots (can also be called on their own) -----
plot_corr_heatmap(df)                              # correlation heatmap
plot_pair_grid_top_corr(df, target="yield", top_n=3)  # pairplot of top-3 features

# --- 3. Full pipeline: trains every model, plots test-set scatter plots,
#        and the final R2 / RMSE / MAE comparison plots ------------------
results = best_model(
    features=X,
    target=y,
    mode="regression",
    grid="No",              # set to "Yes" to also see cross-validation plots
    df=df,
    hue=None,
    pairplot_type="none",    # already showed the overview plots above
)

# A tidy summary table is printed automatically at the end of best_model(),
# but you can also build it again (e.g. to sort differently):
from best_model_toolkit import summarize_results
print(summarize_results(results, mode="regression", sort_by="rmse_val", ascending=True))
