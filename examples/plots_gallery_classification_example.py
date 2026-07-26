"""
Classification plots gallery example for best_model_toolkit.

Runs the full classification pipeline and shows every kind of plot the
package produces for mode="classification":
    1. Overview plot (correlation heatmap, colored context via hue)
    2. Per-model normalized confusion matrix on the test set
    3. Final side-by-side comparison plots (accuracy Score, F1,
       Precision, Recall, and Average Precision / PR curves for models
       that support predict_proba)
    4. A tidy summary table at the end

Run this file directly, or open it in Jupyter / Google Colab and run
cell by cell to see each plot appear one at a time.

    python plots_gallery_classification_example.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from best_model_toolkit import best_model, plot_corr_heatmap, summarize_results

# --- 1. Synthetic binary classification dataset --------------------------
X_raw, y_raw = make_classification(
    n_samples=300,
    n_features=5,
    n_informative=3,
    n_classes=2,
    random_state=0,
)
feature_names = [f"feature_{i}" for i in range(X_raw.shape[1])]
df = pd.DataFrame(X_raw, columns=feature_names)
df["label"] = y_raw

X = df.drop(columns=["label"])
y = df["label"]

# --- 2. Scale features (recommended for SVM/KNN/LogisticRegression) -----
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- 3. Standalone overview plot (can also be called on its own) --------
plot_corr_heatmap(df)

# --- 4. Full pipeline: trains every classifier, plots a confusion matrix
#        for each one, and the final comparison plots -------------------
results = best_model(
    features=X_scaled,
    target=y,
    mode="classification",
    grid="No",              # set to "Yes" to also see cross-validation plots
    df=df,
    hue="label",              # color the overview plot by class
    pairplot_type="none",     # already showed the heatmap above
)

# A tidy summary table is printed automatically at the end of best_model(),
# but you can also build it again (e.g. to sort by a different metric):
print(summarize_results(results, mode="classification", sort_by="f1_val"))
