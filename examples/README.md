# Examples

Ready-to-run scripts demonstrating `best_model_toolkit`. All of them use
synthetic data generated with scikit-learn, so you can run them directly
without needing your own dataset.

| File | What it shows |
|---|---|
| `plots_gallery_example.py` | A quick tour of every regression plot: correlation heatmap, top-correlated pairplot, per-model test scatter plots, final R2/RMSE/MAE comparison, and the summary table. Start here if you just want to see what the package draws. |
| `plots_gallery_classification_example.py` | Same tour, but for classification: confusion matrix per model, and final Score/F1/Precision/Recall comparison plots. |
| `regression_example.py` | Full regression pipeline: scaling features, running `best_model()` with `grid="Yes"`, and reading the results. |
| `classification_example.py` | Full classification pipeline, including `hue` for the overview plot. |
| `standalone_plots_example.py` | Using `plot_corr_heatmap()` and `plot_pair_grid_top_corr()` on their own, without training any models. |

## Running an example

```bash
pip install best-model-toolkit
python plots_gallery_example.py
```

(or open any of these files in Jupyter / Google Colab and run cell by cell)

## What the plots look like

### Overview plots

Correlation heatmap (`pairplot_type="heatmap"`, the default):

![Correlation heatmap](previews/corr_heatmap.png)

Pairplot limited to the top-correlated features (`pairplot_type="top_corr"`):

![Top-correlated pairplot](previews/top_corr_pairplot.png)

### Regression

Per-model test-set scatter plot (predicted vs. actual):

![Regression test scatter](previews/regression_test_scatter.png)

Final comparison across all models:

![R2 comparison](previews/regression_r2_comparison.png)
![RMSE comparison](previews/regression_rmse_comparison.png)
![MAE comparison](previews/regression_mae_comparison.png)

### Classification

Per-model normalized confusion matrix:

![Confusion matrix](previews/classification_confusion_matrix.png)

Final comparison across all models:

![Score comparison](previews/classification_score_comparison.png)
![F1 comparison](previews/classification_f1_comparison.png)
![Precision-recall curve](previews/classification_pr_curve.png)
