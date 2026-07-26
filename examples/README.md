# Examples

Ready-to-run scripts demonstrating `best_model_toolkit`. All of them use
synthetic data generated with scikit-learn, so you can run them directly
without needing your own dataset.

| File | What it shows |
|---|---|
| `regression_example.py` | Full regression pipeline: scaling features, running `best_model()` with `grid="Yes"`, and reading the results. |
| `classification_example.py` | Full classification pipeline, including `hue` for the overview plot. |
| `standalone_plots_example.py` | Using `plot_corr_heatmap()` and `plot_pair_grid_top_corr()` on their own, without training any models. |

## Running an example

```bash
pip install best-model-toolkit
python regression_example.py
```

(or open any of these files in Jupyter / Google Colab and run cell by cell)
