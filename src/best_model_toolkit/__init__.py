"""
best_model_toolkit
===================

Utilities for training, tuning (GridSearchCV) and comparing scikit-learn /
XGBoost models for regression and classification tasks, together with
ready-made plots of performance metrics (R^2, MAE, RMSE, F1, precision,
recall, PR curves, etc).

Main function:

    from best_model_toolkit import best_model

    results = best_model(
        features=X, target=y,
        mode="regression",       # or "classification"
        grid="Yes",               # whether to run GridSearchCV
        df=df, hue=None,
        pairplot_type="heatmap",  # "heatmap" | "top_corr" | "full" | "none"
    )
"""

from .core import (
    best_model,
    test_model_reg,
    test_model_class,
    grid_reg,
    grid_class,
    plot_r2,
    plot_score_reg,
    plot_mae,
    plot_rmse,
    plot_score,
    plot_f1_score,
    plot_score_precision,
    plot_score_recall,
    plot_score_ap,
    plot_score_generic,
    plot_curve_generic,
    plot_cv_r2,
    plot_mae_mse,
    plot_cv_metrics,
    plot_pair_grid_ref,
    plot_pair_grid_class,
    plot_corr_heatmap,
    plot_pair_grid_top_corr,
)

__version__ = "0.2.0"

__all__ = [
    "best_model",
    "test_model_reg",
    "test_model_class",
    "grid_reg",
    "grid_class",
    "plot_r2",
    "plot_score_reg",
    "plot_mae",
    "plot_rmse",
    "plot_score",
    "plot_f1_score",
    "plot_score_precision",
    "plot_score_recall",
    "plot_score_ap",
    "plot_score_generic",
    "plot_curve_generic",
    "plot_cv_r2",
    "plot_mae_mse",
    "plot_cv_metrics",
    "plot_pair_grid_ref",
    "plot_pair_grid_class",
    "plot_corr_heatmap",
    "plot_pair_grid_top_corr",
]
