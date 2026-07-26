# Telco Customer Churn — Regression & Classification

Two Google Colab notebooks that use [`best-model-toolkit`](https://pypi.org/project/best-model-toolkit/)
to analyze the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset from two angles:

- **`Churn_classification.ipynb`** — predicts whether a customer churns (`Churn`: Yes/No), using `MonthlyCharges`, `TotalCharges`, and `tenure` as features.
- **`Churn_regression.ipynb`** — predicts a customer's `TotalCharges`, using `MonthlyCharges` and `tenure` as features.

Both notebooks run every model in `best_model_toolkit` (regression or
classification, depending on the notebook), tune hyperparameters with
`GridSearchCV`, and produce comparison plots plus a summary table so you
can see at a glance which model fits this data best.

## Dataset

Download `data1.csv` (the Telco Customer Churn dataset) from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
before running either notebook. The notebooks expect the file to be named
exactly `data1.csv`.

## How to run

1. Open either notebook in **Google Colab**.
2. Run the first few cells — install the package and import it:
   ```python
   !pip install best-model-toolkit
   ```
   ```python
   from best_model_toolkit import best_model
   ```
3. Run the `files.upload()` cell and choose `data1.csv` when prompted.
4. Run the remaining cells — they clean the data (map `Churn` to 0/1,
   convert `TotalCharges` to numeric, fill missing values with 0) and call
   `best_model(...)`.

## What each notebook does, step by step

1. Uploads `data1.csv` via `google.colab.files.upload()`.
2. Cleans the data:
   - Maps `Churn` (`Yes`/`No`) to `1`/`0` as `Churn1`.
   - Converts `TotalCharges` to numeric (some rows have blank strings) and fills missing values with `0`.
3. Selects features and target:
   - **Classification**: features = `MonthlyCharges`, `TotalCharges`, `tenure`; target = `Churn1`.
   - **Regression**: features = `MonthlyCharges`, `tenure`; target = `TotalCharges`.
4. Calls `best_model(...)`, which trains and compares every built-in model,
   plots an overview of the data, per-model evaluation plots, and a final
   side-by-side comparison — see the
   [best-model-toolkit README](https://pypi.org/project/best-model-toolkit/)
   for the full list of plots and parameters.

## Notes

- Make sure you're on the latest version of the package:
  ```python
  !pip install --upgrade best-model-toolkit
  ```
- `grid="Yes"` is used in both notebooks, so every model also runs
  `GridSearchCV` — this takes longer but shows cross-validation plots too.
  Pass `grid="No"` instead for a much faster run without those plots.
- `TotalCharges` is derived from `MonthlyCharges x tenure` (roughly), so in
  the regression notebook expect very high R² — this is closer to a sanity
  check than a hard prediction problem.
