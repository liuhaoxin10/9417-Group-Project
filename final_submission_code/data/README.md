# Data Description

This directory contains the raw and processed datasets used in the COMP9417 project experiments.

## Datasets

| Dataset | Task | Source |
| --- | --- | --- |
| Wine | Classification | `sklearn.datasets.load_wine()` |
| Divorce | Classification | UCI Machine Learning Repository |
| German Credit | Classification | UCI Machine Learning Repository |
| Bike Sharing | Regression | UCI Bike Sharing Dataset |
| Appliances Energy | Regression | UCI Appliances Energy Prediction Dataset |

## Directory Structure

```text
data/
├── raw/
└── processed/
```

`data/raw/` contains the original data archive used by the project.

`data/processed/` contains the fixed train/validation/test splits used by all model scripts. These processed files are included so the experiments can be rerun without repeating manual data preparation.

## Processed Files

Each dataset has the following processed files:

```text
{dataset}_X_train.csv
{dataset}_X_val.csv
{dataset}_X_test.csv
{dataset}_y_train.csv
{dataset}_y_val.csv
{dataset}_y_test.csv
{dataset}_feature_names.csv
{dataset}_metadata.json
{dataset}_preprocessor.joblib
```

Some datasets also include a `{dataset}_processed_full.csv` file for inspection.

## Preprocessing

Preprocessing is implemented in:

```text
experiments/dataset/preprocess.py
```

The preprocessing pipeline applies:

- missing-value imputation;
- one-hot encoding for categorical features;
- standardization for numerical features;
- date feature extraction for datasets with timestamp columns;
- removal of target-leaking or redundant columns where needed;
- fixed train/validation/test splitting with `random_state = 42`.

To regenerate the processed files from the raw data:

```bash
python experiments/dataset/preprocess.py
```

The regenerated files will be written to:

```text
data/processed/
```

