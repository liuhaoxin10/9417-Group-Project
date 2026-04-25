# COMP9417 Group Project Code Package

This archive contains the code, processed data, and generated outputs used for the COMP9417 group project report on xRFM and tabular baselines.

The report PDF should be submitted separately on Moodle and should not be placed inside this zip file.

## 1. Environment Setup

Install dependencies from the package root:

```bash
pip install -r requirements.txt
```

The xRFM experiments require `xrfm`, `torch`, and the standard scientific Python stack. If a local environment cannot run xRFM, the generated result tables and figures are already included under `outputs/`.

## 2. Data

Raw and processed data are stored under:

```text
data/raw/
data/processed/
```

The processed directory contains fixed train/validation/test splits, feature names, metadata, and fitted preprocessors. These files allow the model scripts to run without repeating manual data preparation.

To regenerate processed data from the raw files:

```bash
python experiments/dataset/preprocess.py
```

## 3. Model Training

Run the baseline models:

```bash
python experiments/baselines/train_xgboost.py
python experiments/baselines/train_lightgbm.py
python experiments/baselines/train_random_forest.py
```

Run xRFM:

```bash
python experiments/xrfm/train_xrfm.py
```

These scripts write model result tables to:

```text
outputs/tables/xgboost_results.csv
outputs/tables/lightgbm_results.csv
outputs/tables/random_forest_results.csv
outputs/tables/xrfm_results.csv
```

## 4. Result Tables

Merge all model results into report-ready tables:

```bash
python experiments/results/merge_all_model_results.py
```

Main outputs:

```text
outputs/tables/all_models_results.csv
outputs/tables/all_models_classification_wide.csv
outputs/tables/all_models_regression_wide.csv
outputs/tables/all_models_classification_summary.csv
outputs/tables/all_models_regression_summary.csv
```

## 5. Subsampling Experiment

Run the Appliances Energy subsampling experiment:

```bash
python experiments/results/subsample_appliances_all_models.py
```

Outputs:

```text
outputs/tables/appliances_subsampling_all.csv
outputs/figures/appliances_subsampling_rmse_all.png
outputs/figures/appliances_subsampling_train_time_all.png
```

## 6. Interpretability

Run interpretability comparisons for a dataset:

```bash
python experiments/xrfm/interpretability_xrfm.py --dataset wine
```

Valid dataset names are:

```text
wine
divorce
german_credit
bike_sharing
appliances_energy
```

Outputs:

```text
outputs/tables/{dataset}_interpretability_comparison.csv
outputs/figures/{dataset}_interpretability_comparison.png
```

## 7. Bonus Experiment

The bonus experiment implements residual-weighted AGOP:

```bash
python experiments/bonus/residual_weighted_agop.py
```

Outputs:

```text
outputs/tables/bonus_residual_agop_direction_comparison.csv
outputs/tables/bonus_residual_agop_performance_comparison.csv
```

## 8. Final Output Check

To regenerate the final manifest and validation checks:

```bash
python experiments/results/prepare_final_outputs.py
```

Outputs:

```text
outputs/tables/final_output_manifest.csv
outputs/tables/final_output_checks.csv
```

## 9. Metric Note

For the multiclass Wine dataset, `wine / xRFM / auc_roc` is intentionally reported as `NaN`. The current xRFM implementation does not provide class probability estimates or continuous decision scores required for standard multiclass AUC-ROC. Accuracy is still reported.

