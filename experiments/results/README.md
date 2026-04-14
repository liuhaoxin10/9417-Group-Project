# Results and Cross-Model Analysis

本目录放跨模型汇总和最终结果生成脚本，不放单个模型的训练代码。

## Scripts

```text
merge_all_model_results.py
```

合并 XGBoost、LightGBM、Random Forest 和 xRFM 的主结果表，输出 all-model summary 和 wide tables。

```text
subsample_appliances_all_models.py
```

在 Appliances Energy 数据集上运行所有模型的 subsampling 对比实验，输出 all-model subsampling 表和图。

```text
prepare_final_outputs.py
```

整合最终交付材料：合并 interpretability 表格，检查主结果和 subsampling 结果是否齐全，并生成输出文件清单。

## Outputs

```text
outputs/tables/all_models_results.csv
outputs/tables/all_models_classification_summary.csv
outputs/tables/all_models_regression_summary.csv
outputs/tables/all_models_classification_wide.csv
outputs/tables/all_models_regression_wide.csv
outputs/tables/appliances_subsampling_all.csv
outputs/tables/interpretability_comparison_all.csv
outputs/tables/final_output_manifest.csv
outputs/tables/final_output_checks.csv
outputs/figures/appliances_subsampling_rmse_all.png
outputs/figures/appliances_subsampling_train_time_all.png
```

## Metric Notes

For classification datasets, the report-required metrics are accuracy and AUC-ROC.

The `wine` dataset is a multi-class classification task. The current xRFM implementation does not provide class probability estimates or continuous decision scores required for standard multi-class AUC-ROC. Therefore, `wine / xRFM / auc_roc` is intentionally reported as `NaN`, while `accuracy` is still reported.

Do not replace this value with an AUC computed from hard class predictions, because that would not be comparable with the probability-based AUC-ROC values from XGBoost, LightGBM, and Random Forest.
