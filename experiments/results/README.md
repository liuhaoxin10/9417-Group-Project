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

## Outputs

```text
outputs/tables/all_models_results.csv
outputs/tables/all_models_classification_summary.csv
outputs/tables/all_models_regression_summary.csv
outputs/tables/all_models_classification_wide.csv
outputs/tables/all_models_regression_wide.csv
outputs/tables/appliances_subsampling_all.csv
outputs/figures/appliances_subsampling_rmse_all.png
outputs/figures/appliances_subsampling_train_time_all.png
```
