# Baseline Model Experiments

本文件说明非 xRFM baseline 模型的训练、评估、输出结果和复现方式。

## 负责范围

本部分负责训练和评估 xRFM 之外的 baseline 模型，用于和 xRFM 进行实验比较。

目前已完成的 baseline 模型包括：

- XGBoost
- LightGBM
- Random Forest

这些模型覆盖了项目要求中的强 tabular baseline。其中 XGBoost 和 LightGBM 是 gradient-boosted tree baselines，Random Forest 是额外 baseline，用于提供非 boosting 的树模型对照。

## 输入数据

所有模型都使用数据预处理同学提供的固定 train/validation/test split。

输入文件位于：

```text
data/processed/
```

每个数据集应包含以下文件：

```text
{dataset}_X_train.csv
{dataset}_X_val.csv
{dataset}_X_test.csv
{dataset}_y_train.csv
{dataset}_y_val.csv
{dataset}_y_test.csv
{dataset}_metadata.json
{dataset}_feature_names.csv
```

当前使用的数据集包括：

| Dataset | Task Type | Notes |
| --- | --- | --- |
| wine | classification | sklearn Wine dataset |
| divorce | classification | UCI Divorce dataset, `d > 50` |
| german_credit | classification | UCI German Credit dataset, mixed feature types |
| bike_sharing | regression | UCI Bike Sharing dataset |
| appliances_energy | regression | UCI Appliances Energy dataset, `n > 10,000` |

## 评估指标

根据项目 PDF 要求，分类任务和回归任务使用不同指标。

分类任务报告：

- Accuracy
- AUC-ROC
- Training time
- Inference time per sample

回归任务报告：

- RMSE
- Training time
- Inference time per sample

因此，在完整结果表中，分类任务的 `rmse` 为空，回归任务的 `accuracy` 和 `auc_roc` 为空。这是正常现象，不代表结果缺失。

## 训练脚本

### XGBoost

脚本：

```text
experiments/baselines/train_xgboost.py
```

运行：

```bash
python experiments/baselines/train_xgboost.py
```

输出：

```text
outputs/tables/xgboost_results.csv
```

### LightGBM

脚本：

```text
experiments/baselines/train_lightgbm.py
```

运行：

```bash
python experiments/baselines/train_lightgbm.py
```

输出：

```text
outputs/tables/lightgbm_results.csv
```

### Random Forest

脚本：

```text
experiments/baselines/train_random_forest.py
```

运行：

```bash
python experiments/baselines/train_random_forest.py
```

输出：

```text
outputs/tables/random_forest_results.csv
```

## 模型训练流程

三个 baseline 脚本使用一致的实验流程：

1. 读取固定的 `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`。
2. 在 validation set 上进行小规模超参数搜索。
3. 选出 validation 表现最好的参数。
4. 使用最佳参数在 `train + validation` 上重新训练最终模型。
5. 只在 test set 上评估一次。
6. 记录 test performance、training time 和 inference time per sample。

这种流程避免了在 test set 上调参，符合 held-out test evaluation 的要求。

## 合并结果表

脚本：

```text
experiments/baselines/merge_baseline_results.py
```

运行：

```bash
python experiments/baselines/merge_baseline_results.py
```

默认合并：

```text
outputs/tables/xgboost_results.csv
outputs/tables/lightgbm_results.csv
outputs/tables/random_forest_results.csv
```

输出：

```text
outputs/tables/baseline_results_all.csv
outputs/tables/baseline_classification_summary.csv
outputs/tables/baseline_regression_summary.csv
outputs/tables/baseline_classification_wide.csv
outputs/tables/baseline_regression_wide.csv
```

其中：

- `baseline_results_all.csv` 是完整 long-format 结果表。
- `baseline_classification_summary.csv` 是分类任务的报告友好表。
- `baseline_regression_summary.csv` 是回归任务的报告友好表。
- `baseline_classification_wide.csv` 是分类任务的宽表，满足 PDF 中 “datasets as rows and (model, metric) pairs as columns” 的格式要求。
- `baseline_regression_wide.csv` 是回归任务的宽表，满足同样格式要求。

## 当前 baseline 结果概览

### Classification

| Dataset | Best Baseline Observation |
| --- | --- |
| wine | 三个模型 Accuracy 和 AUC-ROC 均为 1.0000 |
| divorce | XGBoost 和 LightGBM AUC-ROC 最高，约 0.9965 |
| german_credit | Random Forest AUC-ROC 最高，约 0.8067 |

### Regression

| Dataset | Best Baseline Observation |
| --- | --- |
| bike_sharing | XGBoost RMSE 最低，约 652.56 |
| appliances_energy | Random Forest RMSE 最低，约 63.50 |

这些结果说明 tree-based baselines 在小型分类任务上表现很强，但在不同回归任务和 mixed-type 分类任务上存在差异。

## Subsampling Experiment

项目 PDF 要求在至少一个 `n > 10,000` 的大数据集上，比较不同训练样本数下的 test performance 和 training time。

本部分使用：

```text
appliances_energy
```

脚本：

```text
experiments/baselines/subsample_appliances_baselines.py
```

运行：

```bash
python experiments/baselines/subsample_appliances_baselines.py
```

训练样本数：

```text
1000, 3000, 6000, 10000, 11841
```

输出表：

```text
outputs/tables/appliances_subsampling_baselines.csv
```

输出图：

```text
outputs/figures/appliances_subsampling_rmse.png
outputs/figures/appliances_subsampling_train_time.png
```

实验设置说明：

- 固定 test set 不变；
- 只从 training set 中抽取不同大小的训练子集；
- 使用前面 full baseline validation tuning 得到的最佳参数；
- 不在每个 sample size 上重新调参；
- 记录 test RMSE 和 training time。

当前主要趋势：

- 三个 baseline 的 RMSE 都随训练样本数增加而下降；
- Random Forest 在 full training subset 上 RMSE 最低；
- XGBoost 训练时间较短；
- Random Forest 的训练时间随样本数增长更明显。

## 与 xRFM 结果合并

后续需要 xRFM 同学提供同样格式的结果文件，例如：

```text
outputs/tables/xrfm_results.csv
```

建议字段与 baseline 结果保持一致：

```text
dataset
task_type
model
n_train
n_val
n_test
n_features
best_validation_score
best_params
train_time_sec
inference_time_per_sample_ms
rmse
accuracy
auc_roc
```

其中：

- 分类任务填写 `accuracy` 和 `auc_roc`，`rmse` 留空；
- 回归任务填写 `rmse`，`accuracy` 和 `auc_roc` 留空；
- `model` 建议写 `xRFM`。

如果要生成包含 xRFM 的最终 all-model 结果表，请使用独立的 results 脚本：

```bash
python experiments/results/merge_all_model_results.py
```

该脚本默认读取：

```text
outputs/tables/xgboost_results.csv
outputs/tables/lightgbm_results.csv
outputs/tables/random_forest_results.csv
outputs/tables/xrfm_results.csv
```

并输出：

```text
outputs/tables/all_models_results.csv
outputs/tables/all_models_classification_summary.csv
outputs/tables/all_models_regression_summary.csv
outputs/tables/all_models_classification_wide.csv
outputs/tables/all_models_regression_wide.csv
```

## 后续仍需补充的内容

目前 baseline 训练、结果合并、subsampling baseline 部分已经完成。

后续如果继续整理最终材料，需要和 xRFM 同学对齐：

- xRFM 在 5 个数据集上的完整结果；
- xRFM 在 `appliances_energy` 上的 subsampling 结果；
- xRFM 的 AGOP diagonal，用于 interpretability comparison；
- 最终 report 中的总表和总图。

如果做 interpretability comparison，本部分可以继续补充：

- PCA loadings；
- mutual information scores；
- permutation importance。

AGOP diagonal 应由 xRFM 部分提供。
