# 最终输出文件说明

这份文件用于说明本项目目前已经生成的最终实验数据、表格和图片。写论文的同学可以根据这里列出的文件直接取结果，不需要重新理解所有训练脚本。

## 1. 主实验结果表

- `outputs/tables/all_models_results.csv`：最完整的 long-format 总结果表，每一行是某个模型在某个数据集上的测试结果。
- `outputs/tables/all_models_classification_wide.csv`：分类任务宽表，适合整理成论文中的分类结果表。
- `outputs/tables/all_models_regression_wide.csv`：回归任务宽表，适合整理成论文中的回归结果表。
- `outputs/tables/all_models_classification_summary.csv`：分类任务简化表，方便写 Discussion。
- `outputs/tables/all_models_regression_summary.csv`：回归任务简化表，方便写 Discussion。

主实验比较的模型包括 XGBoost、LightGBM、Random Forest 和 xRFM。分类任务主要看 Accuracy 和 AUC-ROC；回归任务主要看 RMSE。所有任务都记录训练时间和单样本推理时间。

## 2. Wine 数据集上 xRFM AUC-ROC 为空的说明

`wine / xRFM / auc_roc` 被有意保留为 `NaN`，不是漏跑，也不是表格损坏。

原因是 Wine 是多分类任务，标准多分类 AUC-ROC 需要模型输出每个类别的概率或连续分数；当前 xRFM 实现没有提供这类输出，因此无法标准计算该指标。

不要用硬预测类别强行计算 AUC-ROC，因为那样得到的值不能和 XGBoost、LightGBM、Random Forest 基于概率算出来的 AUC-ROC 公平比较。

Discussion 可以这样写：

```text
For the multi-class Wine dataset, the current xRFM implementation does not provide class probability estimates or continuous decision scores required for standard multi-class AUC-ROC. Therefore, AUC-ROC is reported as NaN for xRFM on Wine, while accuracy is still reported. This reflects a practical limitation of the current xRFM implementation for multi-class classification evaluation.
```

## 3. Appliances Energy Subsampling 实验

- `outputs/tables/appliances_subsampling_all.csv`：所有模型的 subsampling 源数据表。
- `outputs/figures/appliances_subsampling_rmse_all.png`：训练样本数增加时，各模型 Test RMSE 的变化。
- `outputs/figures/appliances_subsampling_train_time_all.png`：训练样本数增加时，各模型训练时间的变化。

这个实验使用 `appliances_energy`，训练样本规模为 1000、3000、6000、10000 和 11841。RMSE 越低代表回归预测越好，训练时间越低代表训练效率越高。

当前主要现象是：随着训练样本增加，所有模型 RMSE 总体下降；xRFM 的 RMSE 高于 tree-based baselines，并且训练时间随样本数增加更明显。这可以作为 Discussion 中关于 xRFM 在大数据集上效率和性能限制的证据。

## 4. Interpretability Comparison

- `outputs/tables/interpretability_comparison_all.csv`：合并后的 interpretability 总表。
- `outputs/tables/*_interpretability_comparison.csv`：每个数据集单独的 interpretability 表。
- `outputs/figures/*_interpretability_comparison.png`：每个数据集单独的 interpretability 图。

interpretability 对比包含四种特征重要性：

| 方法 | 含义 |
| --- | --- |
| xRFM_AGOP | xRFM 模型内部的 AGOP diagonal 特征重要性 |
| PCA_Loadings | 第一主成分中的特征 loading，反映数据整体变化方向 |
| Mutual_Info | 单个特征和目标变量之间的信息关联 |
| Permutation | 打乱某个特征后模型性能下降多少 |

所有重要性分数都已经归一化到 0 到 1。它们适合比较同一种方法内部哪些特征更重要；不同方法之间的数值不能当成完全相同意义的绝对值。

论文篇幅只有 4 到 6 页，不一定要放全部 interpretability 图。建议选择 1 到 2 张最有代表性的图，例如 Appliances Energy 和 German Credit。

## 5. Baseline-only 输出

为了保留 baseline 过程记录，当前也保留了只包含 XGBoost、LightGBM 和 Random Forest 的输出：

- `outputs/tables/baseline_results_all.csv`
- `outputs/tables/baseline_classification_summary.csv`
- `outputs/tables/baseline_regression_summary.csv`
- `outputs/tables/baseline_classification_wide.csv`
- `outputs/tables/baseline_regression_wide.csv`
- `outputs/tables/appliances_subsampling_baselines.csv`
- `outputs/figures/appliances_subsampling_rmse.png`
- `outputs/figures/appliances_subsampling_train_time.png`

这些文件不包含 xRFM。最终论文如果要展示和 xRFM 的对比，优先使用带 `_all` 的 all-model 图和表。

## 6. 最终检查文件

- `outputs/tables/final_output_manifest.csv`：最终推荐使用的表格和图片清单。
- `outputs/tables/final_output_checks.csv`：最终输出检查结果。

当前检查结果没有 blocking error，只有一个 note：`wine / xRFM / auc_roc` 是 `NaN`，原因是 xRFM 当前实现没有多分类概率或连续分数输出。

## 7. 复现命令

如果所有模型结果已经生成，只需要重新整理最终表格和清单，可以运行：

```bash
python experiments/results/merge_all_model_results.py
python experiments/results/prepare_final_outputs.py
```

如果需要重新生成 baseline-only 表格，可以运行：

```bash
python experiments/baselines/merge_baseline_results.py
```

如果需要重新生成 all-model subsampling 图，可以运行：

```bash
python experiments/results/subsample_appliances_all_models.py
```

注意：all-model subsampling 需要 xRFM 和 torch 环境。如果当前环境不能运行 xRFM，可以先使用已经生成好的 `_all` 图和表。

## 8. 最终推荐给论文同学使用的文件

```text
outputs/tables/all_models_results.csv
outputs/tables/all_models_classification_wide.csv
outputs/tables/all_models_regression_wide.csv
outputs/tables/appliances_subsampling_all.csv
outputs/tables/interpretability_comparison_all.csv
outputs/figures/appliances_subsampling_rmse_all.png
outputs/figures/appliances_subsampling_train_time_all.png
outputs/figures/*_interpretability_comparison.png
```

## 9. Check Summary

- Missing manifest files: 0
- Blocking check errors: 0

机器可读的详细检查结果见 `outputs/tables/final_output_manifest.csv` 和 `outputs/tables/final_output_checks.csv`。
