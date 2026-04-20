# Bonus 输出结果说明：Residual-weighted AGOP

这份文件用于解释新版 Bonus 实验生成的两个结果表分别代表什么，方便写论文的同学直接使用。

新版 PDF 的 Bonus 要求是提出并分析一个 AGOP framework extension。我们采用的是 PDF 中给出的示例：**Residual-weighted AGOP**。

核心思想是：

```text
standard AGOP 平等看待所有样本；
residual-weighted AGOP 给预测误差更大的样本更高权重。
```

我们使用的权重函数是：

```text
w_i = r_i^2
```

其中：

```text
r_i = y_i - f(x_i)
```

也就是第 `i` 个样本的真实值和模型预测值之间的 residual。

---

## 1. 运行命令

运行新版 Bonus 实验：

```bash
python experiments/bonus/residual_weighted_agop.py
```

运行后会生成两个主要结果文件：

```text
outputs/tables/bonus_residual_agop_direction_comparison.csv
outputs/tables/bonus_residual_agop_performance_comparison.csv
```

---

## 2. Synthetic 数据集是什么意思

本 Bonus 使用的是一个小型 synthetic regression dataset。

目标函数可以理解为：

```text
y = 3 * x0 + 5 * 1{x1 > 0.35} * x2 + noise
```

其中：

| Feature | 作用 |
|---|---|
| `x0_global_linear_signal` | 全局线性信号，大部分样本都受它影响 |
| `x1_residual_gate` | 控制局部交互是否出现的 gating feature |
| `x2_local_interaction` | 只在 `x1 > 0.35` 区域里明显影响目标值 |
| `x3_noise`, `x4_noise`, `x5_noise` | 噪声特征 |

这个 synthetic dataset 的设计目的不是模拟某个真实数据集，而是构造一个清楚的例子，展示：

```text
standard AGOP 可能被全局主信号 x0 主导；
residual-weighted AGOP 可能转向模型难以拟合的局部区域 x1。
```

这正好对应新版 Bonus 的要求：找一个 residual-weighted AGOP 和 standard AGOP 方向不同、且 residual-weighted AGOP 有性能提升的例子。

---

## 3. direction comparison 表是什么意思

文件：

```text
outputs/tables/bonus_residual_agop_direction_comparison.csv
```

这个表主要回答新版 Bonus 的第 iii 点：

```text
Disagreement example:
Identify at least one dataset where residual-weighted AGOP and standard AGOP select different split directions.
```

也就是说，这个表用来证明：

```text
standard AGOP 和 residual-weighted AGOP 确实选出了不同的 split direction。
```

### 3.1 每一列的含义

| Column | 含义 |
|---|---|
| `dataset` | 使用的数据集名称，这里是 `synthetic_residual_interaction` |
| `n_samples` | synthetic 数据集总样本数 |
| `seed` | 随机种子，用于保证实验可复现 |
| `standard_top_feature` | standard AGOP split direction 中权重最大的特征 |
| `residual_weighted_top_feature` | residual-weighted AGOP split direction 中权重最大的特征 |
| `cosine_similarity_abs` | 两个 split direction 的绝对余弦相似度 |
| `disagreement_threshold` | 判断两个方向是否不同的阈值 |
| `disagreement` | 是否认定两个方向不同 |
| `weight_function` | residual weighting 使用的权重函数 |
| `high_residual_sample_share` | 高 residual 样本大概占比，用于说明 residual weighting 关注的是少数困难样本 |
| `brief_explanation` | 对方向差异的简短解释 |

### 3.2 当前结果怎么读

当前结果是：

```text
standard_top_feature = x0_global_linear_signal
residual_weighted_top_feature = x1_residual_gate
cosine_similarity_abs = 0.038346
disagreement = True
```

含义是：

```text
standard AGOP 主要选择了 x0 方向；
residual-weighted AGOP 主要选择了 x1 方向；
两个方向的 cosine similarity 只有 0.038346，几乎完全不同；
因此这是一个明确的 disagreement example。
```

### 3.3 为什么 cosine similarity 很低反而是好事

cosine similarity 表示两个方向有多像：

| Cosine Similarity | 含义 |
|---:|---|
| 接近 1 | 两个方向几乎一样 |
| 接近 0 | 两个方向几乎完全不同 |

新版 Bonus 要求我们找一个两个方法选出不同方向的例子，所以这里希望看到较低的 cosine similarity。

当前：

```text
cosine_similarity_abs = 0.038346
```

说明：

```text
residual-weighted AGOP 确实改变了 standard AGOP 的 split direction。
```

---

## 4. performance comparison 表是什么意思

文件：

```text
outputs/tables/bonus_residual_agop_performance_comparison.csv
```

这个表主要回答新版 Bonus 的第 iv 点：

```text
Performance comparison:
Identify at least one dataset where residual-weighted AGOP leads to improved performance compared to standard AGOP.
```

也就是说，这个表用来证明：

```text
在这个 synthetic example 中，residual-weighted AGOP split 的测试 RMSE 比 standard AGOP split 更低。
```

### 4.1 每一列的含义

| Column | 含义 |
|---|---|
| `method` | 使用的方法 |
| `split_top_feature` | 该方法选择的主要 split feature |
| `test_rmse` | 在 test set 上的 RMSE，越低越好 |
| `rmse_improvement_vs_standard_agop` | 相比 standard AGOP split 的 RMSE 改善幅度 |
| `left_train_size` | split 后左侧 leaf 的训练样本数 |
| `right_train_size` | split 后右侧 leaf 的训练样本数 |

### 4.2 三种 method 分别是什么

| Method | 含义 |
|---|---|
| `single_ridge_no_split` | 不做 AGOP split，只训练一个 Ridge model，用作参考 baseline |
| `standard_agop_split` | 用 standard AGOP 选择 split direction，然后左右两边各训练一个 Ridge model |
| `residual_weighted_agop_split` | 用 residual-weighted AGOP 选择 split direction，然后左右两边各训练一个 Ridge model |

注意：

```text
这里的 Ridge model 只是为了公平评估 split direction 的质量。
```

也就是说，我们比较的是：

```text
如果只改变 split direction，哪种 AGOP 产生的切分更有用？
```

而不是比较不同预测模型的能力。

### 4.3 当前结果怎么读

当前结果是：

```text
standard_agop_split RMSE = 2.258162
residual_weighted_agop_split RMSE = 1.722678
RMSE improvement = 0.535484
```

含义是：

```text
使用 residual-weighted AGOP 选择的 split direction 后，test RMSE 降低了约 0.535。
```

这说明在这个 synthetic setting 中：

```text
residual weighting 改善了 splitting behavior。
```

---

## 5. 可以直接写进论文的结论

可以写：

```text
In the synthetic residual-interaction dataset, standard AGOP selected the globally dominant feature x0 as the main split direction, whereas residual-weighted AGOP selected x1, the gating feature controlling the high-residual interaction region. The absolute cosine similarity between the two directions was only 0.038346, indicating a clear disagreement. Using the same two-leaf Ridge evaluator, residual-weighted AGOP reduced test RMSE from 2.258162 to 1.722678 compared with the standard AGOP split. This suggests that residual weighting can help when the standard AGOP is dominated by global structure but the remaining prediction error is concentrated in a local region.
```

中文理解：

```text
在这个 synthetic 数据集中，standard AGOP 被全局主导特征 x0 影响，所以选择了 x0 方向。
residual-weighted AGOP 更关注预测误差大的区域，因此选择了控制局部交互区域的 x1。
两个方向几乎完全不同，并且 residual-weighted AGOP 的 split 让 RMSE 更低。
```

---

## 6. 不要过度解读

不要写成：

```text
Residual-weighted AGOP is always better than standard AGOP.
```

因为我们只证明了：

```text
在至少一个 synthetic example 中 residual-weighted AGOP 更好。
```

也不要写成：

```text
Residual-weighted AGOP improves the full xRFM model on all datasets.
```

因为这里比较的是一个 controlled two-leaf split evaluator，不是完整 xRFM pipeline。

更稳妥的说法是：

```text
This example suggests that residual-weighting can be useful when the largest remaining errors are concentrated in a local region that is not captured by the globally dominant standard AGOP direction.
```

---

## 7. 和旧版 Bonus 文件的关系

`experiments/bonus/agop_split_from_scratch.py` 是旧版 PDF 要求下写的 standard AGOP from-scratch verification。

现在新版 PDF 已经改成 residual-weighted AGOP extension，因此：

```text
正式 Bonus 应该使用 residual_weighted_agop.py 和两个 bonus_residual_agop_*.csv 文件。
```

旧版 `agop_split_from_scratch.py` 可以保留作为 standard AGOP 的参考实现，但论文中不要把它作为新版 Bonus 的主要内容。
