# Bonus: Residual-weighted AGOP

本目录用于放置新版 PDF 中 Bonus 加分项相关的代码和论文写作说明。

## 1. 新版 PDF 的 Bonus 要求

新版 PDF 的 Bonus 要求已经改为：

```text
Propose and analyse an extension to the AGOP framework.
```

PDF 给出的示例是 residual-weighted AGOP。它要求完成四件事：

1. **Conceptual justification**：解释为什么 residual-weighting 可能改进 standard AGOP splitting criterion；
2. **Implementation**：实现 residual-weighted AGOP，并在小数据集上选择 split direction，同时和 standard AGOP 的方向对比；
3. **Disagreement example**：找一个真实或 synthetic 数据集，让 residual-weighted AGOP 和 standard AGOP 选出不同的方向，并解释原因；
4. **Performance comparison**：找一个 residual-weighted AGOP 比 standard AGOP 表现更好的例子，并解释为什么有效。

## 2. 正式 Bonus 脚本

正式对应新版 PDF Bonus 要求的脚本是：

```bash
python experiments/bonus/residual_weighted_agop.py
```

这个脚本做的事情是：

1. 构造一个小型 synthetic regression dataset；
2. 训练一个 kernel ridge predictor；
3. 计算 standard AGOP；
4. 根据 residual weights `w_i = r_i^2` 计算 residual-weighted AGOP；
5. 分别取两个 AGOP 的 top singular vector 作为 split direction；
6. 比较两个 split direction 的 absolute cosine similarity；
7. 用相同的 two-leaf Ridge evaluator 比较 standard AGOP split 和 residual-weighted AGOP split 的 test RMSE。

默认输出两个表：

```text
outputs/tables/bonus_residual_agop_direction_comparison.csv
outputs/tables/bonus_residual_agop_performance_comparison.csv
```

## 3. 当前运行结果

当前默认运行结果如下：

```text
Dataset: synthetic_residual_interaction
Standard AGOP top feature: x0_global_linear_signal
Residual-weighted AGOP top feature: x1_residual_gate
Absolute cosine similarity: 0.038346
Disagreement: True
Standard AGOP split RMSE: 2.258162
Residual-weighted AGOP split RMSE: 1.722678
RMSE improvement: 0.535484
```

这个结果说明：

- standard AGOP 主要关注全局线性信号 `x0_global_linear_signal`；
- residual-weighted AGOP 主要关注产生高残差区域的 gating 特征 `x1_residual_gate`；
- 两个方向的 absolute cosine similarity 只有 `0.038346`，说明它们几乎是完全不同的方向；
- residual-weighted AGOP split 的 test RMSE 更低，说明在这个 synthetic setting 中 residual weighting 改善了 splitting behavior。

如果需要逐列理解两个 CSV 的含义，请看：

```text
experiments/bonus/bonus_outputs_explained.md
```

## 4. Synthetic 数据集设计

脚本中的 synthetic dataset 有 6 个特征，其中关键特征是：

| Feature | Role |
|---|---|
| `x0_global_linear_signal` | 全局线性信号，大多数样本都受它影响 |
| `x1_residual_gate` | 控制局部交互是否激活的 gating feature |
| `x2_local_interaction` | 只在 `x1` 较大时影响目标值的局部交互特征 |

目标函数可以理解为：

```text
y = 3 * x0 + 5 * 1{x1 > 0.35} * x2 + noise
```

这个设计的直觉是：

- standard AGOP 会被全局主导信号 `x0` 影响；
- 但模型在 `x1 > 0.35` 的局部交互区域更容易出现较大 residual；
- residual-weighted AGOP 会给这些高 residual 样本更高权重，因此更容易选择 `x1` 作为 split direction；
- 按 `x1` 切分后，右侧 leaf 能更好地处理局部交互结构，所以测试 RMSE 更低。

## 5. 论文中建议放在哪里

建议放在 Appendix 或 Bonus Section 中。正文可以简单提一句：

```text
We also attempted the optional bonus task by implementing and analysing a residual-weighted extension of the AGOP splitting criterion. Details are provided in the appendix.
```

Appendix 小节标题可以写：

```text
Appendix: Residual-weighted AGOP Extension
```

## 6. Appendix 中可使用的英文说明

### 6.1 Conceptual justification

```text
The standard AGOP averages gradient outer products uniformly across all training samples. As a result, its leading direction can be dominated by global structure that is already relatively easy for the predictor to fit. We considered a residual-weighted AGOP extension, where each sample receives weight w_i = r_i^2 based on the squared residual of a trained predictor. This gives more influence to regions where the current predictor underfits the data, and may therefore produce split directions that isolate difficult local structure.
```

### 6.2 Implementation

```text
We implemented residual-weighted AGOP by first fitting a kernel ridge predictor, computing residuals r_i = y_i - f(x_i), and setting w_i = r_i^2. We then computed the weighted matrix

AGOP_res(f) = sum_i w_i grad f(x_i) grad f(x_i)^T / sum_i w_i.

The split direction was selected as the top singular vector of this matrix. We compared it against the standard AGOP direction computed from the same fitted predictor.
```

### 6.3 Disagreement and performance comparison

论文中可以放下面这个表：

```text
| Dataset | Standard Top Feature | Residual-weighted Top Feature | Cosine Similarity | Disagreement |
|---|---|---|---:|---|
| synthetic_residual_interaction | x0_global_linear_signal | x1_residual_gate | 0.038346 | Yes |
```

再放一个 performance 表：

```text
| Method | Split Top Feature | Test RMSE |
|---|---|---:|
| Standard AGOP split | x0_global_linear_signal | 2.258162 |
| Residual-weighted AGOP split | x1_residual_gate | 1.722678 |
```

表格下面可以写：

```text
In this synthetic example, the standard AGOP direction was dominated by the global linear signal x0. In contrast, residual-weighted AGOP shifted the split direction toward x1, which controls the region where a local x2 interaction becomes active. The two directions had absolute cosine similarity 0.038346, showing a clear disagreement. Using the residual-weighted direction in the same two-leaf Ridge evaluator reduced test RMSE from 2.258162 to 1.722678. This suggests that residual weighting can improve splitting when the main remaining error is concentrated in a local region that is not captured by the globally dominant AGOP direction.
```

## 7. 代码路径说明

论文 Appendix 最后可以补一句：

```text
The implementation is provided in experiments/bonus/residual_weighted_agop.py. The direction comparison and performance comparison are saved in outputs/tables/bonus_residual_agop_direction_comparison.csv and outputs/tables/bonus_residual_agop_performance_comparison.csv.
```

## 8. 旧版 Bonus 脚本说明

`experiments/bonus/agop_split_from_scratch.py` 是旧版 PDF 要求下写的 standard AGOP from-scratch verification。它可以作为 standard AGOP 计算的参考，但**不是新版 PDF Bonus 的正式提交重点**。
