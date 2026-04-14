# Bonus AGOP Split Verification

本目录用于放置 PDF 中 Bonus 加分项相关的实验代码和论文写作说明。

## 1. 对应 PDF 的 Bonus 要求

PDF 中的 Bonus 要求是：

```text
Implement the AGOP-based splitting criterion from scratch
without using xRFM library internals
and verify it selects the same split direction as the library on a small dataset.
```

也就是说，这部分要证明的不是 xRFM 效果更好，也不是提出新的 AGOP 改进方法，而是：

```text
我们可以手写复现 xRFM 的 AGOP 分裂规则，并且手写结果和 xRFM library 的分裂方向一致。
```

## 2. 代码运行方式

正式对应 PDF Bonus 要求的脚本是：

```bash
python experiments/bonus/agop_split_from_scratch.py
```

这个脚本完成的事情是：

1. 从 `data/processed` 中读取一个小数据集子集；
2. 不调用 `xrfm.rfm_src` 里的内部 AGOP 或 gradient 函数；
3. 手写 l2 Laplace kernel、kernel ridge regression、function gradient、AGOP matrix；
4. 对 AGOP 做 SVD，取第一右奇异向量作为 split direction；
5. 用 xRFM 的公开接口训练同样设置的模型；
6. 从 `model.get_state_dict()` 中读取 xRFM reference split direction；
7. 用 absolute cosine similarity 检查手写方向是否和 xRFM 一致。

默认输出结果表：

```text
outputs/tables/bonus_agop_split_check.csv
```

## 3. 当前实验结果

当前默认运行结果如下：

```text
Dataset: bike_sharing
Absolute cosine similarity: 0.99999999
Passed: True
From-scratch top feature: num__yr
xRFM reference top feature: num__yr
```

结果表中最重要的是：

| 字段 | 含义 |
|---|---|
| `cosine_similarity_abs` | 手写 split direction 和 xRFM reference split direction 的绝对余弦相似度 |
| `passed` | 相似度是否超过预设阈值 |
| `scratch_top_feature` | 手写 AGOP 分裂方向中权重最大的特征 |
| `xrfm_top_feature` | xRFM reference 分裂方向中权重最大的特征 |

这次结果中 `cosine_similarity_abs = 0.99999999`，非常接近 1。因为 split direction 的正负号本身没有意义，所以使用 absolute cosine similarity。这个数值说明两边方向几乎完全一致。

## 4. 论文中建议放在哪里

建议放在论文的 Appendix 或 Bonus Section 中，不需要放在主实验结果里。

正文可以简单提一句：

```text
We also attempted the optional bonus task by implementing the AGOP-based splitting criterion from scratch and verifying that it selected the same split direction as the xRFM library on a small dataset. Details are provided in the appendix.
```

Appendix 小节标题可以写：

```text
Appendix: Bonus AGOP Split Criterion Verification
```

## 5. Appendix 中建议写的内容

Appendix 中建议包含三部分：

1. 一段方法说明；
2. 一个很小的验证表格；
3. 一段结论解释。

### 5.1 方法说明

```text
For the bonus task, we implemented the AGOP-based splitting criterion from scratch instead of calling xRFM's internal AGOP or gradient functions. Specifically, we manually implemented the l2 Laplace kernel, kernel ridge regression, the gradient of the fitted kernel model with respect to input features, the AGOP matrix, and the extraction of the top singular vector as the split direction.

To verify correctness, we compared our hand-written split direction against the split direction produced by the xRFM library under the same kernel, bandwidth, regularization, random seed, and data subset.
```

### 5.2 验证表格

论文中可以放下面这个表：

```text
| Dataset | Samples | Kernel | Bandwidth | Regularization | Cosine Similarity | Passed | Top Feature |
|---|---:|---|---:|---:|---:|---|---|
| Bike Sharing | 120 | l2 Laplace | 10.0 | 0.001 | 0.99999999 | Yes | num__yr |
```

### 5.3 结论解释

表格下面可以写：

```text
The absolute cosine similarity between the from-scratch AGOP split direction and the xRFM reference split direction was 0.99999999. Since a value of 1 indicates identical directions up to sign, this result verifies that our implementation selected the same split direction as the library. Both implementations also identified num__yr as the dominant feature in the split direction.
```

## 6. 代码路径说明

论文 Appendix 最后可以补一句，说明代码和输出在哪里：

```text
The implementation is provided in experiments/bonus/agop_split_from_scratch.py, and the verification output is saved in outputs/tables/bonus_agop_split_check.csv.
```

## 7. 这部分不要怎么写

这部分不要写成：

```text
The bonus experiment shows that xRFM performs better than baseline models.
```

因为这个 Bonus 只验证 AGOP split criterion 的实现正确性，不比较模型预测性能。

也不要写成：

```text
The bonus experiment proves that AGOP improves prediction accuracy.
```

因为这里验证的是 split direction 是否一致，不是预测指标是否提升。
