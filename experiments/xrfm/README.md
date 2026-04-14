# xRFM Experiments

本目录只放 xRFM 相关实验代码，避免和 baseline 模型混在一起。

## Scripts

```text
train_xrfm.py
```

训练 xRFM，并输出：

```text
outputs/tables/xrfm_results.csv
```

```text
interpretability_xrfm.py
```

生成 xRFM AGOP diagonal、PCA loadings、mutual information 和 permutation importance 的可解释性对比结果。

默认输出：

```text
outputs/tables/{dataset}_interpretability_comparison.csv
outputs/figures/{dataset}_interpretability_comparison.png
```

## Example

```bash
python experiments/xrfm/train_xrfm.py
python experiments/xrfm/interpretability_xrfm.py --dataset wine
```
