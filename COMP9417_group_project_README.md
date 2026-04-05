# COMP9417 小组项目

## 项目简介

这个仓库用于我们 COMP9417 小组项目的开发、实验和报告整理。

本项目的整体目标是：

1. 选择若干数据集；
2. 对数据进行清洗、预处理和必要的特征工程；
3. 使用多个机器学习模型进行训练与比较；
4. 对实验结果进行评估、分析和可视化；
5. 完成最终项目报告。

为了方便小组协作，我们将代码、数据、实验记录、结果图表和报告草稿分别放在不同文件夹中。请所有成员尽量按照本 README 中的说明使用仓库结构，以避免文件混乱和重复劳动。

---

## 仓库结构说明

当前仓库的主要结构如下：

```text
9417-Group-Project/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── docs/
│   └── task_allocation.md
├── experiments/
│   ├── dataset1/
│   ├── dataset2/
│   └── dataset3/
├── notebooks/
│   ├── exploration/
│   ├── modeling/
│   └── preprocessing/
├── outputs/
│   ├── figures/
│   └── tables/
├── report/
│   ├── draft/
│   │   ├── introduction.md
│   │   ├── method.md
│   │   ├── outline.md
│   │   └── results.md
│   ├── figures/
│   ├── final/
│   └── tables/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   └── models/
```

---

## 各文件夹用途说明

### 1. `data/`

这个文件夹用于存放项目使用的数据。

#### `data/raw/`

用于存放**原始数据集**。这里的文件应尽量保持原样，不要直接手动修改原始数据。

建议：
- 从公开网站下载的数据原文件放在这里；
- 文件名尽量清晰，例如：
  - `wine_quality.csv`
  - `heart_disease.csv`

#### `data/processed/`

用于存放**预处理后的数据**。例如：
- 清洗后的数据；
- 编码后的数据；
- 划分训练集/测试集后的数据；
- 特征工程处理后的数据。

#### `data/README.md`

用于说明数据相关信息，例如：
- 数据集名称；
- 数据来源；
- 原始文件名；
- 处理后文件名；
- 哪个脚本负责处理该数据。

---

### 2. `docs/`

这个文件夹用于存放项目协作相关文档。

#### `docs/task_allocation.md`

用于记录小组成员分工，例如：
- 谁负责找数据集；
- 谁负责预处理；
- 谁负责模型训练；
- 谁负责结果分析；
- 谁负责报告撰写与整合。

如果后续还有会议记录、进度安排等内容，也可以继续放在 `docs/` 中。

---

### 3. `experiments/`

这个文件夹用于记录不同数据集上的实验过程和实验结果说明。

目前我们使用：
- `experiments/dataset1/`
- `experiments/dataset2/`
- `experiments/dataset3/`

每个子文件夹可以存放：
- 该数据集上跑过的模型说明；
- 参数设置；
- baseline 结果；
- 实验结论草稿；
- 结果记录文档。

例如：
- `lr_baseline.md`：逻辑回归 baseline 的实验记录；
- 后续也可以增加：
  - `svm.md`
  - `random_forest.md`
  - `knn.md`

**注意：** 目前 `dataset1`、`dataset2`、`dataset3` 只是占位名字，后面如果确定了数据集，也可以改成更具体的名字。

---

### 4. `notebooks/`

这个文件夹用于存放 Jupyter Notebook 文件，主要用于探索、试验和可视化。

#### `notebooks/exploration/`

用于存放数据探索 notebook，例如：
- 查看数据分布；
- 检查缺失值；
- 初步可视化；
- 观察类别不平衡情况等。

#### `notebooks/preprocessing/`

用于存放数据预处理 notebook，例如：
- 数据清洗；
- 缺失值处理；
- 编码；
- 标准化；
- 特征工程尝试。

#### `notebooks/modeling/`

用于存放模型试验 notebook，例如：
- 训练某个模型的原型实验；
- 对参数进行初步测试；
- 观察模型表现。

**建议：**
- notebook 更适合做探索和试验；
- 最终正式使用的代码尽量整理到 `src/` 中，而不是只留在 notebook 里。

---

### 5. `outputs/`

这个文件夹用于存放实验生成的输出结果。

#### `outputs/figures/`

用于存放实验生成的图像，例如：
- 数据分布图；
- 模型比较图；
- ROC 曲线；
- 混淆矩阵图；
- 其他可视化图表。

#### `outputs/tables/`

用于存放实验结果表格，例如：
- 模型性能对比表；
- 各数据集结果汇总表；
- 参数实验结果表。

---

### 6. `report/`

这个文件夹用于存放项目报告相关内容。

#### `report/draft/`

用于存放报告草稿文件，目前包括：
- `outline.md`：报告提纲
- `introduction.md`：引言部分
- `method.md`：方法部分
- `results.md`：实验结果与分析部分

后续如果需要，也可以继续增加：
- `discussion.md`
- `conclusion.md`

#### `report/figures/`

用于存放**最终准备插入报告中的图**。通常这里的图应是经过筛选和整理后的版本。

#### `report/tables/`

用于存放**最终准备插入报告中的表格**。

#### `report/final/`

用于存放最终提交版本，例如：
- 最终 PDF；
- 最终 Word 文档；
- 其他最终整理好的提交文件。

---

### 7. `src/`

这个文件夹用于存放项目的正式 Python 代码。

#### `src/data/`

用于存放与数据读取和基础处理相关的代码，例如：
- 读取数据；
- 合并数据；
- 划分训练集和测试集；
- 数据保存。

#### `src/features/`

用于存放特征工程相关代码，例如：
- 特征构建；
- 特征编码；
- 特征选择；
- 标准化和归一化。

#### `src/models/`

用于存放模型训练相关代码，例如：
- 逻辑回归；
- SVM；
- 决策树；
- 随机森林；
- kNN；
- 其他模型训练脚本。

#### `src/evaluation/`

用于存放评估相关代码，例如：
- accuracy / precision / recall / F1 计算；
- confusion matrix；
- cross validation；
- 模型比较；
- 可视化结果生成。

---

## 文件放置规则建议

为了避免仓库混乱，请尽量遵守以下规则：

### 数据相关
- 原始数据放在 `data/raw/`
- 处理后的数据放在 `data/processed/`
- 不要直接修改原始数据文件

### 代码相关
- 正式代码放在 `src/`
- 临时探索或试验可以放在 `notebooks/`
- 不要把大量正式代码直接写在仓库根目录

### 实验相关
- 实验过程记录放在 `experiments/`
- 生成的图放在 `outputs/figures/`
- 生成的表放在 `outputs/tables/`

### 报告相关
- 报告草稿放在 `report/draft/`
- 最终使用的图和表分别放在 `report/figures/` 和 `report/tables/`
- 最终提交版本放在 `report/final/`

---

## 小组协作建议

为了减少冲突，建议大家按下面方式协作：

1. **开始写代码前先更新项目**
   - 先拉取最新版本，避免基于旧代码继续修改。

2. **尽量不要多人同时改同一个文件**
   - 尤其是公共代码文件和报告总文件。

3. **提交前写清楚提交信息**
   - 例如：
     - `add wine dataset preprocessing notebook`
     - `implement svm training script`
     - `update introduction draft`

4. **不要把无关文件提交到仓库**
   - 例如 IDE 配置、缓存文件、临时文件等。

---

## 依赖环境

项目依赖包记录在：

```text
requirements.txt
```

如果后续新增依赖，请及时更新该文件。

---

## 当前工作内容（可后续更新）

### 数据集
- 待补充

### 模型
- 待补充

### 评估指标
- 待补充

### 成员分工
详见：

```text
docs/task_allocation.md
```

---

## 后续维护建议

随着项目推进，建议逐步补充以下内容：
- 在 `data/README.md` 中写清数据来源和说明；
- 在 `docs/task_allocation.md` 中更新每位成员分工；
- 在 `experiments/` 中记录每个数据集和模型的实验过程；
- 在 `report/draft/` 中持续更新报告内容；
- 在本 README 中补充最终使用的数据集、模型和运行方式。

---

## 备注

本仓库当前仍在项目初期搭建阶段，文件结构会随着项目推进继续完善。如果需要新增文件夹或调整内容，请尽量保持整体结构清晰，并提前与组员同步。
