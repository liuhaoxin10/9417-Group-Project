"""
LightGBM baseline training script.

用途：
1. 读取 data/processed 中固定好的 train/validation/test split；
2. 用 validation set 对 LightGBM 做小规模网格调参；
3. 用最佳参数在 train + validation 上重新训练；
4. 只在 test set 上评估一次；
5. 保存统一格式的结果表，方便后续和 XGBoost、Random Forest、xRFM 合并。

运行方式：
    python experiments/baselines/train_lightgbm.py

输出：
    outputs/tables/lightgbm_results.csv
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError as exc:
    raise ImportError(
        "缺少 lightgbm 依赖。请先运行：pip install lightgbm，"
        "或者运行：pip install -r requirements.txt"
    ) from exc


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 当前项目使用的 5 个数据集。task_type 优先从 metadata.json 读取，
# 这里作为兜底配置。
DATASETS = {
    "wine": "classification",
    "divorce": "classification",
    "german_credit": "classification",
    "bike_sharing": "regression",
    "appliances_energy": "regression",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便只跑部分数据集或指定输出路径。"""
    parser = argparse.ArgumentParser(description="Train LightGBM baselines.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="预处理后 train/val/test 文件所在目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/tables/lightgbm_results.csv",
        help="结果表输出路径。",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        help="要训练的数据集名称。默认训练全部数据集。",
    )
    return parser.parse_args()


def load_metadata(processed_dir: Path, dataset_name: str) -> dict[str, Any]:
    """读取 metadata，用来判断任务类型和记录数据规模。"""
    metadata_path = processed_dir / f"{dataset_name}_metadata.json"
    if not metadata_path.exists():
        return {
            "dataset_name": dataset_name,
            "task_type": DATASETS[dataset_name],
        }

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split(processed_dir: Path, dataset_name: str) -> dict[str, Any]:
    """读取一个数据集的固定 train/validation/test split。"""
    metadata = load_metadata(processed_dir, dataset_name)

    # X 文件已经由预处理脚本完成了缺失值填充、标准化和 one-hot 编码。
    X_train = pd.read_csv(processed_dir / f"{dataset_name}_X_train.csv")
    X_val = pd.read_csv(processed_dir / f"{dataset_name}_X_val.csv")
    X_test = pd.read_csv(processed_dir / f"{dataset_name}_X_test.csv")

    # y 文件只有一列 target，这里转成一维数组。
    y_train = pd.read_csv(processed_dir / f"{dataset_name}_y_train.csv")["target"].to_numpy()
    y_val = pd.read_csv(processed_dir / f"{dataset_name}_y_val.csv")["target"].to_numpy()
    y_test = pd.read_csv(processed_dir / f"{dataset_name}_y_test.csv")["target"].to_numpy()

    return {
        "metadata": metadata,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def parameter_grid(task_type: str) -> list[dict[str, Any]]:
    """
    为 LightGBM 准备一个小型调参网格。

    这个网格故意控制得比较小：目标是得到稳定、可复现的 baseline，
    而不是进行非常耗时的大规模超参数搜索。
    """
    common_grid = {
        "n_estimators": [200, 500],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31],
        "min_child_samples": [5, 20],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1.0],
    }

    # 分类和回归使用同一组核心超参数，方便横向比较。
    keys = list(common_grid.keys())
    return [dict(zip(keys, values)) for values in product(*(common_grid[k] for k in keys))]


def make_model(task_type: str, params: dict[str, Any], n_classes: int | None = None):
    """根据任务类型创建 LightGBM 模型。"""
    base_params = {
        **params,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        # force_col_wise 可以减少 LightGBM 自动测试行/列并行策略的额外开销。
        "force_col_wise": True,
    }

    if task_type == "regression":
        return LGBMRegressor(
            objective="regression",
            metric="rmse",
            **base_params,
        )

    if n_classes is None:
        raise ValueError("分类任务必须提供 n_classes。")

    if n_classes <= 2:
        return LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            **base_params,
        )

    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        metric="multi_logloss",
        **base_params,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE，兼容不同 sklearn 版本。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def classification_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """计算二分类或多分类 AUC-ROC；无法计算时返回 NaN。"""
    try:
        if n_classes <= 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        # 如果某个 split 类别不完整，AUC 会无法定义。
        return float("nan")


def validation_score(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_classes: int | None = None,
) -> float:
    """
    返回用于选择最佳超参数的 validation 分数。

    回归：使用负 RMSE，数值越大代表 RMSE 越小；
    分类：优先使用 AUC-ROC，无法计算时退回 Accuracy。
    """
    if task_type == "regression":
        return -rmse(y_true, y_pred)

    assert y_proba is not None
    assert n_classes is not None
    auc = classification_auc(y_true, y_proba, n_classes)
    if np.isnan(auc):
        return float(accuracy_score(y_true, y_pred))
    return auc


def tune_on_validation(data: dict[str, Any], task_type: str) -> dict[str, Any]:
    """在 validation set 上搜索最佳 LightGBM 参数。"""
    X_train = data["X_train"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    label_encoder = None
    n_classes = None

    # 分类任务统一编码到 0 到 K-1，避免模型对标签格式有额外要求。
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_train_fit = label_encoder.fit_transform(y_train)
        y_val_eval = label_encoder.transform(y_val)
        n_classes = len(label_encoder.classes_)
    else:
        y_train_fit = y_train
        y_val_eval = y_val

    best_score = -np.inf
    best_params: dict[str, Any] | None = None

    for params in parameter_grid(task_type):
        model = make_model(task_type, params, n_classes=n_classes)
        model.fit(X_train, y_train_fit)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if task_type == "classification" else None

        score = validation_score(
            task_type=task_type,
            y_true=y_val_eval,
            y_pred=y_pred,
            y_proba=y_proba,
            n_classes=n_classes,
        )

        if score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError("没有找到可用的 LightGBM 参数。")

    return {
        "best_params": best_params,
        "best_validation_score": best_score,
        "label_encoder": label_encoder,
        "n_classes": n_classes,
    }


def train_final_and_evaluate(
    dataset_name: str,
    data: dict[str, Any],
    task_type: str,
    tuning_result: dict[str, Any],
) -> dict[str, Any]:
    """
    用最佳参数在 train + validation 上重新训练，
    然后只在 test set 上评估一次。
    """
    X_train_val = pd.concat([data["X_train"], data["X_val"]], axis=0)
    y_train_val = np.concatenate([data["y_train"], data["y_val"]])
    X_test = data["X_test"]
    y_test = data["y_test"]

    label_encoder: LabelEncoder | None = tuning_result["label_encoder"]
    n_classes: int | None = tuning_result["n_classes"]

    if task_type == "classification":
        assert label_encoder is not None
        y_train_val_fit = label_encoder.transform(y_train_val)
        y_test_eval = label_encoder.transform(y_test)
    else:
        y_train_val_fit = y_train_val
        y_test_eval = y_test

    model = make_model(
        task_type=task_type,
        params=tuning_result["best_params"],
        n_classes=n_classes,
    )

    train_start = time.perf_counter()
    model.fit(X_train_val, y_train_val_fit)
    train_time_sec = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time_sec = time.perf_counter() - predict_start
    inference_time_per_sample_ms = inference_time_sec / len(X_test) * 1000

    result = {
        "dataset": dataset_name,
        "task_type": task_type,
        "model": "LightGBM",
        "n_train": int(len(data["X_train"])),
        "n_val": int(len(data["X_val"])),
        "n_test": int(len(data["X_test"])),
        "n_features": int(data["X_train"].shape[1]),
        "best_validation_score": tuning_result["best_validation_score"],
        "best_params": json.dumps(tuning_result["best_params"], sort_keys=True),
        "train_time_sec": train_time_sec,
        "inference_time_per_sample_ms": inference_time_per_sample_ms,
        "rmse": np.nan,
        "accuracy": np.nan,
        "auc_roc": np.nan,
    }

    if task_type == "regression":
        result["rmse"] = rmse(y_test_eval, y_pred)
    else:
        y_proba = model.predict_proba(X_test)
        result["accuracy"] = float(accuracy_score(y_test_eval, y_pred))
        result["auc_roc"] = classification_auc(y_test_eval, y_proba, n_classes or 2)

    return result


def train_one_dataset(processed_dir: Path, dataset_name: str) -> dict[str, Any]:
    """训练并评估单个数据集。"""
    data = load_split(processed_dir, dataset_name)
    task_type = data["metadata"].get("task_type", DATASETS[dataset_name])

    print(f"\n===== Training LightGBM on {dataset_name} ({task_type}) =====")
    print(
        "split shapes:",
        data["X_train"].shape,
        data["X_val"].shape,
        data["X_test"].shape,
    )

    tuning_result = tune_on_validation(data, task_type)
    result = train_final_and_evaluate(dataset_name, data, task_type, tuning_result)

    print("best params:", tuning_result["best_params"])
    if task_type == "regression":
        print(f"test RMSE: {result['rmse']:.6f}")
    else:
        print(f"test Accuracy: {result['accuracy']:.6f}, test AUC-ROC: {result['auc_roc']:.6f}")
    print(f"train time: {result['train_time_sec']:.3f}s")

    return result


def main() -> None:
    """脚本入口：逐个数据集训练 LightGBM，并保存汇总结果。"""
    args = parse_args()

    # 和 XGBoost 脚本一样：相对路径统一解释为相对于项目根目录，
    # 避免在 PyCharm 或其他工作目录运行时找不到 data/processed。
    if not args.processed_dir.is_absolute():
        args.processed_dir = PROJECT_ROOT / args.processed_dir
    if not args.output.is_absolute():
        args.output = PROJECT_ROOT / args.output

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"未知数据集：{dataset_name}，可选值：{list(DATASETS)}")
        results.append(train_one_dataset(args.processed_dir, dataset_name))

    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved LightGBM results to: {args.output}")


if __name__ == "__main__":
    main()
