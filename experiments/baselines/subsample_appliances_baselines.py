from __future__ import annotations
"""
Subsampling experiment on Appliances Energy for baseline models.

PDF 任务要求：
在一个 n > 10,000 的大数据集上，使用多个训练集大小进行 subsampling，
并画出 test performance 和 training time 随训练样本数 n 的变化。

本脚本只运行 baseline 模型：
- XGBoost
- LightGBM
- Random Forest

运行方式：
    python experiments/baselines/subsample_appliances_baselines.py

输出：
    outputs/tables/appliances_subsampling_baselines.csv
    outputs/figures/appliances_subsampling_rmse.png
    outputs/figures/appliances_subsampling_train_time.png
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("缺少 xgboost 依赖。请先运行：pip install xgboost") from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("缺少 lightgbm 依赖。请先运行：pip install lightgbm") from exc


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "appliances_energy"
MODELS = ["XGBoost", "LightGBM", "Random Forest"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认复现 baseline subsampling。"""
    parser = argparse.ArgumentParser(description="Run baseline Appliances Energy subsampling.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=PROJECT_ROOT / "outputs/tables/baseline_results_all.csv",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=PROJECT_ROOT / "outputs/tables/appliances_subsampling_baselines.csv",
    )
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/figures")
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[1000, 3000, 6000, 10000])
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """把相对路径解释为项目根目录下的路径。"""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_appliances_split(processed_dir: Path) -> dict[str, Any]:
    """读取 Appliances Energy 的固定 train/val/test split。"""
    X_train = pd.read_csv(processed_dir / f"{DATASET_NAME}_X_train.csv")
    X_test = pd.read_csv(processed_dir / f"{DATASET_NAME}_X_test.csv")
    y_train = pd.read_csv(processed_dir / f"{DATASET_NAME}_y_train.csv")["target"].to_numpy()
    y_test = pd.read_csv(processed_dir / f"{DATASET_NAME}_y_test.csv")["target"].to_numpy()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE，和主训练脚本保持一致。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_best_params(baseline_path: Path) -> dict[str, dict[str, Any]]:
    """从 baseline 主结果表中读取 Appliances Energy 的最佳参数。"""
    params: dict[str, dict[str, Any]] = {model_name: {} for model_name in MODELS}
    if not baseline_path.exists():
        return params

    df = pd.read_csv(baseline_path)
    dataset_df = df[df["dataset"] == DATASET_NAME]
    for model_name in MODELS:
        row = dataset_df[dataset_df["model"] == model_name]
        if not row.empty:
            params[model_name] = json.loads(row.iloc[0]["best_params"])
    return params


def make_model(model_name: str, params: dict[str, Any]):
    """根据模型名和已选参数创建 baseline 回归器。"""
    if model_name == "XGBoost":
        return XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "LightGBM":
        return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)
    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    raise ValueError(f"未知模型：{model_name}")


def sample_training_subset(X_train: pd.DataFrame, y_train: np.ndarray, train_size: int):
    """用固定随机种子抽取训练子集，保证不同模型使用同一批样本。"""
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(X_train), size=train_size, replace=False)
    return X_train.iloc[indices], y_train[indices]


def run_one_setting(
    model_name: str,
    params: dict[str, Any],
    X_subset: pd.DataFrame,
    y_subset: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, float]:
    """训练一个 baseline，并在固定 test set 上记录 RMSE 和耗时。"""
    model = make_model(model_name, params)

    train_start = time.perf_counter()
    model.fit(X_subset, y_subset)
    train_time_sec = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time_sec = time.perf_counter() - predict_start

    return {
        "rmse": rmse(y_test, y_pred),
        "train_time_sec": train_time_sec,
        "inference_time_per_sample_ms": inference_time_sec / len(X_test) * 1000,
    }


def save_line_plot(result_df: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: Path) -> None:
    """按模型画出某个指标随训练样本数变化的折线图。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model_name in MODELS:
        model_df = result_df[result_df["model"] == model_name]
        ax.plot(model_df["train_size"], model_df[metric], marker="o", label=model_name)
    ax.legend()
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """脚本入口：运行 Appliances Energy 的 baseline subsampling 实验。"""
    args = parse_args()
    args.processed_dir = resolve_path(args.processed_dir)
    args.baseline_results = resolve_path(args.baseline_results)
    args.output_table = resolve_path(args.output_table)
    args.figures_dir = resolve_path(args.figures_dir)

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_appliances_split(args.processed_dir)
    best_params = load_best_params(args.baseline_results)

    sample_sizes = [s for s in args.sample_sizes if 0 < s <= len(data["X_train"])]
    sample_sizes.append(len(data["X_train"]))
    sample_sizes = sorted(set(sample_sizes))

    rows = []
    for train_size in sample_sizes:
        X_subset, y_subset = sample_training_subset(data["X_train"], data["y_train"], train_size)
        for model_name in MODELS:
            print(f"Training {model_name} with n={train_size}...")
            metrics = run_one_setting(
                model_name,
                best_params[model_name],
                X_subset,
                y_subset,
                data["X_test"],
                data["y_test"],
            )
            rows.append({
                "dataset": DATASET_NAME,
                "model": model_name,
                "train_size": train_size,
                "rmse": metrics["rmse"],
                "train_time_sec": metrics["train_time_sec"],
                "inference_time_per_sample_ms": metrics["inference_time_per_sample_ms"],
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output_table, index=False)
    print(f"Saved subsampling results to: {args.output_table}")

    save_line_plot(
        result_df,
        "rmse",
        "Test RMSE",
        "Baseline Test RMSE vs Training Size",
        args.figures_dir / "appliances_subsampling_rmse.png",
    )
    save_line_plot(
        result_df,
        "train_time_sec",
        "Training Time (seconds)",
        "Baseline Training Time vs Training Size",
        args.figures_dir / "appliances_subsampling_train_time.png",
    )


if __name__ == "__main__":
    main()
