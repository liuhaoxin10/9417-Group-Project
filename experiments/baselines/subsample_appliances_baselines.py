"""
Subsampling experiment on Appliances Energy for baseline models.

PDF 任务要求：
在一个 n > 10,000 的大数据集上，使用多个训练集大小进行 subsampling，
并画出 test performance 和 training time 随训练样本数 n 的变化。

本脚本做的事情：
1. 固定 appliances_energy 的 test set；
2. 从 appliances_energy_X_train.csv 中抽取不同大小的训练子集；
3. 分别训练 XGBoost、LightGBM、Random Forest；
4. 记录每个模型在 test set 上的 RMSE 和训练时间；
5. 输出结果表和两张图。

运行方式：
    python experiments/baselines/subsample_appliances_baselines.py

输出：
    outputs/tables/appliances_subsampling_baselines.csv
    outputs/figures/appliances_subsampling_rmse.png
    outputs/figures/appliances_subsampling_train_time.png
"""

from __future__ import annotations

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
    raise ImportError(
        "缺少 xgboost 依赖。请先运行：pip install xgboost，"
        "或者运行：pip install -r requirements.txt"
    ) from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError(
        "缺少 lightgbm 依赖。请先运行：pip install lightgbm，"
        "或者运行：pip install -r requirements.txt"
    ) from exc


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "appliances_energy"
MODELS = ["XGBoost", "LightGBM", "Random Forest"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便修改样本大小或输出目录。"""
    parser = argparse.ArgumentParser(
        description="Run Appliances Energy subsampling experiment for baseline models."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="预处理后 train/test 文件所在目录。",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=PROJECT_ROOT / "outputs/tables/baseline_results_all.csv",
        help="包含各模型最佳参数的 baseline 总表。",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=PROJECT_ROOT / "outputs/tables/appliances_subsampling_baselines.csv",
        help="subsampling 结果表输出路径。",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/figures",
        help="图像输出目录。",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="*",
        type=int,
        default=[1000, 3000, 6000, 10000],
        help="要抽取的训练集大小；脚本会自动追加 full train size。",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """把相对路径解释为相对于项目根目录，避免从不同目录运行时找不到文件。"""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_appliances_split(processed_dir: Path) -> dict[str, Any]:
    """读取 Appliances Energy 的固定 train/test split。"""
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
    """计算 RMSE，兼容不同 sklearn 版本。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def default_params() -> dict[str, dict[str, Any]]:
    """
    如果 baseline_results_all.csv 不存在，就使用这组兜底参数。

    正常情况下脚本会读取前面调参得到的最佳参数；
    这里只是为了避免缺少结果表时脚本完全不可运行。
    """
    return {
        "XGBoost": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        "LightGBM": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        "Random Forest": {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
    }


def load_best_params(baseline_results_path: Path) -> dict[str, dict[str, Any]]:
    """
    从 baseline_results_all.csv 读取 Appliances Energy 上各模型的最佳参数。

    这样 subsampling 实验只改变训练样本数，不重新调参，
    更容易解释为“数据规模变化对模型表现和训练时间的影响”。
    """
    params = default_params()

    if not baseline_results_path.exists():
        print(f"Warning: 找不到 {baseline_results_path}，将使用默认参数。")
        return params

    result_df = pd.read_csv(baseline_results_path)
    dataset_df = result_df[result_df["dataset"] == DATASET_NAME]

    for model_name in MODELS:
        row = dataset_df[dataset_df["model"] == model_name]
        if row.empty:
            print(f"Warning: {baseline_results_path} 中没有 {model_name}，将使用默认参数。")
            continue
        params[model_name] = json.loads(row.iloc[0]["best_params"])

    return params


def make_model(model_name: str, params: dict[str, Any]):
    """根据模型名称和参数创建回归模型。"""
    if model_name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )

    if model_name == "LightGBM":
        return LGBMRegressor(
            objective="regression",
            metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
            **params,
        )

    if model_name == "Random Forest":
        return RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )

    raise ValueError(f"未知模型：{model_name}")


def make_sample_sizes(requested_sizes: list[int], full_train_size: int) -> list[int]:
    """整理训练样本大小：过滤非法值，并自动追加 full train size。"""
    valid_sizes = [size for size in requested_sizes if 0 < size <= full_train_size]
    valid_sizes.append(full_train_size)
    return sorted(set(valid_sizes))


def sample_training_subset(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    从训练集中固定随机抽样。

    注意：test set 完全不参与抽样和训练，保证最终 test performance 是 held-out。
    """
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
    """训练一个模型在一个样本大小下的结果。"""
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


def run_subsampling_experiment(
    data: dict[str, Any],
    best_params: dict[str, dict[str, Any]],
    sample_sizes: list[int],
) -> pd.DataFrame:
    """对所有模型和所有训练样本大小运行 subsampling 实验。"""
    rows = []

    for train_size in sample_sizes:
        X_subset, y_subset = sample_training_subset(
            data["X_train"],
            data["y_train"],
            train_size,
        )

        for model_name in MODELS:
            print(f"Training {model_name} with n={train_size}...")
            metrics = run_one_setting(
                model_name=model_name,
                params=best_params[model_name],
                X_subset=X_subset,
                y_subset=y_subset,
                X_test=data["X_test"],
                y_test=data["y_test"],
            )

            rows.append({
                "dataset": DATASET_NAME,
                "model": model_name,
                "train_size": train_size,
                "n_test": len(data["X_test"]),
                "n_features": data["X_train"].shape[1],
                "rmse": metrics["rmse"],
                "train_time_sec": metrics["train_time_sec"],
                "inference_time_per_sample_ms": metrics["inference_time_per_sample_ms"],
                "params": json.dumps(best_params[model_name], sort_keys=True),
            })

            print(
                f"  RMSE={metrics['rmse']:.6f}, "
                f"train_time={metrics['train_time_sec']:.3f}s"
            )

    return pd.DataFrame(rows)


def plot_metric(
    result_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """画出某个指标随训练样本数变化的折线图。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for model_name in MODELS:
        model_df = result_df[result_df["model"] == model_name].sort_values("train_size")
        ax.plot(
            model_df["train_size"],
            model_df[metric],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_xlabel("Training sample size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Appliances Energy: {ylabel} vs Training Size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """脚本入口：运行 subsampling 实验并保存表格和图。"""
    args = parse_args()
    args.processed_dir = resolve_path(args.processed_dir)
    args.baseline_results = resolve_path(args.baseline_results)
    args.output_table = resolve_path(args.output_table)
    args.figures_dir = resolve_path(args.figures_dir)

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_appliances_split(args.processed_dir)
    best_params = load_best_params(args.baseline_results)
    sample_sizes = make_sample_sizes(args.sample_sizes, full_train_size=len(data["X_train"]))

    print("Subsampling train sizes:", sample_sizes)
    result_df = run_subsampling_experiment(data, best_params, sample_sizes)
    result_df.to_csv(args.output_table, index=False)
    print(f"Saved subsampling results to: {args.output_table}")

    rmse_path = args.figures_dir / "appliances_subsampling_rmse.png"
    time_path = args.figures_dir / "appliances_subsampling_train_time.png"

    plot_metric(result_df, "rmse", "Test RMSE", rmse_path)
    plot_metric(result_df, "train_time_sec", "Training Time (seconds)", time_path)

    print(f"Saved RMSE figure to: {rmse_path}")
    print(f"Saved training-time figure to: {time_path}")


if __name__ == "__main__":
    main()
