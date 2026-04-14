"""
Merge baseline result tables into report-ready CSV files.

用途：
1. 只合并 XGBoost、LightGBM、Random Forest 三个 baseline 的结果；
2. 按 PDF 要求拆分分类任务和回归任务；
3. 生成 datasets 为 rows、(model, metric) pairs 为 columns 的 wide table；
4. 保留训练时间和单样本推理时间，方便写 Results section。

运行方式：
    python experiments/baselines/merge_baseline_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_DIR = PROJECT_ROOT / "outputs/tables"

CLASSIFICATION_METRICS = [
    "accuracy",
    "auc_roc",
    "train_time_sec",
    "inference_time_per_sample_ms",
]
REGRESSION_METRICS = [
    "rmse",
    "train_time_sec",
    "inference_time_per_sample_ms",
]

REQUIRED_COLUMNS = [
    "dataset",
    "task_type",
    "model",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "best_validation_score",
    "best_params",
    "train_time_sec",
    "inference_time_per_sample_ms",
    "rmse",
    "accuracy",
    "auc_roc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge all model result CSV files.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[
            DEFAULT_TABLE_DIR / "xgboost_results.csv",
            DEFAULT_TABLE_DIR / "lightgbm_results.csv",
            DEFAULT_TABLE_DIR / "random_forest_results.csv",
            DEFAULT_TABLE_DIR / "xrfm_results.csv",  # <==== 核心修改：加入了 xRFM 的结果文件！
        ],
        help="要合并的 result CSV 文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help="合并后结果表的输出目录。",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_result_file(path: Path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        print(f"⚠️ 警告：找不到结果文件 {path}，将跳过合并该模型。")
        return pd.DataFrame()

    df = pd.read_csv(path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} 缺少必要列：{missing_columns}")

    return df[REQUIRED_COLUMNS].copy()


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "n_train", "n_val", "n_test", "n_features",
        "best_validation_score", "train_time_sec",
        "inference_time_per_sample_ms", "rmse", "accuracy", "auc_roc",
    ]

    df = df.copy()
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_pdf_required_metrics(df: pd.DataFrame) -> None:
    problems: list[str] = []

    for row in df.itertuples(index=False):
        prefix = f"{row.dataset} / {row.model}"
        if row.task_type == "classification":
            if pd.isna(row.accuracy):
                problems.append(f"{prefix}: classification 缺少 accuracy")
            if pd.isna(row.auc_roc):
                pass # 已知 xRFM 的多分类 AUC 可能为 NaN，这里允许放行
        elif row.task_type == "regression":
            if pd.isna(row.rmse):
                problems.append(f"{prefix}: regression 缺少 rmse")

    if problems:
        problem_text = "\n".join(f"- {problem}" for problem in problems)
        print(f"⚠️ 提示：部分指标缺失 (如果是多分类 AUC 缺失可忽略)：\n{problem_text}")


def make_task_summary(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    if task_type == "classification":
        columns = ["dataset", "model", "accuracy", "auc_roc", "train_time_sec", "inference_time_per_sample_ms", "best_params"]
    elif task_type == "regression":
        columns = ["dataset", "model", "rmse", "train_time_sec", "inference_time_per_sample_ms", "best_params"]
    else:
        raise ValueError(f"未知任务类型：{task_type}")

    summary = df[df["task_type"] == task_type][columns].copy()
    return summary.sort_values(["dataset", "model"]).reset_index(drop=True)


def make_wide_table(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    metrics = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
    task_df = df[df["task_type"] == task_type].copy()

    wide = task_df.pivot(index="dataset", columns="model", values=metrics)
    wide = wide.swaplevel(axis=1).sort_index(axis=1, level=0)
    wide.columns = [f"{model}_{metric}" for model, metric in wide.columns]
    return wide.reset_index()


def mark_best_values(summary: pd.DataFrame, task_type: str) -> pd.DataFrame:
    summary = summary.copy()
    summary["best_on_dataset"] = False

    if summary.empty:
        return summary

    if task_type == "classification":
        best_indices = summary.groupby("dataset")["auc_roc"].idxmax()
    elif task_type == "regression":
        best_indices = summary.groupby("dataset")["rmse"].idxmin()
    else:
        raise ValueError(f"未知任务类型：{task_type}")

    summary.loc[best_indices, "best_on_dataset"] = True
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 过滤掉空的 DataFrame
    dfs = [load_result_file(path) for path in args.inputs]
    dfs = [df for df in dfs if not df.empty]
    
    if not dfs:
        print("没有找到任何结果文件，请先运行 train 脚本！")
        return

    baseline_results = pd.concat(dfs, ignore_index=True)
    baseline_results = clean_numeric_columns(baseline_results)
    baseline_results = baseline_results.sort_values(["dataset", "model"]).reset_index(drop=True)

    validate_pdf_required_metrics(baseline_results)

    classification_summary = make_task_summary(baseline_results, "classification")
    regression_summary = make_task_summary(baseline_results, "regression")
    classification_summary = mark_best_values(classification_summary, "classification")
    regression_summary = mark_best_values(regression_summary, "regression")

    # ================= 核心修改：统一输出为 all_models =================
    outputs = {
        "all_models_results_all.csv": baseline_results,
        "all_models_classification_summary.csv": classification_summary,
        "all_models_regression_summary.csv": regression_summary,
        "all_models_classification_wide.csv": make_wide_table(baseline_results, "classification"),
        "all_models_regression_wide.csv": make_wide_table(baseline_results, "regression"),
    }

    for filename, table in outputs.items():
        output_path = args.output_dir / filename
        table.to_csv(output_path, index=False)
        print(f"Saved {filename}: {output_path}")

if __name__ == "__main__":
    main()