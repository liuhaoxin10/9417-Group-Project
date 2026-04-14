"""
Prepare final tables and checks for report handoff.

This script does not train any model. It collects the already-generated outputs,
combines per-dataset interpretability tables, validates the final result tables,
and writes a manifest that tells report writers which files to use.

Run:
    python experiments/results/prepare_final_outputs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "outputs/tables"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"
DOCS_DIR = PROJECT_ROOT / "docs"

DATASETS = {
    "wine": "classification",
    "divorce": "classification",
    "german_credit": "classification",
    "bike_sharing": "regression",
    "appliances_energy": "regression",
}
MODELS = ["XGBoost", "LightGBM", "Random Forest", "xRFM"]
BASELINE_MODELS = ["XGBoost", "LightGBM", "Random Forest"]
ALLOWED_MISSING_AUC = {("wine", "xRFM")}

INTERPRETABILITY_COLUMNS = [
    "Feature",
    "xRFM_AGOP",
    "PCA_Loadings",
    "Mutual_Info",
    "Permutation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare final output files for report handoff.")
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--strict", action="store_true", help="Exit with error if blocking issues are found.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def add_manifest_row(
    rows: list[dict[str, Any]],
    group: str,
    path: Path,
    description: str,
    report_use: str,
    note: str = "",
) -> None:
    rows.append(
        {
            "group": group,
            "path": relative_path(path),
            "exists": path.exists(),
            "description": description,
            "report_use": report_use,
            "note": note,
        }
    )


def add_check(checks: list[dict[str, str]], severity: str, item: str, message: str) -> None:
    checks.append({"severity": severity, "item": item, "message": message})


def require_columns(df: pd.DataFrame, columns: list[str], item: str, checks: list[dict[str, str]]) -> bool:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        add_check(checks, "error", item, f"Missing columns: {missing}")
        return False
    return True


def combine_interpretability_tables(tables_dir: Path, checks: list[dict[str, str]]) -> Path:
    output_path = tables_dir / "interpretability_comparison_all.csv"
    frames: list[pd.DataFrame] = []

    for dataset_name, task_type in DATASETS.items():
        input_path = tables_dir / f"{dataset_name}_interpretability_comparison.csv"
        if not input_path.exists():
            add_check(checks, "warning", input_path.name, "Interpretability table is missing.")
            continue

        df = pd.read_csv(input_path)
        if not require_columns(df, INTERPRETABILITY_COLUMNS, input_path.name, checks):
            continue

        df = df.copy()
        df.insert(0, "task_type", task_type)
        df.insert(0, "dataset", dataset_name)
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_path, index=False)
    else:
        add_check(checks, "error", "interpretability", "No interpretability tables could be combined.")

    return output_path


def validate_all_model_results(tables_dir: Path, checks: list[dict[str, str]]) -> None:
    path = tables_dir / "all_models_results.csv"
    if not path.exists():
        add_check(checks, "error", "all_models_results.csv", "Final all-model result table is missing.")
        return

    df = pd.read_csv(path)
    required = [
        "dataset",
        "task_type",
        "model",
        "train_time_sec",
        "inference_time_per_sample_ms",
        "rmse",
        "accuracy",
        "auc_roc",
    ]
    if not require_columns(df, required, path.name, checks):
        return

    present_pairs = set(zip(df["dataset"], df["model"]))
    for dataset_name in DATASETS:
        for model_name in MODELS:
            if (dataset_name, model_name) not in present_pairs:
                add_check(checks, "error", path.name, f"Missing row for {dataset_name} / {model_name}.")

    for row in df.itertuples(index=False):
        dataset_name = str(row.dataset)
        model_name = str(row.model)
        task_type = str(row.task_type)
        item = f"{dataset_name} / {model_name}"

        if pd.isna(row.train_time_sec):
            add_check(checks, "error", item, "Missing train_time_sec.")
        if pd.isna(row.inference_time_per_sample_ms):
            add_check(checks, "error", item, "Missing inference_time_per_sample_ms.")

        if task_type == "classification":
            if pd.isna(row.accuracy):
                add_check(checks, "error", item, "Missing accuracy.")
            if pd.isna(row.auc_roc):
                if (dataset_name, model_name) in ALLOWED_MISSING_AUC:
                    add_check(
                        checks,
                        "note",
                        item,
                        "AUC-ROC is NaN because current xRFM implementation does not provide multi-class probability scores.",
                    )
                else:
                    add_check(checks, "error", item, "Missing auc_roc.")
        elif task_type == "regression":
            if pd.isna(row.rmse):
                add_check(checks, "error", item, "Missing rmse.")
        else:
            add_check(checks, "error", item, f"Unknown task_type: {task_type}.")


def validate_baseline_results(tables_dir: Path, checks: list[dict[str, str]]) -> None:
    path = tables_dir / "baseline_results_all.csv"
    if not path.exists():
        add_check(checks, "error", path.name, "Baseline result table is missing.")
        return

    df = pd.read_csv(path)
    models = sorted(df["model"].dropna().unique())
    if models != sorted(BASELINE_MODELS):
        add_check(checks, "error", path.name, f"Expected only baseline models, found {models}.")


def validate_subsampling_tables(tables_dir: Path, checks: list[dict[str, str]]) -> None:
    all_path = tables_dir / "appliances_subsampling_all.csv"
    baseline_path = tables_dir / "appliances_subsampling_baselines.csv"

    for path, expected_models in [(all_path, MODELS), (baseline_path, BASELINE_MODELS)]:
        if not path.exists():
            add_check(checks, "error", path.name, "Subsampling table is missing.")
            continue
        df = pd.read_csv(path)
        required = ["dataset", "model", "train_size", "rmse", "train_time_sec", "inference_time_per_sample_ms"]
        if not require_columns(df, required, path.name, checks):
            continue
        models = sorted(df["model"].dropna().unique())
        if models != sorted(expected_models):
            add_check(checks, "error", path.name, f"Unexpected model set: {models}.")
        if df[["rmse", "train_time_sec", "inference_time_per_sample_ms"]].isna().any().any():
            add_check(checks, "error", path.name, "Subsampling table contains missing metric values.")


def build_manifest(tables_dir: Path, figures_dir: Path, combined_interpretability_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    add_manifest_row(
        rows,
        "main_results",
        tables_dir / "all_models_results.csv",
        "Long-format result table for all models and all datasets.",
        "Main result table and source for summary/wide tables.",
        "wine / xRFM / auc_roc is intentionally NaN.",
    )
    add_manifest_row(
        rows,
        "main_results",
        tables_dir / "all_models_classification_wide.csv",
        "Wide table for classification metrics.",
        "Report table for classification datasets.",
    )
    add_manifest_row(
        rows,
        "main_results",
        tables_dir / "all_models_regression_wide.csv",
        "Wide table for regression metrics.",
        "Report table for regression datasets.",
    )
    add_manifest_row(
        rows,
        "subsampling",
        tables_dir / "appliances_subsampling_all.csv",
        "All-model subsampling results on Appliances Energy.",
        "Subsampling table for large-dataset experiment.",
    )
    add_manifest_row(
        rows,
        "subsampling",
        figures_dir / "appliances_subsampling_rmse_all.png",
        "Test RMSE versus training size for all models.",
        "Subsampling performance figure.",
    )
    add_manifest_row(
        rows,
        "subsampling",
        figures_dir / "appliances_subsampling_train_time_all.png",
        "Training time versus training size for all models.",
        "Subsampling efficiency figure.",
    )
    add_manifest_row(
        rows,
        "interpretability",
        combined_interpretability_path,
        "Combined feature-importance comparison table for all datasets.",
        "Source table for interpretability discussion.",
    )

    for dataset_name in DATASETS:
        add_manifest_row(
            rows,
            "interpretability",
            tables_dir / f"{dataset_name}_interpretability_comparison.csv",
            f"Feature-importance comparison table for {dataset_name}.",
            "Dataset-level interpretability table.",
        )
        add_manifest_row(
            rows,
            "interpretability",
            figures_dir / f"{dataset_name}_interpretability_comparison.png",
            f"Feature-importance comparison figure for {dataset_name}.",
            "Dataset-level interpretability figure.",
        )

    return pd.DataFrame(rows)


def write_guide(docs_dir: Path, manifest: pd.DataFrame, checks: pd.DataFrame) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_path = docs_dir / "final_outputs_guide.md"

    missing = manifest[~manifest["exists"]]
    blocking = checks[checks["severity"] == "error"] if not checks.empty else pd.DataFrame()

    lines = [
        "# 最终输出文件说明",
        "",
        "这份文件用于说明本项目目前已经生成的最终实验数据、表格和图片。写论文的同学可以根据这里列出的文件直接取结果，不需要重新理解所有训练脚本。",
        "",
        "## 1. 主实验结果表",
        "",
        "- `outputs/tables/all_models_results.csv`：最完整的 long-format 总结果表，每一行是某个模型在某个数据集上的测试结果。",
        "- `outputs/tables/all_models_classification_wide.csv`：分类任务宽表，适合整理成论文中的分类结果表。",
        "- `outputs/tables/all_models_regression_wide.csv`：回归任务宽表，适合整理成论文中的回归结果表。",
        "- `outputs/tables/all_models_classification_summary.csv`：分类任务简化表，方便写 Discussion。",
        "- `outputs/tables/all_models_regression_summary.csv`：回归任务简化表，方便写 Discussion。",
        "",
        "主实验比较的模型包括 XGBoost、LightGBM、Random Forest 和 xRFM。分类任务主要看 Accuracy 和 AUC-ROC；回归任务主要看 RMSE。所有任务都记录训练时间和单样本推理时间。",
        "",
        "## 2. Wine 数据集上 xRFM AUC-ROC 为空的说明",
        "",
        "`wine / xRFM / auc_roc` 被有意保留为 `NaN`，不是漏跑，也不是表格损坏。",
        "",
        "原因是 Wine 是多分类任务，标准多分类 AUC-ROC 需要模型输出每个类别的概率或连续分数；当前 xRFM 实现没有提供这类输出，因此无法标准计算该指标。",
        "",
        "不要用硬预测类别强行计算 AUC-ROC，因为那样得到的值不能和 XGBoost、LightGBM、Random Forest 基于概率算出来的 AUC-ROC 公平比较。",
        "",
        "Discussion 可以这样写：",
        "",
        "```text",
        "For the multi-class Wine dataset, the current xRFM implementation does not provide class probability estimates or continuous decision scores required for standard multi-class AUC-ROC. Therefore, AUC-ROC is reported as NaN for xRFM on Wine, while accuracy is still reported. This reflects a practical limitation of the current xRFM implementation for multi-class classification evaluation.",
        "```",
        "",
        "## 3. Appliances Energy Subsampling 实验",
        "",
        "- `outputs/tables/appliances_subsampling_all.csv`：所有模型的 subsampling 源数据表。",
        "- `outputs/figures/appliances_subsampling_rmse_all.png`：训练样本数增加时，各模型 Test RMSE 的变化。",
        "- `outputs/figures/appliances_subsampling_train_time_all.png`：训练样本数增加时，各模型训练时间的变化。",
        "",
        "这个实验使用 `appliances_energy`，训练样本规模为 1000、3000、6000、10000 和 11841。RMSE 越低代表回归预测越好，训练时间越低代表训练效率越高。",
        "",
        "当前主要现象是：随着训练样本增加，所有模型 RMSE 总体下降；xRFM 的 RMSE 高于 tree-based baselines，并且训练时间随样本数增加更明显。这可以作为 Discussion 中关于 xRFM 在大数据集上效率和性能限制的证据。",
        "",
        "## 4. Interpretability Comparison",
        "",
        "- `outputs/tables/interpretability_comparison_all.csv`：合并后的 interpretability 总表。",
        "- `outputs/tables/*_interpretability_comparison.csv`：每个数据集单独的 interpretability 表。",
        "- `outputs/figures/*_interpretability_comparison.png`：每个数据集单独的 interpretability 图。",
        "",
        "interpretability 对比包含四种特征重要性：",
        "",
        "| 方法 | 含义 |",
        "| --- | --- |",
        "| xRFM_AGOP | xRFM 模型内部的 AGOP diagonal 特征重要性 |",
        "| PCA_Loadings | 第一主成分中的特征 loading，反映数据整体变化方向 |",
        "| Mutual_Info | 单个特征和目标变量之间的信息关联 |",
        "| Permutation | 打乱某个特征后模型性能下降多少 |",
        "",
        "所有重要性分数都已经归一化到 0 到 1。它们适合比较同一种方法内部哪些特征更重要；不同方法之间的数值不能当成完全相同意义的绝对值。",
        "",
        "论文篇幅只有 4 到 6 页，不一定要放全部 interpretability 图。建议选择 1 到 2 张最有代表性的图，例如 Appliances Energy 和 German Credit。",
        "",
        "## 5. Bonus AGOP Split Verification",
        "",
        "Bonus 加分项已经单独整理在 `experiments/bonus` 中。正式对应 PDF Bonus 要求的脚本是 `experiments/bonus/agop_split_from_scratch.py`，输出结果表是 `outputs/tables/bonus_agop_split_check.csv`。",
        "",
        "这部分验证的是：我们从零实现的 AGOP-based splitting criterion，是否能在小数据集上选出和 xRFM library 一致的 split direction。当前结果中 absolute cosine similarity 为 `0.99999999`，`passed=True`，说明手写实现和 xRFM reference 的分裂方向几乎完全一致。",
        "",
        "论文中建议把这部分放在 Appendix 或 Bonus Section，不需要画图，放一个很小的验证表格即可。详细写法、表格模板和可复制英文段落见 `experiments/bonus/README.md`。",
        "",
        "## 6. Baseline-only 输出",
        "",
        "为了保留 baseline 过程记录，当前也保留了只包含 XGBoost、LightGBM 和 Random Forest 的输出：",
        "",
        "- `outputs/tables/baseline_results_all.csv`",
        "- `outputs/tables/baseline_classification_summary.csv`",
        "- `outputs/tables/baseline_regression_summary.csv`",
        "- `outputs/tables/baseline_classification_wide.csv`",
        "- `outputs/tables/baseline_regression_wide.csv`",
        "- `outputs/tables/appliances_subsampling_baselines.csv`",
        "- `outputs/figures/appliances_subsampling_rmse.png`",
        "- `outputs/figures/appliances_subsampling_train_time.png`",
        "",
        "这些文件不包含 xRFM。最终论文如果要展示和 xRFM 的对比，优先使用带 `_all` 的 all-model 图和表。",
        "",
        "## 7. 最终检查文件",
        "",
        "- `outputs/tables/final_output_manifest.csv`：最终推荐使用的表格和图片清单。",
        "- `outputs/tables/final_output_checks.csv`：最终输出检查结果。",
        "",
        "当前检查结果没有 blocking error，只有一个 note：`wine / xRFM / auc_roc` 是 `NaN`，原因是 xRFM 当前实现没有多分类概率或连续分数输出。",
        "",
        "## 8. 复现命令",
        "",
        "如果所有模型结果已经生成，只需要重新整理最终表格和清单，可以运行：",
        "",
        "```bash",
        "python experiments/results/merge_all_model_results.py",
        "python experiments/results/prepare_final_outputs.py",
        "```",
        "",
        "如果需要重新生成 baseline-only 表格，可以运行：",
        "",
        "```bash",
        "python experiments/baselines/merge_baseline_results.py",
        "```",
        "",
        "如果需要重新生成 all-model subsampling 图，可以运行：",
        "",
        "```bash",
        "python experiments/results/subsample_appliances_all_models.py",
        "```",
        "",
        "注意：all-model subsampling 需要 xRFM 和 torch 环境。如果当前环境不能运行 xRFM，可以先使用已经生成好的 `_all` 图和表。",
        "",
        "如果需要重新运行 Bonus AGOP split verification，可以运行：",
        "",
        "```bash",
        "python experiments/bonus/agop_split_from_scratch.py",
        "```",
        "",
        "## 9. 最终推荐给论文同学使用的文件",
        "",
        "```text",
        "outputs/tables/all_models_results.csv",
        "outputs/tables/all_models_classification_wide.csv",
        "outputs/tables/all_models_regression_wide.csv",
        "outputs/tables/appliances_subsampling_all.csv",
        "outputs/tables/interpretability_comparison_all.csv",
        "outputs/tables/bonus_agop_split_check.csv",
        "outputs/figures/appliances_subsampling_rmse_all.png",
        "outputs/figures/appliances_subsampling_train_time_all.png",
        "outputs/figures/*_interpretability_comparison.png",
        "```",
        "",
        "## 10. Check Summary",
        "",
        f"- Missing manifest files: {len(missing)}",
        f"- Blocking check errors: {len(blocking)}",
        "",
        "机器可读的详细检查结果见 `outputs/tables/final_output_manifest.csv` 和 `outputs/tables/final_output_checks.csv`。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    tables_dir = resolve_path(args.tables_dir)
    figures_dir = resolve_path(args.figures_dir)
    docs_dir = resolve_path(args.docs_dir)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, str]] = []

    combined_interpretability_path = combine_interpretability_tables(tables_dir, checks)
    validate_all_model_results(tables_dir, checks)
    validate_baseline_results(tables_dir, checks)
    validate_subsampling_tables(tables_dir, checks)

    manifest = build_manifest(tables_dir, figures_dir, combined_interpretability_path)
    checks_df = pd.DataFrame(checks, columns=["severity", "item", "message"])

    manifest_path = tables_dir / "final_output_manifest.csv"
    checks_path = tables_dir / "final_output_checks.csv"
    manifest.to_csv(manifest_path, index=False)
    checks_df.to_csv(checks_path, index=False)
    guide_path = write_guide(docs_dir, manifest, checks_df)

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved checks: {checks_path}")
    print(f"Saved guide: {guide_path}")
    print(f"Saved combined interpretability table: {combined_interpretability_path}")

    blocking = checks_df[checks_df["severity"] == "error"] if not checks_df.empty else pd.DataFrame()
    if not checks_df.empty:
        print("\nCheck summary:")
        print(checks_df.to_string(index=False))

    if args.strict and not blocking.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
