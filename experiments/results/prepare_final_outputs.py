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
        "# Final Outputs Guide",
        "",
        "This file lists the final tables and figures prepared for the report-writing team.",
        "",
        "## Main Tables",
        "",
        "- `outputs/tables/all_models_results.csv`: long-format results for all models.",
        "- `outputs/tables/all_models_classification_wide.csv`: classification wide table.",
        "- `outputs/tables/all_models_regression_wide.csv`: regression wide table.",
        "",
        "## Subsampling",
        "",
        "- `outputs/tables/appliances_subsampling_all.csv`: all-model subsampling data.",
        "- `outputs/figures/appliances_subsampling_rmse_all.png`: RMSE versus training size.",
        "- `outputs/figures/appliances_subsampling_train_time_all.png`: training time versus training size.",
        "",
        "## Interpretability",
        "",
        "- `outputs/tables/interpretability_comparison_all.csv`: combined interpretability table.",
        "- `outputs/tables/*_interpretability_comparison.csv`: per-dataset interpretability tables.",
        "- `outputs/figures/*_interpretability_comparison.png`: per-dataset interpretability figures.",
        "",
        "## Notes",
        "",
        "- `wine / xRFM / auc_roc` is intentionally reported as `NaN` because the current xRFM implementation does not provide multi-class class probabilities or continuous scores required for standard multi-class AUC-ROC.",
        "- Do not replace the missing Wine xRFM AUC-ROC with an AUC computed from hard class labels, because that is not comparable with probability-based AUC-ROC from the baseline models.",
        "",
        "## Check Summary",
        "",
        f"- Missing manifest files: {len(missing)}",
        f"- Blocking check errors: {len(blocking)}",
        "",
        "See `outputs/tables/final_output_manifest.csv` and `outputs/tables/final_output_checks.csv` for machine-readable details.",
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
