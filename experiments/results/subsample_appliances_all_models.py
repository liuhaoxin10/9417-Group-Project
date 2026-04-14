from __future__ import annotations
"""
Subsampling experiment on Appliances Energy for all models.

运行方式：
    python experiments/results/subsample_appliances_all_models.py

输出：
    outputs/tables/appliances_subsampling_all.csv
    outputs/figures/appliances_subsampling_rmse_all.png
    outputs/figures/appliances_subsampling_train_time_all.png
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
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor
except ImportError:
    pass

try:
    from xrfm import xRFM
except ImportError as exc:
    raise ImportError("缺少 xrfm 依赖。请运行: pip install xrfm") from exc

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "appliances_energy"
MODELS = ["XGBoost", "LightGBM", "Random Forest", "xRFM"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Appliances Energy subsampling experiment.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--baseline-results", type=Path, default=PROJECT_ROOT / "outputs/tables/baseline_results_all.csv")
    parser.add_argument("--xrfm-results", type=Path, default=PROJECT_ROOT / "outputs/tables/xrfm_results.csv")
    parser.add_argument("--output-table", type=Path, default=PROJECT_ROOT / "outputs/tables/appliances_subsampling_all.csv")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/figures")
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[1000, 3000, 6000, 10000])
    return parser.parse_args()

def resolve_path(path: Path) -> Path:
    if path.is_absolute(): return path
    return PROJECT_ROOT / path

def load_appliances_split(processed_dir: Path) -> dict[str, Any]:
    X_train = pd.read_csv(processed_dir / f"{DATASET_NAME}_X_train.csv")
    X_val = pd.read_csv(processed_dir / f"{DATASET_NAME}_X_val.csv")
    X_test = pd.read_csv(processed_dir / f"{DATASET_NAME}_X_test.csv")
    y_train = pd.read_csv(processed_dir / f"{DATASET_NAME}_y_train.csv")["target"].to_numpy()
    y_val = pd.read_csv(processed_dir / f"{DATASET_NAME}_y_val.csv")["target"].to_numpy()
    y_test = pd.read_csv(processed_dir / f"{DATASET_NAME}_y_test.csv")["target"].to_numpy()

    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, 
            "y_train": y_train, "y_val": y_val, "y_test": y_test}

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def load_best_params(baseline_path: Path, xrfm_path: Path) -> dict[str, Any]:
    params = {}
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)
        ds_base = df_base[df_base["dataset"] == DATASET_NAME]
        for m in ["XGBoost", "LightGBM", "Random Forest"]:
            row = ds_base[ds_base["model"] == m]
            if not row.empty: params[m] = json.loads(row.iloc[0]["best_params"])
            else: params[m] = {}
    else:
        for m in ["XGBoost", "LightGBM", "Random Forest"]: params[m] = {}

    if xrfm_path.exists():
        df_xrfm = pd.read_csv(xrfm_path)
        row = df_xrfm[(df_xrfm["dataset"] == DATASET_NAME) & (df_xrfm["model"] == "xRFM")]
        if not row.empty: params["xRFM"] = json.loads(row.iloc[0]["best_params"])
        else: params["xRFM"] = {}
    else:
        params["xRFM"] = {
            'model': {'kernel': 'l2', 'bandwidth': 5.0, 'exponent': 1.0, 'diag': False, 'bandwidth_mode': 'constant'},
            'fit': {'reg': 1e-3, 'iters': 3, 'verbose': False, 'early_stop_rfm': True}
        }
    return params

def make_model(model_name: str, params: dict[str, Any]):
    import torch
    if model_name == "XGBoost":
        return XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "LightGBM":
        return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)
    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "xRFM":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rfm_params = params if 'model' in params else {
            'model': {'kernel': 'l2', 'bandwidth': 5.0, 'exponent': 1.0, 'diag': False, 'bandwidth_mode': 'constant'},
            'fit': {'reg': 1e-3, 'iters': 3, 'verbose': False, 'early_stop_rfm': True}
        }
        return xRFM(rfm_params=rfm_params, device=device, tuning_metric='mse')
    raise ValueError(f"未知模型：{model_name}")

def sample_training_subset(X_train: pd.DataFrame, y_train: np.ndarray, train_size: int):
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(X_train), size=train_size, replace=False)
    return X_train.iloc[indices], y_train[indices]

def run_one_setting(model_name: str, params: dict[str, Any], X_subset: pd.DataFrame, y_subset: np.ndarray, 
                    X_val: pd.DataFrame, y_val: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    model = make_model(model_name, params)
    train_start = time.perf_counter()
    
    if model_name == "xRFM":
        model.fit(
            X_subset.values.astype(np.float32), y_subset.astype(np.float32),
            X_val.values.astype(np.float32), y_val.astype(np.float32)
        )
        train_time_sec = time.perf_counter() - train_start
        predict_start = time.perf_counter()
        y_pred = model.predict(X_test.values.astype(np.float32))
    else:
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

def main() -> None:
    args = parse_args()
    args.processed_dir = resolve_path(args.processed_dir)
    args.baseline_results = resolve_path(args.baseline_results)
    args.xrfm_results = resolve_path(args.xrfm_results)
    args.output_table = resolve_path(args.output_table)
    args.figures_dir = resolve_path(args.figures_dir)

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_appliances_split(args.processed_dir)
    best_params = load_best_params(args.baseline_results, args.xrfm_results)
    
    sample_sizes = [s for s in args.sample_sizes if 0 < s <= len(data["X_train"])]
    sample_sizes.append(len(data["X_train"]))
    sample_sizes = sorted(set(sample_sizes))

    rows = []
    for train_size in sample_sizes:
        X_subset, y_subset = sample_training_subset(data["X_train"], data["y_train"], train_size)
        for model_name in MODELS:
            print(f"Training {model_name} with n={train_size}...")
            metrics = run_one_setting(
                model_name, best_params[model_name], X_subset, y_subset, 
                data["X_val"], data["y_val"], data["X_test"], data["y_test"]
            )
            rows.append({
                "dataset": DATASET_NAME, "model": model_name, "train_size": train_size,
                "rmse": metrics["rmse"], "train_time_sec": metrics["train_time_sec"],
                "inference_time_per_sample_ms": metrics["inference_time_per_sample_ms"]
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output_table, index=False)
    print(f"Saved subsampling results to: {args.output_table}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in MODELS:
        m_df = result_df[result_df["model"] == m]
        ax.plot(m_df["train_size"], m_df["rmse"], marker="o", label=m)
    ax.legend()
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Test RMSE vs Training Size")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "appliances_subsampling_rmse_all.png", dpi=200)

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
    for m in MODELS:
        m_df = result_df[result_df["model"] == m]
        ax2.plot(m_df["train_size"], m_df["train_time_sec"], marker="s", label=m)
    ax2.legend()
    ax2.set_xlabel("Number of Training Samples")
    ax2.set_ylabel("Training Time (seconds)")
    ax2.set_title("Training Time vs Training Size")
    fig2.tight_layout()
    fig2.savefig(args.figures_dir / "appliances_subsampling_train_time_all.png", dpi=200)
if __name__ == "__main__":
    main()
