"""
xRFM main training script.

依据 xRFM 官方最新 API 编写：
1. 统一用 from xrfm import xRFM。
2. 使用 np.float32 格式传入数据。
3. fit() 时强制传入 validation 数据以支持 early stopping。
4. 输出的 CSV 完全兼容 experiments/results/merge_all_model_results.py。

运行方式：
    python experiments/xrfm/train_xrfm.py
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
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xrfm import xRFM

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "wine": "classification",
    "divorce": "classification",
    "german_credit": "classification",
    "bike_sharing": "regression",
    "appliances_energy": "regression",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xRFM model.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tables/xrfm_results.csv")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    return parser.parse_args()

def load_split(processed_dir: Path, dataset_name: str) -> dict[str, Any]:
    X_train = pd.read_csv(processed_dir / f"{dataset_name}_X_train.csv").values.astype(np.float32)
    X_val = pd.read_csv(processed_dir / f"{dataset_name}_X_val.csv").values.astype(np.float32)
    X_test = pd.read_csv(processed_dir / f"{dataset_name}_X_test.csv").values.astype(np.float32)
    y_train = pd.read_csv(processed_dir / f"{dataset_name}_y_train.csv")["target"].to_numpy().astype(np.float32)
    y_val = pd.read_csv(processed_dir / f"{dataset_name}_y_val.csv")["target"].to_numpy().astype(np.float32)
    y_test = pd.read_csv(processed_dir / f"{dataset_name}_y_test.csv")["target"].to_numpy().astype(np.float32)

    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val, "y_test": y_test}

def parameter_grid() -> list[dict[str, Any]]:
    """构建官方建议的 rfm_params 字典网格"""
    bandwidths = [5.0, 10.0]
    iters_list = [3, 5]
    
    grid = []
    for bw in bandwidths:
        for it in iters_list:
            grid.append({
                'model': {'kernel': 'l2', 'bandwidth': bw, 'exponent': 1.0, 'diag': False, 'bandwidth_mode': 'constant'},
                'fit': {'reg': 1e-3, 'iters': it, 'verbose': False, 'early_stop_rfm': True}
            })
    return grid

def make_model(task_type: str, rfm_params: dict[str, Any]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tuning_metric = 'mse' if task_type == 'regression' else 'accuracy'
    
    return xRFM(
        rfm_params=rfm_params,
        device=device,
        tuning_metric=tuning_metric,
        # 如果你想限制叶子节点大小，可以取消下面这行的注释
        # min_subset_size=500 
    )

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_and_evaluate(dataset_name: str, processed_dir: Path) -> dict[str, Any]:
    data = load_split(processed_dir, dataset_name)
    task_type = DATASETS[dataset_name]
    
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    label_encoder, n_classes = None, None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        # 将分类标签编码为 0, 1, 2... 并转回 float32，因为 xRFM 需要 float 格式
        y_train = label_encoder.fit_transform(y_train).astype(np.float32)
        y_val = label_encoder.transform(y_val).astype(np.float32)
        y_test = label_encoder.transform(y_test).astype(np.float32)
        n_classes = len(label_encoder.classes_)

    print(f"\n===== Training xRFM on {dataset_name} ({task_type}) =====")
    
    # --- 1. 网格调参 ---
    best_score = -np.inf if task_type == "classification" else np.inf
    best_params = None
    
    for params in parameter_grid():
        model = make_model(task_type, params)
        # xRFM API 要求必须传入验证集！
        model.fit(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_val)
        
        if task_type == "regression":
            score = rmse(y_val, y_pred)
            if score < best_score: best_score, best_params = score, params
        else:
            score = accuracy_score(y_val, y_pred)
            if score > best_score: best_score, best_params = score, params

    # --- 2. 最终训练评估 ---
    # 注意：既然 xRFM 的 fit 需要 validation 集合做 early stopping，
    # 我们就不应该把 train 和 val 合并，而是保持使用 X_train 和 X_val 再次拟合一次最终模型。
    final_model = make_model(task_type, best_params)
    
    train_start = time.perf_counter()
    final_model.fit(X_train, y_train, X_val, y_val)
    train_time_sec = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_pred = final_model.predict(X_test)
    inference_time_sec = time.perf_counter() - predict_start

    # 生成兼容同学脚本的格式
    result = {
        "dataset": dataset_name, "task_type": task_type, "model": "xRFM",
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "n_features": X_train.shape[1], "best_validation_score": best_score,
        "best_params": json.dumps(best_params, sort_keys=True),
        "train_time_sec": train_time_sec, 
        "inference_time_per_sample_ms": inference_time_sec / len(X_test) * 1000,
        "rmse": np.nan, "accuracy": np.nan, "auc_roc": np.nan,
    }

    if task_type == "regression":
        result["rmse"] = rmse(y_test, y_pred)
        print(f"Test RMSE: {result['rmse']:.6f}")
    else:
        result["accuracy"] = float(accuracy_score(y_test, np.round(y_pred)))
        
        try:
            y_proba = final_model.predict_proba(X_test)
            if n_classes <= 2: 
                if len(y_proba.shape) == 1:
                    result["auc_roc"] = float(roc_auc_score(y_test, y_proba))
                else:
                    result["auc_roc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
            else: 
                result["auc_roc"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        except (AttributeError, ValueError, IndexError) as e:
            # 听取 Review 建议：不支持真实概率就直接记为 NaN，绝不静默造假
            print(f"  [Warning] xRFM 无法输出标准概率 ({e})，AUC-ROC 记录为 NaN。")
            result["auc_roc"] = np.nan
        
        print(f"Test Accuracy: {result['accuracy']:.6f}, Test AUC-ROC: {result['auc_roc']:.6f}")

    return result

def main():
    args = parse_args()
    if not args.processed_dir.is_absolute(): args.processed_dir = PROJECT_ROOT / args.processed_dir
    if not args.output.is_absolute(): args.output = PROJECT_ROOT / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for dataset_name in args.datasets:
        res = train_and_evaluate(dataset_name, args.processed_dir)
        results.append(res)

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"\nSaved xRFM results to: {args.output}")

if __name__ == "__main__":
    main()
