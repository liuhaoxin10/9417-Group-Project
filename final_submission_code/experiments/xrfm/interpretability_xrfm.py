"""
Interpretability comparison script for xRFM.

Run:
    python experiments/xrfm/interpretability_xrfm.py --dataset wine
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error

try:
    from xrfm import xRFM
except ImportError as exc:
    raise ImportError("Missing xrfm dependency. Run: pip install xrfm") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42

DATASETS = {
    "wine": "classification",
    "divorce": "classification",
    "german_credit": "classification",
    "bike_sharing": "regression",
    "appliances_energy": "regression",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interpretability comparison for xRFM.")
    parser.add_argument("--dataset", type=str, default="wine", help="Dataset name for interpretability analysis.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/tables")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/figures")
    return parser.parse_args()

def resolve_path(path: Path) -> Path:
    if path.is_absolute(): return path
    return PROJECT_ROOT / path

def load_data(processed_dir: Path, dataset_name: str) -> dict[str, Any]:
    X_train = pd.read_csv(processed_dir / f"{dataset_name}_X_train.csv").values.astype(np.float32)
    X_val = pd.read_csv(processed_dir / f"{dataset_name}_X_val.csv").values.astype(np.float32)
    X_test = pd.read_csv(processed_dir / f"{dataset_name}_X_test.csv").values.astype(np.float32)
    
    y_train = pd.read_csv(processed_dir / f"{dataset_name}_y_train.csv")["target"].to_numpy().astype(np.float32)
    y_val = pd.read_csv(processed_dir / f"{dataset_name}_y_val.csv")["target"].to_numpy().astype(np.float32)
    y_test = pd.read_csv(processed_dir / f"{dataset_name}_y_test.csv")["target"].to_numpy().astype(np.float32)
    
    feature_names = pd.read_csv(processed_dir / f"{dataset_name}_feature_names.csv")["feature_name"].tolist()

    return {"X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "feature_names": feature_names}

def get_best_params(dataset_name: str) -> dict[str, Any]:
    return {
        'model': {'kernel': 'l2', 'bandwidth': 5.0, 'exponent': 1.0, 'diag': False, 'bandwidth_mode': 'constant'},
        'fit': {'reg': 1e-3, 'iters': 5, 'verbose': False, 'early_stop_rfm': True}
    }

def custom_scorer(estimator, X, y):
    metric = getattr(estimator, 'tuning_metric', 'accuracy')
    y_pred = estimator.predict(X.astype(np.float32))
    if metric == 'mse':
        return -mean_squared_error(y, y_pred)
    else:
        return accuracy_score(y, np.round(y_pred))

def find_agop_importance(model, n_features):
    """Extract AGOP-style feature importances from nested xRFM internals."""
    import torch
    import numpy as np
    
    def extract_diag(val):
        if hasattr(val, 'detach'): 
            val = val.detach().cpu().numpy()
        
        if isinstance(val, np.ndarray):
            if val.ndim == 2 and val.shape[0] == val.shape[1] and val.shape[0] >= n_features:
                return np.diag(val)[:n_features]
            if val.ndim == 1 and val.shape[0] >= n_features:
                return val[:n_features]
        return None

    def extract_from_nested(obj):
        """Recursively unpack lists and dictionaries looking for matrices."""
        diags = []
        if isinstance(obj, (list, tuple)):
            for item in obj:
                diags.extend(extract_from_nested(item))
        elif isinstance(obj, dict):
            for item in obj.values():
                diags.extend(extract_from_nested(item))
        else:
            d = extract_diag(obj)
            if d is not None:
                diags.append(d)
        return diags

    diags = []
    
    try:
        if hasattr(model, 'collect_Ms'):
            val = model.collect_Ms() if callable(model.collect_Ms) else model.collect_Ms
            diags.extend(extract_from_nested(val))
    except Exception as e:
        print(f"  [Warning] collect_Ms failed: {e}")

    if not diags:
        try:
            if hasattr(model, 'collect_best_agops'):
                val = model.collect_best_agops() if callable(model.collect_best_agops) else model.collect_best_agops
                diags.extend(extract_from_nested(val))
        except Exception as e:
            pass

    if not diags:
        for attr in ['models', 'trees']:
            if hasattr(model, attr):
                diags.extend(extract_from_nested(getattr(model, attr)))
                
    if diags:
        print(f"  [Info] Collected {len(diags)} local feature-importance matrices and averaged them.")
        return np.mean(diags, axis=0)

    return None

def main() -> None:
    args = parse_args()
    args.processed_dir = resolve_path(args.processed_dir)
    args.output_dir = resolve_path(args.output_dir)
    args.figures_dir = resolve_path(args.figures_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    task_type = DATASETS[args.dataset]
    print(f"\n===== Running Interpretability Analysis on {args.dataset} ({task_type}) =====")

    data = load_data(args.processed_dir, args.dataset)
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    feature_names = data["feature_names"]
    n_features = X_train.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rfm_params = get_best_params(args.dataset)
    tuning_metric = 'mse' if task_type == 'regression' else 'accuracy'
    
    print("Training xRFM model...")
    model = xRFM(rfm_params=rfm_params, device=device, tuning_metric=tuning_metric)
    model.fit(X_train, y_train, X_val, y_val)
    
    print("Computing AGOP Importances...")
    agop_importance = find_agop_importance(model, n_features)
    
    if agop_importance is None:
        print("\n" + "!"*60)
        print("[Debug] AGOP extraction failed: no dimension-compatible feature matrix was found.")
        print("Available public model attributes:")
        print("1. model attributes: ", [a for a in dir(model) if not a.startswith('_')])
        if hasattr(model, 'model'):
            print("2. model.model attributes: ", [a for a in dir(model.model) if not a.startswith('_')])
        print("!"*60 + "\n")
        raise ValueError("Could not extract an AGOP matrix from the fitted xRFM model.")

    print("Computing PCA Loadings...")
    pca = PCA(n_components=1)
    pca.fit(X_train)
    pca_loadings = np.abs(pca.components_[0])

    print("Computing Mutual Information...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE) if task_type == "classification" else mutual_info_regression(X_train, y_train, random_state=RANDOM_STATE)

    print("Computing Permutation Importance...")
    perm_result = permutation_importance(model, X_test, y_test, scoring=custom_scorer, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    perm_scores = perm_result.importances_mean

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "xRFM_AGOP": agop_importance,
        "PCA_Loadings": pca_loadings,
        "Mutual_Info": mi_scores,
        "Permutation": perm_scores
    })

    for col in ["xRFM_AGOP", "PCA_Loadings", "Mutual_Info", "Permutation"]:
        max_val = df_imp[col].max()
        if max_val > 0: df_imp[col] = df_imp[col] / max_val

    df_imp = df_imp.sort_values(by="xRFM_AGOP", ascending=False).reset_index(drop=True)
    out_csv = args.output_dir / f"{args.dataset}_interpretability_comparison.csv"
    df_imp.to_csv(out_csv, index=False)
    print(f"\nInterpretability results saved to: {out_csv}")

    print("Generating comparison plot...")
    top_n = min(15, len(feature_names))
    plot_df = df_imp.head(top_n)
    x = np.arange(top_n)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, plot_df["xRFM_AGOP"], width, label='xRFM AGOP', color='crimson')
    ax.bar(x - 0.5*width, plot_df["Mutual_Info"], width, label='Mutual Info', color='steelblue')
    ax.bar(x + 0.5*width, plot_df["Permutation"], width, label='Permutation', color='forestgreen')
    ax.bar(x + 1.5*width, plot_df["PCA_Loadings"], width, label='PCA Loadings', color='goldenrod')
    
    ax.set_ylabel('Normalized Importance Score')
    ax.set_title(f'Feature Importance Comparison ({args.dataset.capitalize()} Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels([lbl[:15] + ".." if len(lbl) > 15 else lbl for lbl in plot_df["Feature"]], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    out_fig = args.figures_dir / f"{args.dataset}_interpretability_comparison.png"
    fig.savefig(out_fig, dpi=300)
    print("\nDone. The comparison figure has been generated.")

if __name__ == "__main__":
    main()
