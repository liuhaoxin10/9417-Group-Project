"""
Legacy helper: from-scratch standard AGOP split direction check.

This script corresponds to an older version of the project bonus requirement
and is kept as a reference implementation for standard AGOP.
For the updated bonus experiment, use:

    python experiments/bonus/residual_weighted_agop.py

This helper:
1. Implements the AGOP-based split criterion without calling xRFM internals.
2. Computes a split direction on a small processed dataset.
3. Trains xRFM through its public interface and reads the split direction from
   the public state_dict.
4. Checks whether the two directions match.

The from-scratch portion uses only the public math and standard libraries;
xRFM is used only as a reference check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from xrfm import xRFM


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BonusConfig:
    """Store the core settings for this AGOP split-direction check."""

    dataset: str
    n_samples: int
    n_val_samples: int
    seed: int
    bandwidth: float
    exponent: float
    reg: float
    max_leaf_size: int
    cosine_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a from-scratch AGOP split direction against xRFM."
    )
    parser.add_argument("--dataset", default="bike_sharing")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/tables/bonus_agop_split_check.csv")
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--n-val-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bandwidth", type=float, default=10.0)
    parser.add_argument("--exponent", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--max-leaf-size", type=int, default=80)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_small_split(processed_dir: Path, dataset: str, n_samples: int, n_val_samples: int) -> dict[str, Any]:
    """Load processed data and keep a small subset for the bonus check."""

    x_train_path = processed_dir / f"{dataset}_X_train.csv"
    y_train_path = processed_dir / f"{dataset}_y_train.csv"
    x_val_path = processed_dir / f"{dataset}_X_val.csv"
    y_val_path = processed_dir / f"{dataset}_y_val.csv"
    feature_path = processed_dir / f"{dataset}_feature_names.csv"

    X_train = pd.read_csv(x_train_path).to_numpy(dtype=np.float32)[:n_samples]
    y_train = pd.read_csv(y_train_path)["target"].to_numpy(dtype=np.float32)[:n_samples]
    X_val = pd.read_csv(x_val_path).to_numpy(dtype=np.float32)[:n_val_samples]
    y_val = pd.read_csv(y_val_path)["target"].to_numpy(dtype=np.float32)[:n_val_samples]

    if feature_path.exists():
        feature_names = pd.read_csv(feature_path)["feature_name"].tolist()
    else:
        feature_names = [f"feature_{idx}" for idx in range(X_train.shape[1])]

    if len(X_train) < n_samples:
        raise ValueError(f"{dataset} has fewer than {n_samples} training rows.")
    if len(X_val) < n_val_samples:
        raise ValueError(f"{dataset} has fewer than {n_val_samples} validation rows.")

    return {
        "X_train": X_train,
        "y_train": y_train.reshape(-1, 1),
        "X_val": X_val,
        "y_val": y_val.reshape(-1, 1),
        "feature_names": feature_names,
    }


def xrfm_subset_indices(n_samples: int, seed: int) -> np.ndarray:
    """
    Reproduce the random subset order used by xRFM for AGOP-on-subset.

    xRFM internally calls torch.randperm(len(X)) and then uses the first 95%
    for the AGOP training subset. This function reproduces that public sampling
    rule without calling private xRFM functions.
    """

    torch.manual_seed(seed)
    subset_indices = torch.randperm(n_samples).cpu().numpy()
    subset_train_size = max(int(n_samples * 0.95), 1)
    return subset_indices[:subset_train_size]


def l2_laplace_kernel(X: np.ndarray, Z: np.ndarray, bandwidth: float, exponent: float) -> np.ndarray:
    """Compute the l2 Laplace kernel exp(-||x-z||_2^p / bandwidth^p)."""

    distances = np.linalg.norm(X[:, None, :] - Z[None, :, :], axis=2)
    return np.exp(-(distances ** exponent) / (bandwidth ** exponent))


def fit_kernel_ridge_from_scratch(
    X: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    exponent: float,
    reg: float,
) -> np.ndarray:
    """Fit kernel ridge regression and return dual coefficients."""

    kernel_matrix = l2_laplace_kernel(X, X, bandwidth, exponent)
    kernel_matrix = kernel_matrix.copy()
    kernel_matrix[np.diag_indices_from(kernel_matrix)] += reg
    return np.linalg.solve(kernel_matrix, y)


def function_gradients_from_scratch(
    centers: np.ndarray,
    samples: np.ndarray,
    alpha: np.ndarray,
    bandwidth: float,
    exponent: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute gradients of the fitted kernel model with respect to samples.

    For f(z) = sum_i alpha_i k(x_i, z), the l2 Laplace kernel gradient is:
    grad_z f(z_j) = sum_i alpha_i * c_ij * (z_j - x_i)

    where c_ij = -p / bandwidth^p * k(x_i, z_j) * ||x_i-z_j||^(p-2).
    """

    diff = samples[None, :, :] - centers[:, None, :]
    distances = np.linalg.norm(diff, axis=2)
    kernel_matrix = np.exp(-(distances ** exponent) / (bandwidth ** exponent))

    mask = distances >= eps
    safe_distances = np.maximum(distances, eps)
    factors = kernel_matrix * (safe_distances ** (exponent - 2.0))
    factors *= mask
    factors *= -exponent / (bandwidth ** exponent)

    # alpha: (n_centers, n_outputs)
    # factors: (n_centers, n_samples)
    # diff: (n_centers, n_samples, n_features)
    return np.einsum("io,ij,ijd->ojd", alpha, factors, diff)


def agop_from_scratch(
    centers: np.ndarray,
    samples: np.ndarray,
    alpha: np.ndarray,
    bandwidth: float,
    exponent: float,
) -> np.ndarray:
    """Compute the AGOP matrix and normalize it using the xRFM convention."""

    gradients = function_gradients_from_scratch(
        centers=centers,
        samples=samples,
        alpha=alpha,
        bandwidth=bandwidth,
        exponent=exponent,
    )
    gradients = gradients.reshape(-1, gradients.shape[-1])
    agop = gradients.T @ gradients
    return agop / (np.max(agop) + 1e-30)


def top_split_direction_from_agop(agop: np.ndarray) -> np.ndarray:
    """Use the first right singular vector as the AGOP split direction."""

    _, _, vt = np.linalg.svd(agop, full_matrices=False)
    direction = vt[0]
    return direction / (np.linalg.norm(direction) + 1e-30)


def compute_from_scratch_direction(data: dict[str, Any], config: BonusConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run the full from-scratch AGOP split-direction calculation."""

    subset_indices = xrfm_subset_indices(len(data["X_train"]), config.seed)
    X_subset = data["X_train"][subset_indices].astype(np.float64)
    y_subset = data["y_train"][subset_indices].astype(np.float64)

    alpha = fit_kernel_ridge_from_scratch(
        X=X_subset,
        y=y_subset,
        bandwidth=config.bandwidth,
        exponent=config.exponent,
        reg=config.reg,
    )
    agop = agop_from_scratch(
        centers=X_subset,
        samples=X_subset,
        alpha=alpha,
        bandwidth=config.bandwidth,
        exponent=config.exponent,
    )
    return top_split_direction_from_agop(agop), agop


def make_reference_model(config: BonusConfig) -> xRFM:
    """Build the xRFM model used only for the reference check."""

    default_rfm_params = {
        "model": {
            "kernel": "l2_high_dim",
            "bandwidth": config.bandwidth,
            "exponent": config.exponent,
            "diag": False,
            "bandwidth_mode": "constant",
        },
        "fit": {
            "get_agop_best_model": True,
            "return_best_params": False,
            "reg": config.reg,
            "iters": 0,
            "early_stop_rfm": False,
            "verbose": False,
        },
    }

    return xRFM(
        default_rfm_params=default_rfm_params,
        rfm_params=default_rfm_params,
        max_leaf_size=config.max_leaf_size,
        number_of_splits=1,
        split_method="top_vector_agop_on_subset",
        tuning_metric="mse",
        n_trees=1,
        n_tree_iters=0,
        device="cpu",
        random_state=config.seed,
        use_temperature_tuning=False,
    )


def compute_xrfm_reference_direction(data: dict[str, Any], config: BonusConfig) -> np.ndarray:
    """
    Train xRFM through the public API and read the reference split direction.

    xRFM is used only for checking. The AGOP and direction computation above
    remains fully from scratch.
    """

    model = make_reference_model(config)
    model.fit(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
    )

    state_dict = model.get_state_dict()
    reference_tree = state_dict["param_trees"][0]
    if reference_tree["type"] != "node":
        raise RuntimeError("xRFM reference model did not create a split node. Increase n_samples or lower max_leaf_size.")

    direction = reference_tree["split_direction"].detach().cpu().numpy().astype(np.float64)
    return direction / (np.linalg.norm(direction) + 1e-30)


def sign_invariant_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Use absolute cosine similarity because split direction sign is arbitrary."""

    a_norm = a / (np.linalg.norm(a) + 1e-30)
    b_norm = b / (np.linalg.norm(b) + 1e-30)
    return float(abs(np.dot(a_norm, b_norm)))


def direction_summary(
    scratch_direction: np.ndarray,
    reference_direction: np.ndarray,
    feature_names: list[str],
    config: BonusConfig,
) -> pd.DataFrame:
    """Build a one-row summary for CSV output or appendix use."""

    cosine = sign_invariant_cosine(scratch_direction, reference_direction)
    scratch_top_idx = int(np.argmax(np.abs(scratch_direction)))
    reference_top_idx = int(np.argmax(np.abs(reference_direction)))

    return pd.DataFrame(
        [
            {
                "dataset": config.dataset,
                "n_samples": config.n_samples,
                "seed": config.seed,
                "bandwidth": config.bandwidth,
                "reg": config.reg,
                "cosine_similarity_abs": cosine,
                "cosine_threshold": config.cosine_threshold,
                "passed": cosine >= config.cosine_threshold,
                "scratch_top_feature_index": scratch_top_idx,
                "scratch_top_feature": feature_names[scratch_top_idx],
                "xrfm_top_feature_index": reference_top_idx,
                "xrfm_top_feature": feature_names[reference_top_idx],
            }
        ]
    )


def main() -> None:
    args = parse_args()
    processed_dir = resolve_path(args.processed_dir)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = BonusConfig(
        dataset=args.dataset,
        n_samples=args.n_samples,
        n_val_samples=args.n_val_samples,
        seed=args.seed,
        bandwidth=args.bandwidth,
        exponent=args.exponent,
        reg=args.reg,
        max_leaf_size=args.max_leaf_size,
        cosine_threshold=args.cosine_threshold,
    )

    data = load_small_split(
        processed_dir=processed_dir,
        dataset=config.dataset,
        n_samples=config.n_samples,
        n_val_samples=config.n_val_samples,
    )

    scratch_direction, scratch_agop = compute_from_scratch_direction(data, config)
    reference_direction = compute_xrfm_reference_direction(data, config)

    result = direction_summary(
        scratch_direction=scratch_direction,
        reference_direction=reference_direction,
        feature_names=data["feature_names"],
        config=config,
    )
    result.to_csv(output, index=False)

    row = result.iloc[0]
    print("\n================ AGOP split direction check ================")
    print(f"Dataset: {row['dataset']}")
    print(f"From-scratch AGOP shape: {scratch_agop.shape}")
    print(f"Absolute cosine similarity: {row['cosine_similarity_abs']:.8f}")
    print(f"Threshold: {row['cosine_threshold']:.6f}")
    print(f"Passed: {bool(row['passed'])}")
    print(f"From-scratch top feature: {row['scratch_top_feature']}")
    print(f"xRFM reference top feature: {row['xrfm_top_feature']}")
    print(f"Saved check table to: {output}")

    if not bool(row["passed"]):
        raise RuntimeError(
            "From-scratch direction did not match xRFM reference direction closely enough. "
            "Check kernel parameters, seed, subset size, and xRFM version."
        )


if __name__ == "__main__":
    main()
