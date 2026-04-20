"""
Bonus: residual-weighted AGOP extension.

This script matches the updated project bonus requirement:
1. implement residual-weighted AGOP;
2. compare its split direction with the standard AGOP direction;
3. show a disagreement example;
4. show one setting where the residual-weighted split improves test RMSE.

The experiment uses a small synthetic regression dataset. The target contains:
- a dominant global linear signal in x0;
- a local interaction that only appears when x1 is large.

The standard AGOP is dominated by the global signal. The residual-weighted AGOP
puts more mass on samples that the global predictor underfits, so it can select
the gating direction x1 and create a more useful split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BonusConfig:
    """Store the deterministic settings for the bonus experiment."""

    n_samples: int
    test_size: float
    seed: int
    bandwidth: float
    exponent: float
    kernel_reg: float
    leaf_reg: float
    gate_threshold: float
    interaction_strength: float
    noise_std: float
    disagreement_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residual-weighted AGOP bonus experiment.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/tables")
    parser.add_argument("--n-samples", type=int, default=900)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bandwidth", type=float, default=2.0)
    parser.add_argument("--exponent", type=float, default=1.0)
    parser.add_argument("--kernel-reg", type=float, default=0.1)
    parser.add_argument("--leaf-reg", type=float, default=1.0)
    parser.add_argument("--gate-threshold", type=float, default=0.35)
    parser.add_argument("--interaction-strength", type=float, default=5.0)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--disagreement-threshold", type=float, default=0.95)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_synthetic_dataset(config: BonusConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a controlled dataset where residual weighting has a clear reason to help."""

    rng = np.random.default_rng(config.seed)
    X = rng.normal(size=(config.n_samples, 6))

    global_signal = X[:, 0]
    gate = X[:, 1]
    local_effect = X[:, 2]
    active_region = gate > config.gate_threshold

    y = (
        3.0 * global_signal
        + config.interaction_strength * active_region.astype(float) * local_effect
        + config.noise_std * rng.normal(size=config.n_samples)
    )

    feature_names = [
        "x0_global_linear_signal",
        "x1_residual_gate",
        "x2_local_interaction",
        "x3_noise",
        "x4_noise",
        "x5_noise",
    ]
    return X.astype(np.float64), y.reshape(-1, 1).astype(np.float64), feature_names


def l2_laplace_kernel(X: np.ndarray, Z: np.ndarray, bandwidth: float, exponent: float) -> np.ndarray:
    """Compute exp(-||x-z||_2^p / bandwidth^p)."""

    distances = np.linalg.norm(X[:, None, :] - Z[None, :, :], axis=2)
    return np.exp(-(distances ** exponent) / (bandwidth ** exponent))


def fit_kernel_ridge(
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


def function_gradients(
    centers: np.ndarray,
    samples: np.ndarray,
    alpha: np.ndarray,
    bandwidth: float,
    exponent: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute gradients of the fitted kernel model with respect to input samples."""

    diff = samples[None, :, :] - centers[:, None, :]
    distances = np.linalg.norm(diff, axis=2)
    kernel_matrix = np.exp(-(distances ** exponent) / (bandwidth ** exponent))

    mask = distances >= eps
    safe_distances = np.maximum(distances, eps)
    factors = kernel_matrix * (safe_distances ** (exponent - 2.0))
    factors *= mask
    factors *= -exponent / (bandwidth ** exponent)

    return np.einsum("io,ij,ijd->ojd", alpha, factors, diff)


def normalize_agop(agop: np.ndarray) -> np.ndarray:
    """Scale an AGOP matrix only for stable comparison and reporting."""

    return agop / (np.max(np.abs(agop)) + 1e-30)


def compute_standard_and_residual_agop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: BonusConfig,
) -> dict[str, np.ndarray]:
    """Compute standard AGOP and residual-weighted AGOP from the same predictor."""

    alpha = fit_kernel_ridge(
        X=X_train,
        y=y_train,
        bandwidth=config.bandwidth,
        exponent=config.exponent,
        reg=config.kernel_reg,
    )

    train_kernel = l2_laplace_kernel(X_train, X_train, config.bandwidth, config.exponent)
    y_pred = train_kernel @ alpha
    residuals = (y_train - y_pred).reshape(-1)

    gradients = function_gradients(
        centers=X_train,
        samples=X_train,
        alpha=alpha,
        bandwidth=config.bandwidth,
        exponent=config.exponent,
    ).reshape(-1, X_train.shape[1])

    standard_agop = gradients.T @ gradients / len(X_train)

    # Updated bonus definition: w_i = phi(r_i), with phi(r)=r^2.
    weights = residuals**2
    weighted_agop = (gradients * weights[:, None]).T @ gradients / (weights.sum() + 1e-30)

    return {
        "standard_agop": normalize_agop(standard_agop),
        "residual_agop": normalize_agop(weighted_agop),
        "residuals": residuals,
        "train_predictions": y_pred.reshape(-1),
    }


def top_split_direction(agop: np.ndarray) -> np.ndarray:
    """Use the top singular vector as the AGOP split direction."""

    _, _, vt = np.linalg.svd(agop, full_matrices=False)
    direction = vt[0]
    return direction / (np.linalg.norm(direction) + 1e-30)


def sign_invariant_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Directions v and -v define the same split axis, so use absolute cosine."""

    a = a / (np.linalg.norm(a) + 1e-30)
    b = b / (np.linalg.norm(b) + 1e-30)
    return float(abs(np.dot(a, b)))


def split_masks(X: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, float]:
    """Split samples at the median projection along the selected direction."""

    projections = X @ direction
    threshold = float(np.median(projections))
    return projections <= threshold, threshold


def evaluate_two_leaf_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    direction: np.ndarray,
    leaf_reg: float,
) -> dict[str, float]:
    """Evaluate a split by fitting the same Ridge leaf model on both sides."""

    train_left, threshold = split_masks(X_train, direction)
    test_left = (X_test @ direction) <= threshold
    y_pred = np.zeros(len(y_test), dtype=float)

    for is_left in [True, False]:
        train_mask = train_left if is_left else ~train_left
        test_mask = test_left if is_left else ~test_left
        model = Ridge(alpha=leaf_reg)
        model.fit(X_train[train_mask], y_train[train_mask].ravel())
        y_pred[test_mask] = model.predict(X_test[test_mask])

    return {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "split_threshold": threshold,
        "left_train_size": int(train_left.sum()),
        "right_train_size": int((~train_left).sum()),
    }


def evaluate_single_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    leaf_reg: float,
) -> float:
    """Baseline evaluator without a split, included only for context."""

    model = Ridge(alpha=leaf_reg)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    return float(np.sqrt(mean_squared_error(y_test, y_pred)))


def build_direction_table(
    config: BonusConfig,
    feature_names: list[str],
    standard_direction: np.ndarray,
    residual_direction: np.ndarray,
    residuals: np.ndarray,
) -> pd.DataFrame:
    """Summarize the direction disagreement required by the updated bonus."""

    cosine = sign_invariant_cosine(standard_direction, residual_direction)
    standard_top = int(np.argmax(np.abs(standard_direction)))
    residual_top = int(np.argmax(np.abs(residual_direction)))

    active_residual_share = float(
        np.mean(np.abs(residuals) >= np.quantile(np.abs(residuals), 0.75))
    )

    return pd.DataFrame(
        [
            {
                "dataset": "synthetic_residual_interaction",
                "n_samples": config.n_samples,
                "seed": config.seed,
                "standard_top_feature": feature_names[standard_top],
                "residual_weighted_top_feature": feature_names[residual_top],
                "cosine_similarity_abs": cosine,
                "disagreement_threshold": config.disagreement_threshold,
                "disagreement": cosine < config.disagreement_threshold,
                "weight_function": "phi(r)=r^2",
                "high_residual_sample_share": active_residual_share,
                "brief_explanation": (
                    "Standard AGOP is dominated by the global x0 trend, while residual weighting "
                    "upweights underfit samples from the gated interaction region and shifts the "
                    "split direction toward x1."
                ),
            }
        ]
    )


def build_performance_table(
    no_split_rmse: float,
    standard_eval: dict[str, float],
    residual_eval: dict[str, float],
    standard_direction: np.ndarray,
    residual_direction: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Summarize the performance comparison required by the updated bonus."""

    standard_top = feature_names[int(np.argmax(np.abs(standard_direction)))]
    residual_top = feature_names[int(np.argmax(np.abs(residual_direction)))]
    standard_rmse = standard_eval["test_rmse"]
    residual_rmse = residual_eval["test_rmse"]

    return pd.DataFrame(
        [
            {
                "method": "single_ridge_no_split",
                "split_top_feature": "",
                "test_rmse": no_split_rmse,
                "rmse_improvement_vs_standard_agop": np.nan,
                "left_train_size": np.nan,
                "right_train_size": np.nan,
            },
            {
                "method": "standard_agop_split",
                "split_top_feature": standard_top,
                "test_rmse": standard_rmse,
                "rmse_improvement_vs_standard_agop": 0.0,
                "left_train_size": standard_eval["left_train_size"],
                "right_train_size": standard_eval["right_train_size"],
            },
            {
                "method": "residual_weighted_agop_split",
                "split_top_feature": residual_top,
                "test_rmse": residual_rmse,
                "rmse_improvement_vs_standard_agop": standard_rmse - residual_rmse,
                "left_train_size": residual_eval["left_train_size"],
                "right_train_size": residual_eval["right_train_size"],
            },
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BonusConfig(
        n_samples=args.n_samples,
        test_size=args.test_size,
        seed=args.seed,
        bandwidth=args.bandwidth,
        exponent=args.exponent,
        kernel_reg=args.kernel_reg,
        leaf_reg=args.leaf_reg,
        gate_threshold=args.gate_threshold,
        interaction_strength=args.interaction_strength,
        noise_std=args.noise_std,
        disagreement_threshold=args.disagreement_threshold,
    )

    X, y, feature_names = make_synthetic_dataset(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.seed,
    )

    agop_result = compute_standard_and_residual_agop(X_train, y_train, config)
    standard_direction = top_split_direction(agop_result["standard_agop"])
    residual_direction = top_split_direction(agop_result["residual_agop"])

    standard_eval = evaluate_two_leaf_ridge(
        X_train, y_train, X_test, y_test, standard_direction, config.leaf_reg
    )
    residual_eval = evaluate_two_leaf_ridge(
        X_train, y_train, X_test, y_test, residual_direction, config.leaf_reg
    )
    no_split_rmse = evaluate_single_ridge(
        X_train, y_train, X_test, y_test, config.leaf_reg
    )

    direction_table = build_direction_table(
        config=config,
        feature_names=feature_names,
        standard_direction=standard_direction,
        residual_direction=residual_direction,
        residuals=agop_result["residuals"],
    )
    performance_table = build_performance_table(
        no_split_rmse=no_split_rmse,
        standard_eval=standard_eval,
        residual_eval=residual_eval,
        standard_direction=standard_direction,
        residual_direction=residual_direction,
        feature_names=feature_names,
    )

    direction_path = output_dir / "bonus_residual_agop_direction_comparison.csv"
    performance_path = output_dir / "bonus_residual_agop_performance_comparison.csv"
    direction_table.to_csv(direction_path, index=False)
    performance_table.to_csv(performance_path, index=False)

    row = direction_table.iloc[0]
    residual_perf = performance_table[performance_table["method"] == "residual_weighted_agop_split"].iloc[0]
    standard_perf = performance_table[performance_table["method"] == "standard_agop_split"].iloc[0]

    print("\n================ Residual-weighted AGOP bonus ================")
    print(f"Dataset: {row['dataset']}")
    print(f"Standard AGOP top feature: {row['standard_top_feature']}")
    print(f"Residual-weighted AGOP top feature: {row['residual_weighted_top_feature']}")
    print(f"Absolute cosine similarity: {row['cosine_similarity_abs']:.6f}")
    print(f"Disagreement: {bool(row['disagreement'])}")
    print(f"Standard AGOP split RMSE: {standard_perf['test_rmse']:.6f}")
    print(f"Residual-weighted AGOP split RMSE: {residual_perf['test_rmse']:.6f}")
    print(f"RMSE improvement: {residual_perf['rmse_improvement_vs_standard_agop']:.6f}")
    print(f"Saved direction comparison to: {direction_path}")
    print(f"Saved performance comparison to: {performance_path}")

    if not bool(row["disagreement"]):
        raise RuntimeError("Residual-weighted AGOP did not disagree with standard AGOP.")
    if residual_perf["rmse_improvement_vs_standard_agop"] <= 0:
        raise RuntimeError("Residual-weighted AGOP did not improve over standard AGOP.")


if __name__ == "__main__":
    main()
