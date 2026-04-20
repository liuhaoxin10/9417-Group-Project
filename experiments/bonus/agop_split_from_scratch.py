"""
Legacy helper: from-scratch standard AGOP split direction check.

这个脚本对应旧版 PDF 里的 Bonus 要求，保留为 standard AGOP 的参考实现。
新版 PDF 的正式 Bonus 脚本请使用：

    python experiments/bonus/residual_weighted_agop.py

本脚本完成的旧版检查包括：
1. 不依赖 xRFM 的内部 AGOP / gradient 函数，手写 AGOP-based splitting criterion；
2. 在一个小数据集上计算 split direction；
3. 用 xRFM 的公开训练接口跑同样设置，并读取公开 state_dict 里的 split direction；
4. 检查两者方向是否一致。

注意：
- from-scratch 部分只使用 numpy / pandas / sklearn 思路，不调用 xrfm.rfm_src；
- xRFM 只作为 reference，用来核对我们的手写方向是否和 library 一致。
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
    """保存本次 Bonus 检查需要用到的核心参数。"""

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
    """读取已经预处理好的数据，并截取小样本用于 Bonus 验证。"""

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
        raise ValueError(f"{dataset} 的训练样本不足 {n_samples} 行。")
    if len(X_val) < n_val_samples:
        raise ValueError(f"{dataset} 的验证样本不足 {n_val_samples} 行。")

    return {
        "X_train": X_train,
        "y_train": y_train.reshape(-1, 1),
        "X_val": X_val,
        "y_val": y_val.reshape(-1, 1),
        "feature_names": feature_names,
    }


def xrfm_subset_indices(n_samples: int, seed: int) -> np.ndarray:
    """
    复现 xRFM 在 _get_agop_on_subset 中使用的随机子集顺序。

    xRFM 内部会调用 torch.randperm(len(X))，然后取前 95% 作为 AGOP
    计算模型的训练部分。这里不调用 xRFM 内部函数，只复现这个公开可读的随机抽样规则。
    """

    torch.manual_seed(seed)
    subset_indices = torch.randperm(n_samples).cpu().numpy()
    subset_train_size = max(int(n_samples * 0.95), 1)
    return subset_indices[:subset_train_size]


def l2_laplace_kernel(X: np.ndarray, Z: np.ndarray, bandwidth: float, exponent: float) -> np.ndarray:
    """手写 l2 Laplace kernel: exp(-||x-z||_2^p / bandwidth^p)。"""

    distances = np.linalg.norm(X[:, None, :] - Z[None, :, :], axis=2)
    return np.exp(-(distances ** exponent) / (bandwidth ** exponent))


def fit_kernel_ridge_from_scratch(
    X: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    exponent: float,
    reg: float,
) -> np.ndarray:
    """手写 kernel ridge regression，得到 dual coefficients alpha。"""

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
    手写核模型对输入样本的梯度。

    对 f(z) = sum_i alpha_i k(x_i, z)，l2 Laplace kernel 的梯度为：
    grad_z f(z_j) = sum_i alpha_i * c_ij * (z_j - x_i)

    其中 c_ij = -p / bandwidth^p * k(x_i, z_j) * ||x_i-z_j||^(p-2)。
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
    """手写 AGOP 矩阵，并按 xRFM 的做法除以最大值做尺度归一化。"""

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
    """取 AGOP 的第一右奇异向量，和 xRFM 的 top_vector_agop_on_subset 逻辑一致。"""

    _, _, vt = np.linalg.svd(agop, full_matrices=False)
    direction = vt[0]
    return direction / (np.linalg.norm(direction) + 1e-30)


def compute_from_scratch_direction(data: dict[str, Any], config: BonusConfig) -> tuple[np.ndarray, np.ndarray]:
    """完整执行手写 AGOP split direction 计算。"""

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
    """构造只用于 reference check 的 xRFM 模型。"""

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
    用 xRFM 的公开接口训练，并从 get_state_dict() 中读取 reference split direction。

    这里的 xRFM 只用于核对答案；AGOP 和 direction 的计算仍然由上面的
    from-scratch 函数完成。
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
    """split direction 的正负号没有意义，所以用绝对 cosine similarity。"""

    a_norm = a / (np.linalg.norm(a) + 1e-30)
    b_norm = b / (np.linalg.norm(b) + 1e-30)
    return float(abs(np.dot(a_norm, b_norm)))


def direction_summary(
    scratch_direction: np.ndarray,
    reference_direction: np.ndarray,
    feature_names: list[str],
    config: BonusConfig,
) -> pd.DataFrame:
    """整理一行结果，方便写入 CSV 或直接复制进 appendix。"""

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
