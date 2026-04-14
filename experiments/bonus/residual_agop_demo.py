"""
COMP9417 Bonus Task: Residual-weighted AGOP
满足 PDF 要求的 (ii) Implementation 和 (iii) Disagreement example
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from xrfm.rfm_src.recursive_feature_machine import RFM

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_toy_dataset():
    # 使用 bike_sharing 的一个小子集来做实验 (容易产生残差)
    processed_dir = PROJECT_ROOT / "data/processed"
    X = pd.read_csv(processed_dir / "bike_sharing_X_train.csv").values[:300]
    y = pd.read_csv(processed_dir / "bike_sharing_y_train.csv")["target"].values[:300]
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return X, y

def main():
    X, y = load_toy_dataset()
    print(f"Dataset loaded: X shape {X.shape}, y shape {y.shape}")

    # 1. 训练一个基础模型（故意加大正则化 reg=10.0，不让它完美拟合，从而产生残差）
    model = RFM(kernel='l2', bandwidth=5.0, iters=0, device='cpu')
    model.reg = 10.0
    
    # [修复]：手动补充直接调用底层函数所缺失的初始化属性
    model.fit_using_eigenpro = False 
    model.center_grads = False
    
    model.fit_predictor(X, y)
    
    # 获取预测值和残差权重 w_i = (y - y_pred)^2
    y_pred = model.kernel(X, X) @ model.weights
    residuals = torch.abs(y - y_pred)
    weights = residuals ** 2
    weights = weights / (torch.sum(weights) + 1e-8)  # 归一化

    # 2. 计算标准 AGOP (Standard AGOP)
    grads = model.kernel_obj.get_function_grads(X, X, model.weights.t(), mat=None)
    grads = grads.reshape(-1, grads.shape[-1])  # Shape: (N, D)
    standard_agop = (grads.T @ grads) / len(X)
    
    # 3. 计算残差加权 AGOP (Residual-weighted AGOP)
    # 根据 PDF 公式: sum(w_i * grad_i * grad_i.T) / sum(w_i)
    weighted_grads = grads * torch.sqrt(weights)
    residual_agop = weighted_grads.T @ weighted_grads

    # 4. 获取分裂方向 (Top Eigenvector)
    _, std_eigenvectors = torch.linalg.eigh(standard_agop)
    std_split_dir = std_eigenvectors[:, -1]
    
    _, res_eigenvectors = torch.linalg.eigh(residual_agop)
    res_split_dir = res_eigenvectors[:, -1]

    # 5. 验证是否产生 Disagreement (任务 iii)
    cos_sim = torch.abs(torch.dot(std_split_dir, res_split_dir) / 
                       (torch.linalg.norm(std_split_dir) * torch.linalg.norm(res_split_dir)))
    
    print("\n" + "="*50)
    print("Bonus Task (iii): Disagreement Example")
    print("="*50)
    print(f"Cosine Similarity between Split Directions: {cos_sim.item():.4f}")
    if cos_sim.item() < 0.95:
        print("✅ SUCCESS: The standard AGOP and residual-weighted AGOP select DIFFERENT split directions!")
        
        # 看看哪些特征最被看重
        std_top_feature = torch.argmax(torch.abs(std_split_dir)).item()
        res_top_feature = torch.argmax(torch.abs(res_split_dir)).item()
        print(f"Standard AGOP Top Feature Index: {std_top_feature}")
        print(f"Residual AGOP Top Feature Index: {res_top_feature}")

if __name__ == "__main__":
    main()