"""
DSCLR 评分模块 - 支持多种评分策略
1. DSCLR-Classic: 原始减法惩罚 S_final = S_base - alpha * ReLU(S_neg - tau)
2. DSCLR-Micro: 正向主导型差分惩罚 S_final = S_pos * (1 - alpha * Softplus(S_neg - S_pos + margin))
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScoringConfig:
    """评分配置"""
    method: str = "classic"  # "classic" 或 "micro"
    
    # Classic 方法参数
    alpha: float = 1.0
    tau: float = 0.5
    
    # Micro 方法参数
    micro_alpha: float = 2.0
    margin: float = 0.1
    beta: float = 20.0  # Softplus 的陡峭度


def dsclr_classic_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    alpha: float = 1.0,
    tau: float = 0.5,
    neg_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    DSCLR 经典评分方法（减法惩罚）
    
    S_final = S_base - alpha * ReLU(S_neg - tau)
    
    Args:
        S_base: 基础得分 [num_queries, num_docs]
        S_neg: 负向得分 [num_queries, num_docs]
        alpha: 惩罚强度
        tau: 惩罚阈值
        neg_mask: 负向掩码 [num_queries, num_docs]，用于保护 [NONE] 查询
    
    Returns:
        S_final: 最终得分 [num_queries, num_docs]
    """
    penalty = torch.relu(S_neg - tau)
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_base - alpha * penalty
    
    return S_final


def dsclr_micro_score(
    S_pos: torch.Tensor,
    S_neg: torch.Tensor,
    alpha: float = 2.0,
    margin: float = 0.1,
    beta: float = 20.0,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Micro 评分方法（正向主导型差分惩罚）
    
    S_final = S_pos * (1 - alpha * Softplus(S_neg - S_pos + margin))
    
    核心思想：
    1. 差分保护：只有当负向分数快要追上正向分数时，才触发惩罚
    2. 乘法保序：高质量文档对惩罚的"耐受力"更强，保护头部好文的相对顺序
    
    Args:
        S_pos: 正向得分 [num_queries, num_docs]
        S_neg: 负向得分 [num_queries, num_docs]
        alpha: 惩罚强度
        margin: 差分边界，控制惩罚触发时机
        beta: Softplus 的陡峭度
        neg_mask: 负向掩码 [num_queries, num_docs]，用于保护 [NONE] 查询
    
    Returns:
        S_final: 最终得分 [num_queries, num_docs]
        penalty_ratio: 惩罚比例 [num_queries, num_docs]
    """
    diff = S_neg - S_pos + margin
    
    penalty_ratio = torch.log(1 + torch.exp(beta * diff)) / beta
    
    if neg_mask is not None:
        penalty_ratio = penalty_ratio * neg_mask
    
    S_final = S_pos * (1 - alpha * penalty_ratio)
    
    return S_final, penalty_ratio


def dsclr_micro_v2_score(
    S_pos: torch.Tensor,
    S_neg: torch.Tensor,
    alpha: float = 0.5,
    margin: float = -0.1,
    beta: float = 20.0,
    max_penalty: float = 0.5,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Micro V2: 改进版 - 降低alpha + 负margin + 惩罚上限
    
    改进点：
    1. 降低 alpha（默认0.5），减少惩罚强度
    2. 使用负 margin（默认-0.1），只有当 S_neg 明显超过 S_pos 时才惩罚
    3. 添加惩罚上限 max_penalty，避免分数变为负数
    
    S_final = S_pos * (1 - clamp(alpha * penalty_ratio, 0, max_penalty))
    """
    diff = S_neg - S_pos + margin
    
    penalty_ratio = torch.log(1 + torch.exp(beta * diff)) / beta
    
    if neg_mask is not None:
        penalty_ratio = penalty_ratio * neg_mask
    
    penalty = torch.clamp(alpha * penalty_ratio, 0, max_penalty)
    
    S_final = S_pos * (1 - penalty)
    
    return S_final, penalty_ratio


def dsclr_micro_v3_score(
    S_pos: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 0.5,
    margin: float = -0.05,
    beta: float = 20.0,
    max_penalty: float = 0.5,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Micro V3: 结合动态τ机制
    
    改进点：
    1. 结合经典 DSCLR 的动态 τ 机制
    2. 只有当 S_neg > τ 时才考虑惩罚
    3. 使用差分惩罚，但以 τ 为基准
    
    S_final = S_pos * (1 - clamp(alpha * Softplus(S_neg - max(S_pos, tau) + margin), 0, max_penalty))
    """
    S_threshold = torch.maximum(S_pos, tau.unsqueeze(1))
    diff = S_neg - S_threshold + margin
    
    penalty_ratio = torch.log(1 + torch.exp(beta * diff)) / beta
    
    if neg_mask is not None:
        penalty_ratio = penalty_ratio * neg_mask
    
    penalty = torch.clamp(alpha * penalty_ratio, 0, max_penalty)
    
    S_final = S_pos * (1 - penalty)
    
    return S_final, penalty_ratio


def dsclr_hybrid_score(
    S_pos: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 0.3,
    margin: float = -0.05,
    beta: float = 20.0,
    max_penalty: float = 0.3,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Hybrid: 混合策略（经典减法 + 差分惩罚）
    
    改进点：
    1. 结合经典减法惩罚和差分惩罚
    2. 只有当 S_neg > τ 时才惩罚
    3. 惩罚强度由差分决定
    
    S_final = S_pos - alpha * ReLU(S_neg - tau) * Softplus(S_neg - S_pos + margin)
    """
    classic_penalty = torch.relu(S_neg - tau.unsqueeze(1))
    
    diff = S_neg - S_pos + margin
    diff_penalty = torch.log(1 + torch.exp(beta * diff)) / beta
    
    if neg_mask is not None:
        classic_penalty = classic_penalty * neg_mask
        diff_penalty = diff_penalty * neg_mask
    
    total_penalty = torch.clamp(alpha * classic_penalty * diff_penalty, 0, max_penalty)
    
    S_final = S_pos - total_penalty
    
    return S_final, diff_penalty


def dsclr_hybrid_v2_score(
    S_pos: torch.Tensor,
    S_neg: torch.Tensor,
    alpha: float = 2.0,
    margin: float = 0.1,
    beta: float = 20.0,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Hybrid V2: 动态差分大锤
    
    核心思想：结合 Original 的"绝对杀伤力"和 Micro 的"精准识别"
    - 使用 S_pos - Margin 作为动态护盾
    - 好文章有高 S_pos，护盾也高，不容易被惩罚
    - 烂文章的 S_pos 和 S_neg 接近，护盾低，容易被惩罚
    - 使用减法逻辑，确保烂文排名暴跌
    
    公式：S_final = S_pos - alpha * Softplus(S_neg - (S_pos - Margin))
    
    Args:
        S_pos: 正向得分 [num_queries, num_docs]
        S_neg: 负向得分 [num_queries, num_docs]
        alpha: 惩罚强度（推荐 2.0）
        margin: 安全余量（推荐 0.1）
        beta: Softplus 的陡峭度
        neg_mask: 负向掩码 [num_queries, num_docs]，用于保护 [NONE] 查询
    
    Returns:
        S_final: 最终得分 [num_queries, num_docs]
        penalty: 惩罚值 [num_queries, num_docs]
    """
    dynamic_tau = S_pos - margin
    
    overflow = S_neg - dynamic_tau
    
    penalty = alpha * torch.log(1 + torch.exp(beta * overflow)) / beta
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_pos - penalty
    
    return S_final, penalty


def dsclr_hybrid_v3_score(
    S_pos: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    margin: float = -0.15,
    beta: float = 20.0,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Hybrid V3: 结合动态τ机制 + 减法惩罚
    
    核心思想：
    1. 使用原始 DSCLR 的动态 τ 作为基础阈值
    2. 使用 S_pos - margin 作为动态护盾
    3. 取两者的最大值作为最终护盾
    4. 使用减法逻辑，确保烂文排名暴跌
    
    公式：S_final = S_pos - alpha * Softplus(S_neg - max(S_pos - margin, tau))
    
    Args:
        S_pos: 正向得分 [num_queries, num_docs]
        S_neg: 负向得分 [num_queries, num_docs]
        tau: 动态阈值 [num_queries]
        alpha: 惩罚强度（推荐 1.0）
        margin: 安全余量（推荐 -0.15，负值表示更严格的保护）
        beta: Softplus 的陡峭度
        neg_mask: 负向掩码 [num_queries, num_docs]，用于保护 [NONE] 查询
    
    Returns:
        S_final: 最终得分 [num_queries, num_docs]
        penalty: 惩罚值 [num_queries, num_docs]
    """
    dynamic_tau = S_pos - margin
    
    final_tau = torch.maximum(dynamic_tau, tau.unsqueeze(1))
    
    overflow = S_neg - final_tau
    
    penalty = alpha * torch.log(1 + torch.exp(beta * overflow)) / beta
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_pos - penalty
    
    return S_final, penalty


def dsclr_softplus_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 20.0,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Softplus: 动态τ + Softplus 惩罚
    
    核心思想：
    1. 使用原始 DSCLR 的动态 τ 作为阈值
    2. 使用 Softplus 替代 ReLU，实现平滑惩罚
    3. 使用减法逻辑，确保烂文排名暴跌
    
    公式：S_final = S_base - alpha * Softplus(S_neg - tau)
    
    Args:
        S_base: 基础得分 [num_queries, num_docs]
        S_neg: 负向得分 [num_queries, num_docs]
        tau: 动态阈值 [num_queries]
        alpha: 惩罚强度（推荐 1.0）
        beta: Softplus 的陡峭度
        neg_mask: 负向掩码 [num_queries, num_docs]，用于保护 [NONE] 查询
    
    Returns:
        S_final: 最终得分 [num_queries, num_docs]
        penalty: 惩罚值 [num_queries, num_docs]
    """
    overflow = S_neg - tau.unsqueeze(1)
    
    penalty = alpha * torch.log(1 + torch.exp(beta * overflow)) / beta
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_base - penalty
    
    return S_final, penalty


def compute_scores(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    config: ScoringConfig,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    统一评分接口
    
    Args:
        S_base: 基础得分 [num_queries, num_docs]
        S_neg: 负向得分 [num_queries, num_docs]
        config: 评分配置
        neg_mask: 负向掩码
    
    Returns:
        S_final: 最终得分
        penalty_info: 惩罚信息（可选）
    """
    if config.method == "micro":
        S_final, penalty_ratio = dsclr_micro_score(
            S_base, S_neg,
            alpha=config.micro_alpha,
            margin=config.margin,
            beta=config.beta,
            neg_mask=neg_mask
        )
        return S_final, penalty_ratio
    else:
        S_final = dsclr_classic_score(
            S_base, S_neg,
            alpha=config.alpha,
            tau=config.tau,
            neg_mask=neg_mask
        )
        return S_final, None


def analyze_score_distribution(
    S_pos: torch.Tensor,
    S_neg: torch.Tensor,
    margin: float = 0.1
) -> dict:
    """
    分析得分分布，帮助理解差分惩罚的效果
    
    Args:
        S_pos: 正向得分
        S_neg: 负向得分
        margin: 差分边界
    
    Returns:
        分布统计信息
    """
    diff = S_neg - S_pos + margin
    
    protected = (diff < 0).sum().item()
    penalized = (diff >= 0).sum().item()
    total = diff.numel()
    
    return {
        "protected_docs": protected,
        "protected_ratio": protected / total,
        "penalized_docs": penalized,
        "penalized_ratio": penalized / total,
        "diff_mean": diff.mean().item(),
        "diff_std": diff.std().item(),
        "diff_min": diff.min().item(),
        "diff_max": diff.max().item(),
    }
