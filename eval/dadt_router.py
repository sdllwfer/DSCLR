"""
DADT (Distribution-Aware Dynamic Threshold) 统计分布路由模块
免训练的动态阈值计算接口

使用方式:
    from dadt_router import get_dadt_params, run_dadt_grid_search
    
    # 单点调用
    dynamic_alpha, dynamic_tau = get_dadt_params(neg_scores, base_alpha=2.0, gamma=1.0)
    S_final = S_base - dynamic_alpha * max(0, S_neg - dynamic_tau)
"""

import torch
import numpy as np
from typing import Union, List, Tuple, Optional


def get_dadt_params(
    neg_scores: Union[torch.Tensor, np.ndarray, List[float]],
    base_alpha: float = 2.0,
    gamma: float = 1.0
) -> Tuple[float, float]:
    """
    DADT 核心接口：基于负样本分布计算动态阈值
    
    Args:
        neg_scores: 负样本相似度分数 (Top-K 负向流)
        base_alpha: 基础惩罚力度
        gamma: 标准差乘数（控制阈值偏离均值的程度）
        
    Returns:
        (dynamic_alpha, dynamic_tau): 动态参数元组
        - dynamic_alpha: 保持为 base_alpha（DADT 只调整 tau）
        - dynamic_tau: mu + gamma * sigma（基于负样本分布统计量）
        
    安全保护:
        - 空数组检查：返回 (base_alpha, 0.0)
        - 除零保护：sigma=0 时返回 (base_alpha, mu)
    """
    # 统一转换为 numpy 数组
    if isinstance(neg_scores, torch.Tensor):
        scores = neg_scores.detach().cpu().numpy()
    elif isinstance(neg_scores, list):
        scores = np.array(neg_scores)
    else:
        scores = neg_scores
    
    # 空数组安全检查
    if scores is None or len(scores) == 0:
        return (base_alpha, 0.0)
    
    # 展平为一维
    scores = scores.flatten()
    
    # 计算统计量
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    
    # 除零保护
    if sigma < 1e-8:
        dynamic_tau = mu
    else:
        dynamic_tau = mu + gamma * sigma
    
    return (base_alpha, dynamic_tau)


def run_dadt_grid_search(
    evaluate_fn,
    gamma_list: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
    alpha_list: List[float] = [1.5, 2.0, 2.5],
    verbose: bool = True
) -> List[dict]:
    """
    DADT Grid Search 执行器
    
    遍历 (gamma, alpha) 组合空间，执行评估并返回结果汇总
    
    Args:
        evaluate_fn: 评估函数，接收 (alpha, gamma, get_dadt_params) 返回 metrics dict
        gamma_list: gamma 参数列表，默认 [0.0, 0.5, 1.0, 1.5, 2.0]
        alpha_list: alpha 参数列表，默认 [1.5, 2.0, 2.5]
        verbose: 是否打印进度
        
    Returns:
        List[dict]: 每个组合的评估结果，包含 gamma, alpha, p_mrr, changed_ndcg@5
    """
    results = []
    total = len(gamma_list) * len(alpha_list)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DADT Grid Search: {len(gamma_list)} gammas × {len(alpha_list)} alphas = {total} 组合")
        print(f"{'='*60}")
    
    for gamma in gamma_list:
        for alpha in alpha_list:
            try:
                # 调用评估函数（外部实现具体评估逻辑）
                metrics = evaluate_fn(alpha=alpha, gamma=gamma, dadt_fn=get_dadt_params)
                
                result = {
                    'gamma': gamma,
                    'alpha': alpha,
                    'p_mrr': metrics.get('p_mrr', 0.0),
                    'changed_ndcg@5': metrics.get('changed_ndcg@5', 0.0),
                    'og_ndcg@5': metrics.get('og_ndcg@5', 0.0),
                    'full_metrics': metrics
                }
                results.append(result)
                
                if verbose:
                    print(f"  γ={gamma:.1f}, α={alpha:.1f} → p-MRR={result['p_mrr']:.4f}, nDCG@5={result['changed_ndcg@5']:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"  γ={gamma:.1f}, α={alpha:.1f} → ERROR: {e}")
                results.append({
                    'gamma': gamma,
                    'alpha': alpha,
                    'p_mrr': 0.0,
                    'changed_ndcg@5': 0.0,
                    'error': str(e)
                })
    
    if verbose:
        print_dadt_summary_table(results)
    
    return results


def print_dadt_summary_table(results: List[dict]):
    """
    打印 DADT Grid Search 结果汇总表格 (Markdown 格式)
    """
    print(f"\n{'='*70}")
    print("DADT Grid Search 结果汇总")
    print(f"{'='*70}")
    
    # Markdown 表格头
    print("\n| Gamma | Alpha | p-MRR | Changed nDCG@5 | OG nDCG@5 |")
    print("|-------|-------|-------|----------------|-----------|")
    
    # 按 gamma 分组排序
    for r in sorted(results, key=lambda x: (x['gamma'], x['alpha'])):
        gamma = r['gamma']
        alpha = r['alpha']
        p_mrr = r.get('p_mrr', 0.0)
        changed_ndcg = r.get('changed_ndcg@5', 0.0)
        og_ndcg = r.get('og_ndcg@5', 0.0)
        
        print(f"| {gamma:5.1f} | {alpha:5.1f} | {p_mrr:7.4f} | {changed_ndcg:14.4f} | {og_ndcg:9.4f} |")
    
    # 找出最佳组合
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_p_mrr = max(valid_results, key=lambda x: x['p_mrr'])
        best_ndcg = max(valid_results, key=lambda x: x['changed_ndcg@5'])
        
        print(f"\n{'='*70}")
        print("最佳组合:")
        print(f"  最高 p-MRR:       γ={best_p_mrr['gamma']:.1f}, α={best_p_mrr['alpha']:.1f} → {best_p_mrr['p_mrr']:.4f}")
        print(f"  最高 Changed nDCG@5: γ={best_ndcg['gamma']:.1f}, α={best_ndcg['alpha']:.1f} → {best_ndcg['changed_ndcg@5']:.4f}")
        print(f"{'='*70}\n")


# 便捷函数：从负样本嵌入计算分数后调用 DADT
def compute_dadt_threshold_from_embeddings(
    q_minus_emb: torch.Tensor,
    doc_embeddings: torch.Tensor,
    gamma: float = 1.0
) -> float:
    """
    从 Q- 和文档嵌入直接计算 DADT 阈值
    
    Args:
        q_minus_emb: Q- 查询向量 [dim]
        doc_embeddings: 文档向量矩阵 [num_docs, dim]
        gamma: 标准差乘数
        
    Returns:
        dynamic_tau: 计算得到的动态阈值
    """
    # 计算所有文档与 Q- 的相似度
    with torch.no_grad():
        neg_scores = torch.matmul(doc_embeddings, q_minus_emb)
    
    _, dynamic_tau = get_dadt_params(neg_scores, base_alpha=1.0, gamma=gamma)
    return dynamic_tau
