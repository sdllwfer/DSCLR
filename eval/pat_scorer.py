"""
PAT (Positive-Aware Tolerance) 双流交叉门控计分器
纯数学计算模块，无任何业务耦合

核心公式:
    dynamic_tau = tau_base + lambda_weight * S_base
    penalty = alpha * relu(S_neg - dynamic_tau)
    S_final = S_base - penalty

分段惩罚函数 (Piecewise Penalty):
    - 保护 og_rank <= top_k 的文档，减轻惩罚力度
    - 避免 p-MRR 和 nDCG 的 trade-off
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, List


class PAT_Scorer:
    """
    PAT 计分器 - 基于正向得分的自适应容忍度机制

    核心思想:
    - tau 不再是静态阈值，而是随 S_base 动态调整
    - 正向得分越高，容忍度越高（dynamic_tau 上调）
    - 从而实现对高置信度查询的差异化惩罚
    """

    @staticmethod
    def compute(
        S_base: Union[torch.Tensor, float],
        S_neg: Union[torch.Tensor, float],
        alpha: float,
        tau_base: float,
        lambda_weight: float
    ) -> Union[torch.Tensor, float]:
        """
        PAT 计分核心计算

        Args:
            S_base: 正向得分 (Q+ 与文档相似度)
            S_neg: 负向得分 (Q- 与文档相似度)
            alpha: 惩罚力度
            tau_base: 基础阈值
            lambda_weight: 正向权重 (控制 tau 随 S_base 的偏移程度)

        Returns:
            S_final: 最终得分
        """
        dynamic_tau = tau_base + lambda_weight * S_base

        if isinstance(S_neg, torch.Tensor):
            penalty = alpha * torch.relu(S_neg - dynamic_tau)
        else:
            penalty = alpha * np.maximum(0, S_neg - dynamic_tau)

        S_final = S_base - penalty
        return S_final

    @staticmethod
    def compute_with_og_rank_protection(
        S_base: Union[torch.Tensor, np.ndarray],
        S_neg: Union[torch.Tensor, np.ndarray],
        og_ranks: np.ndarray,
        alpha: float,
        tau_base: float,
        lambda_weight: float,
        top_k: int = 5,
        protection_factor: float = 0.0
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        PAT 计分 + OG排名保护

        分段惩罚策略:
        - og_rank <= top_k: 保护文档，penalty * protection_factor (默认完全不惩罚)
        - og_rank > top_k: 正常惩罚

        Args:
            S_base: 正向得分
            S_neg: 负向得分
            og_ranks: OG模式下的排名 (1-indexed)
            alpha: 惩罚力度
            tau_base: 基础阈值
            lambda_weight: 正向权重
            top_k: 保护阈值 (默认 5)
            protection_factor: 保护系数 (默认 0.0, 即完全不惩罚 top_k)

        Returns:
            S_final: 最终得分
        """
        dynamic_tau = tau_base + lambda_weight * S_base

        if isinstance(S_neg, torch.Tensor):
            penalty = alpha * torch.relu(S_neg - dynamic_tau)
            og_ranks_t = torch.tensor(og_ranks, dtype=torch.float32)
            mask_protected = (og_ranks_t <= top_k).float()
            protected_penalty = penalty * ((1 - mask_protected) * 1.0 + mask_protected * protection_factor)
            S_final = S_base - protected_penalty
        else:
            penalty = alpha * np.maximum(0, S_neg - dynamic_tau)
            mask_protected = (og_ranks <= top_k).astype(np.float32)
            protected_penalty = penalty * ((1 - mask_protected) * 1.0 + mask_protected * protection_factor)
            S_final = S_base - protected_penalty

        return S_final

    @staticmethod
    def compute_hybrid(
        S_base: Union[torch.Tensor, np.ndarray],
        S_neg: Union[torch.Tensor, np.ndarray],
        og_ranks: np.ndarray,
        alpha: float,
        tau_base: float,
        lambda_weight: float,
        top_k: int = 5,
        protection_factor: float = 0.0,
        boost_ndcg_factor: float = 0.5
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        混合策略: 结合保护和boost

        策略说明:
        1. 保护 og_rank <= top_k 的文档 (完全不惩罚)
        2. 对 og_rank 6-20 的文档进行轻微 boost (帮助进入 top-5)

        Args:
            S_base: 正向得分
            S_neg: 负向得分
            og_ranks: OG模式下的排名
            alpha: 惩罚力度
            tau_base: 基础阈值
            lambda_weight: 正向权重
            top_k: 保护阈值
            protection_factor: 保护系数 (默认 0.0，完全不惩罚)
            boost_ndcg_factor: nDCG boost 系数

        Returns:
            S_final: 最终得分
        """
        dynamic_tau = tau_base + lambda_weight * S_base

        if isinstance(S_neg, torch.Tensor):
            penalty = alpha * torch.relu(S_neg - dynamic_tau)
            og_ranks_t = torch.tensor(og_ranks, dtype=torch.float32)

            mask_protected = (og_ranks_t <= top_k).float()
            mask_boost = ((og_ranks_t > top_k) & (og_ranks_t <= 20)).float()

            protected_penalty = penalty * ((1 - mask_protected) * 1.0 + mask_protected * protection_factor)

            S_final = S_base - protected_penalty + mask_boost * boost_ndcg_factor * S_base
        else:
            penalty = alpha * np.maximum(0, S_neg - dynamic_tau)

            mask_protected = (og_ranks <= top_k).astype(np.float32)
            mask_boost = ((og_ranks > top_k) & (og_ranks <= 20)).astype(np.float32)

            protected_penalty = penalty * ((1 - mask_protected) * 1.0 + mask_protected * protection_factor)

            S_final = S_base - protected_penalty + mask_boost * boost_ndcg_factor * S_base

        return S_final

    @staticmethod
    def compute_vectorized(
        S_base: torch.Tensor,
        S_neg: torch.Tensor,
        alpha: float,
        tau_base: float,
        lambda_weight: float
    ) -> torch.Tensor:
        """
        向量化版本 - 适用于批量计算

        Args:
            S_base: 正向得分向量 (batch_size,)
            S_neg: 负向得分向量 (batch_size,)
            alpha: 惩罚力度
            tau_base: 基础阈值
            lambda_weight: 正向权重

        Returns:
            S_final: 最终得分向量 (batch_size,)
        """
        dynamic_tau = tau_base + lambda_weight * S_base
        penalty = alpha * torch.relu(S_neg - dynamic_tau)
        S_final = S_base - penalty
        return S_final


def run_pat_grid_search_evaluation(
    model_name: str,
    task_name: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: str = None,
    use_cache: bool = True,
    alpha_list: list = None,
    tau_base_list: list = None,
    lambda_list: list = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    PAT 极限 Grid Search 评测入口

    搜索空间:
    - alpha_list = [3.0, 4.0, 5.0, 6.0]
    - tau_base_list = [0.4, 0.5]
    - lambda_list = [0.3, 0.5, 0.7]

    目标: 寻找 p-MRR > 0.154 且 nDCG 不崩盘的组合
    """
    import os
    import sys
    import json
    import logging
    import random
    import numpy as np
    from typing import List, Optional

    # 设置路径
    sys.path.insert(0, '/home/luwa/Documents/DSCLR')
    from eval.engine_grid_search import (
        GridSearchEngine, load_cached_embeddings, DEFAULT_CACHE_DIR
    )

    # 默认参数
    if alpha_list is None:
        alpha_list = [3.0, 4.0, 5.0, 6.0]
    if tau_base_list is None:
        tau_base_list = [0.4, 0.5]
    if lambda_list is None:
        lambda_list = [0.3, 0.5, 0.7]

    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*70)
    logger.info("🎯 PAT (Positive-Aware Tolerance) Extreme Grid Search")
    logger.info("="*70)
    logger.info(f"Alpha 列表: {alpha_list}")
    logger.info(f"Tau_base 列表: {tau_base_list}")
    logger.info(f"Lambda 列表: {lambda_list}")
    logger.info(f"总组合数: {len(alpha_list)} × {len(tau_base_list)} × {len(lambda_list)} = {len(alpha_list) * len(tau_base_list) * len(lambda_list)}")
    logger.info("目标: p-MRR > 0.154 且 nDCG 不崩盘")
    logger.info("="*70)

    # 创建引擎
    engine = GridSearchEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        seed=seed
    )

    # 加载数据
    logger.info("\n📊 加载评测数据...")
    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    q_raw_og, q_raw_changed = engine.data_loader.load_raw_queries()

    # 编码文档
    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)

    cached_data = None
    if use_cache:
        cached_data = load_cached_embeddings(
            cache_dir or DEFAULT_CACHE_DIR, task_name, model_name
        )

    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        if set(cached_doc_ids) == set(all_doc_ids):
            logger.info(f"✅ 使用缓存的文档向量 ({len(cached_doc_ids)} 个)")
            engine.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
        else:
            logger.warning(f"⚠️ 缓存文档ID不匹配，重新编码...")
            engine._encode_documents(corpus, all_doc_ids)
    else:
        engine._encode_documents(corpus, all_doc_ids)

    # 加载 dual queries
    logger.info("🔤 加载并编码 dual queries...")
    dual_queries_cache = os.path.join(
        "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v4",
        f"dual_queries_v4_{task_name}.jsonl"
    )

    q_plus_og = {}
    q_plus_changed = {}
    q_minus_changed = {}
    neg_mask_dict = {}

    if os.path.exists(dual_queries_cache):
        logger.info(f"📂 加载 dual queries: {dual_queries_cache}")
        with open(dual_queries_cache, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item['qid']
                q_plus = item['q_plus']
                q_minus = item['q_minus']

                if item['query_type'] == 'og':
                    q_plus_og[qid] = q_plus
                else:
                    q_plus_changed[qid] = q_plus
                    q_minus_changed[qid] = q_minus
                    neg_mask_dict[qid] = 0.0 if q_minus == "[NONE]" else 1.0

        logger.info(f"✅ 加载完成: {len(q_plus_og)} OG, {len(q_plus_changed)} Changed")
    else:
        logger.warning(f"⚠️ 未找到 dual queries 缓存")
        q_plus_og = q_og
        q_plus_changed = q_changed
        q_minus_changed = q_changed
        neg_mask_dict = {qid: 1.0 for qid in q_changed}

    # 编码查询
    q_og_items = list(q_plus_og.items())
    q_og_list = [item[1] for item in q_og_items]
    q_og_emb = engine.encoder.encode_queries(q_og_list, batch_size=batch_size)
    q_og_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_og_items)}

    q_changed_items = list(q_plus_changed.items())
    q_plus_list = [item[1] for item in q_changed_items]
    q_minus_list = [q_minus_changed.get(item[0], item[1]) for item in q_changed_items]

    q_plus_changed_emb = engine.encoder.encode_queries(q_plus_list, batch_size=batch_size)
    q_minus_changed_emb = engine.encoder.encode_queries(q_minus_list, batch_size=batch_size)
    q_changed_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_changed_items)}

    neg_mask = torch.tensor([neg_mask_dict.get(item[0], 1.0) for item in q_changed_items],
                                   dtype=torch.float32)

    # 执行 PAT Grid Search
    all_results = []

    for alpha in alpha_list:
        for tau_base in tau_base_list:
            for lambda_weight in lambda_list:
                logger.info(f"\n🔍 测试: α={alpha:.1f}, τ_base={tau_base:.1f}, λ={lambda_weight:.1f}")

                result = engine._evaluate_params_pat(
                    q_og_emb=q_og_emb,
                    q_plus_changed_emb=q_plus_changed_emb,
                    q_minus_changed_emb=q_minus_changed_emb,
                    candidates=candidates,
                    corpus=corpus,
                    alpha=alpha,
                    tau_base=tau_base,
                    lambda_weight=lambda_weight,
                    q_og_qid_to_idx=q_og_qid_to_idx,
                    q_changed_qid_to_idx=q_changed_qid_to_idx,
                    neg_mask=neg_mask
                )

                all_results.append({
                    'alpha': alpha,
                    'tau_base': tau_base,
                    'lambda': lambda_weight,
                    'p_mrr': result['p_mrr'],
                    'og_ndcg@5': result['og_ndcg@5'],
                    'changed_ndcg@5': result['changed_ndcg@5'],
                    'og_mrr': result['og_mrr'],
                    'changed_mrr': result['changed_mrr']
                })

                logger.info(f"   p-MRR={result['p_mrr']:.4f}, Changed nDCG@5={result['changed_ndcg@5']:.4f}")

    # 打印汇总表格
    _print_pat_summary_table(all_results)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "pat_grid_search_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'alpha_list': alpha_list,
            'tau_base_list': tau_base_list,
            'lambda_list': lambda_list,
            'results': all_results
        }, f, indent=2)
    logger.info(f"\n💾 PAT 结果已保存: {results_file}")

    csv_file = os.path.join(output_dir, "pat_grid_search_results.csv")
    with open(csv_file, 'w') as f:
        f.write("alpha,tau_base,lambda,p_mrr,og_ndcg@5,changed_ndcg@5,og_mrr,changed_mrr\n")
        for r in all_results:
            f.write(f"{r['alpha']:.4f},{r['tau_base']:.4f},{r['lambda']:.4f},"
                   f"{r['p_mrr']:.4f},{r['og_ndcg@5']:.4f},{r['changed_ndcg@5']:.4f},"
                   f"{r['og_mrr']:.4f},{r['changed_mrr']:.4f}\n")
    logger.info(f"💾 CSV 报告已保存: {csv_file}")

    return {
        'results': all_results,
        'best_p_mrr': max(all_results, key=lambda x: x['p_mrr']),
        'best_ndcg': max(all_results, key=lambda x: x['changed_ndcg@5'])
    }


def run_pat_protected_grid_search_evaluation(
    model_name: str,
    task_name: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: str = None,
    use_cache: bool = True,
    alpha_list: list = None,
    tau_base_list: list = None,
    lambda_list: list = None,
    top_k: int = 5,
    protection_factor: float = 0.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    PAT 极限 Grid Search 评测入口 (OG排名保护版)

    分段惩罚策略:
    - og_rank <= top_k: 保护文档，penalty * protection_factor
    - og_rank > top_k: 正常惩罚
    """
    import os
    import sys
    import json
    import logging
    import random
    import numpy as np
    from typing import List, Optional

    sys.path.insert(0, '/home/luwa/Documents/DSCLR')
    from eval.engine_grid_search import (
        GridSearchEngine, load_cached_embeddings, DEFAULT_CACHE_DIR
    )

    if alpha_list is None:
        alpha_list = [3.0, 4.0, 5.0, 6.0]
    if tau_base_list is None:
        tau_base_list = [0.4, 0.5]
    if lambda_list is None:
        lambda_list = [0.3, 0.5, 0.7]

    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*70)
    logger.info("🛡️ PAT (OG排名保护) Extreme Grid Search")
    logger.info("="*70)
    logger.info(f"Alpha 列表: {alpha_list}")
    logger.info(f"Tau_base 列表: {tau_base_list}")
    logger.info(f"Lambda 列表: {lambda_list}")
    logger.info(f"Top_k 保护阈值: {top_k}")
    logger.info(f"Protection Factor: {protection_factor}")
    logger.info(f"总组合数: {len(alpha_list)} × {len(tau_base_list)} × {len(lambda_list)} = {len(alpha_list) * len(tau_base_list) * len(lambda_list)}")
    logger.info("目标: p-MRR > 0.154 且 nDCG 不崩盘")
    logger.info("="*70)

    engine = GridSearchEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        seed=seed
    )

    logger.info("\n📊 加载评测数据...")
    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    q_raw_og, q_raw_changed = engine.data_loader.load_raw_queries()

    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)

    cached_data = None
    if use_cache:
        cached_data = load_cached_embeddings(
            cache_dir or DEFAULT_CACHE_DIR, task_name, model_name
        )

    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        if set(cached_doc_ids) == set(all_doc_ids):
            logger.info(f"✅ 使用缓存的文档向量 ({len(cached_doc_ids)} 个)")
            engine.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
        else:
            logger.warning(f"⚠️ 缓存文档ID不匹配，重新编码...")
            engine._encode_documents(corpus, all_doc_ids)
    else:
        engine._encode_documents(corpus, all_doc_ids)

    logger.info("🔤 加载并编码 dual queries...")
    dual_queries_cache = os.path.join(
        "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v4",
        f"dual_queries_v4_{task_name}.jsonl"
    )

    q_plus_og = {}
    q_plus_changed = {}
    q_minus_changed = {}
    neg_mask_dict = {}

    if os.path.exists(dual_queries_cache):
        logger.info(f"📂 加载 dual queries: {dual_queries_cache}")
        with open(dual_queries_cache, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item['qid']
                q_plus = item['q_plus']
                q_minus = item['q_minus']
                if item['query_type'] == 'og':
                    q_plus_og[qid] = q_plus
                else:
                    q_plus_changed[qid] = q_plus
                    q_minus_changed[qid] = q_minus
                    neg_mask_dict[qid] = 0.0 if q_minus == "[NONE]" else 1.0
        logger.info(f"✅ 加载完成: {len(q_plus_og)} OG, {len(q_plus_changed)} Changed")
    else:
        logger.warning(f"⚠️ 未找到 dual queries 缓存")
        q_plus_og = q_og
        q_plus_changed = q_changed
        q_minus_changed = q_changed
        neg_mask_dict = {qid: 1.0 for qid in q_changed}

    q_og_items = list(q_plus_og.items())
    q_og_list = [item[1] for item in q_og_items]
    q_og_emb = engine.encoder.encode_queries(q_og_list, batch_size=batch_size)
    q_og_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_og_items)}

    q_changed_items = list(q_plus_changed.items())
    q_plus_list = [item[1] for item in q_changed_items]
    q_minus_list = [q_minus_changed.get(item[0], item[1]) for item in q_changed_items]
    q_plus_changed_emb = engine.encoder.encode_queries(q_plus_list, batch_size=batch_size)
    q_minus_changed_emb = engine.encoder.encode_queries(q_minus_list, batch_size=batch_size)
    q_changed_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_changed_items)}
    neg_mask = torch.tensor([neg_mask_dict.get(item[0], 1.0) for item in q_changed_items], dtype=torch.float32)

    all_results = []

    for alpha in alpha_list:
        for tau_base in tau_base_list:
            for lambda_weight in lambda_list:
                logger.info(f"\n🔍 测试: α={alpha:.1f}, τ_base={tau_base:.1f}, λ={lambda_weight:.1f}")

                result = engine._evaluate_params_pat_protected(
                    q_og_emb=q_og_emb,
                    q_plus_changed_emb=q_plus_changed_emb,
                    q_minus_changed_emb=q_minus_changed_emb,
                    candidates=candidates,
                    corpus=corpus,
                    alpha=alpha,
                    tau_base=tau_base,
                    lambda_weight=lambda_weight,
                    q_og_qid_to_idx=q_og_qid_to_idx,
                    q_changed_qid_to_idx=q_changed_qid_to_idx,
                    neg_mask=neg_mask,
                    top_k=top_k,
                    protection_factor=protection_factor
                )

                all_results.append({
                    'alpha': alpha,
                    'tau_base': tau_base,
                    'lambda': lambda_weight,
                    'p_mrr': result['p_mrr'],
                    'og_ndcg@5': result['og_ndcg@5'],
                    'changed_ndcg@5': result['changed_ndcg@5'],
                    'og_mrr': result['og_mrr'],
                    'changed_mrr': result['changed_mrr']
                })

                logger.info(f"   p-MRR={result['p_mrr']:.4f}, Changed nDCG@5={result['changed_ndcg@5']:.4f}")

    _print_pat_summary_table(all_results)

    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "pat_protected_grid_search_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'alpha_list': alpha_list,
            'tau_base_list': tau_base_list,
            'lambda_list': lambda_list,
            'top_k': top_k,
            'protection_factor': protection_factor,
            'results': all_results
        }, f, indent=2)
    logger.info(f"\n💾 PAT Protected 结果已保存: {results_file}")

    csv_file = os.path.join(output_dir, "pat_protected_grid_search_results.csv")
    with open(csv_file, 'w') as f:
        f.write("alpha,tau_base,lambda,p_mrr,og_ndcg@5,changed_ndcg@5,og_mrr,changed_mrr\n")
        for r in all_results:
            f.write(f"{r['alpha']:.4f},{r['tau_base']:.4f},{r['lambda']:.4f},"
                   f"{r['p_mrr']:.4f},{r['og_ndcg@5']:.4f},{r['changed_ndcg@5']:.4f},"
                   f"{r['og_mrr']:.4f},{r['changed_mrr']:.4f}\n")
    logger.info(f"💾 CSV 报告已保存: {csv_file}")

    return {
        'results': all_results,
        'best_p_mrr': max(all_results, key=lambda x: x['p_mrr']),
        'best_ndcg': max(all_results, key=lambda x: x['changed_ndcg@5'])
    }


def run_pat_hybrid_grid_search_evaluation(
    model_name: str,
    task_name: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: str = None,
    use_cache: bool = True,
    alpha_list: list = None,
    tau_base_list: list = None,
    lambda_list: list = None,
    top_k: int = 5,
    protection_factor: float = 0.0,
    boost_ndcg_factor: float = 0.5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    PAT 极限 Grid Search 评测入口 (混合策略版)

    混合策略:
    - og_rank <= top_k: 保护文档，penalty * protection_factor (默认完全不惩罚)
    - og_rank 6-20: 轻微 boost (帮助进入 top-5)
    """
    import os
    import sys
    import json
    import logging
    import random
    import numpy as np
    from typing import List, Optional

    sys.path.insert(0, '/home/luwa/Documents/DSCLR')
    from eval.engine_grid_search import (
        GridSearchEngine, load_cached_embeddings, DEFAULT_CACHE_DIR
    )

    if alpha_list is None:
        alpha_list = [3.0, 4.0, 5.0, 6.0]
    if tau_base_list is None:
        tau_base_list = [0.4, 0.5]
    if lambda_list is None:
        lambda_list = [0.3, 0.5, 0.7]

    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*70)
    logger.info("🚀 PAT (混合策略) Extreme Grid Search")
    logger.info("="*70)
    logger.info(f"Alpha 列表: {alpha_list}")
    logger.info(f"Tau_base 列表: {tau_base_list}")
    logger.info(f"Lambda 列表: {lambda_list}")
    logger.info(f"Top_k 保护阈值: {top_k}")
    logger.info(f"Protection Factor: {protection_factor}")
    logger.info(f"Boost Factor: {boost_ndcg_factor}")
    logger.info(f"总组合数: {len(alpha_list)} × {len(tau_base_list)} × {len(lambda_list)} = {len(alpha_list) * len(tau_base_list) * len(lambda_list)}")
    logger.info("目标: p-MRR > 0.154 且 nDCG 不崩盘")
    logger.info("="*70)

    engine = GridSearchEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        seed=seed
    )

    logger.info("\n📊 加载评测数据...")
    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    q_raw_og, q_raw_changed = engine.data_loader.load_raw_queries()

    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)

    cached_data = None
    if use_cache:
        cached_data = load_cached_embeddings(
            cache_dir or DEFAULT_CACHE_DIR, task_name, model_name
        )

    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        if set(cached_doc_ids) == set(all_doc_ids):
            logger.info(f"✅ 使用缓存的文档向量 ({len(cached_doc_ids)} 个)")
            engine.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
        else:
            logger.warning(f"⚠️ 缓存文档ID不匹配，重新编码...")
            engine._encode_documents(corpus, all_doc_ids)
    else:
        engine._encode_documents(corpus, all_doc_ids)

    logger.info("🔤 加载并编码 dual queries...")
    dual_queries_cache = os.path.join(
        "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v4",
        f"dual_queries_v4_{task_name}.jsonl"
    )

    q_plus_og = {}
    q_plus_changed = {}
    q_minus_changed = {}
    neg_mask_dict = {}

    if os.path.exists(dual_queries_cache):
        logger.info(f"📂 加载 dual queries: {dual_queries_cache}")
        with open(dual_queries_cache, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item['qid']
                q_plus = item['q_plus']
                q_minus = item['q_minus']
                if item['query_type'] == 'og':
                    q_plus_og[qid] = q_plus
                else:
                    q_plus_changed[qid] = q_plus
                    q_minus_changed[qid] = q_minus
                    neg_mask_dict[qid] = 0.0 if q_minus == "[NONE]" else 1.0
        logger.info(f"✅ 加载完成: {len(q_plus_og)} OG, {len(q_plus_changed)} Changed")
    else:
        logger.warning(f"⚠️ 未找到 dual queries 缓存")
        q_plus_og = q_og
        q_plus_changed = q_changed
        q_minus_changed = q_changed
        neg_mask_dict = {qid: 1.0 for qid in q_changed}

    q_og_items = list(q_plus_og.items())
    q_og_list = [item[1] for item in q_og_items]
    q_og_emb = engine.encoder.encode_queries(q_og_list, batch_size=batch_size)
    q_og_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_og_items)}

    q_changed_items = list(q_plus_changed.items())
    q_plus_list = [item[1] for item in q_changed_items]
    q_minus_list = [q_minus_changed.get(item[0], item[1]) for item in q_changed_items]
    q_plus_changed_emb = engine.encoder.encode_queries(q_plus_list, batch_size=batch_size)
    q_minus_changed_emb = engine.encoder.encode_queries(q_minus_list, batch_size=batch_size)
    q_changed_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_changed_items)}
    neg_mask = torch.tensor([neg_mask_dict.get(item[0], 1.0) for item in q_changed_items], dtype=torch.float32)

    all_results = []

    for alpha in alpha_list:
        for tau_base in tau_base_list:
            for lambda_weight in lambda_list:
                logger.info(f"\n🔍 测试: α={alpha:.1f}, τ_base={tau_base:.1f}, λ={lambda_weight:.1f}")

                result = engine._evaluate_params_pat_hybrid(
                    q_og_emb=q_og_emb,
                    q_plus_changed_emb=q_plus_changed_emb,
                    q_minus_changed_emb=q_minus_changed_emb,
                    candidates=candidates,
                    corpus=corpus,
                    alpha=alpha,
                    tau_base=tau_base,
                    lambda_weight=lambda_weight,
                    q_og_qid_to_idx=q_og_qid_to_idx,
                    q_changed_qid_to_idx=q_changed_qid_to_idx,
                    neg_mask=neg_mask,
                    top_k=top_k,
                    protection_factor=protection_factor,
                    boost_ndcg_factor=boost_ndcg_factor
                )

                all_results.append({
                    'alpha': alpha,
                    'tau_base': tau_base,
                    'lambda': lambda_weight,
                    'p_mrr': result['p_mrr'],
                    'og_ndcg@5': result['og_ndcg@5'],
                    'changed_ndcg@5': result['changed_ndcg@5'],
                    'og_mrr': result['og_mrr'],
                    'changed_mrr': result['changed_mrr']
                })

                logger.info(f"   p-MRR={result['p_mrr']:.4f}, Changed nDCG@5={result['changed_ndcg@5']:.4f}")

    _print_pat_summary_table(all_results)

    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "pat_hybrid_grid_search_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'alpha_list': alpha_list,
            'tau_base_list': tau_base_list,
            'lambda_list': lambda_list,
            'top_k': top_k,
            'protection_factor': protection_factor,
            'boost_ndcg_factor': boost_ndcg_factor,
            'results': all_results
        }, f, indent=2)
    logger.info(f"\n💾 PAT Hybrid 结果已保存: {results_file}")

    csv_file = os.path.join(output_dir, "pat_hybrid_grid_search_results.csv")
    with open(csv_file, 'w') as f:
        f.write("alpha,tau_base,lambda,p_mrr,og_ndcg@5,changed_ndcg@5,og_mrr,changed_mrr\n")
        for r in all_results:
            f.write(f"{r['alpha']:.4f},{r['tau_base']:.4f},{r['lambda']:.4f},"
                   f"{r['p_mrr']:.4f},{r['og_ndcg@5']:.4f},{r['changed_ndcg@5']:.4f},"
                   f"{r['og_mrr']:.4f},{r['changed_mrr']:.4f}\n")
    logger.info(f"💾 CSV 报告已保存: {csv_file}")

    return {
        'results': all_results,
        'best_p_mrr': max(all_results, key=lambda x: x['p_mrr']),
        'best_ndcg': max(all_results, key=lambda x: x['changed_ndcg@5'])
    }


def _print_pat_summary_table(results: List[Dict]):
    """打印 PAT Grid Search 结果汇总表格 (Markdown 格式)"""
    print("\n" + "="*80)
    print("🎯 PAT Extreme Grid Search 结果汇总")
    print("="*80)

    # Markdown 表格头
    print("\n| Alpha | Tau_base | Lambda | p-MRR | Changed nDCG@5 | OG nDCG@5 |")
    print("|-------|----------|--------|-------|----------------|-----------|")

    # 按 p_mrr 排序
    for r in sorted(results, key=lambda x: -x['p_mrr']):
        marker = " ★" if r['p_mrr'] > 0.154 else ""
        print(f"| {r['alpha']:5.1f} | {r['tau_base']:8.1f} | {r['lambda']:6.1f} | "
              f"{r['p_mrr']:7.4f} | {r['changed_ndcg@5']:14.4f} | {r['og_ndcg@5']:9.4f} |{marker}")

    # 找出最佳组合
    valid_results = [r for r in results if r.get('p_mrr', 0) > 0]
    if valid_results:
        best_p_mrr = max(valid_results, key=lambda x: x['p_mrr'])
        best_ndcg = max(valid_results, key=lambda x: x['changed_ndcg@5'])

        print("\n" + "="*80)
        print("🏆 最佳组合:")
        print(f"   最高 p-MRR:          α={best_p_mrr['alpha']:.1f}, τ_base={best_p_mrr['tau_base']:.1f}, λ={best_p_mrr['lambda']:.1f} → {best_p_mrr['p_mrr']:.4f}")
        print(f"   最高 Changed nDCG@5: α={best_ndcg['alpha']:.1f}, τ_base={best_ndcg['tau_base']:.1f}, λ={best_ndcg['lambda']:.1f} → {best_ndcg['changed_ndcg@5']:.4f}")

        # 检查是否有组合超过基线
        above_baseline = [r for r in valid_results if r['p_mrr'] > 0.154]
        if above_baseline:
            print(f"\n🎉 找到 {len(above_baseline)} 个组合超过基线 0.154!")
            for r in sorted(above_baseline, key=lambda x: -x['p_mrr']):
                print(f"   α={r['alpha']:.1f}, τ_base={r['tau_base']:.1f}, λ={r['lambda']:.1f} → p-MRR={r['p_mrr']:.4f}")
        else:
            print(f"\n⚠️ 未找到超过基线 0.154 的组合")

        print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="PAT 极限 Grid Search 评测")
    parser.add_argument("--model_name", type=str, default="castorini/repllama-v1-7b-lora-passage")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--alpha_list", type=str, default="3.0,4.0,5.0,6.0")
    parser.add_argument("--tau_base_list", type=str, default="0.4,0.5")
    parser.add_argument("--lambda_list", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    alpha_list = [float(x) for x in args.alpha_list.split(",")]
    tau_base_list = [float(x) for x in args.tau_base_list.split(",")]
    lambda_list = [float(x) for x in args.lambda_list.split(",")]

    results = run_pat_grid_search_evaluation(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        alpha_list=alpha_list,
        tau_base_list=tau_base_list,
        lambda_list=lambda_list,
        seed=args.seed
    )
