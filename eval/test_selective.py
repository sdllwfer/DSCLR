#!/usr/bin/env python
"""测试选择性惩罚策略 - 只惩罚真正有害的文档"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.models import ModelFactory
from eval.metrics.evaluator import DataLoader, FollowIREvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"


def dsclr_selective_penalty_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    base_threshold: float = 0.5,
    neg_boost: float = 0.1,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Selective: 选择性惩罚
    
    核心思想：
    - 只惩罚"真正有害"的文档：S_base 低（本身不相关）且 S_neg 高（与负向约束相似）
    - 使用组合条件：penalty_weight = sigmoid((S_neg - S_base - margin) * scale)
    
    公式：penalty = alpha * penalty_weight * Softplus(S_neg - tau)
    """
    overflow = S_neg - tau.unsqueeze(1)
    soft_penalty = torch.log(1 + torch.exp(beta * overflow)) / beta
    
    harm_score = S_neg - S_base + neg_boost
    penalty_weight = torch.sigmoid(harm_score * 10)
    
    low_base_mask = torch.sigmoid((base_threshold - S_base) * 10)
    
    penalty = alpha * penalty_weight * low_base_mask * soft_penalty
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_base - penalty
    
    return S_final, penalty


def dsclr_harm_aware_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    gamma: float = 1.0,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Harm-Aware: 危害感知惩罚
    
    核心思想：
    - 计算每个文档的"危害分数" = S_neg - S_base
    - 危害分数越高，惩罚越重
    - 使用指数函数放大高危害文档的惩罚
    
    公式：penalty = alpha * exp(gamma * max(0, S_neg - S_base)) * Softplus(S_neg - tau)
    """
    overflow = S_neg - tau.unsqueeze(1)
    soft_penalty = torch.log(1 + torch.exp(beta * overflow)) / beta
    
    harm = torch.relu(S_neg - S_base)
    harm_weight = torch.exp(gamma * harm)
    
    penalty = alpha * harm_weight * soft_penalty
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_base - penalty
    
    return S_final, penalty


def dsclr_triple_condition_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    base_threshold: float = 0.5,
    neg_threshold: float = 0.3,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Triple-Condition: 三重条件惩罚
    
    核心思想：
    - 只惩罚同时满足三个条件的文档：
      1. S_base < base_threshold（本身不相关）
      2. S_neg > neg_threshold（与负向约束相似）
      3. S_neg > tau（超过动态阈值）
    
    公式：penalty = alpha * condition_mask * Softplus(S_neg - tau)
    """
    overflow = S_neg - tau.unsqueeze(1)
    soft_penalty = torch.log(1 + torch.exp(beta * overflow)) / beta
    
    cond1 = (S_base < base_threshold).float()
    cond2 = (S_neg > neg_threshold).float()
    cond3 = (S_neg > tau.unsqueeze(1)).float()
    
    condition_mask = cond1 * cond2 * cond3
    
    penalty = alpha * condition_mask * soft_penalty
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_base - penalty
    
    return S_final, penalty


def get_model_cache_dir(base_cache_dir: str, model_name: str) -> str:
    if "mistral" in model_name.lower():
        model_subdir = "e5-mistral-7b"
    elif "bge" in model_name.lower():
        model_subdir = "bge-large-en"
    else:
        model_subdir = model_name.split("/")[-1].replace("-", "_")
    return os.path.join(base_cache_dir, model_subdir)


def get_model_name_short(model_name: str) -> str:
    if "mistral" in model_name.lower():
        return "e5-mistral-7b"
    elif "bge" in model_name.lower():
        return "bge-large-en"
    return model_name.split("/")[-1].replace("-", "_")


def load_cached_embeddings(cache_dir: str, task_name: str, model_name: str) -> Optional[Tuple[torch.Tensor, List[str]]]:
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    model_name_short = get_model_name_short(model_name)
    cache_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")
    
    if os.path.exists(cache_file) and os.path.exists(ids_file):
        try:
            data = np.load(cache_file, allow_pickle=True).item()
            with open(ids_file, 'r') as f:
                doc_ids = json.load(f)
            
            embeddings = []
            for doc_id in doc_ids:
                if doc_id in data:
                    emb = data[doc_id]
                    if hasattr(emb, 'numpy'):
                        emb = emb.numpy()
                    embeddings.append(emb)
                else:
                    raise ValueError(f"Missing embedding for {doc_id}")
            
            embeddings = np.array(embeddings, dtype=np.float32)
            logger.info(f"   ✅ 使用缓存的文档嵌入: {cache_file}")
            return torch.tensor(embeddings, dtype=torch.float32), doc_ids
        except Exception as e:
            logger.warning(f"   ⚠️ 加载缓存失败: {e}")
    
    return None


class DSCLRSelectiveEvaluator:
    """DSCLR Selective 评估器"""
    
    def __init__(
        self,
        model_name: str,
        task_name: str,
        device: str = "cuda:0",
        batch_size: int = 16,
        cache_dir: str = None,
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        
        self.encoder = ModelFactory.create(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        
        self.data_loader = DataLoader(task_name)
        self.evaluator = FollowIREvaluator(task_name)
        
        from model.reformulator import QueryReformulator
        self.reformulator = QueryReformulator(
            task_name=task_name,
            use_cache=True,
            cache_dir="/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v5"
        )
        
        self.queries_og = None
        self.queries_changed = None
        self.qrels = None
        self.qrel_diff = None
        self.corpus = None
        self.candidates = None
        
        self.raw_queries_og = None
        self.raw_queries_changed = None
        
        self.corpus_embeddings = None
        self.doc_ids = None
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        self.queries_og, self.queries_changed = self.data_loader.load_queries()
        self.qrels = self.data_loader.load_qrels()
        self.qrel_diff = self.data_loader.load_qrel_diff()
        self.corpus = self.data_loader.load_corpus()
        self.candidates = self.data_loader.load_candidates()
        self.raw_queries_og, self.raw_queries_changed = self.data_loader.load_raw_queries()
        
        cached = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)
        if cached is None:
            raise ValueError("请先缓存文档向量")
        self.corpus_embeddings, self.doc_ids = cached
        self.corpus_embeddings = self.corpus_embeddings.to(self.device)
    
    def encode_queries(self, queries: Dict[str, str]) -> Tuple[torch.Tensor, List[str]]:
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        embeddings = self.encoder.encode_queries(query_texts)
        return torch.tensor(embeddings, device=self.device, dtype=torch.float32), query_ids
    
    def _prepare_dual_queries(
        self,
        queries: Dict[str, str],
        raw_queries: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str], List[str], torch.Tensor, List[str]]:
        query_ids = []
        q_plus_list = []
        q_minus_list = []
        q_original_list = []
        
        for qid in queries.keys():
            raw = raw_queries.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            q_original = f"{query_text} {instruction}".strip() if query_text else queries.get(qid, "")
            
            try:
                idx = int(qid.split('-')[0])
            except:
                idx = 0
            
            query_type = "og" if qid.endswith("-og") else "changed"
            q_plus, q_minus = self.reformulator.reformulate(
                qid=qid, idx=idx, query=query_text, instruction=instruction, query_type=query_type
            )
            
            query_ids.append(qid)
            q_plus_list.append(q_plus)
            q_minus_list.append(q_minus)
            q_original_list.append(q_original)
        
        neg_mask = torch.tensor(
            [0.0 if qm == "[NONE]" or not qm.strip() else 1.0 for qm in q_minus_list],
            device=self.device, dtype=torch.float32
        )
        
        return q_plus_list, q_minus_list, q_original_list, neg_mask, query_ids
    
    def compute_dynamic_tau(
        self,
        original_embeddings: torch.Tensor,
        q_plus_embeddings: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        delta: float = 0.0
    ) -> torch.Tensor:
        cos_orig_neg = F.cosine_similarity(original_embeddings, q_minus_embeddings)
        cos_plus_neg = F.cosine_similarity(q_plus_embeddings, q_minus_embeddings)
        return torch.minimum(cos_orig_neg, cos_plus_neg) + delta
    
    def evaluate(
        self,
        method: str = "selective",
        alpha: float = 1.0,
        delta: float = 0.0,
        beta: float = 50.0,
        base_threshold: float = 0.5,
        neg_boost: float = 0.1,
        gamma: float = 1.0,
        neg_threshold: float = 0.3,
        top_k: int = 1000
    ) -> Dict:
        og_embeddings, query_ids_og = self.encode_queries(self.queries_og)
        changed_embeddings, query_ids_changed = self.encode_queries(self.queries_changed)
        
        q_plus_list, q_minus_list, original_query_list, neg_mask, _ = self._prepare_dual_queries(
            self.queries_changed, self.raw_queries_changed
        )
        
        q_plus_embeddings = torch.tensor(
            self.encoder.encode_queries(q_plus_list), device=self.device, dtype=torch.float32
        )
        q_minus_embeddings = torch.tensor(
            self.encoder.encode_queries(q_minus_list), device=self.device, dtype=torch.float32
        )
        original_embeddings = torch.tensor(
            self.encoder.encode_queries(original_query_list), device=self.device, dtype=torch.float32
        )
        
        tau = self.compute_dynamic_tau(
            original_embeddings, q_plus_embeddings, q_minus_embeddings, delta
        )
        
        with torch.no_grad():
            S_og = torch.mm(og_embeddings, self.corpus_embeddings.t())
            S_base = torch.mm(original_embeddings, self.corpus_embeddings.t())
            S_neg = torch.mm(q_minus_embeddings, self.corpus_embeddings.t())
            
            S_final_og = S_og
            
            if method == "selective":
                S_final_changed, penalty = dsclr_selective_penalty_score(
                    S_base, S_neg, tau, alpha, beta, base_threshold, neg_boost, neg_mask=neg_mask.unsqueeze(1)
                )
            elif method == "harm_aware":
                S_final_changed, penalty = dsclr_harm_aware_score(
                    S_base, S_neg, tau, alpha, beta, gamma, neg_mask=neg_mask.unsqueeze(1)
                )
            elif method == "triple":
                S_final_changed, penalty = dsclr_triple_condition_score(
                    S_base, S_neg, tau, alpha, beta, base_threshold, neg_threshold, neg_mask=neg_mask.unsqueeze(1)
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        
        results_og = self._extract_results(S_final_og, query_ids_og, self.doc_ids, top_k)
        results_changed = self._extract_results(S_final_changed, query_ids_changed, self.doc_ids, top_k)
        
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "method": method,
            "alpha": alpha,
            "delta": delta,
            "beta": beta,
            "base_threshold": base_threshold,
            "neg_boost": neg_boost,
            "gamma": gamma,
            "neg_threshold": neg_threshold,
            "map_at_1000": metrics["changed"]["map_at_1000"],
            "ndcg_at_5": metrics["changed"]["ndcg_at_5"],
            "p_mrr": metrics["p-MRR"],
        }
    
    def _extract_results(
        self,
        scores: torch.Tensor,
        query_ids: List[str],
        doc_ids: List[str],
        top_k: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for i, qid in enumerate(query_ids):
            query_scores = scores[i].cpu().numpy()
            top_indices = np.argsort(query_scores)[::-1][:top_k]
            results[qid] = {doc_ids[idx]: float(query_scores[idx]) for idx in top_indices}
        return results


if __name__ == "__main__":
    evaluator = DSCLRSelectiveEvaluator(
        model_name="samaya-ai/RepLLaMA-reproduced",
        task_name="Core17InstructionRetrieval",
        device="cuda:0",
        batch_size=16,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    
    output_dir = "eval/results/core17_selective"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print("\n" + "="*60)
    print("测试 Selective Penalty")
    print("="*60)
    for alpha in [1.0, 2.0, 3.0]:
        for base_threshold in [0.3, 0.4, 0.5, 0.6]:
            for neg_boost in [0.0, 0.1, 0.2]:
                for delta in [0.0, -0.05]:
                    result = evaluator.evaluate(
                        method="selective",
                        alpha=alpha,
                        delta=delta,
                        beta=50,
                        base_threshold=base_threshold,
                        neg_boost=neg_boost
                    )
                    results.append(result)
                    print(f"  alpha={alpha}, base_th={base_threshold}, neg_boost={neg_boost}, delta={delta}: MAP={result['map_at_1000']:.4f}, p-MRR={result['p_mrr']:.4f}")
    
    print("\n" + "="*60)
    print("测试 Harm-Aware Penalty")
    print("="*60)
    for alpha in [0.5, 1.0, 1.5]:
        for gamma in [0.5, 1.0, 2.0, 3.0]:
            for delta in [0.0, -0.05]:
                result = evaluator.evaluate(
                    method="harm_aware",
                    alpha=alpha,
                    delta=delta,
                    beta=50,
                    gamma=gamma
                )
                results.append(result)
                print(f"  alpha={alpha}, gamma={gamma}, delta={delta}: MAP={result['map_at_1000']:.4f}, p-MRR={result['p_mrr']:.4f}")
    
    print("\n" + "="*60)
    print("测试 Triple-Condition Penalty")
    print("="*60)
    for alpha in [1.0, 2.0, 3.0]:
        for base_threshold in [0.3, 0.4, 0.5]:
            for neg_threshold in [0.2, 0.3, 0.4]:
                for delta in [0.0, -0.05]:
                    result = evaluator.evaluate(
                        method="triple",
                        alpha=alpha,
                        delta=delta,
                        beta=50,
                        base_threshold=base_threshold,
                        neg_threshold=neg_threshold
                    )
                    results.append(result)
                    print(f"  alpha={alpha}, base_th={base_threshold}, neg_th={neg_threshold}, delta={delta}: MAP={result['map_at_1000']:.4f}, p-MRR={result['p_mrr']:.4f}")
    
    output_file = os.path.join(output_dir, "grid_search_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\n" + "="*60)
    print("🏆 Top 10 by MAP:")
    print("="*60)
    sorted_by_map = sorted(results, key=lambda x: x['map_at_1000'], reverse=True)
    for r in sorted_by_map[:10]:
        print(f"  {r['method']}: alpha={r['alpha']}, delta={r['delta']}: MAP={r['map_at_1000']:.4f}, p-MRR={r['p_mrr']:.4f}")
    
    print("\n" + "="*60)
    print("🏆 Top 10 by p-MRR (MAP > 0.23):")
    print("="*60)
    filtered = [r for r in results if r['map_at_1000'] > 0.23]
    sorted_by_pmrr = sorted(filtered, key=lambda x: x['p_mrr'], reverse=True)
    for r in sorted_by_pmrr[:10]:
        print(f"  {r['method']}: alpha={r['alpha']}, delta={r['delta']}: MAP={r['map_at_1000']:.4f}, p-MRR={r['p_mrr']:.4f}")
    
    logger.info(f"\n💾 结果已保存: {output_file}")
