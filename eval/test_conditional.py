#!/usr/bin/env python
"""测试条件惩罚策略"""

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


def dsclr_conditional_penalty_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    threshold: float = 0.7,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Conditional: 条件惩罚
    
    核心思想：
    - 只惩罚 S_base < threshold 的文档（低质量文档）
    - 高质量文档（S_base >= threshold）不受惩罚
    
    公式：
    - 如果 S_base >= threshold: penalty = 0
    - 如果 S_base < threshold: penalty = alpha * Softplus(S_neg - tau)
    """
    overflow = S_neg - tau.unsqueeze(1)
    
    soft_penalty = torch.log(1 + torch.exp(beta * overflow)) / beta
    
    condition_mask = (S_base < threshold).float()
    
    penalty = alpha * condition_mask * soft_penalty
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_base - penalty
    
    return S_final, penalty


def dsclr_gradual_protection_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    threshold: float = 0.7,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Gradual-Protection: 渐进保护
    
    核心思想：
    - 高质量文档（S_base 高）受到更多保护
    - 低质量文档（S_base 低）受到更少保护
    
    公式：penalty = alpha * (1 - sigmoid((S_base - threshold) * 10)) * Softplus(S_neg - tau)
    """
    overflow = S_neg - tau.unsqueeze(1)
    
    soft_penalty = torch.log(1 + torch.exp(beta * overflow)) / beta
    
    protection = torch.sigmoid((S_base - threshold) * 10)
    
    penalty = alpha * (1 - protection) * soft_penalty
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_base - penalty
    
    return S_final, penalty


def dsclr_smart_penalty_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DSCLR-Smart: 智能惩罚
    
    核心思想：
    - 如果 S_base >> S_neg（好文档，负向感低）：不惩罚
    - 如果 S_base ≈ S_neg（可疑文档）：惩罚
    - 如果 S_base << S_neg（坏文档，负向感高）：重罚
    
    公式：penalty = alpha * sigmoid(S_neg - S_base) * Softplus(S_neg - tau)
    """
    overflow = S_neg - tau.unsqueeze(1)
    
    soft_penalty = torch.log(1 + torch.exp(beta * overflow)) / beta
    
    suspicion = torch.sigmoid((S_neg - S_base) * 10)
    
    penalty = alpha * suspicion * soft_penalty
    
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
            embeddings = np.load(cache_file)
            with open(ids_file, 'r') as f:
                doc_ids = json.load(f)
            return torch.tensor(embeddings, dtype=torch.float32), doc_ids
        except:
            pass
    
    return None


class DSCLRConditionalEvaluator:
    """DSCLR Conditional 评估器"""
    
    def __init__(
        self,
        model_name: str,
        task_name: str,
        device: str = "cuda:0",
        batch_size: int = 128,
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
        if cached:
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
        method: str = "conditional",
        alpha: float = 1.0,
        delta: float = 0.0,
        beta: float = 50.0,
        threshold: float = 0.7,
        top_k: int = 1000
    ) -> Dict:
        corpus_embeddings, doc_ids = self.encode_corpus(self.corpus)
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
            S_og = torch.mm(og_embeddings, corpus_embeddings.t())
            S_base = torch.mm(original_embeddings, corpus_embeddings.t())
            S_neg = torch.mm(q_minus_embeddings, corpus_embeddings.t())
            
            S_final_og = S_og
            
            if method == "conditional":
                S_final_changed, penalty = dsclr_conditional_penalty_score(
                    S_base, S_neg, tau, alpha, beta, threshold, neg_mask=neg_mask.unsqueeze(1)
                )
            elif method == "gradual":
                S_final_changed, penalty = dsclr_gradual_protection_score(
                    S_base, S_neg, tau, alpha, beta, threshold, neg_mask=neg_mask.unsqueeze(1)
                )
            elif method == "smart":
                S_final_changed, penalty = dsclr_smart_penalty_score(
                    S_base, S_neg, tau, alpha, beta, neg_mask=neg_mask.unsqueeze(1)
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        
        results_og = self._extract_results(S_final_og, query_ids_og, doc_ids, top_k)
        results_changed = self._extract_results(S_final_changed, query_ids_changed, doc_ids, top_k)
        
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "method": method,
            "alpha": alpha,
            "delta": delta,
            "beta": beta,
            "threshold": threshold,
            "map_at_1000": metrics["changed"]["map_at_1000"],
            "ndcg_at_5": metrics["changed"]["ndcg_at_5"],
            "p_mrr": metrics["p-MRR"],
        }
    
    def encode_corpus(self, corpus: Dict[str, Dict[str, str]]) -> Tuple[torch.Tensor, List[str]]:
        if self.corpus_embeddings is not None and self.doc_ids is not None:
            return self.corpus_embeddings, self.doc_ids
        
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[doc_id]['text'] for doc_id in doc_ids]
        
        if hasattr(self.encoder, 'encode_documents'):
            embeddings = self.encoder.encode_documents(doc_texts)
        else:
            embeddings = self.encoder.encode_corpus(doc_texts)
        
        return torch.tensor(embeddings, device=self.device), doc_ids
    
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
    evaluator = DSCLRConditionalEvaluator(
        model_name="samaya-ai/RepLLaMA-reproduced",
        task_name="Core17InstructionRetrieval",
        device="cuda:0",
        batch_size=16,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    
    output_dir = "eval/results/core17_conditional"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print("\n" + "="*60)
    print("测试 Conditional Penalty")
    print("="*60)
    for alpha in [1.0, 2.0, 3.0]:
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            for delta in [0.0, -0.05]:
                result = evaluator.evaluate(
                    method="conditional",
                    alpha=alpha,
                    delta=delta,
                    beta=50,
                    threshold=threshold
                )
                results.append(result)
                print(f"  alpha={alpha}, threshold={threshold}, delta={delta}: MAP={result['map_at_1000']:.4f}, p-MRR={result['p_mrr']:.4f}")
    
    print("\n" + "="*60)
    print("测试 Gradual Protection")
    print("="*60)
    for alpha in [1.0, 2.0]:
        for threshold in [0.5, 0.6, 0.7]:
            for delta in [0.0, -0.05]:
                result = evaluator.evaluate(
                    method="gradual",
                    alpha=alpha,
                    delta=delta,
                    beta=50,
                    threshold=threshold
                )
                results.append(result)
                print(f"  alpha={alpha}, threshold={threshold}, delta={delta}: MAP={result['map_at_1000']:.4f}, p-MRR={result['p_mrr']:.4f}")
    
    print("\n" + "="*60)
    print("测试 Smart Penalty")
    print("="*60)
    for alpha in [1.0, 2.0, 3.0]:
        for delta in [0.0, -0.05, -0.1]:
            result = evaluator.evaluate(
                method="smart",
                alpha=alpha,
                delta=delta,
                beta=50
            )
            results.append(result)
            print(f"  alpha={alpha}, delta={delta}: MAP={result['map_at_1000']:.4f}, p-MRR={result['p_mrr']:.4f}")
    
    output_file = os.path.join(output_dir, "grid_search_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\n" + "="*60)
    print("🏆 Top 10 by MAP:")
    print("="*60)
    sorted_by_map = sorted(results, key=lambda x: x['map_at_1000'], reverse=True)
    for r in sorted_by_map[:10]:
        threshold = r.get('threshold', 'N/A')
        print(f"  {r['method']}: alpha={r['alpha']}, delta={r['delta']}, threshold={threshold}: MAP={r['map_at_1000']:.4f}, p-MRR={r['p_mrr']:.4f}")
    
    print("\n" + "="*60)
    print("🏆 Top 10 by p-MRR (MAP > 0.23):")
    print("="*60)
    filtered = [r for r in results if r['map_at_1000'] > 0.23]
    sorted_by_pmrr = sorted(filtered, key=lambda x: x['p_mrr'], reverse=True)
    for r in sorted_by_pmrr[:10]:
        threshold = r.get('threshold', 'N/A')
        print(f"  {r['method']}: alpha={r['alpha']}, delta={r['delta']}, threshold={threshold}: MAP={r['map_at_1000']:.4f}, p-MRR={r['p_mrr']:.4f}")
    
    logger.info(f"\n💾 结果已保存: {output_file}")
