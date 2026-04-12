"""
批量评估多种 DSCLR 改进方案
同时测试 V2, V3, Hybrid 等方法
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/home/luwa/Documents/DSCLR')

from eval.models.encoder import ModelFactory
from eval.metrics.evaluator import FollowIREvaluator
from model.dsclr_scoring import (
    dsclr_micro_v2_score,
    dsclr_micro_v3_score,
    dsclr_hybrid_score,
    analyze_score_distribution
)


def get_model_cache_dir(cache_dir: str, model_name: str) -> str:
    if "RepLLaMA-reproduced" in model_name:
        return os.path.join(cache_dir, "RepLLaMA_reproduced")
    model_name_short = model_name.split('/')[-1].lower().replace('-', '_')
    return os.path.join(cache_dir, model_name_short)


def get_model_name_short(model_name: str) -> str:
    if "RepLLaMA-reproduced" in model_name:
        return "RepLLaMA_reproduced"
    return model_name.split('/')[-1].lower().replace('-', '_')


def load_cached_embeddings(cache_dir: str, task_name: str, model_name: str) -> Optional[Tuple[torch.Tensor, List[str]]]:
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    model_name_short = get_model_name_short(model_name)
    cache_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")
    
    if os.path.exists(cache_file) and os.path.exists(ids_file):
        logger.info(f"📂 加载缓存的文档向量: {cache_file}")
        
        try:
            embeddings = np.load(cache_file)
            with open(ids_file, 'r') as f:
                doc_ids = json.load(f)
            logger.info(f"✅ 缓存加载成功: {len(doc_ids)} 个文档, shape={embeddings.shape}")
            return torch.tensor(embeddings, dtype=torch.float32), doc_ids
        except:
            try:
                data = np.load(cache_file, allow_pickle=True)
                if data.dtype == np.object_ and len(data.shape) == 0:
                    embedding_dict = data.item()
                    with open(ids_file, 'r') as f:
                        doc_ids = json.load(f)
                    
                    embeddings_list = []
                    for doc_id in doc_ids:
                        if doc_id in embedding_dict:
                            embeddings_list.append(embedding_dict[doc_id])
                    
                    if embeddings_list:
                        embeddings = torch.stack([torch.tensor(e, dtype=torch.float32) for e in embeddings_list])
                        logger.info(f"✅ 缓存加载成功 (dict格式): {len(doc_ids)} 个文档, shape={embeddings.shape}")
                        return embeddings, doc_ids
            except Exception as e:
                logger.warning(f"⚠️ 缓存加载失败: {e}")
    
    logger.info(f"⚠️ 未找到缓存: {cache_file}")
    return None


class MultiMethodEvaluator:
    def __init__(
        self,
        model_name: str,
        task_name: str,
        device: str = "cuda:0",
        batch_size: int = 64,
        cache_dir: str = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        logger.info(f"🚀 初始化评估器: {model_name} on {device}")
        
        self.encoder = ModelFactory.create(
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )
        
        self.evaluator = FollowIREvaluator(task_name=task_name)
        
        from model.reformulator import QueryReformulator
        self.reformulator = QueryReformulator(
            task_name=task_name,
            use_cache=True,
            cache_dir="/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v5"
        )
        
        self.corpus_embeddings = None
        self.doc_ids = None
        
        self._load_data()
    
    def _load_data(self):
        self.queries_og, self.queries_changed = self.evaluator.data_loader.load_queries()
        self.corpus = self.evaluator.data_loader.load_corpus()
        self.top_ranked = self.evaluator.data_loader.load_qrel_diff()
        
        self.raw_queries_og, self.raw_queries_changed = self.evaluator.data_loader.load_raw_queries()
        
        cached = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)
        if cached is not None:
            self.corpus_embeddings, self.doc_ids = cached
            self.corpus_embeddings = self.corpus_embeddings.to(self.device)
            logger.info(f"   ✅ 使用缓存的文档嵌入")
        else:
            raise ValueError("请先缓存文档向量")
    
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
                qid=qid,
                idx=idx,
                query=query_text,
                instruction=instruction,
                query_type=query_type
            )
            
            query_ids.append(qid)
            q_plus_list.append(q_plus)
            q_minus_list.append(q_minus)
            q_original_list.append(q_original)
        
        neg_mask = torch.tensor(
            [0.0 if qm == "[NONE]" or not qm.strip() else 1.0 for qm in q_minus_list],
            device=self.device,
            dtype=torch.float32
        )
        
        return q_plus_list, q_minus_list, q_original_list, neg_mask, query_ids
    
    def parse_qplus_qminus(self, query_text: str) -> Tuple[str, str]:
        if "Q+:" in query_text and "Q-:" in query_text:
            parts = query_text.split("Q-:")
            q_plus_part = parts[0].replace("Q+:", "").strip()
            q_minus = parts[1].strip() if len(parts) > 1 else "[NONE]"
        elif "Q+:" in query_text:
            q_plus_part = query_text.replace("Q+:", "").strip()
            q_minus = "[NONE]"
        else:
            q_plus_part = query_text
            q_minus = "[NONE]"
        
        return q_plus_part, q_minus
    
    def compute_dynamic_tau(self, q_plus_emb: torch.Tensor, q_minus_emb: torch.Tensor, 
                           original_emb: torch.Tensor, delta: float = -0.15) -> torch.Tensor:
        cos_qplus_qminus = torch.nn.functional.cosine_similarity(q_plus_emb, q_minus_emb, dim=-1)
        cos_orig_qminus = torch.nn.functional.cosine_similarity(original_emb, q_minus_emb, dim=-1)
        tau = torch.minimum(cos_qplus_qminus, cos_orig_qminus) + delta
        return tau
    
    def _extract_results(self, scores: torch.Tensor, query_ids: List[str], 
                        doc_ids: List[str], top_k: int = 1000) -> Dict[str, Dict[str, float]]:
        results = {}
        for i, qid in enumerate(query_ids):
            query_scores = scores[i].cpu().numpy()
            top_indices = np.argsort(query_scores)[::-1][:top_k]
            results[qid] = {doc_ids[idx]: float(query_scores[idx]) for idx in top_indices}
        return results
    
    def evaluate_v2(self, alpha: float, margin: float, max_penalty: float) -> Dict:
        logger.info(f"🧠 DSCLR-Micro V2 评估 (alpha={alpha}, margin={margin}, max_penalty={max_penalty})...")
        
        og_embeddings, query_ids_og = self.encode_queries(self.queries_og)
        changed_embeddings, query_ids_changed = self.encode_queries(self.queries_changed)
        
        q_plus_list, q_minus_list, original_query_list, neg_mask, _ = self._prepare_dual_queries(
            self.queries_changed, self.raw_queries_changed
        )
        
        q_plus_embeddings = torch.tensor(
            self.encoder.encode_queries(q_plus_list),
            device=self.device,
            dtype=torch.float32
        )
        q_minus_embeddings = torch.tensor(
            self.encoder.encode_queries(q_minus_list),
            device=self.device,
            dtype=torch.float32
        )
        original_embeddings = torch.tensor(
            self.encoder.encode_queries(original_query_list),
            device=self.device,
            dtype=torch.float32
        )
        
        with torch.no_grad():
            S_og = torch.mm(og_embeddings, self.corpus_embeddings.t())
            S_base = torch.mm(original_embeddings, self.corpus_embeddings.t())
            S_neg = torch.mm(q_minus_embeddings, self.corpus_embeddings.t())
            
            S_final_og = S_og
            S_final_changed, penalty_ratio = dsclr_micro_v2_score(
                S_base, S_neg, alpha, margin, 20.0, max_penalty, neg_mask.unsqueeze(1)
            )
        
        results_og = self._extract_results(S_final_og, query_ids_og, self.doc_ids)
        results_changed = self._extract_results(S_final_changed, query_ids_changed, self.doc_ids)
        
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "method": "v2",
            "alpha": alpha,
            "margin": margin,
            "max_penalty": max_penalty,
            "map_at_1000": metrics["changed"]["map_at_1000"],
            "ndcg_at_5": metrics["changed"]["ndcg_at_5"],
            "p_mrr": metrics["p-MRR"]
        }
    
    def evaluate_v3(self, alpha: float, margin: float, max_penalty: float, delta: float) -> Dict:
        logger.info(f"🧠 DSCLR-Micro V3 评估 (alpha={alpha}, margin={margin}, delta={delta})...")
        
        og_embeddings, query_ids_og = self.encode_queries(self.queries_og)
        changed_embeddings, query_ids_changed = self.encode_queries(self.queries_changed)
        
        q_plus_list, q_minus_list, original_query_list, neg_mask, _ = self._prepare_dual_queries(
            self.queries_changed, self.raw_queries_changed
        )
        
        q_plus_embeddings = torch.tensor(
            self.encoder.encode_queries(q_plus_list),
            device=self.device,
            dtype=torch.float32
        )
        q_minus_embeddings = torch.tensor(
            self.encoder.encode_queries(q_minus_list),
            device=self.device,
            dtype=torch.float32
        )
        original_embeddings = torch.tensor(
            self.encoder.encode_queries(original_query_list),
            device=self.device,
            dtype=torch.float32
        )
        
        tau = self.compute_dynamic_tau(q_plus_embeddings, q_minus_embeddings, original_embeddings, delta)
        
        with torch.no_grad():
            S_og = torch.mm(og_embeddings, self.corpus_embeddings.t())
            S_base = torch.mm(original_embeddings, self.corpus_embeddings.t())
            S_neg = torch.mm(q_minus_embeddings, self.corpus_embeddings.t())
            
            S_final_og = S_og
            S_final_changed, penalty_ratio = dsclr_micro_v3_score(
                S_base, S_neg, tau, alpha, margin, 20.0, max_penalty, neg_mask.unsqueeze(1)
            )
        
        results_og = self._extract_results(S_final_og, query_ids_og, self.doc_ids)
        results_changed = self._extract_results(S_final_changed, query_ids_changed, self.doc_ids)
        
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "method": "v3",
            "alpha": alpha,
            "margin": margin,
            "max_penalty": max_penalty,
            "delta": delta,
            "map_at_1000": metrics["changed"]["map_at_1000"],
            "ndcg_at_5": metrics["changed"]["ndcg_at_5"],
            "p_mrr": metrics["p-MRR"]
        }
    
    def evaluate_hybrid(self, alpha: float, margin: float, max_penalty: float, delta: float) -> Dict:
        logger.info(f"🧠 DSCLR-Hybrid 评估 (alpha={alpha}, margin={margin}, delta={delta})...")
        
        og_embeddings, query_ids_og = self.encode_queries(self.queries_og)
        changed_embeddings, query_ids_changed = self.encode_queries(self.queries_changed)
        
        q_plus_list, q_minus_list, original_query_list, neg_mask, _ = self._prepare_dual_queries(
            self.queries_changed, self.raw_queries_changed
        )
        
        q_plus_embeddings = torch.tensor(
            self.encoder.encode_queries(q_plus_list),
            device=self.device,
            dtype=torch.float32
        )
        q_minus_embeddings = torch.tensor(
            self.encoder.encode_queries(q_minus_list),
            device=self.device,
            dtype=torch.float32
        )
        original_embeddings = torch.tensor(
            self.encoder.encode_queries(original_query_list),
            device=self.device,
            dtype=torch.float32
        )
        
        tau = self.compute_dynamic_tau(q_plus_embeddings, q_minus_embeddings, original_embeddings, delta)
        
        with torch.no_grad():
            S_og = torch.mm(og_embeddings, self.corpus_embeddings.t())
            S_base = torch.mm(original_embeddings, self.corpus_embeddings.t())
            S_neg = torch.mm(q_minus_embeddings, self.corpus_embeddings.t())
            
            S_final_og = S_og
            S_final_changed, penalty_ratio = dsclr_hybrid_score(
                S_base, S_neg, tau, alpha, margin, 20.0, max_penalty, neg_mask.unsqueeze(1)
            )
        
        results_og = self._extract_results(S_final_og, query_ids_og, self.doc_ids)
        results_changed = self._extract_results(S_final_changed, query_ids_changed, self.doc_ids)
        
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "method": "hybrid",
            "alpha": alpha,
            "margin": margin,
            "max_penalty": max_penalty,
            "delta": delta,
            "map_at_1000": metrics["changed"]["map_at_1000"],
            "ndcg_at_5": metrics["changed"]["ndcg_at_5"],
            "p_mrr": metrics["p-MRR"]
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--output_dir", type=str, default="eval/results/multi_method_test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = MultiMethodEvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        device=args.device,
        batch_size=args.batch_size
    )
    
    results = []
    
    logger.info("\n" + "="*60)
    logger.info("测试 V2 方法: 降低alpha + 负margin + 惩罚上限")
    logger.info("="*60)
    
    v2_configs = [
        {"alpha": 0.3, "margin": -0.1, "max_penalty": 0.3},
        {"alpha": 0.5, "margin": -0.1, "max_penalty": 0.4},
        {"alpha": 0.5, "margin": -0.15, "max_penalty": 0.3},
        {"alpha": 0.7, "margin": -0.1, "max_penalty": 0.5},
        {"alpha": 0.3, "margin": -0.05, "max_penalty": 0.3},
    ]
    
    for config in v2_configs:
        result = evaluator.evaluate_v2(**config)
        results.append(result)
        logger.info(f"   MAP={result['map_at_1000']:.4f}, nDCG={result['ndcg_at_5']:.4f}, pMRR={result['p_mrr']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("测试 V3 方法: 结合动态τ机制")
    logger.info("="*60)
    
    v3_configs = [
        {"alpha": 0.3, "margin": -0.05, "max_penalty": 0.3, "delta": -0.15},
        {"alpha": 0.5, "margin": -0.05, "max_penalty": 0.4, "delta": -0.15},
        {"alpha": 0.5, "margin": -0.1, "max_penalty": 0.3, "delta": -0.1},
        {"alpha": 0.7, "margin": -0.05, "max_penalty": 0.5, "delta": -0.2},
    ]
    
    for config in v3_configs:
        result = evaluator.evaluate_v3(**config)
        results.append(result)
        logger.info(f"   MAP={result['map_at_1000']:.4f}, nDCG={result['ndcg_at_5']:.4f}, pMRR={result['p_mrr']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("测试 Hybrid 方法: 经典减法 + 差分惩罚")
    logger.info("="*60)
    
    hybrid_configs = [
        {"alpha": 0.2, "margin": -0.05, "max_penalty": 0.2, "delta": -0.15},
        {"alpha": 0.3, "margin": -0.05, "max_penalty": 0.3, "delta": -0.15},
        {"alpha": 0.3, "margin": -0.1, "max_penalty": 0.3, "delta": -0.1},
        {"alpha": 0.5, "margin": -0.05, "max_penalty": 0.4, "delta": -0.2},
    ]
    
    for config in hybrid_configs:
        result = evaluator.evaluate_hybrid(**config)
        results.append(result)
        logger.info(f"   MAP={result['map_at_1000']:.4f}, nDCG={result['ndcg_at_5']:.4f}, pMRR={result['p_mrr']:.4f}")
    
    results.sort(key=lambda x: x["map_at_1000"], reverse=True)
    
    output_file = os.path.join(args.output_dir, "multi_method_results.json")
    with open(output_file, 'w') as f:
        json.dump({"results": results, "best": results[0]}, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("🏆 最佳结果")
    logger.info("="*60)
    best = results[0]
    logger.info(f"方法: {best['method']}")
    logger.info(f"参数: {best}")
    logger.info(f"MAP@1000: {best['map_at_1000']:.4f}")
    logger.info(f"nDCG@5: {best['ndcg_at_5']:.4f}")
    logger.info(f"pMRR: {best['p_mrr']:.4f}")
    logger.info(f"\n💾 结果已保存: {output_file}")


if __name__ == "__main__":
    main()
