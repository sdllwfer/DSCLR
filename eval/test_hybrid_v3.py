#!/usr/bin/env python
"""测试 DSCLR-Hybrid V3"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dsclr_scoring import dsclr_hybrid_v3_score, analyze_score_distribution
from eval.models import ModelFactory
from eval.metrics.evaluator import DataLoader, FollowIREvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"


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


class DSCLRHybridV3Evaluator:
    """DSCLR-Hybrid V3 评估器"""
    
    def __init__(
        self,
        model_name: str,
        task_name: str,
        device: str = "cuda:0",
        batch_size: int = 128,
        cache_dir: str = None,
        sbase_mode: str = "original"
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.sbase_mode = sbase_mode
        
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
        logger.info(f"📂 加载数据: {self.task_name}")
        
        self.queries_og, self.queries_changed = self.data_loader.load_queries()
        self.qrels = self.data_loader.load_qrels()
        self.qrel_diff = self.data_loader.load_qrel_diff()
        self.corpus = self.data_loader.load_corpus()
        self.candidates = self.data_loader.load_candidates()
        
        self.raw_queries_og, self.raw_queries_changed = self.data_loader.load_raw_queries()
        
        logger.info(f"   OG 查询: {len(self.queries_og)}")
        logger.info(f"   Changed 查询: {len(self.queries_changed)}")
        logger.info(f"   文档数: {len(self.corpus)}")
        
        cached = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)
        if cached:
            self.corpus_embeddings, self.doc_ids = cached
            self.corpus_embeddings = self.corpus_embeddings.to(self.device)
            logger.info(f"   ✅ 使用缓存的文档嵌入")
    
    def encode_queries(self, queries: Dict[str, str]) -> Tuple[torch.Tensor, List[str]]:
        """编码查询"""
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        embeddings = self.encoder.encode_queries(query_texts)
        
        return torch.tensor(embeddings, device=self.device, dtype=torch.float32), query_ids
    
    def encode_corpus(self, corpus: Dict[str, Dict[str, str]]) -> Tuple[torch.Tensor, List[str]]:
        """编码文档"""
        if self.corpus_embeddings is not None and self.doc_ids is not None:
            return self.corpus_embeddings, self.doc_ids
        
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[doc_id]['text'] for doc_id in doc_ids]
        
        if hasattr(self.encoder, 'encode_documents'):
            embeddings = self.encoder.encode_documents(doc_texts)
        else:
            embeddings = self.encoder.encode_corpus(doc_texts)
        
        return torch.tensor(embeddings, device=self.device), doc_ids
    
    def _prepare_dual_queries(
        self,
        queries: Dict[str, str],
        raw_queries: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str], List[str], torch.Tensor, List[str]]:
        """使用 reformulator 准备 Q+ 和 Q-"""
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
    
    def compute_dynamic_tau(
        self,
        original_embeddings: torch.Tensor,
        q_plus_embeddings: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        delta: float = -0.15
    ) -> torch.Tensor:
        """计算动态 τ"""
        cos_orig_neg = F.cosine_similarity(original_embeddings, q_minus_embeddings)
        cos_plus_neg = F.cosine_similarity(q_plus_embeddings, q_minus_embeddings)
        
        tau = torch.minimum(cos_orig_neg, cos_plus_neg) + delta
        
        return tau
    
    def evaluate(
        self,
        alpha: float = 1.0,
        margin: float = -0.15,
        delta: float = -0.15,
        top_k: int = 1000
    ) -> Dict:
        """评估"""
        logger.info(f"🔍 编码文档...")
        corpus_embeddings, doc_ids = self.encode_corpus(self.corpus)
        
        logger.info(f"🔍 编码 OG 查询...")
        og_embeddings, query_ids_og = self.encode_queries(self.queries_og)
        
        logger.info(f"🔍 编码 Changed 查询...")
        changed_embeddings, query_ids_changed = self.encode_queries(self.queries_changed)
        
        q_plus_list, q_minus_list, original_query_list, neg_mask, _ = self._prepare_dual_queries(
            self.queries_changed, self.raw_queries_changed
        )
        
        logger.info(f"🔍 编码 Q+ 和 Q-...")
        logger.info(f"   Q- 有效数: {neg_mask.sum().item():.0f} / {len(q_minus_list)}")
        
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
        
        tau = self.compute_dynamic_tau(
            original_embeddings, q_plus_embeddings, q_minus_embeddings, delta
        )
        logger.info(f"   动态 τ 范围: [{tau.min().item():.4f}, {tau.max().item():.4f}]")
        
        logger.info(f"🧠 DSCLR-Hybrid V3 评估 (alpha={alpha}, margin={margin}, delta={delta})...")
        
        with torch.no_grad():
            S_og = torch.mm(og_embeddings, corpus_embeddings.t())
            
            if self.sbase_mode == "original":
                S_base = torch.mm(original_embeddings, corpus_embeddings.t())
            else:
                S_base = torch.mm(q_plus_embeddings, corpus_embeddings.t())
            
            S_neg = torch.mm(q_minus_embeddings, corpus_embeddings.t())
            
            S_final_og = S_og
            
            S_final_changed, penalty = dsclr_hybrid_v3_score(
                S_base, S_neg, tau, alpha, margin, neg_mask=neg_mask.unsqueeze(1)
            )
            
            distribution = analyze_score_distribution(S_base, S_neg, margin)
            logger.info(f"   得分分布: protected={distribution['protected_ratio']:.2%}, "
                       f"penalized={distribution['penalized_ratio']:.2%}")
        
        results_og = self._extract_results(S_final_og, query_ids_og, doc_ids, top_k)
        results_changed = self._extract_results(S_final_changed, query_ids_changed, doc_ids, top_k)
        
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "alpha": alpha,
            "margin": margin,
            "delta": delta,
            "map_at_1000": metrics["changed"]["map_at_1000"],
            "ndcg_at_5": metrics["changed"]["ndcg_at_5"],
            "p_mrr": metrics["p-MRR"],
            "tau_range": [tau.min().item(), tau.max().item()],
            "distribution": distribution
        }
    
    def _extract_results(
        self,
        scores: torch.Tensor,
        query_ids: List[str],
        doc_ids: List[str],
        top_k: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """提取 Top-K 结果"""
        results = {}
        for i, qid in enumerate(query_ids):
            query_scores = scores[i].cpu().numpy()
            top_indices = np.argsort(query_scores)[::-1][:top_k]
            results[qid] = {doc_ids[idx]: float(query_scores[idx]) for idx in top_indices}
        return results
    
    def grid_search(
        self,
        alphas: List[float],
        margins: List[float],
        deltas: List[float],
        top_k: int = 1000
    ) -> Dict:
        """网格搜索"""
        logger.info(f"🔍 开始网格搜索...")
        logger.info(f"   Alpha 范围: {alphas}")
        logger.info(f"   Margin 范围: {margins}")
        logger.info(f"   Delta 范围: {deltas}")
        
        results = []
        best_map = 0
        best_params = None
        
        for alpha in alphas:
            for margin in margins:
                for delta in deltas:
                    result = self.evaluate(alpha=alpha, margin=margin, delta=delta, top_k=top_k)
                    results.append(result)
                    
                    logger.info(f"   alpha={alpha}, margin={margin}, delta={delta}: "
                               f"MAP={result['map_at_1000']:.4f}, "
                               f"p-MRR={result['p_mrr']:.4f}")
                    
                    if result['map_at_1000'] > best_map:
                        best_map = result['map_at_1000']
                        best_params = {"alpha": alpha, "margin": margin, "delta": delta}
        
        return {
            "results": results,
            "best_params": best_params,
            "best_map": best_map
        }


if __name__ == "__main__":
    evaluator = DSCLRHybridV3Evaluator(
        model_name="samaya-ai/RepLLaMA-reproduced",
        task_name="Core17InstructionRetrieval",
        device="cuda:0",
        batch_size=64,
        cache_dir=DEFAULT_CACHE_DIR,
        sbase_mode="original"
    )
    
    output_dir = "eval/results/core17_hybrid_v3"
    os.makedirs(output_dir, exist_ok=True)
    
    alphas = [0.5, 1.0, 1.5, 2.0]
    margins = [-0.2, -0.15, -0.1, 0.0]
    deltas = [-0.2, -0.15, -0.1]
    
    results = evaluator.grid_search(alphas=alphas, margins=margins, deltas=deltas)
    
    output_file = os.path.join(output_dir, "grid_search_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"\n💾 结果已保存: {output_file}")
    logger.info(f"\n{'='*60}")
    logger.info(f"🏆 最佳参数: alpha={results['best_params']['alpha']}, "
               f"margin={results['best_params']['margin']}, "
               f"delta={results['best_params']['delta']}")
    logger.info(f"   最佳 MAP@1000: {results['best_map']:.4f}")
    logger.info(f"{'='*60}")
