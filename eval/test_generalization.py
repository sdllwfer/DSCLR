#!/usr/bin/env python
"""跨数据集泛化性测试 - 使用统一参数"""

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

TASKS = ["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"]


def dsclr_double_threshold_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    margin: float = 0.05,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    overflow = S_neg - tau.unsqueeze(1)
    soft_penalty = torch.log(1 + torch.exp(beta * overflow)) / beta
    
    protection_threshold = tau.unsqueeze(1) - margin
    protection_mask = (S_neg < protection_threshold).float()
    
    penalty = alpha * (1 - protection_mask) * soft_penalty
    
    if neg_mask is not None:
        penalty = penalty * neg_mask
    
    S_final = S_base - penalty
    return S_final, penalty


def dsclr_softplus_score(
    S_base: torch.Tensor,
    S_neg: torch.Tensor,
    tau: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 50.0,
    neg_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    overflow = S_neg - tau.unsqueeze(1)
    penalty = alpha * torch.log(1 + torch.exp(beta * overflow)) / beta
    
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
            data = np.load(cache_file, allow_pickle=True)
            with open(ids_file, 'r') as f:
                doc_ids = json.load(f)
            
            if data.ndim == 0:
                data = data.item()
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
            else:
                embeddings = data.astype(np.float32)
            
            logger.info(f"   ✅ 使用缓存的文档嵌入: {cache_file}")
            return torch.tensor(embeddings, dtype=torch.float32), doc_ids
        except Exception as e:
            logger.warning(f"   ⚠️ 加载缓存失败: {e}")
    
    return None


class GeneralizationEvaluator:
    """跨数据集泛化性评估器"""
    
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
            raise ValueError(f"请先缓存 {self.task_name} 的文档向量")
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
        method: str = "double_threshold",
        alpha: float = 1.0,
        delta: float = 0.0,
        beta: float = 50.0,
        margin: float = 0.05,
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
            
            if method == "double_threshold":
                S_final_changed, penalty = dsclr_double_threshold_score(
                    S_base, S_neg, tau, alpha, beta, margin, neg_mask=neg_mask.unsqueeze(1)
                )
            elif method == "softplus":
                S_final_changed, penalty = dsclr_softplus_score(
                    S_base, S_neg, tau, alpha, beta, neg_mask=neg_mask.unsqueeze(1)
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        
        results_og = self._extract_results(S_final_og, query_ids_og, self.doc_ids, top_k)
        results_changed = self._extract_results(S_final_changed, query_ids_changed, self.doc_ids, top_k)
        
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "task": self.task_name,
            "method": method,
            "alpha": alpha,
            "delta": delta,
            "beta": beta,
            "margin": margin,
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


def get_baseline(task_name: str) -> Dict:
    """获取基线结果（原始 RepLLaMA）"""
    baseline_file = f"/home/luwa/Documents/DSCLR/eval/results/baselines/{task_name}_baseline.json"
    if os.path.exists(baseline_file):
        with open(baseline_file) as f:
            return json.load(f)
    return {"map_at_1000": 0.0, "p_mrr": 0.0}


def main():
    model_name = "samaya-ai/RepLLaMA-reproduced"
    device = "cuda:0"
    batch_size = 16
    
    param_configs = [
        {"method": "double_threshold", "alpha": 1.0, "delta": 0.0, "beta": 50, "margin": 0.02, "desc": "最高MAP"},
        {"method": "double_threshold", "alpha": 2.0, "delta": 0.0, "beta": 50, "margin": 0.02, "desc": "平衡点"},
        {"method": "double_threshold", "alpha": 2.0, "delta": 0.0, "beta": 50, "margin": 0.05, "desc": "平衡点v2"},
        {"method": "softplus", "alpha": 1.0, "delta": -0.1, "beta": 20, "margin": 0.05, "desc": "最高p-MRR"},
        {"method": "softplus", "alpha": 1.0, "delta": 0.0, "beta": 20, "margin": 0.05, "desc": "Softplus基线"},
        {"method": "softplus", "alpha": 1.5, "delta": 0.0, "beta": 50, "margin": 0.05, "desc": "Softplus平衡"},
    ]
    
    all_results = {}
    
    for task_name in TASKS:
        print(f"\n{'='*70}")
        print(f"📊 测试数据集: {task_name}")
        print(f"{'='*70}")
        
        try:
            evaluator = GeneralizationEvaluator(
                model_name=model_name,
                task_name=task_name,
                device=device,
                batch_size=batch_size,
                cache_dir=DEFAULT_CACHE_DIR,
            )
            
            task_results = []
            
            for config in param_configs:
                print(f"\n  测试: {config['desc']} ({config['method']}, α={config['alpha']}, δ={config['delta']})")
                
                result = evaluator.evaluate(
                    method=config["method"],
                    alpha=config["alpha"],
                    delta=config["delta"],
                    beta=config["beta"],
                    margin=config.get("margin", 0.05)
                )
                result["desc"] = config["desc"]
                task_results.append(result)
                
                print(f"    MAP={result['map_at_1000']:.4f}, p-MRR={result['p_mrr']:.4f}")
            
            all_results[task_name] = task_results
            
        except Exception as e:
            logger.error(f"  ❌ 测试 {task_name} 失败: {e}")
            all_results[task_name] = []
    
    output_dir = "eval/results/generalization_test"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "generalization_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print("\n" + "="*70)
    print("📊 泛化性分析总结")
    print("="*70)
    
    print("\n基线对比 (FollowIR 论文):")
    print("-"*70)
    baselines = {
        "Core17InstructionRetrieval": {"map": 0.23, "ndcg": 0.0, "p_mrr": 0.0, "primary": "map"},
        "Robust04InstructionRetrieval": {"map": 0.283, "ndcg": 0.0, "p_mrr": 0.0, "primary": "map"},
        "News21InstructionRetrieval": {"map": 0.0, "ndcg": 0.285, "p_mrr": 0.0, "primary": "ndcg"},
    }
    
    for task, baseline in baselines.items():
        if baseline["primary"] == "map":
            print(f"  {task}: change-MAP >= {baseline['map']}")
        else:
            print(f"  {task}: change-nDCG@5 >= {baseline['ndcg']}")
    
    print("\n" + "-"*70)
    print("各配置在三个数据集上的表现:")
    print("-"*70)
    
    for config in param_configs:
        print(f"\n📌 {config['desc']} ({config['method']}, α={config['alpha']}, δ={config['delta']}, margin={config.get('margin', 0.05)})")
        
        all_pass = True
        for task_name in TASKS:
            if task_name in all_results and all_results[task_name]:
                result = all_results[task_name][param_configs.index(config)]
                baseline = baselines.get(task_name, {"map": 0, "ndcg": 0, "p_mrr": 0, "primary": "map"})
                
                if baseline["primary"] == "map":
                    primary_improve = result['map_at_1000'] >= baseline['map']
                    primary_value = result['map_at_1000']
                    primary_target = baseline['map']
                    primary_name = "MAP"
                else:
                    primary_improve = result['ndcg_at_5'] >= baseline['ndcg']
                    primary_value = result['ndcg_at_5']
                    primary_target = baseline['ndcg']
                    primary_name = "nDCG@5"
                
                pmrr_improve = result['p_mrr'] > 0
                
                status = "✅" if (primary_improve and pmrr_improve) else "❌"
                if not (primary_improve and pmrr_improve):
                    all_pass = False
                
                print(f"    {task_name[:20]:20s}: {primary_name}={primary_value:.4f} (target>={primary_target:.3f}) {'✅' if primary_improve else '❌'}, "
                      f"p-MRR={result['p_mrr']:.4f} {'✅' if pmrr_improve else '❌'} {status}")
        
        if all_pass:
            print(f"    🎉 该配置在所有数据集上都超越基线！")
    
    logger.info(f"\n💾 结果已保存: {output_file}")


if __name__ == "__main__":
    main()
