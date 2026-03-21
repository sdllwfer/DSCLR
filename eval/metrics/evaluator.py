"""
评估指标计算模块
实现 FollowIR 相关的评测指标计算
使用 MTEB 官方接口确保计算正确性
"""

import logging
import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')

import mteb
import datasets
from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change

logger = logging.getLogger(__name__)


class DataLoader:
    """FollowIR 数据加载器 - 专门用于评测"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.dataset_path = self._get_dataset_path(task_name)
        
    def _get_dataset_path(self, task_name: str) -> str:
        path_map = {
            "Core17InstructionRetrieval": "jhu-clsp/core17-instructions-mteb",
            "Robust04InstructionRetrieval": "jhu-clsp/robust04-instructions-mteb",
            "News21InstructionRetrieval": "jhu-clsp/news21-instructions-mteb",
        }
        return path_map.get(task_name, "")
    
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """加载 qrels - 相关性标签"""
        ds_qrels = datasets.load_dataset(self.dataset_path, 'default')
        q_split = 'test' if 'test' in ds_qrels else list(ds_qrels.keys())[0]
        qrels = {}
        for item in ds_qrels[q_split]:
            qid = item.get('query-id', item.get('query_id', ''))
            doc_id = str(item.get('corpus-id', item.get('doc_id', '')))
            relevance = int(item.get('score', item.get('relevance', 1)))
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = relevance
        
        logger.info(f"   ✅ 加载 qrels: {len(qrels)} 个查询")
        return qrels
    
    def load_qrel_diff(self) -> Dict[str, List[str]]:
        """加载 qrel_diff - 记录哪些文档的相关性发生了变化"""
        ds_diff = datasets.load_dataset(self.dataset_path, 'qrel_diff')
        diff_splits = [k for k in ds_diff.keys() if 'qrel' in k or 'diff' in k]
        d_split = diff_splits[0] if diff_splits else list(ds_diff.keys())[0]
        changed_qrels = {}
        for item in ds_diff[d_split]:
            qid = item.get('query-id', item.get('query_id', ''))
            corpus_ids = item.get('corpus-ids', item.get('results', []))
            if corpus_ids:
                changed_qrels[qid] = [str(cid) for cid in corpus_ids]
        
        logger.info(f"   ✅ 加载 qrel_diff: {len(changed_qrels)} 个查询有变化的文档")
        return changed_qrels
    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        """加载文档集合"""
        ds_c = datasets.load_dataset(self.dataset_path, 'corpus')
        c_split = 'corpus' if 'corpus' in ds_c else 'train'
        corpus = {}
        for d in ds_c[c_split]:
            doc_id = str(d.get('_id', d.get('id')))
            corpus[doc_id] = {'text': str(d.get('text', ''))}
        
        logger.info(f"   ✅ 加载 {len(corpus)} 个文档")
        return corpus
    
    def load_queries(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """加载查询和指令"""
        ds_q = datasets.load_dataset(self.dataset_path, 'queries')
        q_split = 'queries' if 'queries' in ds_q else 'train'
        
        ds_inst = datasets.load_dataset(self.dataset_path, 'instruction')
        i_split = 'instruction' if 'instruction' in ds_inst else 'train'
        
        instruction_dict = {}
        for item in ds_inst[i_split]:
            qid = str(item.get('query-id', ''))
            instruction_dict[qid] = str(item.get('instruction', ''))
        
        q_og = {}
        q_changed = {}
        
        for q in ds_q[q_split]:
            full_qid = str(q.get('_id', q.get('id', '')))
            query_text = q.get('text', '')
            inst = instruction_dict.get(full_qid, "")
            
            if full_qid.endswith('-og'):
                q_og[full_qid] = f"{query_text} {inst}".strip()
            elif full_qid.endswith('-changed'):
                q_changed[full_qid] = f"{query_text} {inst}".strip()
        
        logger.info(f"   ✅ 加载 {len(q_og)} 个 og 查询, {len(q_changed)} 个 changed 查询")
        return q_og, q_changed
    
    def load_raw_queries(self):
        """加载原始 query 和 instruction，用于 reformulator
        
        Returns:
            (og_queries, changed_queries) 其中每个是 {qid: (query, instruction)} 的字典
        """
        ds_q = datasets.load_dataset(self.dataset_path, 'queries')
        q_split = 'queries' if 'queries' in ds_q else 'train'
        
        ds_inst = datasets.load_dataset(self.dataset_path, 'instruction')
        i_split = 'instruction' if 'instruction' in ds_inst else 'train'
        
        instruction_dict = {}
        for item in ds_inst[i_split]:
            qid = str(item.get('query-id', ''))
            instruction_dict[qid] = str(item.get('instruction', ''))
        
        q_og_raw = {}
        q_changed_raw = {}
        
        for q in ds_q[q_split]:
            full_qid = str(q.get('_id', q.get('id', '')))
            query_text = q.get('text', '')
            inst = instruction_dict.get(full_qid, "")
            
            if full_qid.endswith('-og'):
                q_og_raw[full_qid] = (query_text, inst)
            elif full_qid.endswith('-changed'):
                q_changed_raw[full_qid] = (query_text, inst)
        
        return q_og_raw, q_changed_raw
    
    def load_candidates(self) -> Dict[str, List[str]]:
        """加载候选文档"""
        ds_top = datasets.load_dataset(self.dataset_path, 'top_ranked')
        available_splits = list(ds_top.keys())
        t_split = available_splits[0] if available_splits else None
        
        candidates = {}
        if t_split:
            for item in ds_top[t_split]:
                full_qid = str(item.get('query-id', item.get('query_id', item.get('qid', ''))))
                base_qid = full_qid.replace('-og', '').replace('-changed', '')
                results_list = item.get('corpus-ids', item.get('results', []))
                
                if base_qid not in candidates:
                    candidates[base_qid] = [str(did) for did in results_list]
        
        if candidates:
            avg_cand = sum(len(v) for v in candidates.values()) / len(candidates)
            logger.info(f"   ✅ 加载 {len(candidates)} 个查询的候选文档, 平均 {avg_cand:.0f} 个/查询")
        
        return candidates


class FollowIREvaluator:
    """FollowIR 评测器 - 使用 MTEB 官方接口"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.data_loader = DataLoader(task_name)
        
    def evaluate(
        self,
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """计算 FollowIR 所有指标
        
        使用 MTEB 官方的 evaluate_p_mrr_change 函数确保正确性
        """
        logger.info("📊 计算 FollowIR 评测指标...")
        
        qrels = self.data_loader.load_qrels()
        changed_qrels = self.data_loader.load_qrel_diff()
        
        results = {**results_og, **results_changed}
        
        k_values = [1, 3, 5, 10, 100, 1000]
        
        scores = evaluate_p_mrr_change(
            qrels=qrels,
            results=results,
            changed_qrels=changed_qrels,
            k_values=k_values,
        )
        
        metrics = {
            "p-MRR": scores.get('p-MRR', 0),
            "original": {
                "ndcg_at_1": scores.get('og', {}).get('ndcg_at_1', 0),
                "ndcg_at_5": scores.get('og', {}).get('ndcg_at_5', 0),
                "ndcg_at_10": scores.get('og', {}).get('ndcg_at_10', 0),
                "map_at_1000": scores.get('og', {}).get('map_at_1000', 0),
            },
            "changed": {
                "ndcg_at_1": scores.get('changed', {}).get('ndcg_at_1', 0),
                "ndcg_at_5": scores.get('changed', {}).get('ndcg_at_5', 0),
                "ndcg_at_10": scores.get('changed', {}).get('ndcg_at_10', 0),
                "map_at_1000": scores.get('changed', {}).get('map_at_1000', 0),
            },
            "full_scores": scores
        }
        
        logger.info(f"   p-MRR: {metrics['p-MRR']:.4f}")
        logger.info(f"   og nDCG@5: {metrics['original']['ndcg_at_5']:.4f}")
        logger.info(f"   changed nDCG@5: {metrics['changed']['ndcg_at_5']:.4f}")
        
        return metrics


class MetricsRegistry:
    """指标注册表"""
    
    _metrics: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, metric_class: type):
        cls._metrics[name] = metric_class
    
    @classmethod
    def create(cls, task_name: str) -> FollowIREvaluator:
        return FollowIREvaluator(task_name)


MetricsRegistry.register("followir", FollowIREvaluator)
