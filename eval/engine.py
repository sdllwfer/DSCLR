"""
FollowIR 评测引擎
整合数据加载、模型编码、检索和指标计算
"""

import os
import sys
import logging
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from eval.models import ModelFactory, DenseRetriever
from eval.metrics import DataLoader as MetricsDataLoader, FollowIREvaluator
from eval.output import OutputManager

logger = logging.getLogger(__name__)


class FollowIRDataLoader:
    """FollowIR 数据加载器 - 检索用"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self._init_metrics_loader()
        
    def _init_metrics_loader(self):
        """初始化指标计算用的数据加载器"""
        self.metrics_loader = MetricsDataLoader(self.task_name)
        
    def load(self):
        """加载完整数据集"""
        logger.info(f"📂 加载数据集: {self.task_name}")
        
        self.corpus = self.metrics_loader.load_corpus()
        self.q_og, self.q_changed = self.metrics_loader.load_queries()
        self.candidates = self.metrics_loader.load_candidates()
        
        logger.info(f"✅ 数据加载完成: {len(self.corpus)} 文档, "
                   f"{len(self.q_og)} og查询, {len(self.q_changed)} changed查询")
        
        return self.corpus, self.q_og, self.q_changed, self.candidates
    
    def get_query_count(self) -> Dict[str, int]:
        return {
            "og_queries": len(self.q_og),
            "changed_queries": len(self.q_changed),
            "total_queries": len(self.q_og) + len(self.q_changed)
        }
    
    def get_candidate_stats(self) -> Dict[str, float]:
        if not self.candidates:
            return {"count": 0, "avg_per_query": 0.0}
        
        return {
            "count": len(self.candidates),
            "avg_per_query": sum(len(v) for v in self.candidates.values()) / len(self.candidates),
            "min_per_query": min(len(v) for v in self.candidates.values()),
            "max_per_query": max(len(v) for v in self.candidates.values())
        }


class FollowIREvaluatorEngine:
    """FollowIR 评测引擎"""
    
    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        device: str = "cuda",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        seed: int = 42
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.seed = seed
        
        self._setup_seed()
        self._init_components()
    
    def _setup_seed(self) -> None:
        """设置随机种子"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
    
    def _init_components(self) -> None:
        """初始化各组件"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.encoder = ModelFactory.create(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            max_seq_length=self.max_seq_length
        )
        
        self.retriever = DenseRetriever(self.encoder)
        self.output_manager = OutputManager(self.output_dir)
        
        self.data_loader = FollowIRDataLoader(self.task_name)
        
        logger.info(f"✅ 评测引擎初始化完成")
        logger.info(f"   模型: {self.model_name}")
        logger.info(f"   任务: {self.task_name}")
        logger.info(f"   输出: {self.output_dir}")
    
    def run(self) -> Dict[str, Any]:
        """运行完整评测流程"""
        logger.info("=" * 60)
        logger.info("🚀 开始 FollowIR 评测")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        
        query_stats = self.data_loader.get_query_count()
        candidate_stats = self.data_loader.get_candidate_stats()
        
        logger.info(f"📊 数据统计:")
        logger.info(f"   og 查询: {query_stats['og_queries']}")
        logger.info(f"   changed 查询: {query_stats['changed_queries']}")
        logger.info(f"   候选文档: {candidate_stats['avg_per_query']:.0f} 个/查询")
        
        logger.info("📚 编码候选文档...")
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        doc_texts = [corpus[did]['text'] for did in all_doc_ids]
        
        self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
        
        logger.info("🔍 执行检索: og 查询...")
        results_og = self._run_retrieval(q_og, candidates)
        
        logger.info("🔍 执行检索: changed 查询...")
        results_changed = self._run_retrieval(q_changed, candidates)
        
        logger.info("📊 计算评测指标...")
        evaluator = FollowIREvaluator(self.task_name)
        metrics = evaluator.evaluate(results_og, results_changed)
        
        self.output_manager.save_results(
            results_og=results_og,
            results_changed=results_changed,
            task_name=self.task_name,
            model_name=self.model_name,
            metrics=metrics
        )
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("📊 评测结果:")
        logger.info(f"   p-MRR: {metrics.get('p-MRR', 0):.4f}")
        logger.info(f"   og nDCG@5: {metrics.get('original', {}).get('ndcg_at_5', 0):.4f}")
        logger.info(f"   changed nDCG@5: {metrics.get('changed', {}).get('ndcg_at_5', 0):.4f}")
        logger.info(f"   耗时: {elapsed_time:.1f}秒")
        logger.info("=" * 60)
        
        return metrics
    
    def _get_all_candidate_doc_ids(self, candidates: Dict[str, List[str]]) -> List[str]:
        """获取所有候选文档ID"""
        all_doc_ids_set = set()
        for doc_ids in candidates.values():
            all_doc_ids_set.update(doc_ids)
        return list(all_doc_ids_set)
    
    def _run_retrieval(
        self,
        queries: Dict[str, str],
        candidates: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """执行检索"""
        results = {}
        
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        query_embeddings = self.encoder.encode_queries(query_texts, self.batch_size)
        
        for idx, qid in enumerate(tqdm(query_ids, desc="检索")):
            base_qid = qid.replace('-og', '').replace('-changed', '')
            
            if base_qid not in candidates or not candidates[base_qid]:
                logger.warning(f"⚠️ 查询 {qid} 没有候选文档")
                continue
            
            doc_ids = candidates[base_qid]
            scores = self.retriever.compute_scores(query_embeddings[idx], doc_ids)
            
            results[qid] = scores
        
        return results


class EvaluationRunner:
    """评测运行器 - 支持多任务评测"""
    
    def __init__(
        self,
        model_name: str,
        tasks: List[str],
        output_base_dir: str,
        **kwargs
    ):
        self.model_name = model_name
        self.tasks = tasks
        self.output_base_dir = output_base_dir
        self.kwargs = kwargs
        
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """运行所有任务评测"""
        logger.info("=" * 60)
        logger.info(f"🚀 开始批量评测: {len(self.tasks)} 个任务")
        logger.info(f"   模型: {self.model_name}")
        logger.info(f"   任务: {', '.join(self.tasks)}")
        logger.info("=" * 60)
        
        for task in self.tasks:
            logger.info(f"\n{'='*60}")
            logger.info(f"📋 评测任务: {task}")
            logger.info(f"{'='*60}")
            
            output_dir = os.path.join(self.output_base_dir, self._sanitize_filename(f"{self.model_name}_{task}"))
            
            evaluator = FollowIREvaluatorEngine(
                model_name=self.model_name,
                task_name=task,
                output_dir=output_dir,
                **self.kwargs
            )
            
            metrics = evaluator.run()
            self.results[task] = metrics
        
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self) -> None:
        """生成汇总报告"""
        from eval.output import ReportGenerator
        
        summary_gen = ReportGenerator(self.output_base_dir)
        
        summary_gen.generate_summary_report(self.results, "summary.json")
        summary_gen.generate_markdown_report(self.results, self.model_name, "report.md")
        
        logger.info(f"\n📊 汇总结果:")
        for task, metrics in self.results.items():
            logger.info(f"   {task}: p-MRR = {metrics.get('p-MRR', 0):.4f}")
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名"""
        return name.replace('/', '_').replace(':', '_')
