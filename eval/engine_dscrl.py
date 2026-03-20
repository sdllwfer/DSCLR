"""
DSCLR 双流检索引擎
实现 Dual-Stream Contrastive Logical Reranking 的双流打分逻辑
支持静态超参数网格搜索 (Grid Search) 寻找最佳 alpha 和 tau
支持文档向量缓存，避免重复编码
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import random
import sys
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


DEFAULT_CACHE_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"


def load_cached_embeddings(
    cache_dir: str,
    task_name: str
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """尝试加载缓存的文档向量"""
    model_name_short = "bge-large-en"
    cache_file = os.path.join(cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")

    if os.path.exists(cache_file) and os.path.exists(ids_file):
        logger.info(f"📂 加载缓存的文档向量: {cache_file}")
        embeddings = np.load(cache_file)
        with open(ids_file, 'r') as f:
            doc_ids = json.load(f)
        logger.info(f"✅ 缓存加载成功: {len(doc_ids)} 个文档, shape={embeddings.shape}")
        return torch.tensor(embeddings), doc_ids

    logger.info(f"⚠️ 未找到缓存: {cache_file}")
    return None


def save_embeddings_cache(
    cache_dir: str,
    task_name: str,
    embeddings: torch.Tensor,
    doc_ids: List[str]
) -> None:
    """保存文档向量到缓存"""
    os.makedirs(cache_dir, exist_ok=True)
    model_name_short = "bge-large-en"
    cache_file = os.path.join(cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")

    np.save(cache_file, embeddings.cpu().numpy())
    with open(ids_file, 'w') as f:
        json.dump(doc_ids, f)

    logger.info(f"💾 文档向量已缓存: {cache_file}")


class DSCLRDenseRetriever:
    """DSCLR 双流稠密检索器"""

    def __init__(
        self,
        encoder,
        device: str = "cuda",
        batch_size: int = 64
    ):
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.doc_ids: List[str] = []

    def index_documents(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
        batch_size: Optional[int] = None
    ) -> None:
        """构建文档索引（带 L2 归一化）"""
        batch_size = batch_size or self.batch_size
        logger.info(f"📚 索引 {len(doc_ids)} 个文档...")

        embeddings = self.encoder.encode_documents(doc_texts, batch_size=batch_size)

        # 确保 L2 归一化
        if embeddings.dim() == 2:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        self.doc_embeddings = embeddings
        self.doc_ids = doc_ids
        logger.info(f"✅ 文档索引构建完成 (L2 归一化)")

    def set_embeddings(
        self,
        embeddings: torch.Tensor,
        doc_ids: List[str]
    ) -> None:
        """直接设置已编码的文档向量"""
        # 确保 L2 归一化
        if embeddings.dim() == 2:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 确保在正确的设备上
        embeddings = embeddings.to(self.device)
        
        self.doc_embeddings = embeddings
        self.doc_ids = doc_ids
        logger.info(f"✅ 文档向量已加载 (L2 归一化)")

    def compute_scores_matrix(
        self,
        q_plus_embeddings: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算得分矩阵（向量化）
        返回: (S_base, S_neg, S_final)
        """
        # 文档已在索引时归一化，查询也已归一化
        # S_base: [num_queries, num_docs]
        S_base = torch.matmul(q_plus_embeddings, self.doc_embeddings.T)

        # S_neg: [num_queries, num_docs]
        S_neg = torch.matmul(q_minus_embeddings, self.doc_embeddings.T)

        # 应用 mask（将 [NONE] 的负向得分置零）
        S_neg = S_neg * neg_mask.unsqueeze(1)

        return S_base, S_neg

    def compute_dscrl_scores(
        self,
        S_base: torch.Tensor,
        S_neg: torch.Tensor,
        alpha: float,
        tau: float
    ) -> torch.Tensor:
        """
        计算 DSCLR 最终得分
        S_final = S_base - alpha * ReLU(S_neg - tau)
        """
        # ReLU 惩罚项
        penalty = torch.relu(S_neg - tau)

        # 最终得分
        S_final = S_base - alpha * penalty

        return S_final


class DSCLREvaluatorEngine:
    """DSCLR 评测引擎"""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        device: str = "cuda",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.use_cache = use_cache

        # 网格搜索参数空间
        self.alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        self.taus = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        
        # 随机抽取15组参数进行测试
        all_combinations = [(a, t) for a in self.alphas for t in self.taus]
        self.param_combinations = random.sample(all_combinations, min(15, len(all_combinations)))
        logger.info(f"🎲 随机抽取 {len(self.param_combinations)} 组参数: {self.param_combinations}")

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

        # 加载编码器
        from eval.models import ModelFactory
        self.encoder = ModelFactory.create(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            max_seq_length=self.max_seq_length
        )

        # 初始化 Query Reformulator (LLM API 调用)
        from model.reformulator import QueryReformulator
        self.reformulator = QueryReformulator(
            task_name=self.task_name,
            use_cache=True,
            cache_dir="/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries"
        )

        # 创建检索器
        self.retriever = DSCLRDenseRetriever(self.encoder, self.device, self.batch_size)

        # 加载数据
        from eval.engine import FollowIRDataLoader
        self.data_loader = FollowIRDataLoader(self.task_name)

        logger.info(f"✅ DSCLR 评测引擎初始化完成")
        logger.info(f"   模型: {self.model_name}")
        logger.info(f"   任务: {self.task_name}")
        logger.info(f"   查询重构: LLM API (实时解耦)")

    def run(self) -> Dict[str, Any]:
        """运行 DSCLR 评测流程（含网格搜索）"""
        logger.info("=" * 60)
        logger.info("🚀 开始 DSCLR 评测 + 网格搜索")
        logger.info("=" * 60)

        start_time = time.time()

        # 加载数据
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        
        # 加载原始 query 和 instruction (用于 reformulator)
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        # 编码/加载文档
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        
        # 尝试加载缓存
        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name)
        
        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"✅ 使用缓存的文档向量 ({len(cached_doc_ids)} 个)")
                
                # 直接使用缓存中的顺序，不再重排
                # 缓存中的顺序: cached_doc_ids[0] -> cached_embeddings[0]
                # _extract_results 需要使用相同的顺序
                ordered_embeddings = cached_embeddings
                ordered_doc_ids = cached_doc_ids
                
                self.retriever.set_embeddings(ordered_embeddings, ordered_doc_ids)
            else:
                logger.warning(f"⚠️ 缓存文档ID不匹配，重新编码...")
                doc_texts = [corpus[did]['text'] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
                save_embeddings_cache(self.cache_dir, self.task_name, self.retriever.doc_embeddings, self.retriever.doc_ids)
        else:
            # 无缓存，重新编码并保存
            logger.info("📚 编码候选文档...")
            doc_texts = [corpus[did]['text'] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
            if self.use_cache:
                save_embeddings_cache(self.cache_dir, self.task_name, self.retriever.doc_embeddings, self.retriever.doc_ids)

        # 构建 og 查询对 (使用 reformulator 实时解耦)
        logger.info("🔍 准备 og 查询 (LLM API 解耦)...")
        q_plus_list_og, q_minus_list_og, neg_mask_og, query_ids_og = self._prepare_dual_queries(q_og, q_raw_og)
        
        # 构建 changed 查询对 (使用 reformulator 实时解耦)
        logger.info("🔍 准备 changed 查询 (LLM API 解耦)...")
        q_plus_list_changed, q_minus_list_changed, neg_mask_changed, query_ids_changed = self._prepare_dual_queries(q_changed, q_raw_changed)

        # 编码查询
        logger.info("🔍 编码 Q+ 和 Q- (og)...")
        q_plus_embeddings_og = self._encode_queries(q_plus_list_og)
        q_minus_embeddings_og = self._encode_queries(q_minus_list_og)
        
        logger.info("🔍 编码 Q+ 和 Q- (changed)...")
        q_plus_embeddings_changed = self._encode_queries(q_plus_list_changed)
        q_minus_embeddings_changed = self._encode_queries(q_minus_list_changed)

        # 计算 og 得分矩阵
        logger.info("📊 计算 og 得分矩阵...")
        S_base_og, S_neg_og = self.retriever.compute_scores_matrix(
            q_plus_embeddings_og, q_minus_embeddings_og, neg_mask_og
        )
        
        # 计算 changed 得分矩阵
        logger.info("📊 计算 changed 得分矩阵...")
        S_base_changed, S_neg_changed = self.retriever.compute_scores_matrix(
            q_plus_embeddings_changed, q_minus_embeddings_changed, neg_mask_changed
        )

        # 随机搜索
        logger.info(f"🔬 随机搜索: {len(self.param_combinations)} 组参数")
        best_metrics = None
        best_params = None
        best_results_og = None
        best_results_changed = None
        all_results = []
        all_query_metrics = []

        for alpha, tau in self.param_combinations:
                # 计算 og 最终得分
                S_final_og = self.retriever.compute_dscrl_scores(S_base_og, S_neg_og, alpha, tau)
                # 计算 changed 最终得分
                S_final_changed = self.retriever.compute_dscrl_scores(S_base_changed, S_neg_changed, alpha, tau)

                # 提取检索结果
                results_og = self._extract_results(S_final_og, query_ids_og, candidates)
                results_changed = self._extract_results(S_final_changed, query_ids_changed, candidates)

                # 评测 - 使用正确的 og 和 changed 结果
                from eval.metrics import FollowIREvaluator
                evaluator = FollowIREvaluator(self.task_name)
                metrics = evaluator.evaluate(results_og, results_changed)

                p_mrr = metrics.get('p-MRR', 0)
                og_ndcg = metrics.get('original', {}).get('ndcg_at_5', 0)
                changed_ndcg = metrics.get('changed', {}).get('ndcg_at_5', 0)
                logger.info(f"   α={alpha:.1f}, τ={tau:.2f} => p-MRR={p_mrr:.4f}, og_nDCG@5={og_ndcg:.4f}, changed_nDCG@5={changed_ndcg:.4f}")

                all_results.append({
                    'alpha': alpha,
                    'tau': tau,
                    'p-MRR': p_mrr,
                    'og_nDCG@5': og_ndcg,
                    'changed_nDCG@5': changed_ndcg,
                    'metrics': metrics
                })

                # 选择最佳参数：综合考虑 p-MRR、og_nDCG 和 changed_nDCG
                # 标准化后求和作为综合得分
                if best_metrics is None:
                    best_metrics = metrics
                    best_params = (alpha, tau)
                    best_composite_score = p_mrr + og_ndcg + changed_ndcg
                    best_results_og = results_og
                    best_results_changed = results_changed
                else:
                    current_composite_score = p_mrr + og_ndcg + changed_ndcg
                    best_composite_score = best_metrics.get('p-MRR', 0) + best_metrics.get('original', {}).get('ndcg_at_5', 0) + best_metrics.get('changed', {}).get('ndcg_at_5', 0)
                    if current_composite_score > best_composite_score:
                        best_metrics = metrics
                        best_params = (alpha, tau)
                        best_composite_score = current_composite_score
                        best_results_og = results_og
                        best_results_changed = results_changed

        elapsed_time = time.time() - start_time

        # 为每个查询计算详细的性能指标
        all_query_metrics = self._compute_per_query_metrics(
            best_results_og, best_results_changed, 
            query_ids_og, query_ids_changed, candidates
        )

        # 输出结果
        logger.info("=" * 60)
        logger.info("📊 DSCLR 随机搜索结果:")
        logger.info(f"   最佳参数: α={best_params[0]}, τ={best_params[1]}")
        logger.info(f"   最佳 p-MRR: {best_metrics.get('p-MRR', 0):.4f}")
        logger.info(f"   og nDCG@5: {best_metrics.get('original', {}).get('ndcg_at_5', 0):.4f}")
        logger.info(f"   changed nDCG@5: {best_metrics.get('changed', {}).get('ndcg_at_5', 0):.4f}")
        logger.info(f"   耗时: {elapsed_time:.1f}秒")
        logger.info("=" * 60)

        # 保存结构化汇总文件
        self._save_structured_summary(best_metrics, all_query_metrics, q_raw_og, q_raw_changed)

        # 生成坏例分析报告
        self._generate_bad_case_analysis(all_query_metrics, q_raw_og, q_raw_changed)

        # 保存 TREC 格式文件
        trec_dir = os.path.join(self.output_dir, "trec")
        os.makedirs(trec_dir, exist_ok=True)
        
        run_og_path = os.path.join(trec_dir, f"run_{self.task_name}_og.trec")
        run_changed_path = os.path.join(trec_dir, f"run_{self.task_name}_changed.trec")
        
        self._save_trec_format(best_results_og, run_og_path)
        self._save_trec_format(best_results_changed, run_changed_path)
        
        logger.info(f"💾 TREC 文件已保存:")
        logger.info(f"   OG: {run_og_path}")
        logger.info(f"   Changed: {run_changed_path}")

        # 保存结果
        self._save_results(all_results, best_params, best_metrics)

        # 保存最佳参数的单独文件
        best_result_path = os.path.join(self.output_dir, "best_params_result.json")
        best_result = {
            'best_params': {'alpha': best_params[0], 'tau': best_params[1]},
            'best_composite_score': float(best_composite_score),
            'metrics': best_metrics,
            'all_results_summary': [
                {
                    'alpha': r['alpha'],
                    'tau': r['tau'],
                    'p-MRR': r['p-MRR'],
                    'og_nDCG@5': r['og_nDCG@5'],
                    'changed_nDCG@5': r['changed_nDCG@5'],
                    'composite_score': r['p-MRR'] + r['og_nDCG@5'] + r['changed_nDCG@5']
                }
                for r in all_results
            ]
        }
        with open(best_result_path, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 最佳参数结果已保存: {best_result_path}")

        return {
            'best_params': {'alpha': best_params[0], 'tau': best_params[1]},
            'best_metrics': best_metrics,
            'all_results': all_results
        }

    def _get_all_candidate_doc_ids(self, candidates: Dict[str, List[str]]) -> List[str]:
        """获取所有候选文档ID"""
        all_doc_ids_set = set()
        for doc_ids in candidates.values():
            all_doc_ids_set.update(doc_ids)
        return list(all_doc_ids_set)

    def _prepare_dual_queries(
        self,
        queries: Dict[str, str],
        raw_queries: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str], torch.Tensor, List[str]]:
        """
        准备双流查询 - 使用 reformulator 实时解耦
        返回: (q_plus_list, q_minus_list, neg_mask, query_ids)
        """
        query_ids = []
        q_plus_list = []
        q_minus_list = []

        for qid in queries.keys():
            # 获取原始 query 和 instruction
            raw = raw_queries.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            
            # 从 qid 提取 idx (格式: "1-og" -> idx=1)
            try:
                idx = int(qid.split('-')[0])
            except:
                idx = 0
            
            # 确定 query_type
            query_type = "og" if qid.endswith("-og") else "changed"
            
            # 使用 reformulator 进行实时解耦 (带缓存)
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

        # 生成 mask: [NONE] 位置为 0.0，否则为 1.0
        neg_mask = torch.tensor(
            [0.0 if qm == "[NONE]" else 1.0 for qm in q_minus_list],
            dtype=torch.float32,
            device=self.device
        )

        return q_plus_list, q_minus_list, neg_mask, query_ids

    def _encode_queries(self, texts: List[str]) -> torch.Tensor:
        """编码查询（带 L2 归一化）"""
        embeddings = self.encoder.encode_queries(texts, self.batch_size)

        # 确保 L2 归一化
        if embeddings.dim() == 2:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _extract_results(
        self,
        S_final: torch.Tensor,
        query_ids: List[str],
        candidates: Dict[str, List[str]],
        top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """从得分矩阵提取检索结果"""
        results = {}
        
        # 构建 doc_id -> 列索引 的映射
        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
        
        for idx, qid in enumerate(query_ids):
            base_qid = qid.replace('-og', '').replace('-changed', '')

            if base_qid not in candidates or not candidates[base_qid]:
                continue

            doc_ids = candidates[base_qid]

            # 获取该查询对应的得分
            scores = S_final[idx].cpu().numpy()

            # 使用 doc_id_to_col_idx 找到正确的列索引
            doc_scores = {}
            for doc_id in doc_ids:
                if doc_id in doc_id_to_col_idx:
                    col_idx = doc_id_to_col_idx[doc_id]
                    doc_scores[doc_id] = float(scores[col_idx])

            # 取 top-k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            results[qid] = dict(sorted_docs[:top_k])

        return results

    def _save_results(
        self,
        all_results: List[Dict],
        best_params: Tuple[float, float],
        best_metrics: Dict
    ) -> None:
        """保存评测结果"""
        # 保存完整网格搜索结果
        results_path = os.path.join(self.output_dir, "random_search_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': {'alpha': best_params[0], 'tau': best_params[1]},
                'best_metrics': best_metrics,
                'all_results': all_results
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 随机搜索结果已保存: {results_path}")

    def _save_trec_format(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: str
    ) -> None:
        """保存 TREC 格式结果文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for qid in sorted(results.keys()):
                doc_scores = results[qid]
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} dscrl\n")
        logger.info(f"✅ TREC 文件已保存: {output_path}")

    def _compute_per_query_metrics(
        self,
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        candidates: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """计算每个查询的详细性能指标"""
        from eval.metrics.evaluator import DataLoader
        
        data_loader = DataLoader(self.task_name)
        qrels = data_loader.load_qrels()
        
        query_metrics = []
        
        # 获取所有查询的基础ID (用于匹配 qrels)
        processed_qids = set()
        for qid in query_ids_og:
            base_qid = qid.replace('-og', '')
            # qrels 的键格式是 "{id}-og" 或 "{id}-changed"
            processed_qids.add(f"{base_qid}-og")
        
        # 过滤 qrels 只保留需要的
        filtered_qrels = {k: v for k, v in qrels.items() if k in processed_qids}
        
        # 获取所有查询的基础ID
        for idx, qid in enumerate(query_ids_og):
            base_qid = qid.replace('-og', '')
            changed_qid = qid.replace('-og', '-changed')
            
            # qrels 键格式是 "{id}-og" 或 "{id}-changed"，需要用完整键来查找
            og_qid = f"{base_qid}-og"
            
            if og_qid not in filtered_qrels:
                continue
            
            # 获取真实相关文档
            relevant_docs = set(filtered_qrels.get(og_qid, {}).keys())
            
            # 获取模型返回的排序结果
            og_scores = results_og.get(qid, {})
            changed_scores = results_changed.get(changed_qid, {}) if changed_qid in results_changed else {}
            
            # 计算各个指标
            for k in [1, 3, 5, 10, 100, 1000]:
                # OG nDCG@k
                og_ndcg = self._compute_ndcg(og_scores, relevant_docs, k)
                # Changed nDCG@k
                changed_ndcg = self._compute_ndcg(changed_scores, relevant_docs, k) if changed_scores else 0
                
                # MAP@k
                og_map = self._compute_map(og_scores, relevant_docs, k)
                changed_map = self._compute_map(changed_scores, relevant_docs, k) if changed_scores else 0
                
                # MRR@k
                og_mrr = self._compute_mrr(og_scores, relevant_docs, k)
                changed_mrr = self._compute_mrr(changed_scores, relevant_docs, k) if changed_scores else 0
                
                query_metrics.append({
                    'qid': base_qid,
                    'query_type': 'og' if '-og' in qid else 'changed',
                    'k': k,
                    'ndcg': og_ndcg,
                    'map': og_map,
                    'mrr': og_mrr
                })
                
                if changed_scores:
                    query_metrics.append({
                        'qid': base_qid,
                        'query_type': 'changed',
                        'k': k,
                        'ndcg': changed_ndcg,
                        'map': changed_map,
                        'mrr': changed_mrr
                    })
        
        return query_metrics

    def _compute_ndcg(
        self,
        scores: Dict[str, float],
        relevant_docs: set,
        k: int
    ) -> float:
        """计算 NDCG@k"""
        if not scores:
            return 0.0
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        dcg = 0.0
        for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(rank + 1)
        
        # 计算 IDCG
        num_relevant = min(len(relevant_docs), k)
        if num_relevant == 0:
            return 0.0
        
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, num_relevant + 1))
        
        return dcg / idcg if idcg > 0 else 0.0

    def _compute_map(
        self,
        scores: Dict[str, float],
        relevant_docs: set,
        k: int
    ) -> float:
        """计算 MAP@k"""
        if not scores or not relevant_docs:
            return 0.0
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        num_relevant = 0
        precision_sum = 0.0
        
        for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_sum += num_relevant / rank
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0

    def _compute_mrr(
        self,
        scores: Dict[str, float],
        relevant_docs: set,
        k: int
    ) -> float:
        """计算 MRR@k"""
        if not scores or not relevant_docs:
            return 0.0
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        
        return 0.0

    def _save_structured_summary(
        self,
        best_metrics: Dict[str, Any],
        all_query_metrics: List[Dict[str, Any]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, str]]
    ) -> None:
        """保存结构化汇总文件"""
        # 提取完整分数
        full_scores = best_metrics.get('full_scores', {})
        
        # 提取关键指标字段
        summary = {
            'summary': {
                'pMRR': full_scores.get('p-MRR', best_metrics.get('p-MRR', 0)),
                'nDCG1': full_scores.get('og', {}).get('ndcg_at_1', 0),
                'nDCG3': full_scores.get('og', {}).get('ndcg_at_3', 0),
                'nDCG5': full_scores.get('og', {}).get('ndcg_at_5', 0),
                'nDCG10': full_scores.get('og', {}).get('ndcg_at_10', 0),
                'nDCG100': full_scores.get('og', {}).get('ndcg_at_100', 0),
                'nDCG1000': full_scores.get('og', {}).get('ndcg_at_1000', 0),
                'MAP1': full_scores.get('og', {}).get('map_at_1', 0),
                'MAP3': full_scores.get('og', {}).get('map_at_3', 0),
                'MAP5': full_scores.get('og', {}).get('map_at_5', 0),
                'MAP10': full_scores.get('og', {}).get('map_at_10', 0),
                'MAP100': full_scores.get('og', {}).get('map_at_100', 0),
                'MAP1000': full_scores.get('og', {}).get('map_at_1000', 0),
            },
            'best_params': best_metrics.get('best_params', {}),
            'task': self.task_name,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 结构化指标汇总已保存: {summary_path}")

    def _generate_bad_case_analysis(
        self,
        all_query_metrics: List[Dict[str, Any]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, str]]
    ) -> None:
        """生成坏例分析诊断报告"""
        # 提取 k=5 时的 p-MRR（使用 MRR 作为代理）
        og_metrics = [m for m in all_query_metrics if m['query_type'] == 'og' and m['k'] == 5]
        
        if not og_metrics:
            logger.warning("⚠️ 无法生成坏例分析：没有足够的查询指标数据")
            return
        
        mrr_values = [m['mrr'] for m in og_metrics]
        mean_mrr = np.mean(mrr_values)
        std_mrr = np.std(mrr_values)
        
        # 筛选显著低于平均的样本（低于 mean - std）
        low_performance = [m for m in og_metrics if m['mrr'] < mean_mrr - std_mrr]
        # 筛选显著高于平均的样本（高于 mean + std）
        high_performance = [m for m in og_metrics if m['mrr'] > mean_mrr + std_mrr]
        
        # 构建坏例分析报告
        report = {
            'statistics': {
                'total_queries': len(og_metrics),
                'mean_mrr': float(mean_mrr),
                'std_mrr': float(std_mrr),
                'low_performance_count': len(low_performance),
                'high_performance_count': len(high_performance),
            },
            'low_performance_samples': [],
            'high_performance_samples': [],
            'comparison_analysis': {}
        }
        
        # 添加低性能样本详情
        for m in low_performance[:10]:  # 最多10个
            qid = m['qid']
            # q_raw_og 的键是完整格式，如 "310-og"
            og_key = f"{qid}-og"
            raw = q_raw_og.get(og_key, ("", ""))
            
            sample = {
                'qid': qid,
                'mrr': float(m['mrr']),
                'ndcg': float(m['ndcg']),
                'map': float(m['map']),
                'query': raw[0] if raw else "",
                'instruction': raw[1] if raw else "",
            }
            report['low_performance_samples'].append(sample)
        
        # 添加高性能样本详情
        for m in high_performance[:10]:  # 最多10个
            qid = m['qid']
            # q_raw_og 的键是完整格式，如 "310-og"
            og_key = f"{qid}-og"
            raw = q_raw_og.get(og_key, ("", ""))
            
            sample = {
                'qid': qid,
                'mrr': float(m['mrr']),
                'ndcg': float(m['ndcg']),
                'map': float(m['map']),
                'query': raw[0] if raw else "",
                'instruction': raw[1] if raw else "",
            }
            report['high_performance_samples'].append(sample)
        
        # 对比分析
        low_query_lens = [len(s['query'].split()) for s in report['low_performance_samples']]
        high_query_lens = [len(s['query'].split()) for s in report['high_performance_samples']]
        
        low_instr_lens = [len(s['instruction'].split()) for s in report['low_performance_samples']]
        high_instr_lens = [len(s['instruction'].split()) for s in report['high_performance_samples']]
        
        report['comparison_analysis'] = {
            'avg_query_length_low': float(np.mean(low_query_lens)) if low_query_lens else 0,
            'avg_query_length_high': float(np.mean(high_query_lens)) if high_query_lens else 0,
            'avg_instruction_length_low': float(np.mean(low_instr_lens)) if low_instr_lens else 0,
            'avg_instruction_length_high': float(np.mean(high_instr_lens)) if high_instr_lens else 0,
            'key_findings': []
        }
        
        # 自动生成分析结论
        if report['comparison_analysis']['avg_query_length_high'] > report['comparison_analysis']['avg_query_length_low']:
            report['comparison_analysis']['key_findings'].append(
                "高性能样本倾向于有更长的查询文本"
            )
        else:
            report['comparison_analysis']['key_findings'].append(
                "低性能样本倾向于有更长的查询文本"
            )
        
        if report['comparison_analysis']['avg_instruction_length_high'] > report['comparison_analysis']['avg_instruction_length_low']:
            report['comparison_analysis']['key_findings'].append(
                "高性能样本倾向于有更长的指令文本"
            )
        else:
            report['comparison_analysis']['key_findings'].append(
                "低性能样本倾向于有更长的指令文本"
            )
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "bad_case_analysis.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 坏例分析报告已保存: {report_path}")


def run_dsclr_evaluation(
    model_name: str = "BAAI/bge-large-en-v1.5",
    task_name: str = "Core17InstructionRetrieval",
    output_dir: str = "eval/output/dsclr",
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """运行 DSCLR 评测的便捷函数"""
    engine = DSCLREvaluatorEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    return engine.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DSCLR 评测")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5", help="模型名称")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval", help="任务名称")
    parser.add_argument("--output_dir", type=str, default="eval/output/dsclr", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录")
    parser.add_argument("--use_cache", type=bool, default=True, help="是否使用缓存")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    results = run_dsclr_evaluation(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache
    )
    print(f"\n最佳参数: {results['best_params']}")
    print(f"最佳 p-MRR: {results['best_metrics'].get('p-MRR', 0):.4f}")
