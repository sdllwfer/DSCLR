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


def get_model_cache_dir(base_cache_dir: str, model_name: str) -> str:
    """根据模型名称获取模型专属的缓存目录"""
    if "mistral" in model_name.lower():
        model_subdir = "e5-mistral-7b"
    elif "bge" in model_name.lower():
        model_subdir = "bge-large-en"
    else:
        model_subdir = model_name.split("/")[-1].replace("-", "_")
    
    return os.path.join(base_cache_dir, model_subdir)


def get_model_name_short(model_name: str) -> str:
    """从模型全名获取短名称用于缓存"""
    if "mistral" in model_name.lower():
        return "e5-mistral-7b"
    elif "bge" in model_name.lower():
        return "bge-large-en"
    else:
        # 默认使用模型名称的最后一部分
        return model_name.split("/")[-1].replace("-", "_")


def load_cached_embeddings(
    cache_dir: str,
    task_name: str,
    model_name: str
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """尝试加载缓存的文档向量"""
    # 使用模型专属的缓存目录
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    model_name_short = get_model_name_short(model_name)
    cache_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")

    if os.path.exists(cache_file) and os.path.exists(ids_file):
        logger.info(f"📂 加载缓存的文档向量: {cache_file}")
        
        # 尝试加载为 numpy 数组
        try:
            embeddings = np.load(cache_file)
            with open(ids_file, 'r') as f:
                doc_ids = json.load(f)
            logger.info(f"✅ 缓存加载成功: {len(doc_ids)} 个文档, shape={embeddings.shape}")
            return torch.tensor(embeddings), doc_ids
        except:
            # 可能是 dict 格式（E5-Mistral 保存的格式）
            try:
                data = np.load(cache_file, allow_pickle=True)
                if data.dtype == np.object_ and len(data.shape) == 0:
                    embedding_dict = data.item()
                    with open(ids_file, 'r') as f:
                        doc_ids = json.load(f)
                    
                    # 按 doc_ids 顺序提取 embeddings
                    embeddings_list = []
                    for doc_id in doc_ids:
                        if doc_id in embedding_dict:
                            embeddings_list.append(embedding_dict[doc_id])
                    
                    if embeddings_list:
                        embeddings = torch.stack(embeddings_list)
                        logger.info(f"✅ 缓存加载成功 (dict格式): {len(doc_ids)} 个文档, shape={embeddings.shape}")
                        return embeddings, doc_ids
            except Exception as e:
                logger.warning(f"⚠️ 缓存加载失败: {e}")

    logger.info(f"⚠️ 未找到缓存: {cache_file}")
    return None


def save_embeddings_cache(
    cache_dir: str,
    task_name: str,
    model_name: str,
    embeddings: torch.Tensor,
    doc_ids: List[str]
) -> None:
    """保存文档向量到缓存"""
    # 使用模型专属的缓存目录
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    os.makedirs(model_cache_dir, exist_ok=True)
    model_name_short = get_model_name_short(model_name)
    cache_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")

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
        logger.info(f"   [set_embeddings] 输入设备: {embeddings.device}, 目标设备: {self.device}")
        
        # 确保 L2 归一化
        if embeddings.dim() == 2:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 确保在正确的设备上
        embeddings = embeddings.to(self.device)
        logger.info(f"   [set_embeddings] 转移后设备: {embeddings.device}")
        
        self.doc_embeddings = embeddings
        self.doc_ids = doc_ids
        logger.info(f"✅ 文档向量已加载 (L2 归一化)")

    def compute_base_scores(
        self,
        q_plus_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        计算基础得分矩阵（仅 S_base，用于 OG 查询）
        返回: S_base
        """
        # 确保在同一设备上
        device = self.doc_embeddings.device
        if q_plus_embeddings.device != device:
            q_plus_embeddings = q_plus_embeddings.to(device)
        
        # S_base: [num_queries, num_docs]
        S_base = torch.matmul(q_plus_embeddings, self.doc_embeddings.T)
        return S_base

    def compute_scores_matrix(
        self,
        q_plus_embeddings: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算得分矩阵（向量化，用于 Changed 查询）
        返回: (S_base, S_neg)
        """
        # 调试设备信息
        logger.info(f"   [compute_scores_matrix] q_plus: {q_plus_embeddings.device}, doc_emb: {self.doc_embeddings.device}")
        
        # 确保所有张量在同一设备上（使用 doc_embeddings 的设备作为目标）
        device = self.doc_embeddings.device
        if q_plus_embeddings.device != device:
            logger.warning(f"   设备不匹配！将查询 embeddings 转移到 {device}")
            q_plus_embeddings = q_plus_embeddings.to(device)
            q_minus_embeddings = q_minus_embeddings.to(device)
            neg_mask = neg_mask.to(device)
        
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
        计算 DSCLR 最终得分（静态版本）
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
        use_cache: bool = True,
        alphas: Optional[str] = None,
        taus: Optional[str] = None,
        num_samples: int = 15
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

        # 网格搜索参数空间（支持自定义）
        default_alphas = "0.0,0.5,1.0,2.0,3.0,5.0"
        default_taus = "0.5,0.6,0.7,0.8,0.9,0.95"

        alphas_list = [float(a.strip()) for a in (alphas or default_alphas).split(",")]
        taus_list = [float(t.strip()) for t in (taus or default_taus).split(",")]
        self.num_samples = num_samples

        all_combinations = [(a, t) for a in alphas_list for t in taus_list]
        self.param_combinations = random.sample(all_combinations, min(self.num_samples, len(all_combinations)))
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
            cache_dir="/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v4"
        )

        # 创建检索器
        self.retriever = DSCLRDenseRetriever(self.encoder, self.device, self.batch_size)

        # 加载数据
        from eval.engine import FollowIRDataLoader
        self.data_loader = FollowIRDataLoader(self.task_name)

        # 初始化 MLP (用于动态推理)
        self.mlp = None
        self.use_mlp = False

        logger.info(f"✅ DSCLR 评测引擎初始化完成")
        logger.info(f"   模型: {self.model_name}")
        logger.info(f"   任务: {self.task_name}")
        logger.info(f"   查询重构: LLM API (实时解耦)")

    def compute_dscrl_scores_dynamic(
        self,
        S_base: torch.Tensor,
        S_neg: torch.Tensor,
        q_minus_embeddings: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 DSCLR 最终得分（动态 MLP 版本）
        
        使用绝对值扣分，并通过 neg_mask 保护 [NONE] 查询
        """
        if self.mlp is None:
            raise RuntimeError("MLP model not loaded!")
        
        # 确保 q_minus_embeddings 在正确的设备上
        device = next(self.mlp.parameters()).device
        if q_minus_embeddings.device != device:
            q_minus_embeddings = q_minus_embeddings.to(device)
        
        # 转换为 float32 以匹配 MLP 权重
        q_minus_fp32 = q_minus_embeddings.float()
        
        # 根据模型名称确定 encoder_type
        encoder_type = 'mistral' if 'mistral' in self.model_name.lower() else 'bge'
        
        alpha, tau = self.mlp(q_minus_fp32, encoder_type=encoder_type)
        
        # 扩展维度用于计算
        alpha_expanded = alpha.unsqueeze(1)
        tau_expanded = tau.unsqueeze(1)
        neg_mask_expanded = neg_mask.unsqueeze(1)
        
        # 计算惩罚项（统一使用线性 ReLU）
        penalty = torch.relu(S_neg - tau_expanded)
        
        # 应用绝对值扣分，使用 neg_mask 保护 [NONE] 查询
        # 如果 neg_mask=0（即 [NONE]），则惩罚为 0，S_final = S_base
        S_final = S_base - alpha_expanded * penalty * neg_mask_expanded

        return S_final, alpha, tau

    def run(self, mlp_model_path: Optional[str] = None, mlp_hidden_dim: int = 256) -> Dict[str, Any]:
        """运行 DSCLR 评测流程（含网格搜索或动态 MLP）
        
        Args:
            mlp_model_path: 如果提供，则使用动态 MLP 推理；否则使用网格搜索
            mlp_hidden_dim: MLP隐藏层维度 (默认: 256)
        """
        logger.info("=" * 60)
        logger.info("🚀 开始 DSCLR 评测")
        logger.info("=" * 60)

        start_time = time.time()

        # 初始化 MLP (如果提供了模型路径)
        if mlp_model_path:
            logger.info(f"🧠 加载动态 MLP 模型: {mlp_model_path}")
            from model.dsclr_mlp import DSCLR_MLP
            
            # 根据模型名称确定嵌入维度
            if "mistral" in self.model_name.lower() or "repllama" in self.model_name.lower():
                embed_dim = 4096
                logger.info(f"   检测到 {self.model_name} 模型，使用嵌入维度: {embed_dim}")
            else:
                embed_dim = 1024
                logger.info(f"   使用默认嵌入维度: {embed_dim}")
            
            # 使用指定的 hidden_dim
            self.mlp = DSCLR_MLP(input_dim=embed_dim, hidden_dim=mlp_hidden_dim).to(self.device)
            self.mlp.load_state_dict(torch.load(mlp_model_path, map_location=self.device))
            self.mlp.eval()
            self.use_mlp = True
            logger.info(f"✅ MLP 模型加载成功 (hidden_dim={mlp_hidden_dim})，进入动态推理模式")
        else:
            self.use_mlp = False
            logger.info("🔬 使用静态网格搜索模式")

        # 加载数据
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        
        # 加载原始 query 和 instruction (用于 reformulator)
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        # 编码/加载文档
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        
        # 尝试加载缓存
        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)
        
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
                save_embeddings_cache(self.cache_dir, self.task_name, self.model_name, self.retriever.doc_embeddings, self.retriever.doc_ids)
        else:
            # 无缓存，重新编码并保存
            logger.info("📚 编码候选文档...")
            doc_texts = [corpus[did]['text'] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
            if self.use_cache:
                save_embeddings_cache(self.cache_dir, self.task_name, self.model_name, self.retriever.doc_embeddings, self.retriever.doc_ids)

        # 构建 og 查询对 (仅提取 Q+，节省算力)
        logger.info("🔍 准备 og 查询 (仅 Q+)...")
        q_plus_list_og, query_ids_og = self._prepare_single_queries(q_og, q_raw_og)
        
        # 构建 changed 查询对 (使用 reformulator 实时解耦，需要 Q+ 和 Q-)
        logger.info("🔍 准备 changed 查询 (Q+ 和 Q-)...")
        q_plus_list_changed, q_minus_list_changed, q_original_list_changed, neg_mask_changed, query_ids_changed = self._prepare_dual_queries(q_changed, q_raw_changed)

        # 编码查询
        logger.info("🔍 编码 OG 查询 (仅 Q+)...")
        q_plus_embeddings_og = self._encode_queries(q_plus_list_og)

        logger.info("🔍 编码 Changed 查询 (Q+ 和 Q- 和原始查询)...")
        q_plus_embeddings_changed = self._encode_queries(q_plus_list_changed)
        q_minus_embeddings_changed = self._encode_queries(q_minus_list_changed)
        q_original_embeddings_changed = self._encode_queries(q_original_list_changed)

        # 计算 og 得分矩阵 (仅 S_base，无 S_neg)
        logger.info("📊 计算 og 得分矩阵 (仅 S_base)...")
        S_base_og = self.retriever.compute_base_scores(q_plus_embeddings_og)

        # 计算 changed 得分矩阵
        # S_base: 使用原始 query+instruction
        # S_neg: 使用 Q- (负查询)
        logger.info("📊 计算 changed 得分矩阵 (S_base=原查询, S_neg=Q-)...")
        S_base_changed = self.retriever.compute_base_scores(q_original_embeddings_changed)
        q_minus_embeddings_changed = q_minus_embeddings_changed.to(self.retriever.doc_embeddings.device)
        S_neg_changed = torch.matmul(q_minus_embeddings_changed, self.retriever.doc_embeddings.T)
        S_neg_changed = S_neg_changed * neg_mask_changed.unsqueeze(1)

        # 动态推理 或 静态网格搜索
        if self.use_mlp:
            logger.info("🧠 使用动态 MLP 进行推理...")
            with torch.no_grad():
                # 【物理隔离】OG 查询直接使用 S_base，不经过任何 MLP 惩罚！
                S_final_og = S_base_og
                pred_alpha_og = torch.zeros(len(query_ids_og), device=self.device)
                pred_tau_og = torch.zeros(len(query_ids_og), device=self.device)
                
                # 只有 Changed 查询才进入动态门控计算
                S_final_changed, pred_alpha_changed, pred_tau_changed = self.compute_dscrl_scores_dynamic(
                    S_base_changed, S_neg_changed, q_minus_embeddings_changed, neg_mask_changed
                )
            
            logger.info(f"   OG 查询: 物理隔离，直接使用 S_base (无惩罚)")
            logger.info(f"   Changed 查询: 动态预测 avg_alpha={pred_alpha_changed.mean().item():.4f}, avg_tau={pred_tau_changed.mean().item():.4f}")
            
            # 提取检索结果
            results_og = self._extract_results(S_final_og, query_ids_og, candidates)
            results_changed = self._extract_results(S_final_changed, query_ids_changed, candidates)
            
            # 评测
            from eval.metrics import FollowIREvaluator
            evaluator = FollowIREvaluator(self.task_name)
            best_metrics = evaluator.evaluate(results_og, results_changed)
            best_params = {'alpha': 'dynamic', 'tau': 'dynamic'}
            
            # 计算单查询指标
            all_query_metrics = self._compute_per_query_metrics(
                results_og, results_changed, 
                query_ids_og, query_ids_changed, candidates
            )
            
            all_results = [{
                'mode': 'dynamic_mlp',
                'p-MRR': best_metrics.get('p-MRR', 0),
                'og_nDCG@5': best_metrics.get('original', {}).get('ndcg_at_5', 0),
                'changed_nDCG@5': best_metrics.get('changed', {}).get('ndcg_at_5', 0),
                'metrics': best_metrics
            }]
        else:
            # 静态网格搜索
            logger.info(f"🔬 随机搜索: {len(self.param_combinations)} 组参数")
            best_metrics = None
            best_params = None
            best_results_og = None
            best_results_changed = None
            all_results = []
            all_query_metrics = []

            for alpha, tau in self.param_combinations:
                    # 【物理隔离】OG 查询直接使用 S_base，不经过任何惩罚！
                    S_final_og = S_base_og
                    # 只有 Changed 查询才应用 DSCLR 惩罚
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
                        'metrics': metrics,
                        'results_changed': {qid: list(scores.items()) for qid, scores in results_changed.items()},
                        'S_base_changed': S_base_changed.cpu().numpy().tolist() if S_base_changed is not None else None,
                        'S_neg_changed': S_neg_changed.cpu().numpy().tolist() if S_neg_changed is not None else None
                    })

                    # 选择最佳参数：综合考虑 p-MRR、og_nDCG 和 changed_nDCG
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
            
            # 为每个查询计算详细的性能指标
            all_query_metrics = self._compute_per_query_metrics(
                best_results_og, best_results_changed, 
                query_ids_og, query_ids_changed, candidates
            )

        elapsed_time = time.time() - start_time

        # 输出结果
        logger.info("=" * 60)
        if self.use_mlp:
            logger.info("🧠 DSCLR 动态 MLP 推理结果:")
        else:
            logger.info("📊 DSCLR 随机搜索结果:")
            logger.info(f"   最佳参数: α={best_params[0]}, τ={best_params[1]}")
        og_metrics = best_metrics.get('original', {})
        changed_metrics = best_metrics.get('changed', {})
        logger.info(f"   p-MRR: {best_metrics.get('p-MRR', 0):.4f}")
        logger.info(f"   OG - nDCG@1: {og_metrics.get('ndcg_at_1', 0):.4f}, nDCG@5: {og_metrics.get('ndcg_at_5', 0):.4f}, nDCG@10: {og_metrics.get('ndcg_at_10', 0):.4f}")
        logger.info(f"   Changed - nDCG@1: {changed_metrics.get('ndcg_at_1', 0):.4f}, nDCG@5: {changed_metrics.get('ndcg_at_5', 0):.4f}, nDCG@10: {changed_metrics.get('ndcg_at_10', 0):.4f}")
        logger.info(f"   耗时: {elapsed_time:.1f}秒")
        logger.info("=" * 60)

        # 保存结构化汇总文件
        self._save_structured_summary(best_metrics, all_query_metrics, q_raw_og, q_raw_changed)
        
        # 保存所有参数的简化汇总
        self._save_all_params_summary(all_results)

        # 生成坏例分析报告
        # 将 corpus 合并到 candidates 中，方便后续坏例分析使用文档文本
        candidates_with_text = {}
        for qid, doc_ids in candidates.items():
            for doc_id in doc_ids:
                if doc_id not in candidates_with_text:
                    candidates_with_text[doc_id] = corpus.get(doc_id, {'text': ''})
        
        # 构建 q_minus_map 用于坏例分析
        q_minus_map = {}
        for i, qid in enumerate(query_ids_changed):
            q_minus_map[qid] = q_minus_list_changed[i]
        
        self._generate_bad_case_analysis(
            all_query_metrics, q_raw_og, q_raw_changed, 
            candidates_with_text, all_results, q_minus_map
        )

        # 保存 TREC 格式文件
        trec_dir = os.path.join(self.output_dir, "trec")
        os.makedirs(trec_dir, exist_ok=True)
        
        run_og_path = os.path.join(trec_dir, f"run_{self.task_name}_og.trec")
        run_changed_path = os.path.join(trec_dir, f"run_{self.task_name}_changed.trec")
        
        # 保存 TREC 格式文件
        self._save_trec_format(results_og, run_og_path)
        self._save_trec_format(results_changed, run_changed_path)
        
        logger.info(f"💾 TREC 文件已保存:")
        logger.info(f"   OG: {run_og_path}")
        logger.info(f"   Changed: {run_changed_path}")

        # 保存结果
        if self.use_mlp:
            # MLP 模式：只保存测试指标结果
            result_path = os.path.join(self.output_dir, "mlp_results.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'dynamic_mlp',
                    'metrics': best_metrics
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 MLP 测试结果已保存: {result_path}")
        else:
            # 静态网格搜索模式：保存完整结果
            self._save_results(all_results, best_params, best_metrics, all_query_metrics)

        return {
            'best_params': {'alpha': 'dynamic', 'tau': 'dynamic'} if self.use_mlp else {'alpha': best_params[0], 'tau': best_params[1]},
            'best_metrics': best_metrics
        }

    def _get_all_candidate_doc_ids(self, candidates: Dict[str, List[str]]) -> List[str]:
        """获取所有候选文档ID"""
        all_doc_ids_set = set()
        for doc_ids in candidates.values():
            all_doc_ids_set.update(doc_ids)
        return list(all_doc_ids_set)

    def _get_bulletproof_mask(self, q_minus_text: str) -> float:
        """防弹级掩码生成函数
        
        拦截所有可能的 LLM 废话输出，确保 [NONE] 查询不受惩罚
        """
        if not q_minus_text:
            return 0.0
        text = str(q_minus_text).strip().upper()
        # 拦截所有可能的无效输出
        if text in ["[NONE]", "NONE", "NULL", "N/A", "", "[NONE]", "NONE"]:
            return 0.0
        return 1.0

    def _prepare_single_queries(
        self,
        queries: Dict[str, str],
        raw_queries: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str]]:
        """
        准备单流查询（仅 Q+）- 用于 OG 查询，节省算力
        返回: (q_plus_list, query_ids)
        """
        query_ids = []
        q_plus_list = []

        for qid in queries.keys():
            # 获取原始 query 和 instruction
            raw = raw_queries.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            
            # OG 查询使用 query + instruction 拼接（与基线评估一致）
            q_plus = f"{query_text} {instruction}".strip() if query_text else queries.get(qid, "")

            query_ids.append(qid)
            q_plus_list.append(q_plus)

        logger.info(f"   单流查询准备完成: {len(query_ids)} 个")
        return q_plus_list, query_ids

    def _prepare_dual_queries(
        self,
        queries: Dict[str, str],
        raw_queries: Dict[str, Tuple[str, str]]
    ) -> Tuple[List[str], List[str], List[str], torch.Tensor, List[str]]:
        """
        准备双流查询 - 使用 reformulator 实时解耦
        返回: (q_plus_list, q_minus_list, q_original_list, neg_mask, query_ids)
        其中 q_original_list 是原始 query + instruction，用于计算 S_base
        """
        query_ids = []
        q_plus_list = []
        q_minus_list = []
        q_original_list = []

        for qid in queries.keys():
            # 获取原始 query 和 instruction
            raw = raw_queries.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            
            # 原始查询 = query + instruction（用于 S_base 计算）
            q_original = f"{query_text} {instruction}".strip() if query_text else queries.get(qid, "")

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
            q_original_list.append(q_original)

        # 【调试逻辑】强制 OG 查询跳过 MLP，即使 Q- 不为 None
        # 这样可以让 OG nDCG 与原始检索模型对比，验证掩码逻辑
        def get_debug_mask(qm, qid):
            base_mask = self._get_bulletproof_mask(qm)
            if qid.endswith("-og"):
                # 强制 OG 查询 mask = 0，跳过 MLP
                return 0.0
            return base_mask
        
        neg_mask = torch.tensor(
            [get_debug_mask(qm, qid) for qm, qid in zip(q_minus_list, query_ids)],
            dtype=torch.float32,
            device=self.device
        )
        
        # 统计调试信息
        og_count = sum(1 for qid in query_ids if qid.endswith("-og"))
        changed_count = len(query_ids) - og_count
        logger.info(f"【调试模式】OG 查询强制跳过 MLP: {og_count} 个, Changed 查询: {changed_count} 个")

        return q_plus_list, q_minus_list, q_original_list, neg_mask, query_ids

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

            # 获取该查询对应的得分 (转换为 float32，避免 BFloat16 问题)
            scores = S_final[idx].cpu().float().numpy()

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
        best_params,
        best_metrics: Dict,
        all_query_metrics: Optional[List[Dict]] = None
    ) -> None:
        """保存评测结果"""
        results_path = os.path.join(self.output_dir, "random_search_results.json")
        
        if self.use_mlp:
            save_data = {
                'best_params': best_params,
                'mode': 'dynamic_mlp',
                'best_metrics': best_metrics,
                'all_results': all_results
            }
        else:
            save_data = {
                'best_params': {'alpha': best_params[0], 'tau': best_params[1]},
                'best_metrics': best_metrics,
                'all_results': all_results,
                'all_query_metrics': all_query_metrics or []
            }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 结果已保存: {results_path}")

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

    def _save_all_params_summary(
        self,
        all_results: List[Dict[str, Any]]
    ) -> None:
        """保存所有参数的简化汇总"""
        summary_path = os.path.join(self.output_dir, "all_params_summary.csv")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("alpha,tau,pMRR,og_nDCG1,og_nDCG5,og_nDCG10,og_nDCG100,og_MAP1,og_MAP5,og_MAP10,og_MAP100,changed_nDCG1,changed_nDCG5,changed_nDCG10,changed_nDCG100,changed_MAP1,changed_MAP5,changed_MAP10,changed_MAP100\n")
            
            for result in all_results:
                alpha = result.get('alpha', 0)
                tau = result.get('tau', 0)
                p_mrr = result.get('p-MRR', 0)
                metrics = result.get('metrics', {})
                
                full_scores = metrics.get('full_scores', {})
                og_metrics = full_scores.get('og', {})
                changed_metrics = full_scores.get('changed', {})
                
                line = f"{alpha},{tau},{p_mrr:.6f},"
                line += f"{og_metrics.get('ndcg_at_1', 0):.6f},"
                line += f"{og_metrics.get('ndcg_at_5', 0):.6f},"
                line += f"{og_metrics.get('ndcg_at_10', 0):.6f},"
                line += f"{og_metrics.get('ndcg_at_100', 0):.6f},"
                line += f"{og_metrics.get('map_at_1', 0):.6f},"
                line += f"{og_metrics.get('map_at_5', 0):.6f},"
                line += f"{og_metrics.get('map_at_10', 0):.6f},"
                line += f"{og_metrics.get('map_at_100', 0):.6f},"
                line += f"{changed_metrics.get('ndcg_at_1', 0):.6f},"
                line += f"{changed_metrics.get('ndcg_at_5', 0):.6f},"
                line += f"{changed_metrics.get('ndcg_at_10', 0):.6f},"
                line += f"{changed_metrics.get('ndcg_at_100', 0):.6f},"
                line += f"{changed_metrics.get('map_at_1', 0):.6f},"
                line += f"{changed_metrics.get('map_at_5', 0):.6f},"
                line += f"{changed_metrics.get('map_at_10', 0):.6f},"
                line += f"{changed_metrics.get('map_at_100', 0):.6f}\n"
                
                f.write(line)
        
        logger.info(f"💾 所有参数汇总已保存: {summary_path}")

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
        q_raw_changed: Dict[str, Tuple[str, str]],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        q_minus_map: Dict[str, str]
    ) -> None:
        """生成极端坏例分析诊断报告"""
        report_path = os.path.join(self.output_dir, "bad_case_analysis.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🔍 DSCLR 极端坏例分析报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            report_data = self._analyze_extreme_cases(
                all_query_metrics, q_raw_og, q_raw_changed, 
                candidates, all_results, q_minus_map
            )
            
            f.write(report_data['markdown'])
        
        json_report_path = os.path.join(self.output_dir, "bad_case_analysis.json")
        with open(json_report_path, 'w', encoding='utf-8') as json_f:
            json.dump(report_data['json'], json_f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 坏例分析报告已保存: {report_path}")
        logger.info(f"💾 JSON格式报告已保存: {json_report_path}")
    
    def _analyze_extreme_cases(
        self,
        all_query_metrics: List[Dict[str, Any]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, str]],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        q_minus_map: Dict[str, str]
    ) -> Dict[str, Any]:
        """分析极端坏例"""
        
        result_params = {r['alpha']: r['tau'] for r in all_results}
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        alpha_5_tau_7_result = next((r for r in all_results if r['alpha'] == 5.0 and abs(r['tau'] - 0.7) < 0.01), None)
        alpha_1_tau_8_result = next((r for r in all_results if r['alpha'] == 1.0 and abs(r['tau'] - 0.8) < 0.01), None)
        
        if not alpha_0_result:
            return {'markdown': '# ❌ 错误：未找到 α=0.0 的结果\n', 'json': {'error': 'Missing alpha=0.0 results'}}
        
        og_metrics = [m for m in all_query_metrics if m['query_type'] == 'og' and m['k'] == 5]
        
        query_neg_scores = self._compute_query_negative_scores(
            q_raw_changed, candidates, all_results, q_minus_map
        )
        
        selected_queries = self._select_extreme_queries(
            og_metrics, query_neg_scores, q_raw_og, q_raw_changed
        )
        
        markdown_output = self._generate_query_analysis_markdown(
            selected_queries, candidates, all_results,
            alpha_0_result, alpha_5_tau_7_result, alpha_1_tau_8_result,
            q_raw_changed, query_neg_scores
        )
        
        json_output = self._generate_query_analysis_json(
            selected_queries, candidates, all_results,
            alpha_0_result, alpha_5_tau_7_result, alpha_1_tau_8_result,
            q_raw_changed, query_neg_scores
        )
        
        return {'markdown': markdown_output, 'json': json_output}
    
    def _compute_query_negative_scores(
        self,
        q_raw_changed: Dict[str, Tuple[str, str]],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        q_minus_map: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """计算每个Query的正负向得分 - 使用 reformulator 提取的 Q-"""
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        
        if not alpha_0_result:
            return {}
        
        from eval.metrics.evaluator import DataLoader
        data_loader = DataLoader(self.task_name)
        qrels = data_loader.load_qrels()
        
        query_neg_scores = {}
        
        for qid in [k.split('-')[0] for k in q_raw_changed.keys()]:
            changed_key = f"{qid}-changed"
            
            if changed_key not in q_raw_changed:
                continue
            
            # 从 q_minus_map 获取 reformulator 提取的 Q-
            q_minus = q_minus_map.get(changed_key, "[NONE]")
            
            if q_minus == "[NONE]" or not q_minus:
                query_neg_scores[qid] = {
                    'neg_words': [],
                    'doc_scores': {},
                    'relevant_docs': set(),
                    'irrelevant_docs': set()
                }
                continue
            
            # Q- 是逗号分隔的负向词列表
            neg_words_list = [w.strip() for w in q_minus.split(',') if w.strip()]
            
            query_neg_scores[qid] = {
                'neg_words': neg_words_list,
                'doc_scores': {},
                'relevant_docs': set(),
                'irrelevant_docs': set()
            }
            
            results_changed = alpha_0_result.get('results_changed', {})
            
            changed_key_result = f"{qid}-changed"
            if changed_key_result in results_changed:
                for doc_id, score in results_changed[changed_key_result][:50]:
                    doc = candidates.get(doc_id, {})
                    doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                    
                    # 从 qrels 获取相关性标注
                    qrel_key = f"{qid}-changed"
                    relevance = qrels.get(qrel_key, {}).get(doc_id, 0)
                    
                    if relevance > 0:
                        query_neg_scores[qid]['relevant_docs'].add(doc_id)
                    else:
                        query_neg_scores[qid]['irrelevant_docs'].add(doc_id)
                    
                    # 检查文档是否包含负向词
                    neg_word_found = None
                    for neg_word in neg_words_list:
                        if neg_word.lower() in doc_text.lower():
                            neg_word_found = neg_word
                            break
                    
                    if neg_word_found:
                        query_neg_scores[qid]['doc_scores'][doc_id] = {
                            'score': score,
                            'neg_word': neg_word_found,
                            'text_snippet': doc_text[:200]
                        }
        
        return query_neg_scores
    
    def _select_extreme_queries(
        self,
        og_metrics: List[Dict[str, Any]],
        query_neg_scores: Dict[str, Dict[str, float]],
        q_raw_og: Dict[str, Tuple[str, str]],
        q_raw_changed: Dict[str, Tuple[str, Tuple[str, str]]]
    ) -> List[Dict[str, Any]]:
        """选择4种极端类型的Query"""
        
        query_type_scores = {}
        for m in og_metrics:
            qid = str(m['qid'])
            mrr = m['mrr']
            
            neg_info = query_neg_scores.get(qid, {})
            avg_neg_score = 0.0
            
            if neg_info.get('doc_scores'):
                scores = [d['score'] for d in neg_info['doc_scores'].values()]
                avg_neg_score = np.mean(scores) if scores else 0.0
            
            query_type_scores[qid] = {
                'mrr': mrr,
                'avg_neg_score': avg_neg_score,
                'neg_info': neg_info
            }
        
        high_noise = []
        low_noise = []
        entity_entangled = []
        logical_negation = []
        
        for qid, scores in query_type_scores.items():
            avg_neg = scores['avg_neg_score']
            
            neg_info = scores.get('neg_info', {})
            neg_words = neg_info.get('neg_words', [])
            
            has_entity_neg = any(
                any(c.isalpha() and len(c) > 3 for c in w.split()) 
                for w in neg_words
            )
            
            is_logical_only = all(
                any(neg in w.lower() for neg in ['not', 'no', 'without', 'except', '除了', '不要', '非', '无'])
                for w in neg_words
            ) if neg_words else True
            
            if avg_neg > 0.65:
                high_noise.append((qid, scores, 'high_noise'))
            elif avg_neg < 0.55 and avg_neg > 0:
                low_noise.append((qid, scores, 'low_noise'))
            
            if is_logical_only and neg_words:
                logical_negation.append((qid, scores, 'logical_negation'))
        
        entity_keywords = ['病', '癌', '基因', '细胞', '蛋白', '病毒', '遗传', '突', '转基因', 
                          'cancer', 'gene', 'protein', 'virus', 'genetic', 'mutant']
        
        for qid, scores in query_type_scores.items():
            og_key = f"{qid}-og"
            raw = q_raw_og.get(og_key, ("", ""))
            query_text = raw[0].lower() if raw else ""
            
            has_entity = any(kw in query_text for kw in entity_keywords)
            
            if has_entity:
                neg_info = scores.get('neg_info', {})
                if neg_info.get('neg_words'):
                    entity_entangled.append((qid, scores, 'entity_entangled'))
        
        selected = []
        
        if high_noise:
            selected.append(high_noise[0])
        if low_noise:
            selected.append(low_noise[0])
        if entity_entangled:
            selected.append(entity_entangled[0])
        if logical_negation:
            selected.append(logical_negation[0])
        
        remaining = []
        for qid, scores, qtype in high_noise[1:]:
            if qid not in [s[0] for s in selected]:
                remaining.append((qid, scores, qtype))
        for qid, scores, qtype in low_noise[1:]:
            if qid not in [s[0] for s in selected]:
                remaining.append((qid, scores, qtype))
        for qid, scores, qtype in entity_entangled[1:]:
            if qid not in [s[0] for s in selected]:
                remaining.append((qid, scores, qtype))
        for qid, scores, qtype in logical_negation[1:]:
            if qid not in [s[0] for s in selected]:
                remaining.append((qid, scores, qtype))
        
        selected.extend(remaining[:max(0, 8 - len(selected))])
        
        return selected
    
    def _generate_query_analysis_markdown(
        self,
        selected_queries: List[Tuple],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        alpha_0_result: Optional[Dict],
        alpha_5_tau_7_result: Optional[Dict],
        alpha_1_tau_8_result: Optional[Dict],
        q_raw_changed: Dict[str, Tuple[str, str]],
        query_neg_scores: Dict[str, Dict]
    ) -> str:
        """生成Markdown格式的分析报告"""
        
        markdown = "## 📋 选中的极端Query分析\n\n"
        
        type_names = {
            'high_noise': '🔴 高底噪型',
            'low_noise': '🟢 低底噪型',
            'entity_entangled': '🟠 实体纠缠型',
            'logical_negation': '🔵 纯逻辑否定型'
        }
        
        for qid, scores, qtype in selected_queries:
            changed_key = f"{qid}-changed"
            raw = q_raw_changed.get(changed_key, ("", ""))
            query_text, _ = raw
            
            neg_info = query_neg_scores.get(qid, {})
            neg_words_list = neg_info.get('neg_words', [])
            q_minus = ', '.join(neg_words_list) if neg_words_list else '[NONE]'
            
            markdown += f"### {type_names.get(qtype, qtype)} - Q{qid}\n\n"
            markdown += f"**Query**: {query_text}\n\n"
            markdown += f"**负向词 (Q-)**: {q_minus}\n\n"
            markdown += f"**当前MRR**: {scores['mrr']:.4f}\n\n"
            markdown += f"**平均负向得分**: {scores['avg_neg_score']:.4f}\n\n"
            
            markdown += f"---\n\n"
            markdown += f"#### 【数据组 A：假阳性（漏网的烂文）】 α=0.0\n\n"
            
            fp_docs = self._extract_false_positives(
                qid, candidates, alpha_0_result, neg_info
            )
            
            for i, doc in enumerate(fp_docs[:3], 1):
                markdown += f"**A-{i}**: `doc_id={doc['doc_id']}`\n"
                markdown += f"- Snippet: {doc['snippet']}\n"
                markdown += f"- $S_{{pos}}$: {doc['S_pos']:.4f}, $S_{{neg\_proj}}$: {doc['S_neg_proj']:.4f}\n\n"
            
            markdown += f"---\n\n"
            markdown += f"#### 【数据组 B：假阴性（冤死的极品好文）】 α=5.0, τ=0.7\n\n"
            
            fn_docs = self._extract_false_negatives(
                qid, candidates, all_results, neg_info
            )
            
            for i, doc in enumerate(fn_docs[:3], 1):
                markdown += f"**B-{i}**: `doc_id={doc['doc_id']}`\n"
                markdown += f"- Snippet: {doc['snippet']}\n"
                markdown += f"- $S_{{pos}}$: {doc['S_pos']:.4f}, $S_{{neg\_proj}}$: {doc['S_neg_proj']:.4f}\n"
                markdown += f"- Penalty: {doc['penalty']:.4f}\n"
                markdown += f"- 原排名: {doc['original_rank']}, 现排名: {doc['current_rank']}\n\n"
            
            markdown += f"---\n\n"
            markdown += f"#### 【数据组 C：当前最优参数下的残留误差】 α=1.0, τ=0.8\n\n"
            
            if alpha_1_tau_8_result:
                c_docs = self._extract_optimal_residual_errors(
                    qid, candidates, alpha_1_tau_8_result, all_results
                )
                
                if c_docs['false_positive']:
                    doc = c_docs['false_positive']
                    markdown += f"**C-1 漏网烂文**: `doc_id={doc['doc_id']}`\n"
                    markdown += f"- $S_{{pos}}$: {doc['S_pos']:.4f}, $S_{{neg\_proj}}$: {doc['S_neg_proj']:.4f}\n"
                    markdown += f"- 排名: {doc['rank']}\n\n"
                
                if c_docs['false_negative']:
                    doc = c_docs['false_negative']
                    markdown += f"**C-2 冤枉好文**: `doc_id={doc['doc_id']}`\n"
                    markdown += f"- $S_{{pos}}$: {doc['S_pos']:.4f}, $S_{{neg\_proj}}$: {doc['S_neg_proj']:.4f}\n"
                    markdown += f"- 排名: {doc['rank']}\n\n"
            
            markdown += f"---\n\n"
            markdown += f"#### 【数据组 D：特征倒挂点】❗最高优先级\n\n"
            
            inversion = self._extract_feature_inversions(
                qid, candidates, all_results, neg_info
            )
            
            if inversion:
                markdown += f"**倒挂文档对**:\n\n"
                markdown += f"- **好文** (相关): `doc_id={inversion['good']['doc_id']}`\n"
                markdown += f"  - Snippet: {inversion['good']['snippet']}\n"
                markdown += f"  - $S_{{neg\_proj}}$: {inversion['good']['S_neg_proj']:.4f}\n\n"
                markdown += f"- **烂文** (不相关): `doc_id={inversion['bad']['doc_id']}`\n"
                markdown += f"  - Snippet: {inversion['bad']['snippet']}\n"
                markdown += f"  - $S_{{neg\_proj}}$: {inversion['bad']['S_neg_proj']:.4f}\n\n"
                markdown += f"- **差值**: {inversion['diff']:.4f}\n\n"
            else:
                markdown += f"未找到特征倒挂点\n\n"
            
            markdown += "---\n\n"
        
        return markdown
    
    def _generate_query_analysis_json(
        self,
        selected_queries: List[Tuple],
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        alpha_0_result: Optional[Dict],
        alpha_5_tau_7_result: Optional[Dict],
        alpha_1_tau_8_result: Optional[Dict],
        q_raw_changed: Dict[str, Tuple[str, str]],
        query_neg_scores: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """生成JSON格式的分析报告"""
        
        json_output = {
            'statistics': {
                'total_selected_queries': len(selected_queries)
            },
            'queries': []
        }
        
        type_names = {
            'high_noise': '高底噪型',
            'low_noise': '低底噪型',
            'entity_entangled': '实体纠缠型',
            'logical_negation': '纯逻辑否定型'
        }
        
        for qid, scores, qtype in selected_queries:
            changed_key = f"{qid}-changed"
            raw = q_raw_changed.get(changed_key, ("", ""))
            query_text, _ = raw
            
            neg_info = query_neg_scores.get(qid, {})
            neg_words_list = neg_info.get('neg_words', [])
            q_minus = ', '.join(neg_words_list) if neg_words_list else '[NONE]'
            
            query_data = {
                'qid': qid,
                'type': type_names.get(qtype, qtype),
                'query': query_text,
                'negative_words': q_minus,
                'current_mrr': float(scores['mrr']),
                'avg_neg_score': float(scores['avg_neg_score']),
                'data_group_A': [],
                'data_group_B': [],
                'data_group_C': {},
                'data_group_D': None
            }
            
            fp_docs = self._extract_false_positives(
                qid, candidates, alpha_0_result, neg_info
            )
            query_data['data_group_A'] = fp_docs[:3]
            
            fn_docs = self._extract_false_negatives(
                qid, candidates, all_results, neg_info
            )
            query_data['data_group_B'] = fn_docs[:3]
            
            if alpha_1_tau_8_result:
                c_docs = self._extract_optimal_residual_errors(
                    qid, candidates, alpha_1_tau_8_result, all_results
                )
                query_data['data_group_C'] = c_docs
            
            inversion = self._extract_feature_inversions(
                qid, candidates, all_results, neg_info
            )
            query_data['data_group_D'] = inversion
            
            json_output['queries'].append(query_data)
        
        return json_output
    
    def _extract_false_positives(
        self,
        qid: str,
        candidates: Dict[str, Any],
        alpha_0_result: Optional[Dict],
        neg_info: Dict
    ) -> List[Dict[str, Any]]:
        """提取假阳性文档（数据组A）"""
        
        if not alpha_0_result:
            return []
        
        results_changed = alpha_0_result.get('results_changed', {})
        
        changed_key = f"{qid}-changed"
        if changed_key not in results_changed:
            return []
        
        fp_docs = []
        
        for doc_id, score in results_changed[changed_key][:10]:
            doc = candidates.get(doc_id, {})
            doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
            
            neg_words = neg_info.get('neg_words', [])
            neg_word_found = None
            
            for neg_word in neg_words:
                if neg_word.lower() in doc_text.lower():
                    neg_word_found = neg_word
                    break
            
            if neg_word_found and neg_word_found in neg_info.get('doc_scores', {}):
                doc_score_info = neg_info['doc_scores'][doc_id]
                
                fp_docs.append({
                    'doc_id': doc_id,
                    'snippet': doc_text[:150],
                    'S_pos': float(score),
                    'S_neg_proj': float(doc_score_info['score']),
                    'neg_word': neg_word_found
                })
        
        return fp_docs
    
    def _extract_false_negatives(
        self,
        qid: str,
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        neg_info: Dict
    ) -> List[Dict[str, Any]]:
        """提取假阴性文档（数据组B）"""
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        alpha_5_result = next((r for r in all_results if r['alpha'] == 5.0 and abs(r['tau'] - 0.7) < 0.01), None)
        
        if not alpha_0_result or not alpha_5_result:
            return []
        
        results_0 = alpha_0_result.get('results_changed', {}).get(f"{qid}-changed", [])
        results_5 = alpha_5_result.get('results_changed', {}).get(f"{qid}-changed", [])
        
        fn_docs = []
        
        for i, (doc_id, score_0) in enumerate(results_0[:10]):
            rank_0 = i + 1
            
            rank_5 = None
            score_5 = None
            for j, (d_id, s) in enumerate(results_5):
                if d_id == doc_id:
                    rank_5 = j + 1
                    score_5 = s
                    break
            
            if rank_5 and rank_5 > 50:
                doc = candidates.get(doc_id, {})
                doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                
                neg_word = neg_info.get('doc_scores', {}).get(doc_id, {}).get('neg_word', '')
                
                penalty = abs(score_0 - score_5) if score_5 else 0
                
                fn_docs.append({
                    'doc_id': doc_id,
                    'snippet': doc_text[:150],
                    'S_pos': float(score_0),
                    'S_neg_proj': float(score_5),
                    'penalty': float(penalty),
                    'original_rank': rank_0,
                    'current_rank': rank_5
                })
        
        return fn_docs
    
    def _extract_optimal_residual_errors(
        self,
        qid: str,
        candidates: Dict[str, Any],
        alpha_1_tau_8_result: Dict,
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """提取最优参数下的残留误差（数据组C）"""
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        
        if not alpha_0_result:
            return {}
        
        results_optimal = alpha_1_tau_8_result.get('results_changed', {}).get(f"{qid}-changed", [])
        results_0 = alpha_0_result.get('results_changed', {}).get(f"{qid}-changed", [])
        
        best_fp = None
        worst_fn = None
        
        for i, (doc_id, score) in enumerate(results_optimal[:20]):
            if doc_id in results_0:
                rank_0 = next((j for j, (d, _) in enumerate(results_0) if d == doc_id), None)
                
                if rank_0 is not None and rank_0 < 10:
                    doc = candidates.get(doc_id, {})
                    doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                    
                    best_fp = {
                        'doc_id': doc_id,
                        'S_pos': float(score),
                        'S_neg_proj': 0.0,
                        'rank': i + 1
                    }
                    break
        
        for i, (doc_id, score) in enumerate(results_optimal):
            if doc_id in results_0:
                rank_0 = next((j for j, (d, _) in enumerate(results_0) if d == doc_id), None)
                
                if rank_0 is not None and rank_0 < 10:
                    if worst_fn is None or i > worst_fn.get('rank', 0):
                        worst_fn = {
                            'doc_id': doc_id,
                            'S_pos': float(score),
                            'S_neg_proj': 0.0,
                            'rank': i + 1
                        }
        
        return {
            'false_positive': best_fp,
            'false_negative': worst_fn
        }
    
    def _extract_feature_inversions(
        self,
        qid: str,
        candidates: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        neg_info: Dict
    ) -> Optional[Dict[str, Any]]:
        """提取特征倒挂点（数据组D）"""
        
        alpha_0_result = next((r for r in all_results if r['alpha'] == 0.0), None)
        
        if not alpha_0_result:
            return None
        
        results_changed = alpha_0_result.get('results_changed', {}).get(f"{qid}-changed", [])
        
        doc_neg_scores = neg_info.get('doc_scores', {})
        
        relevant_docs = []
        irrelevant_docs = []
        
        for doc_id, score in results_changed:
            if doc_id in doc_neg_scores:
                doc_score_info = doc_neg_scores[doc_id]
                
                relevant_docs.append({
                    'doc_id': doc_id,
                    'S_neg_proj': doc_score_info['score']
                })
        
        for doc_id, score in results_changed[:100]:
            if doc_id not in doc_neg_scores:
                doc = candidates.get(doc_id, {})
                doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                
                irrelevant_docs.append({
                    'doc_id': doc_id,
                    'S_neg_proj': float(score),  # 使用实际的 S_neg_proj 得分
                    'snippet': doc_text[:150]
                })
        
        if not relevant_docs or not irrelevant_docs:
            return None
        
        relevant_docs_sorted = sorted(relevant_docs, key=lambda x: x['S_neg_proj'], reverse=True)
        
        for rel_doc in relevant_docs_sorted[:10]:
            rel_doc['snippet'] = candidates.get(rel_doc['doc_id'], {}).get('text', '')[:150] if isinstance(candidates.get(rel_doc['doc_id'], {}), dict) else str(candidates.get(rel_doc['doc_id'], ''))[:150]
            
            for irr_doc in irrelevant_docs[:20]:
                if rel_doc['S_neg_proj'] > irr_doc['S_neg_proj']:
                    return {
                        'good': rel_doc,
                        'bad': irr_doc,
                        'diff': float(rel_doc['S_neg_proj'] - irr_doc['S_neg_proj'])
                    }
        
        return None


def run_dsclr_evaluation(
    model_name: str = "BAAI/bge-large-en-v1.5",
    task_name: str = "Core17InstructionRetrieval",
    output_dir: str = "eval/output/dsclr",
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    mlp_model_path: Optional[str] = None,
    mlp_hidden_dim: int = 256,
    alphas: Optional[str] = None,
    taus: Optional[str] = None,
    num_samples: int = 15
) -> Dict[str, Any]:
    """运行 DSCLR 评测的便捷函数"""
    engine = DSCLREvaluatorEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        use_cache=use_cache,
        alphas=alphas,
        taus=taus,
        num_samples=num_samples
    )
    return engine.run(mlp_model_path=mlp_model_path, mlp_hidden_dim=mlp_hidden_dim)


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
    parser.add_argument("--mlp_model_path", type=str, default=None, help="MLP模型路径 (可选，使用动态MLP推理)")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256, help="MLP隐藏层维度 (默认: 256)")
    parser.add_argument("--alphas", type=str, default=None, help="Alpha 搜索范围，逗号分隔 (默认: 0.0,0.5,1.0,2.0,3.0,5.0)")
    parser.add_argument("--taus", type=str, default=None, help="Tau 搜索范围，逗号分隔 (默认: 0.5,0.6,0.7,0.8,0.9,0.95)")
    parser.add_argument("--num_samples", type=int, default=15, help="随机抽样数量 (默认: 15)")

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
        use_cache=args.use_cache,
        mlp_model_path=args.mlp_model_path,
        mlp_hidden_dim=args.mlp_hidden_dim,
        alphas=args.alphas,
        taus=args.taus,
        num_samples=args.num_samples
    )
    print(f"\n最终 p-MRR: {results['best_metrics'].get('p-MRR', 0):.4f}")
