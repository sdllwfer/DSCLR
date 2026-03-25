"""
DSCLR 网格搜索评测引擎
实现基于固定参数组合的网格搜索评测（替代MLP动态计算）
支持断点续跑和实时结果记录
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 在线模式以加载数据集 - 必须在这些变量被设置之前导入
import os
for key in ['HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    if key in os.environ:
        del os.environ[key]

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

# ============================================================
# 预定义网格参数配置
# ============================================================
PREDEFINED_GRIDS = {
    "conservative": {
        "description": "保守参数组合 - 较小的Alpha和较高的Tau，减少惩罚力度",
        "params": [
            (0.5, 0.50), (0.5, 0.55), (0.5, 0.60),
            (0.8, 0.50), (0.8, 0.55), (0.8, 0.60),
            (1.0, 0.50), (1.0, 0.55), (1.0, 0.60),
        ]
    },
    "balanced": {
        "description": "平衡参数组合 - 适中的Alpha和Tau，平衡Precision和Recall",
        "params": [
            (0.8, 0.45), (0.8, 0.50), (0.8, 0.55),
            (1.0, 0.45), (1.0, 0.50), (1.0, 0.55),
            (1.2, 0.45), (1.2, 0.50), (1.2, 0.55),
        ]
    },
    "aggressive": {
        "description": "激进参数组合 - 较大的Alpha和较低的Tau，增强惩罚力度",
        "params": [
            (1.0, 0.40), (1.0, 0.45), (1.0, 0.50),
            (1.2, 0.40), (1.2, 0.45), (1.2, 0.50),
            (1.5, 0.40), (1.5, 0.45), (1.5, 0.50),
        ]
    },
    "fine_grained": {
        "description": "细粒度参数组合 - 更密集的参数网格",
        "params": [
            (0.6, 0.40), (0.6, 0.45), (0.6, 0.50), (0.6, 0.55),
            (0.8, 0.40), (0.8, 0.45), (0.8, 0.50), (0.8, 0.55),
            (1.0, 0.40), (1.0, 0.45), (1.0, 0.50), (1.0, 0.55),
            (1.2, 0.40), (1.2, 0.45), (1.2, 0.50), (1.2, 0.55),
        ]
    },
    "repllama_25": {
        "description": "RepLLaMA专用25组实验 - Tau探底盘[0.50-0.90], Alpha探火力[0.5-2.0]",
        "params": [
            # Alpha=0.5 (温和削弱)
            (0.5, 0.50), (0.5, 0.60), (0.5, 0.70), (0.5, 0.80), (0.5, 0.90),
            # Alpha=0.8 (轻度削弱)
            (0.8, 0.50), (0.8, 0.60), (0.8, 0.70), (0.8, 0.80), (0.8, 0.90),
            # Alpha=1.2 (标准惩罚)
            (1.2, 0.50), (1.2, 0.60), (1.2, 0.70), (1.2, 0.80), (1.2, 0.90),
            # Alpha=1.5 (强力惩罚)
            (1.5, 0.50), (1.5, 0.60), (1.5, 0.70), (1.5, 0.80), (1.5, 0.90),
            # Alpha=2.0 (重炮轰击)
            (2.0, 0.50), (2.0, 0.60), (2.0, 0.70), (2.0, 0.80), (2.0, 0.90),
        ]
    }
}


def get_model_cache_dir(base_cache_dir: str, model_name: str) -> str:
    """根据模型名称获取模型专属的缓存目录"""
    # 检查实际存在的目录
    if os.path.exists(os.path.join(base_cache_dir, "repllama_v1_7b_lora_passage")):
        return os.path.join(base_cache_dir, "repllama_v1_7b_lora_passage")
    
    if "mistral" in model_name.lower():
        model_subdir = "e5-mistral-7b"
    elif "bge" in model_name.lower():
        model_subdir = "bge-large-en"
    elif "repllama" in model_name.lower():
        model_subdir = "repllama-v1-7b"
    else:
        model_subdir = model_name.split("/")[-1].replace("-", "_")
    
    return os.path.join(base_cache_dir, model_subdir)


def get_model_name_short(model_name: str) -> str:
    """从模型全名获取短名称用于缓存"""
    if "mistral" in model_name.lower():
        return "e5-mistral-7b"
    elif "bge" in model_name.lower():
        return "bge-large-en"
    elif "repllama" in model_name.lower():
        return "repllama_v1_7b_lora_passage"
    else:
        return model_name.split("/")[-1].replace("-", "_")


def load_cached_embeddings(
    cache_dir: str,
    task_name: str,
    model_name: str
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """尝试加载缓存的文档向量 - 支持多种格式"""
    import torch
    import numpy as np
    
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    model_name_short = get_model_name_short(model_name)
    
    # 首先尝试加载 .pt 格式的缓存（engine_dscrl.py 生成的格式）
    # 检查多种可能的文件名
    pt_candidates = [
        os.path.join(model_cache_dir, f"{task_name}_repllama_corpus_fixed.pt"),
        os.path.join(model_cache_dir, f"{task_name}_repllama_corpus.pt"),
        os.path.join(model_cache_dir, f"{task_name}_repllama_v1_7b_lora_passage_corpus.pt"),
    ]
    for pt_cache_file in pt_candidates:
        if os.path.exists(pt_cache_file):
            logger.info(f"📂 加载 PT 格式缓存: {pt_cache_file}")
            try:
                data = torch.load(pt_cache_file, map_location='cpu', weights_only=False)
                if isinstance(data, dict) and 'documents' in data and 'doc_ids' in data:
                    embeddings = data['documents']
                    doc_ids = data['doc_ids']
                    logger.info(f"✅ PT 缓存加载成功: {len(doc_ids)} 个文档, shape={embeddings.shape}")
                    return embeddings, doc_ids
            except Exception as e:
                logger.warning(f"⚠️ PT 缓存加载失败: {e}")
    
    # 尝试 .npy 格式
    cache_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")

    if os.path.exists(cache_file) and os.path.exists(ids_file):
        logger.info(f"📂 加载 NPY 缓存: {cache_file}")
        
        try:
            embeddings = np.load(cache_file)
            with open(ids_file, 'r') as f:
                doc_ids = json.load(f)
            logger.info(f"✅ NPY 缓存加载成功: {len(doc_ids)} 个文档, shape={embeddings.shape}")
            return torch.tensor(embeddings), doc_ids
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
                        embeddings = torch.stack(embeddings_list)
                        logger.info(f"✅ NPY 缓存加载成功 (dict格式): {len(doc_ids)} 个文档, shape={embeddings.shape}")
                        return embeddings, doc_ids
            except Exception as e:
                logger.warning(f"⚠️ NPY 缓存加载失败: {e}")

    logger.info(f"⚠️ 未找到缓存文件")
    return None


def save_embeddings_cache(
    cache_dir: str,
    task_name: str,
    model_name: str,
    embeddings: torch.Tensor,
    doc_ids: List[str]
):
    """保存文档向量缓存"""
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    model_name_short = get_model_name_short(model_name)
    
    os.makedirs(model_cache_dir, exist_ok=True)
    
    cache_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_embeddings.npy")
    ids_file = os.path.join(model_cache_dir, f"{task_name}_{model_name_short}_corpus_ids.json")
    
    np.save(cache_file, embeddings.cpu().numpy())
    with open(ids_file, 'w') as f:
        json.dump(doc_ids, f)
    
    logger.info(f"💾 文档向量已缓存: {cache_file}")


def parse_grid_params(grid_params_str: str) -> List[Tuple[float, float]]:
    """
    解析网格参数字符串
    格式: "alpha1,tau1;alpha2,tau2;..."
    示例: "0.5,0.45;1.0,0.5;1.2,0.55"
    """
    params = []
    for pair in grid_params_str.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        try:
            alpha, tau = map(float, pair.split(","))
            params.append((alpha, tau))
        except ValueError:
            logger.warning(f"⚠️ 忽略无效参数对: {pair}")
    return params


class GridSearchCheckpoint:
    """
    网格搜索检查点管理器
    支持断点续跑和实时结果记录
    """
    
    def __init__(self, output_dir: str, grid_params: List[Tuple[float, float]]):
        self.output_dir = output_dir
        self.grid_params = grid_params
        self.checkpoint_file = os.path.join(output_dir, "grid_search_checkpoint.json")
        self.results_file = os.path.join(output_dir, "grid_search_results.json")
        
        # 已完成的参数组合索引
        self.completed_indices = set()
        self.results = []
        
        # 尝试加载已有检查点
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """加载已有检查点"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                self.completed_indices = set(data.get("completed_indices", []))
                logger.info(f"📂 加载检查点: 已完成 {len(self.completed_indices)}/{len(self.grid_params)} 组参数")
            except Exception as e:
                logger.warning(f"⚠️ 检查点加载失败: {e}")
        
        # 加载已有结果
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    self.results = json.load(f)
                logger.info(f"📂 加载已有结果: {len(self.results)} 条记录")
            except Exception as e:
                logger.warning(f"⚠️ 结果文件加载失败: {e}")
    
    def is_completed(self, param_index: int) -> bool:
        """检查某组参数是否已完成"""
        return param_index in self.completed_indices
    
    def save_result(self, param_index: int, alpha: float, tau: float, result: Dict):
        """保存单组参数的结果（实时写入）"""
        # 标记为已完成
        self.completed_indices.add(param_index)
        
        # 构建结果记录
        result_record = {
            "param_index": param_index,
            "alpha": alpha,
            "tau": tau,
            "timestamp": datetime.now().isoformat(),
            **result
        }
        
        # 更新内存中的结果列表
        # 如果已存在则更新，否则添加
        existing_idx = None
        for i, r in enumerate(self.results):
            if r.get("param_index") == param_index:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.results[existing_idx] = result_record
        else:
            self.results.append(result_record)
        
        # 实时写入文件
        self._flush()
    
    def _flush(self):
        """将当前状态写入磁盘"""
        try:
            # 保存检查点
            checkpoint_data = {
                "completed_indices": list(self.completed_indices),
                "total_params": len(self.grid_params),
                "last_update": datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # 保存结果
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            logger.error(f"❌ 检查点保存失败: {e}")
    
    def get_progress(self) -> Tuple[int, int]:
        """获取进度信息 (已完成, 总数)"""
        return len(self.completed_indices), len(self.grid_params)


class GridSearchEngine:
    """
    DSCLR 网格搜索评测引擎
    使用固定参数组合替代MLP动态计算
    """
    
    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        device: str = "cuda",
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        seed: int = 42
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化评估组件"""
        # 动态导入以避免循环依赖
        sys.path.insert(0, '/home/luwa/Documents/DSCLR')
        from eval.engine import FollowIRDataLoader
        from eval.models import ModelFactory
        from eval.metrics.evaluator import FollowIREvaluator
        
        # 使用与 engine_dscrl.py 相同的数据加载器（加载本地数据）
        self.data_loader = FollowIRDataLoader(self.task_name)
        
        # 加载编码器
        logger.info(f"📥 加载编码器: {self.model_name}")
        self.encoder = ModelFactory.create(self.model_name, device=self.device)
        
        # 初始化检索器
        from eval.models.encoder import DenseRetriever
        self.retriever = DenseRetriever(self.encoder)
        
        # 初始化评估器
        self.evaluator = FollowIREvaluator(self.task_name)
        
        logger.info(f"✅ 组件初始化完成")
    
    def run_grid_search(
        self,
        grid_params: List[Tuple[float, float]],
        resume_checkpoint: Optional[str] = None
    ) -> Dict:
        """
        执行网格搜索评测
        
        Args:
            grid_params: 参数组合列表 [(alpha1, tau1), (alpha2, tau2), ...]
            resume_checkpoint: 检查点文件路径（用于断点续跑）
        
        Returns:
            包含所有结果的字典
        """
        logger.info(f"🚀 启动网格搜索评测")
        logger.info(f"   参数组合数: {len(grid_params)}")
        logger.info(f"   输出目录: {self.output_dir}")
        
        # 初始化检查点管理器
        checkpoint = GridSearchCheckpoint(self.output_dir, grid_params)
        
        # 如果指定了外部检查点文件，加载它
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            logger.info(f"📂 从外部检查点恢复: {resume_checkpoint}")
            checkpoint.checkpoint_file = resume_checkpoint
            checkpoint._load_checkpoint()
        
        # 加载数据
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()
        
        # 编码/加载文档
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        
        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)
        
        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"✅ 使用缓存的文档向量 ({len(cached_doc_ids)} 个)")
                self.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
            else:
                logger.warning(f"⚠️ 缓存文档ID不匹配，重新编码...")
                self._encode_documents(corpus, all_doc_ids)
        else:
            self._encode_documents(corpus, all_doc_ids)
        
        # 加载 dual queries (Q+ 和 Q-) - v2 版本
        logger.info("🔤 加载并编码 dual queries...")
        dual_queries_cache = os.path.join(
            "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v4",
            f"dual_queries_v4_{self.task_name}.jsonl"
        )
        
        # 加载 dual queries 数据
        q_plus_og = {}  # OG 查询的 Q+
        q_plus_changed = {}  # Changed 查询的 Q+
        q_minus_changed = {}  # Changed 查询的 Q-
        neg_mask = {}  # 标记是否为 [NONE]
        
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
                        neg_mask[qid] = 0.0 if q_minus == "[NONE]" else 1.0
            
            logger.info(f"✅ 加载完成: {len(q_plus_og)} OG, {len(q_plus_changed)} Changed")
        else:
            logger.warning(f"⚠️ 未找到 dual queries 缓存，使用原始查询")
            q_plus_og = q_og
            q_plus_changed = q_changed
            q_minus_changed = q_changed
            neg_mask = {qid: 1.0 for qid in q_changed}
        
        # 编码 OG 查询 (仅 Q+)
        q_og_items = list(q_plus_og.items())
        q_og_list = [item[1] for item in q_og_items]
        q_og_emb = self.encoder.encode_queries(q_og_list, batch_size=self.batch_size)
        q_og_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_og_items)}
        
        # 编码 Changed 查询 (Q+ 和 Q-)
        q_changed_items = list(q_plus_changed.items())
        q_plus_list = [item[1] for item in q_changed_items]
        q_minus_list = [q_minus_changed.get(item[0], item[1]) for item in q_changed_items]
        
        q_plus_changed_emb = self.encoder.encode_queries(q_plus_list, batch_size=self.batch_size)
        q_minus_changed_emb = self.encoder.encode_queries(q_minus_list, batch_size=self.batch_size)
        q_changed_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_changed_items)}
        
        # 创建 neg_mask 张量
        neg_mask_tensor = torch.tensor([neg_mask.get(item[0], 1.0) for item in q_changed_items], 
                                       dtype=torch.float32)
        
        # 执行网格搜索
        all_results = []
        best_result = None
        best_score = -float('inf')
        
        for idx, (alpha, tau) in enumerate(grid_params):
            # 检查是否已完成
            if checkpoint.is_completed(idx):
                logger.info(f"⏭️  跳过已完成参数 [{idx+1}/{len(grid_params)}]: alpha={alpha:.2f}, tau={tau:.2f}")
                continue
            
            logger.info(f"\n🔍 测试参数 [{idx+1}/{len(grid_params)}]: alpha={alpha:.2f}, tau={tau:.2f}")
            
            try:
                # 执行评测
                result = self._evaluate_params(
                    q_og_emb, q_plus_changed_emb, q_minus_changed_emb, 
                    candidates, corpus, alpha, tau, 
                    q_og_qid_to_idx, q_changed_qid_to_idx, neg_mask_tensor
                )
                
                # 记录结果
                result_summary = {
                    "p_mrr": result["p_mrr"],
                    "og_ndcg@5": result["og_ndcg@5"],
                    "changed_ndcg@5": result["changed_ndcg@5"],
                    "og_mrr": result["og_mrr"],
                    "changed_mrr": result["changed_mrr"]
                }
                
                # 实时保存结果
                checkpoint.save_result(idx, alpha, tau, result_summary)
                all_results.append({
                    "alpha": alpha,
                    "tau": tau,
                    **result_summary
                })
                
                # 更新最佳结果
                if result["p_mrr"] > best_score:
                    best_score = result["p_mrr"]
                    best_result = {
                        "alpha": alpha,
                        "tau": tau,
                        **result_summary
                    }
                
                logger.info(f"   ✅ p-MRR: {result['p_mrr']:.4f}, OG nDCG@5: {result['og_ndcg@5']:.4f}, Changed nDCG@5: {result['changed_ndcg@5']:.4f}")
                
                # 坏例检测 (仅在指定参数时)
                if alpha == 2.0 and tau == 0.70:
                    logger.info("🚨 执行坏例检测...")
                    qids_list = list(candidates.keys())
                    bad_cases = self._detect_bad_cases(
                        qids_list, candidates, corpus,
                        result["scores_og"], result["scores_changed"], result["scores_q_plus"],
                        q_plus_changed, q_minus_changed, q_raw_changed,
                        alpha, tau
                    )
                    
                    if bad_cases:
                        # 保存坏例报告
                        bad_case_file = os.path.join(self.output_dir, "bad_cases_diagnosis.jsonl")
                        with open(bad_case_file, 'w', encoding='utf-8') as f:
                            for case in bad_cases:
                                f.write(json.dumps(case, ensure_ascii=False) + '\n')
                        logger.info(f"🚨 发现 {len(bad_cases)} 个坏例，已保存到: {bad_case_file}")
                    else:
                        logger.info("✅ 未发现坏例 (rank_before<=10 且 rank_after>20 的受害者文档)")
                
            except Exception as e:
                logger.error(f"   ❌ 参数测试失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成最终报告
        completed, total = checkpoint.get_progress()
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 网格搜索完成: {completed}/{total} 组参数")
        
        if best_result:
            logger.info(f"🏆 最佳参数组合:")
            logger.info(f"   Alpha: {best_result['alpha']:.2f}")
            logger.info(f"   Tau: {best_result['tau']:.2f}")
            logger.info(f"   p-MRR: {best_result['p_mrr']:.4f}")
            logger.info(f"   OG nDCG@5: {best_result['og_ndcg@5']:.4f}")
            logger.info(f"   Changed nDCG@5: {best_result['changed_ndcg@5']:.4f}")
        
        # 保存最终报告
        self._save_final_report(all_results, best_result, grid_params)
        
        return {
            "all_results": all_results,
            "best_result": best_result,
            "grid_params": grid_params
        }
    
    def _encode_documents(self, corpus: Dict, doc_ids: List[str]):
        """编码文档"""
        logger.info(f"📝 编码 {len(doc_ids)} 个文档...")
        doc_texts = [corpus[did]['text'] for did in doc_ids]
        self.retriever.index_documents(doc_ids, doc_texts, self.batch_size)
        
        # 保存缓存
        if self.use_cache:
            save_embeddings_cache(
                self.cache_dir, self.task_name, self.model_name,
                self.retriever.doc_embeddings, self.retriever.doc_ids
            )
    
    def _get_all_candidate_doc_ids(self, candidates: Dict) -> List[str]:
        """获取所有候选文档ID"""
        all_doc_ids = set()
        for qid, doc_list in candidates.items():
            all_doc_ids.update(doc_list)
        return list(all_doc_ids)
    
    def _evaluate_params(
        self,
        q_og_emb: torch.Tensor,
        q_plus_changed_emb: torch.Tensor,
        q_minus_changed_emb: torch.Tensor,
        candidates: Dict,
        corpus: Dict,
        alpha: float,
        tau: float,
        q_og_qid_to_idx: Dict[str, int],
        q_changed_qid_to_idx: Dict[str, int],
        neg_mask: torch.Tensor
    ) -> Dict:
        """
        使用指定参数执行评测
        
        实现DSCLR双流打分逻辑:
        S_final = S_base - alpha * ReLU(S_neg - tau)
        其中:
        - S_base = sim(Q+, D)  [Q+ 与文档的相似度]
        - S_neg = sim(Q-, D)   [Q- 与文档的相似度]
        """
        results_og = {}
        results_changed = {}
        
        qids = list(candidates.keys())
        
        for qid in qids:
            doc_ids = candidates[qid]
            
            # 获取文档向量 - doc_embeddings 是字典 {doc_id: embedding}
            doc_emb_list = []
            for did in doc_ids:
                doc_emb_list.append(self.retriever.doc_embeddings[did])
            doc_emb = torch.stack(doc_emb_list).to(self.device)
            
            # 获取查询向量 - 使用映射找到正确的索引
            og_qid = f"{qid}-og"
            changed_qid = f"{qid}-changed"
            og_idx = q_og_qid_to_idx.get(og_qid, 0)
            changed_idx = q_changed_qid_to_idx.get(changed_qid, 0)
            
            og_emb = q_og_emb[og_idx].to(self.device)
            q_plus_emb = q_plus_changed_emb[changed_idx].to(self.device)
            q_minus_emb = q_minus_changed_emb[changed_idx].to(self.device)
            
            # 获取 neg_mask (是否为 [NONE])
            neg_mask_val = neg_mask[changed_idx].item()
            
            # 计算相似度 - 使用点积（向量已归一化）
            # OG 查询使用自身的 embedding
            sim_og = torch.matmul(og_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            
            # DSCLR 打分: S_base = sim(Q+, D), S_neg = sim(Q-, D)
            S_base = torch.matmul(q_plus_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_neg = torch.matmul(q_minus_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            
            # OG模式直接返回 sim_og
            scores_og = sim_og.cpu().numpy()
            
            # Changed模式: DSCLR 打分
            # S_final = S_base - alpha * ReLU(S_neg - tau)
            # 如果 neg_mask=0 ([NONE])，则不应用惩罚
            if neg_mask_val > 0:
                penalty = torch.relu(S_neg - tau)
                scores_changed = (S_base - alpha * penalty).cpu().numpy()
            else:
                # [NONE] 情况，不应用惩罚
                scores_changed = S_base.cpu().numpy()
            
            # 存储结果 - 注意：需要添加 -og 和 -changed 后缀
            results_og[f"{qid}-og"] = {did: float(scores_og[i]) for i, did in enumerate(doc_ids)}
            results_changed[f"{qid}-changed"] = {did: float(scores_changed[i]) for i, did in enumerate(doc_ids)}
            
            # 保存 S_base (Q+ 分数) 用于坏例检测
            if not hasattr(self, '_scores_q_plus'):
                self._scores_q_plus = {}
            self._scores_q_plus[f"{qid}-changed"] = {did: float(S_base[i].item()) for i, did in enumerate(doc_ids)}
        
        # 评估结果 - FollowIREvaluator 需要同时传入 og 和 changed 结果
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "p_mrr": metrics.get("p-MRR", 0.0),
            "og_ndcg@5": metrics.get("original", {}).get("ndcg_at_5", 0.0),
            "changed_ndcg@5": metrics.get("changed", {}).get("ndcg_at_5", 0.0),
            "og_mrr": metrics.get("original", {}).get("mrr_at_10", 0.0),
            "changed_mrr": metrics.get("changed", {}).get("mrr_at_10", 0.0),
            "full_metrics": metrics,
            "scores_og": results_og,
            "scores_changed": results_changed,
            "scores_q_plus": self._scores_q_plus
        }
    
    def _evaluate_params_dadt(
        self,
        q_og_emb: torch.Tensor,
        q_plus_changed_emb: torch.Tensor,
        q_minus_changed_emb: torch.Tensor,
        candidates: Dict,
        corpus: Dict,
        base_alpha: float,
        gamma: float,
        q_og_qid_to_idx: Dict[str, int],
        q_changed_qid_to_idx: Dict[str, int],
        neg_mask: torch.Tensor
    ) -> Dict:
        """
        DADT (Distribution-Aware Dynamic Threshold) 评估
        
        使用统计分布动态计算 tau:
        tau = mu + gamma * sigma
        其中 mu, sigma 是 S_neg (Q- 与文档相似度) 的均值和标准差
        
        复用现有 _evaluate_params 的核心逻辑，仅替换 tau 计算方式
        """
        results_og = {}
        results_changed = {}
        
        qids = list(candidates.keys())
        
        for qid in qids:
            doc_ids = candidates[qid]
            
            # 获取文档向量
            doc_emb_list = []
            for did in doc_ids:
                doc_emb_list.append(self.retriever.doc_embeddings[did])
            doc_emb = torch.stack(doc_emb_list).to(self.device)
            
            # 获取查询向量
            og_qid = f"{qid}-og"
            changed_qid = f"{qid}-changed"
            og_idx = q_og_qid_to_idx.get(og_qid, 0)
            changed_idx = q_changed_qid_to_idx.get(changed_qid, 0)
            
            og_emb = q_og_emb[og_idx].to(self.device)
            q_plus_emb = q_plus_changed_emb[changed_idx].to(self.device)
            q_minus_emb = q_minus_changed_emb[changed_idx].to(self.device)
            
            # 获取 neg_mask
            neg_mask_val = neg_mask[changed_idx].item()
            
            # 计算相似度
            sim_og = torch.matmul(og_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_base = torch.matmul(q_plus_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_neg = torch.matmul(q_minus_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            
            # OG模式
            scores_og = sim_og.cpu().numpy()
            
            # Changed模式: DADT 动态阈值
            if neg_mask_val > 0:
                # 【DADT核心】基于 S_neg 分布动态计算 tau
                dynamic_alpha, dynamic_tau = get_dadt_params(S_neg, base_alpha=base_alpha, gamma=gamma)
                
                # 复用现有计分公式
                penalty = torch.relu(S_neg - dynamic_tau)
                scores_changed = (S_base - dynamic_alpha * penalty).cpu().numpy()
            else:
                # [NONE] 情况，不应用惩罚
                scores_changed = S_base.cpu().numpy()
            
            # 存储结果
            results_og[f"{qid}-og"] = {did: float(scores_og[i]) for i, did in enumerate(doc_ids)}
            results_changed[f"{qid}-changed"] = {did: float(scores_changed[i]) for i, did in enumerate(doc_ids)}
        
        # 评估结果
        metrics = self.evaluator.evaluate(results_og, results_changed)
        
        return {
            "p_mrr": metrics.get("p-MRR", 0.0),
            "og_ndcg@5": metrics.get("original", {}).get("ndcg_at_5", 0.0),
            "changed_ndcg@5": metrics.get("changed", {}).get("ndcg_at_5", 0.0),
            "og_mrr": metrics.get("original", {}).get("mrr_at_10", 0.0),
            "changed_mrr": metrics.get("changed", {}).get("mrr_at_10", 0.0),
            "full_metrics": metrics,
            "scores_og": results_og,
            "scores_changed": results_changed
        }

    def _evaluate_params_pat(
        self,
        q_og_emb: torch.Tensor,
        q_plus_changed_emb: torch.Tensor,
        q_minus_changed_emb: torch.Tensor,
        candidates: Dict,
        corpus: Dict,
        alpha: float,
        tau_base: float,
        lambda_weight: float,
        q_og_qid_to_idx: Dict[str, int],
        q_changed_qid_to_idx: Dict[str, int],
        neg_mask: torch.Tensor
    ) -> Dict:
        """
        PAT (Positive-Aware Tolerance) 评估

        使用正向得分自适应调整阈值:
        dynamic_tau = tau_base + lambda_weight * S_base
        penalty = alpha * relu(S_neg - dynamic_tau)
        S_final = S_base - penalty
        """
        from eval.pat_scorer import PAT_Scorer

        results_og = {}
        results_changed = {}

        qids = list(candidates.keys())

        for qid in qids:
            doc_ids = candidates[qid]

            doc_emb_list = []
            for did in doc_ids:
                doc_emb_list.append(self.retriever.doc_embeddings[did])
            doc_emb = torch.stack(doc_emb_list).to(self.device)

            og_qid = f"{qid}-og"
            changed_qid = f"{qid}-changed"
            og_idx = q_og_qid_to_idx.get(og_qid, 0)
            changed_idx = q_changed_qid_to_idx.get(changed_qid, 0)

            og_emb = q_og_emb[og_idx].to(self.device)
            q_plus_emb = q_plus_changed_emb[changed_idx].to(self.device)
            q_minus_emb = q_minus_changed_emb[changed_idx].to(self.device)

            neg_mask_val = neg_mask[changed_idx].item()

            sim_og = torch.matmul(og_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_base = torch.matmul(q_plus_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_neg = torch.matmul(q_minus_emb.unsqueeze(0), doc_emb.T).squeeze(0)

            scores_og = sim_og.cpu().numpy()

            if neg_mask_val > 0:
                # 【PAT核心】动态阈值计算
                dynamic_tau = tau_base + lambda_weight * S_base
                penalty = alpha * torch.relu(S_neg - dynamic_tau)
                scores_changed = (S_base - penalty).cpu().numpy()
            else:
                scores_changed = S_base.cpu().numpy()

            results_og[f"{qid}-og"] = {did: float(scores_og[i]) for i, did in enumerate(doc_ids)}
            results_changed[f"{qid}-changed"] = {did: float(scores_changed[i]) for i, did in enumerate(doc_ids)}

        metrics = self.evaluator.evaluate(results_og, results_changed)

        return {
            "p_mrr": metrics.get("p-MRR", 0.0),
            "og_ndcg@5": metrics.get("original", {}).get("ndcg_at_5", 0.0),
            "changed_ndcg@5": metrics.get("changed", {}).get("ndcg_at_5", 0.0),
            "og_mrr": metrics.get("original", {}).get("mrr_at_10", 0.0),
            "changed_mrr": metrics.get("changed", {}).get("mrr_at_10", 0.0),
            "full_metrics": metrics,
            "scores_og": results_og,
            "scores_changed": results_changed
        }

    def _evaluate_params_pat_protected(
        self,
        q_og_emb: torch.Tensor,
        q_plus_changed_emb: torch.Tensor,
        q_minus_changed_emb: torch.Tensor,
        candidates: Dict,
        corpus: Dict,
        alpha: float,
        tau_base: float,
        lambda_weight: float,
        q_og_qid_to_idx: Dict[str, int],
        q_changed_qid_to_idx: Dict[str, int],
        neg_mask: torch.Tensor,
        top_k: int = 5,
        protection_factor: float = 0.3
    ) -> Dict:
        """
        PAT 评估 + OG排名保护分段惩罚

        策略:
        - og_rank <= top_k: 保护文档，penalty * protection_factor
        - og_rank > top_k: 正常惩罚
        """
        from eval.pat_scorer import PAT_Scorer
        import numpy as np

        results_og = {}
        results_changed = {}
        og_ranks_map = {}

        qids = list(candidates.keys())

        for qid in qids:
            doc_ids = candidates[qid]
            doc_emb_list = []
            for did in doc_ids:
                doc_emb_list.append(self.retriever.doc_embeddings[did])
            doc_emb = torch.stack(doc_emb_list).to(self.device)

            og_qid = f"{qid}-og"
            changed_qid = f"{qid}-changed"
            og_idx = q_og_qid_to_idx.get(og_qid, 0)
            changed_idx = q_changed_qid_to_idx.get(changed_qid, 0)

            og_emb = q_og_emb[og_idx].to(self.device)
            q_plus_emb = q_plus_changed_emb[changed_idx].to(self.device)
            q_minus_emb = q_minus_changed_emb[changed_idx].to(self.device)

            neg_mask_val = neg_mask[changed_idx].item()

            sim_og = torch.matmul(og_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_base = torch.matmul(q_plus_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_neg = torch.matmul(q_minus_emb.unsqueeze(0), doc_emb.T).squeeze(0)

            scores_og_np = sim_og.cpu().numpy()

            og_sorted_indices = np.argsort(-scores_og_np)
            og_ranks = np.zeros(len(doc_ids), dtype=np.int32)
            for rank, idx in enumerate(og_sorted_indices):
                og_ranks[idx] = rank + 1

            og_ranks_map[changed_qid] = og_ranks

            if neg_mask_val > 0:
                S_base_np = S_base.cpu().numpy()
                S_neg_np = S_neg.cpu().numpy()
                scores_changed_np = PAT_Scorer.compute_with_og_rank_protection(
                    S_base=S_base_np,
                    S_neg=S_neg_np,
                    og_ranks=og_ranks,
                    alpha=alpha,
                    tau_base=tau_base,
                    lambda_weight=lambda_weight,
                    top_k=top_k,
                    protection_factor=protection_factor
                )
            else:
                scores_changed_np = S_base.cpu().numpy()

            results_og[f"{qid}-og"] = {did: float(scores_og_np[i]) for i, did in enumerate(doc_ids)}
            results_changed[f"{qid}-changed"] = {did: float(scores_changed_np[i]) for i, did in enumerate(doc_ids)}

        metrics = self.evaluator.evaluate(results_og, results_changed)

        return {
            "p_mrr": metrics.get("p-MRR", 0.0),
            "og_ndcg@5": metrics.get("original", {}).get("ndcg_at_5", 0.0),
            "changed_ndcg@5": metrics.get("changed", {}).get("ndcg_at_5", 0.0),
            "og_mrr": metrics.get("original", {}).get("mrr_at_10", 0.0),
            "changed_mrr": metrics.get("changed", {}).get("mrr_at_10", 0.0),
            "full_metrics": metrics,
            "scores_og": results_og,
            "scores_changed": results_changed
        }

    def _evaluate_params_pat_hybrid(
        self,
        q_og_emb: torch.Tensor,
        q_plus_changed_emb: torch.Tensor,
        q_minus_changed_emb: torch.Tensor,
        candidates: Dict,
        corpus: Dict,
        alpha: float,
        tau_base: float,
        lambda_weight: float,
        q_og_qid_to_idx: Dict[str, int],
        q_changed_qid_to_idx: Dict[str, int],
        neg_mask: torch.Tensor,
        top_k: int = 5,
        protection_factor: float = 0.3,
        boost_ndcg_factor: float = 0.5
    ) -> Dict:
        """
        PAT 评估 + 混合策略

        策略:
        1. 保护 og_rank <= top_k 的文档
        2. 对 og_rank 5-20 的文档进行轻微 boost
        """
        from eval.pat_scorer import PAT_Scorer
        import numpy as np

        results_og = {}
        results_changed = {}

        qids = list(candidates.keys())

        for qid in qids:
            doc_ids = candidates[qid]
            doc_emb_list = []
            for did in doc_ids:
                doc_emb_list.append(self.retriever.doc_embeddings[did])
            doc_emb = torch.stack(doc_emb_list).to(self.device)

            og_qid = f"{qid}-og"
            changed_qid = f"{qid}-changed"
            og_idx = q_og_qid_to_idx.get(og_qid, 0)
            changed_idx = q_changed_qid_to_idx.get(changed_qid, 0)

            og_emb = q_og_emb[og_idx].to(self.device)
            q_plus_emb = q_plus_changed_emb[changed_idx].to(self.device)
            q_minus_emb = q_minus_changed_emb[changed_idx].to(self.device)

            neg_mask_val = neg_mask[changed_idx].item()

            sim_og = torch.matmul(og_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_base = torch.matmul(q_plus_emb.unsqueeze(0), doc_emb.T).squeeze(0)
            S_neg = torch.matmul(q_minus_emb.unsqueeze(0), doc_emb.T).squeeze(0)

            scores_og_np = sim_og.cpu().numpy()

            og_sorted_indices = np.argsort(-scores_og_np)
            og_ranks = np.zeros(len(doc_ids), dtype=np.int32)
            for rank, idx in enumerate(og_sorted_indices):
                og_ranks[idx] = rank + 1

            if neg_mask_val > 0:
                S_base_np = S_base.cpu().numpy()
                S_neg_np = S_neg.cpu().numpy()
                scores_changed_np = PAT_Scorer.compute_hybrid(
                    S_base=S_base_np,
                    S_neg=S_neg_np,
                    og_ranks=og_ranks,
                    alpha=alpha,
                    tau_base=tau_base,
                    lambda_weight=lambda_weight,
                    top_k=top_k,
                    protection_factor=protection_factor,
                    boost_ndcg_factor=boost_ndcg_factor
                )
            else:
                scores_changed_np = S_base.cpu().numpy()

            results_og[f"{qid}-og"] = {did: float(scores_og_np[i]) for i, did in enumerate(doc_ids)}
            results_changed[f"{qid}-changed"] = {did: float(scores_changed_np[i]) for i, did in enumerate(doc_ids)}

        metrics = self.evaluator.evaluate(results_og, results_changed)

        return {
            "p_mrr": metrics.get("p-MRR", 0.0),
            "og_ndcg@5": metrics.get("original", {}).get("ndcg_at_5", 0.0),
            "changed_ndcg@5": metrics.get("changed", {}).get("ndcg_at_5", 0.0),
            "og_mrr": metrics.get("original", {}).get("mrr_at_10", 0.0),
            "changed_mrr": metrics.get("changed", {}).get("mrr_at_10", 0.0),
            "full_metrics": metrics,
            "scores_og": results_og,
            "scores_changed": results_changed
        }

    def _detect_bad_cases(
        self,
        qids: List[str],
        candidates: Dict,
        corpus: Dict,
        scores_og: Dict[str, np.ndarray],
        scores_changed: Dict[str, np.ndarray],
        scores_q_plus: Dict[str, np.ndarray],
        q_plus_changed: Dict[str, str],
        q_minus_changed: Dict[str, str],
        q_raw_changed: Dict[str, Tuple[str, str]],
        alpha: float,
        tau: float
    ) -> List[Dict]:
        """
        检测 DSCLR 坏例（受害者文档）
        
        触发条件：
        1. is_relevant == True (在 Ground Truth 中是相关文档)
        2. rank_before <= 10 (基线排名在前10)
        3. rank_after > 20 (惩罚后暴跌到20名开外)
        """
        bad_cases = []
        
        # 加载 qrels 获取相关文档信息
        qrels = self.evaluator.data_loader.load_qrels()
        changed_qrels = self.evaluator.data_loader.load_qrel_diff()
        
        for qid in qids:
            changed_qid = f"{qid}-changed"
            
            # 获取相关文档列表
            relevant_docs = set()
            for doc_id in candidates.get(qid, []):
                # 在 qrels 中查找相关文档
                qid_key = f"{qid}-og"  # OG qid
                if qid_key in qrels and doc_id in qrels[qid_key]:
                    relevant_docs.add(doc_id)
            
            if not relevant_docs:
                continue
            
            # 获取得分
            doc_ids = candidates[qid]
            og_qid = f"{qid}-og"
            scores_base = scores_q_plus.get(changed_qid, {})  # S_base = sim(Q+, D)
            scores_changed_arr = scores_changed.get(changed_qid, {})
            
            if not scores_base or not scores_changed_arr:
                continue
            
            # 计算排名 - rank_before 用 S_base (Q+ 的相似度)
            og_sorted = sorted(doc_ids, key=lambda d: scores_base.get(d, float('-inf')), reverse=True)
            changed_sorted = sorted(doc_ids, key=lambda d: scores_changed_arr.get(d, float('-inf')), reverse=True)
            
            og_rank = {doc: rank + 1 for rank, doc in enumerate(og_sorted)}
            changed_rank = {doc: rank + 1 for rank, doc in enumerate(changed_sorted)}
            
            # 检查每个相关文档
            for doc_id in relevant_docs:
                rank_before = og_rank.get(doc_id, float('inf'))
                rank_after = changed_rank.get(doc_id, float('inf'))
                
                # 触发条件检查
                if rank_before <= 10 and rank_after > 20:
                    # 获取查询信息
                    q_plus = q_plus_changed.get(changed_qid, "")
                    q_minus = q_minus_changed.get(changed_qid, "")
                    raw_query, instruction = q_raw_changed.get(changed_qid, ("", ""))
                    
                    # 获取文档文本
                    doc_text = corpus.get(doc_id, {}).get('text', '')[:200]
                    
                    # 获取得分
                    score_base = scores_base.get(doc_id, 0.0)
                    score_neg = scores_changed_arr.get(doc_id, 0.0)  # 实际是 S_neg
                    
                    # 计算惩罚
                    penalty = max(0, score_neg - tau)
                    final_penalty = alpha * penalty
                    
                    bad_case = {
                        "context": {
                            "query_id": changed_qid,
                            "original_query": raw_query,
                            "instruction": instruction,
                            "q_plus": q_plus,
                            "q_minus": q_minus
                        },
                        "victim_doc": {
                            "doc_id": doc_id,
                            "doc_text": doc_text,
                            "is_relevant": True
                        },
                        "physics": {
                            "rank_before": int(rank_before),
                            "rank_after": int(rank_after),
                            "score_base": float(score_base),
                            "score_neg": float(score_neg),
                            "tau": float(tau),
                            "alpha": float(alpha),
                            "final_penalty": float(final_penalty)
                        }
                    }
                    bad_cases.append(bad_case)
        
        return bad_cases
    
    def _calculate_p_mrr(self, metrics_og: Dict, metrics_changed: Dict) -> float:
        """计算p-MRR (Protection MRR)"""
        og_mrr = metrics_og.get("mrr_at_10", 0.0)
        changed_mrr = metrics_changed.get("mrr_at_10", 0.0)
        
        if og_mrr > 0:
            return (changed_mrr - og_mrr) / og_mrr
        return 0.0
    
    def _save_final_report(
        self,
        all_results: List[Dict],
        best_result: Optional[Dict],
        grid_params: List[Tuple[float, float]]
    ):
        """保存最终报告"""
        report = {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "timestamp": datetime.now().isoformat(),
            "total_params": len(grid_params),
            "completed_params": len(all_results),
            "best_result": best_result,
            "all_results": all_results
        }
        
        report_file = os.path.join(self.output_dir, "grid_search_final_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 最终报告已保存: {report_file}")
        
        # 同时保存CSV格式便于分析
        self._save_csv_report(all_results)
    
    def _save_csv_report(self, all_results: List[Dict]):
        """保存CSV格式的报告"""
        csv_file = os.path.join(self.output_dir, "grid_search_results.csv")
        
        with open(csv_file, 'w') as f:
            # 写入表头
            f.write("alpha,tau,p_mrr,og_ndcg@5,changed_ndcg@5,og_mrr,changed_mrr\n")
            
            # 写入数据
            for r in all_results:
                f.write(f"{r['alpha']:.4f},{r['tau']:.4f},{r['p_mrr']:.4f},"
                       f"{r['og_ndcg@5']:.4f},{r['changed_ndcg@5']:.4f},"
                       f"{r['og_mrr']:.4f},{r['changed_mrr']:.4f}\n")
        
        logger.info(f"💾 CSV报告已保存: {csv_file}")


def run_grid_search_evaluation(
    model_name: str,
    task_name: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    grid_params: Optional[List[Tuple[float, float]]] = None,
    predefined_grid: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    seed: int = 42
) -> Dict:
    """
    便捷的网格搜索评测入口函数
    """
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
    
    # 确定参数组合
    if grid_params is not None:
        params = grid_params
    elif predefined_grid is not None:
        if predefined_grid not in PREDEFINED_GRIDS:
            raise ValueError(f"未知的预定义网格: {predefined_grid}. "
                           f"可用选项: {list(PREDEFINED_GRIDS.keys())}")
        grid_config = PREDEFINED_GRIDS[predefined_grid]
        params = grid_config["params"]
        logger.info(f"📋 使用预定义网格: {predefined_grid}")
        logger.info(f"   {grid_config['description']}")
    else:
        # 默认使用平衡网格
        params = PREDEFINED_GRIDS["balanced"]["params"]
        logger.info("📋 使用默认平衡网格")
    
    return engine.run_grid_search(params, resume_checkpoint)


# ============================================================
# DADT (Distribution-Aware Dynamic Threshold) 统计分布路由
# ============================================================

def get_dadt_params(
    neg_scores: torch.Tensor,
    base_alpha: float = 2.0,
    gamma: float = 1.0
) -> Tuple[float, float]:
    """
    DADT 核心接口：基于负样本分布计算动态阈值
    
    Args:
        neg_scores: 负样本相似度分数 (Top-K 负向流)
        base_alpha: 基础惩罚力度
        gamma: 标准差乘数（控制阈值偏离均值的程度）
        
    Returns:
        (dynamic_alpha, dynamic_tau): 动态参数元组
    """
    # 空数组安全检查
    if neg_scores is None or len(neg_scores) == 0:
        return (base_alpha, 0.0)
    
    # 转换为 numpy 并展平
    scores = neg_scores.detach().cpu().numpy().flatten()
    
    # 计算统计量
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    
    # 除零保护
    if sigma < 1e-8:
        dynamic_tau = mu
    else:
        dynamic_tau = mu + gamma * sigma
    
    return (base_alpha, dynamic_tau)


def run_dadt_grid_search_evaluation(
    model_name: str,
    task_name: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    gamma_list: List[float] = None,
    alpha_list: List[float] = None,
    seed: int = 42
) -> Dict:
    """
    DADT Grid Search 评测入口
    
    遍历 (gamma, alpha) 组合空间，使用统计分布动态计算 tau
    
    Args:
        gamma_list: gamma 参数列表，默认 [0.0, 0.5, 1.0, 1.5, 2.0]
        alpha_list: alpha 参数列表，默认 [1.5, 2.0, 2.5]
    """
    # 默认参数
    if gamma_list is None:
        gamma_list = [0.0, 0.5, 1.0, 1.5, 2.0]
    if alpha_list is None:
        alpha_list = [1.5, 2.0, 2.5]
    
    logger.info("\n" + "="*70)
    logger.info("🔬 DADT (Distribution-Aware Dynamic Threshold) Grid Search")
    logger.info("="*70)
    logger.info(f"Gamma 列表: {gamma_list}")
    logger.info(f"Alpha 列表: {alpha_list}")
    logger.info(f"总组合数: {len(gamma_list)} × {len(alpha_list)} = {len(gamma_list) * len(alpha_list)}")
    logger.info("="*70)
    
    # 创建引擎（复用现有基础设施）
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
    
    # 加载数据（复用现有数据加载流程）
    logger.info("\n📊 加载评测数据...")
    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    q_raw_og, q_raw_changed = engine.data_loader.load_raw_queries()
    
    # 编码/加载文档
    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)
    
    cached_data = None
    if use_cache:
        cached_data = load_cached_embeddings(cache_dir or DEFAULT_CACHE_DIR, task_name, model_name)
    
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
    
    # 加载 dual queries (Q+ 和 Q-)
    logger.info("🔤 加载并编码 dual queries...")
    dual_queries_cache = os.path.join(
        "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v4",
        f"dual_queries_v4_{task_name}.jsonl"
    )
    
    # 加载 dual queries 数据
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
        logger.warning(f"⚠️ 未找到 dual queries 缓存，使用原始查询")
        q_plus_og = q_og
        q_plus_changed = q_changed
        q_minus_changed = q_changed
        neg_mask_dict = {qid: 1.0 for qid in q_changed}
    
    # 编码 OG 查询 (仅 Q+)
    q_og_items = list(q_plus_og.items())
    q_og_list = [item[1] for item in q_og_items]
    q_og_emb = engine.encoder.encode_queries(q_og_list, batch_size=batch_size)
    q_og_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_og_items)}
    
    # 编码 Changed 查询 (Q+ 和 Q-)
    q_changed_items = list(q_plus_changed.items())
    q_plus_list = [item[1] for item in q_changed_items]
    q_minus_list = [q_minus_changed.get(item[0], item[1]) for item in q_changed_items]
    
    q_plus_changed_emb = engine.encoder.encode_queries(q_plus_list, batch_size=batch_size)
    q_minus_changed_emb = engine.encoder.encode_queries(q_minus_list, batch_size=batch_size)
    q_changed_qid_to_idx = {item[0]: idx for idx, item in enumerate(q_changed_items)}
    
    # 创建 neg_mask 张量
    neg_mask = torch.tensor([neg_mask_dict.get(item[0], 1.0) for item in q_changed_items], 
                                   dtype=torch.float32)
    
    # 执行 DADT Grid Search
    all_results = []
    total = len(gamma_list) * len(alpha_list)
    
    for gamma in gamma_list:
        for alpha in alpha_list:
            # 使用 DADT 模式执行评估
            result = engine._evaluate_params_dadt(
                q_og_emb=q_og_emb,
                q_plus_changed_emb=q_plus_changed_emb,
                q_minus_changed_emb=q_minus_changed_emb,
                candidates=candidates,
                corpus=corpus,
                base_alpha=alpha,
                gamma=gamma,
                q_og_qid_to_idx=q_og_qid_to_idx,
                q_changed_qid_to_idx=q_changed_qid_to_idx,
                neg_mask=neg_mask
            )
            
            all_results.append({
                'gamma': gamma,
                'alpha': alpha,
                'p_mrr': result['p_mrr'],
                'og_ndcg@5': result['og_ndcg@5'],
                'changed_ndcg@5': result['changed_ndcg@5'],
                'og_mrr': result['og_mrr'],
                'changed_mrr': result['changed_mrr']
            })
            
            logger.info(f"  γ={gamma:.1f}, α={alpha:.1f} → p-MRR={result['p_mrr']:.4f}, "
                       f"Changed nDCG@5={result['changed_ndcg@5']:.4f}")
    
    # 打印汇总表格
    _print_dadt_summary_table(all_results)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 结果
    results_file = os.path.join(output_dir, "dadt_grid_search_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'gamma_list': gamma_list,
            'alpha_list': alpha_list,
            'results': all_results
        }, f, indent=2)
    logger.info(f"\n💾 DADT 结果已保存: {results_file}")
    
    # CSV 结果
    csv_file = os.path.join(output_dir, "dadt_grid_search_results.csv")
    with open(csv_file, 'w') as f:
        f.write("gamma,alpha,p_mrr,og_ndcg@5,changed_ndcg@5,og_mrr,changed_mrr\n")
        for r in all_results:
            f.write(f"{r['gamma']:.4f},{r['alpha']:.4f},{r['p_mrr']:.4f},"
                   f"{r['og_ndcg@5']:.4f},{r['changed_ndcg@5']:.4f},"
                   f"{r['og_mrr']:.4f},{r['changed_mrr']:.4f}\n")
    logger.info(f"💾 CSV 报告已保存: {csv_file}")
    
    return {
        'results': all_results,
        'best_p_mrr': max(all_results, key=lambda x: x['p_mrr']),
        'best_ndcg': max(all_results, key=lambda x: x['changed_ndcg@5'])
    }


def _print_dadt_summary_table(results: List[Dict]):
    """打印 DADT Grid Search 结果汇总表格 (Markdown 格式)"""
    print("\n" + "="*70)
    print("DADT Grid Search 结果汇总")
    print("="*70)
    
    # Markdown 表格头
    print("\n| Gamma | Alpha | p-MRR | Changed nDCG@5 | OG nDCG@5 |")
    print("|-------|-------|-------|----------------|-----------|")
    
    # 按 gamma 分组排序
    for r in sorted(results, key=lambda x: (x['gamma'], x['alpha'])):
        print(f"| {r['gamma']:5.1f} | {r['alpha']:5.1f} | {r['p_mrr']:7.4f} | "
              f"{r['changed_ndcg@5']:14.4f} | {r['og_ndcg@5']:9.4f} |")
    
    # 找出最佳组合
    valid_results = [r for r in results if r.get('p_mrr', 0) > 0]
    if valid_results:
        best_p_mrr = max(valid_results, key=lambda x: x['p_mrr'])
        best_ndcg = max(valid_results, key=lambda x: x['changed_ndcg@5'])
        
        print("\n" + "="*70)
        print("最佳组合:")
        print(f"  最高 p-MRR:          γ={best_p_mrr['gamma']:.1f}, α={best_p_mrr['alpha']:.1f} → {best_p_mrr['p_mrr']:.4f}")
        print(f"  最高 Changed nDCG@5: γ={best_ndcg['gamma']:.1f}, α={best_ndcg['alpha']:.1f} → {best_ndcg['changed_ndcg@5']:.4f}")
        print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DSCLR 网格搜索评测")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5", help="模型名称")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval", help="任务名称")
    parser.add_argument("--output_dir", type=str, default="eval/output/grid_search", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录")
    parser.add_argument("--use_cache", type=bool, default=True, help="是否使用缓存")
    parser.add_argument("--grid_params", type=str, default=None, help="自定义网格参数，格式: 'alpha1,tau1;alpha2,tau2;...'")
    parser.add_argument("--predefined_grid", type=str, default=None, 
                       choices=["conservative", "balanced", "aggressive", "fine_grained", "repllama_25"],
                       help="使用预定义网格参数")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="检查点文件路径（断点续跑）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dadt", action="store_true", help="启用 DADT (Distribution-Aware Dynamic Threshold) 模式")
    parser.add_argument("--gamma_list", type=str, default="0.0,0.5,1.0,1.5,2.0", help="DADT gamma 参数列表，逗号分隔")
    parser.add_argument("--alpha_list", type=str, default="1.5,2.0,2.5", help="DADT alpha 参数列表，逗号分隔")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # DADT 模式
    if args.dadt:
        gamma_list = [float(x) for x in args.gamma_list.split(",")]
        alpha_list = [float(x) for x in args.alpha_list.split(",")]
        
        results = run_dadt_grid_search_evaluation(
            model_name=args.model_name,
            task_name=args.task_name,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            use_cache=args.use_cache,
            gamma_list=gamma_list,
            alpha_list=alpha_list,
            seed=args.seed
        )
    else:
        # 标准网格搜索模式
        grid_params = None
        if args.grid_params:
            grid_params = parse_grid_params(args.grid_params)
            logger.info(f"📋 使用自定义网格参数: {len(grid_params)} 组")
        
        results = run_grid_search_evaluation(
            model_name=args.model_name,
            task_name=args.task_name,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            use_cache=args.use_cache,
            grid_params=grid_params,
            predefined_grid=args.predefined_grid,
            resume_checkpoint=args.resume_checkpoint,
            seed=args.seed
        )
    
    logger.info("\n" + "="*60)
    logger.info("✅ 网格搜索评测完成!")
    logger.info(f"📁 结果保存于: {args.output_dir}")
