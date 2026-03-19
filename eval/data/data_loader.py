"""
FollowIR 数据加载模块
负责加载评测所需的语料库、查询、指令和候选文档
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import datasets
import mteb

logger = logging.getLogger(__name__)


class DataLoader:
    """FollowIR 数据加载器"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.q_og: Dict[str, str] = {}
        self.q_changed: Dict[str, str] = {}
        self.candidates: Dict[str, List[str]] = {}
        self.instructions: Dict[str, str] = {}
        
    def load(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """加载完整数据集
        
        Returns:
            Tuple of (corpus, q_og, q_changed, candidates)
        """
        logger.info(f"📂 加载数据集: {self.task_name}")
        
        task = mteb.get_task(self.task_name)
        dataset_path = task.metadata.dataset["path"]
        
        self._load_corpus(dataset_path)
        self._load_queries_and_instructions(dataset_path)
        self._load_candidates(dataset_path)
        
        logger.info(f"✅ 数据加载完成: {len(self.corpus)} 文档, "
                   f"{len(self.q_og)} og查询, {len(self.q_changed)} changed查询")
        
        return self.corpus, self.q_og, self.q_changed, self.candidates
    
    def _load_corpus(self, dataset_path: str) -> None:
        """加载文档集合"""
        try:
            ds_c = datasets.load_dataset(dataset_path, 'corpus', trust_remote_code=True)
            c_split = 'corpus' if 'corpus' in ds_c else 'train'
            
            for d in ds_c[c_split]:
                doc_id = str(d.get('_id', d.get('id')))
                self.corpus[doc_id] = {'text': str(d.get('text', ''))}
            
            logger.info(f"   ✅ 加载 {len(self.corpus)} 个文档")
        except Exception as e:
            logger.error(f"加载语料库失败: {e}")
            raise
    
    def _load_queries_and_instructions(self, dataset_path: str) -> None:
        """加载查询和指令"""
        try:
            ds_q = datasets.load_dataset(dataset_path, 'queries', trust_remote_code=True)
            q_split = 'queries' if 'queries' in ds_q else 'train'
            
            ds_inst = datasets.load_dataset(dataset_path, 'instruction', trust_remote_code=True)
            i_split = 'instruction' if 'instruction' in ds_inst else 'train'
            
            instruction_dict = {}
            for inst_item in ds_inst[i_split]:
                qid = str(inst_item.get('query-id', ''))
                inst_text = str(inst_item.get('instruction', ''))
                instruction_dict[qid] = inst_text
                self.instructions[qid] = inst_text
            
            for q in ds_q[q_split]:
                full_qid = str(q.get('_id', q.get('id', '')))
                query_text = q.get('text', '')
                inst = instruction_dict.get(full_qid, "")
                
                if full_qid.endswith('-og'):
                    self.q_og[full_qid] = f"{query_text} {inst}".strip()
                elif full_qid.endswith('-changed'):
                    self.q_changed[full_qid] = f"{query_text} {inst}".strip()
            
            logger.info(f"   ✅ 加载 {len(self.q_og)} 个 og 查询, {len(self.q_changed)} 个 changed 查询")
        except Exception as e:
            logger.error(f"加载查询和指令失败: {e}")
            raise
    
    def _load_candidates(self, dataset_path: str) -> None:
        """加载候选文档"""
        try:
            ds_top = datasets.load_dataset(dataset_path, 'top_ranked', trust_remote_code=True)
            available_splits = list(ds_top.keys())
            t_split = available_splits[0] if available_splits else None
            
            if t_split:
                for item in ds_top[t_split]:
                    full_qid = str(item.get('query-id', item.get('query_id', item.get('qid', ''))))
                    base_qid = full_qid.replace('-og', '').replace('-changed', '')
                    results_list = item.get('corpus-ids', item.get('results', []))
                    
                    if base_qid not in self.candidates:
                        self.candidates[base_qid] = [str(did) for did in results_list]
                
                if self.candidates:
                    avg_cand = sum(len(v) for v in self.candidates.values()) / len(self.candidates)
                    logger.info(f"   ✅ 加载 {len(self.candidates)} 个查询的候选文档, 平均 {avg_cand:.0f} 个/查询")
        except Exception as e:
            logger.warning(f"加载候选文档失败: {e}")
    
    def get_all_queries(self) -> Dict[str, str]:
        """获取所有查询（og + changed）"""
        return {**self.q_og, **self.q_changed}
    
    def get_query_count(self) -> Dict[str, int]:
        """获取查询统计信息"""
        return {
            "og_queries": len(self.q_og),
            "changed_queries": len(self.q_changed),
            "total_queries": len(self.q_og) + len(self.q_changed)
        }
    
    def get_candidate_stats(self) -> Dict[str, float]:
        """获取候选文档统计信息"""
        if not self.candidates:
            return {"count": 0, "avg_per_query": 0.0}
        
        return {
            "count": len(self.candidates),
            "avg_per_query": sum(len(v) for v in self.candidates.values()) / len(self.candidates),
            "min_per_query": min(len(v) for v in self.candidates.values()),
            "max_per_query": max(len(v) for v in self.candidates.values())
        }


class DataLoaderFactory:
    """数据加载器工厂"""
    
    _loaders: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, loader_class: type):
        """注册数据加载器"""
        cls._loaders[name] = loader_class
    
    @classmethod
    def create(cls, task_name: str, loader_type: str = "followir") -> DataLoader:
        """创建数据加载器"""
        if loader_type not in cls._loaders:
            raise ValueError(f"未知加载器类型: {loader_type}")
        return cls._loaders[loader_type](task_name)


DataLoaderFactory.register("followir", DataLoader)
