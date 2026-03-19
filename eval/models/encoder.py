"""
模型加载模块
支持多种类型的稠密检索模型
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class BaseEncoder(ABC):
    """编码器基类"""
    
    @abstractmethod
    def encode_queries(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码查询文本"""
        pass
    
    @abstractmethod
    def encode_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码文档文本"""
        pass


class SentenceTransformerEncoder(BaseEncoder):
    """基于 SentenceTransformer 的编码器"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"📥 加载模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
            
        logger.info(f"✅ 模型加载完成: {model_name}")
    
    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码查询"""
        batch_size = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            **kwargs
        )
        return embeddings
    
    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码文档"""
        batch_size = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            **kwargs
        )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.model.get_sentence_embedding_dimension()


class ModelFactory:
    """模型工厂"""
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type):
        """注册模型类"""
        cls._models[name] = model_class
    
    @classmethod
    def create(cls, model_name: str, **kwargs) -> BaseEncoder:
        """创建模型实例"""
        if model_name in cls._models:
            return cls._models[model_name](**kwargs)
        
        return SentenceTransformerEncoder(
            model_name=model_name,
            **kwargs
        )
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """获取可用模型列表"""
        return list(cls._models.keys())


ModelFactory.register("sentence_transformer", SentenceTransformerEncoder)


class DenseRetriever:
    """稠密检索器"""
    
    def __init__(self, encoder: BaseEncoder):
        self.encoder = encoder
        self.doc_embeddings: Dict[str, torch.Tensor] = {}
    
    def index_documents(self, doc_ids: List[str], doc_texts: List[str], batch_size: int = 64) -> None:
        """构建文档索引"""
        logger.info(f"📚 索引 {len(doc_ids)} 个文档...")
        
        embeddings = self.encoder.encode_documents(doc_texts, batch_size=batch_size)
        
        self.doc_embeddings = {doc_id: embeddings[idx] for idx, doc_id in enumerate(doc_ids)}
        logger.info(f"✅ 文档索引构建完成")
    
    def search(
        self,
        query_embeddings: torch.Tensor,
        doc_ids: List[str],
        top_k: int = 100
    ) -> List[List[Dict[str, Any]]]:
        """执行检索"""
        doc_emb = torch.stack([self.doc_embeddings[did] for did in doc_ids]).to(query_embeddings.device)
        
        scores = torch.matmul(query_embeddings, doc_emb.T)
        
        results = []
        for query_scores in scores:
            _, top_indices = torch.topk(query_scores, min(top_k, len(doc_ids)))
            
            query_results = []
            for idx in top_indices:
                doc_id = doc_ids[idx.item()]
                score = query_scores[idx.item()].item()
                query_results.append({
                    "doc_id": doc_id,
                    "score": score
                })
            results.append(query_results)
        
        return results
    
    def compute_scores(
        self,
        query_embedding: torch.Tensor,
        doc_ids: List[str]
    ) -> Dict[str, float]:
        """计算单个查询与文档的得分"""
        doc_emb = torch.stack([self.doc_embeddings[did] for did in doc_ids]).to(query_embedding.device)
        
        scores = torch.matmul(query_embedding, doc_emb.T).squeeze(0)
        
        return {doc_ids[idx]: scores[idx].item() for idx in range(len(doc_ids))}
