"""
RepLLaMA 编码器
使用正确的 prompt template 和 last token 提取
"""

import os
import logging
from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .encoder import BaseEncoder

logger = logging.getLogger(__name__)


class RepLLaMAEncoder(BaseEncoder):
    """RepLLaMA 编码器 - 使用正确的 prompt template"""
    
    def __init__(
        self,
        model_name: str = "castorini/repllama-v1-7b-lora-passage",
        device: str = "cuda",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
        base_model_path: str = "/home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf"
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length or 512
        self.base_model_path = base_model_path
        
        logger.info(f"📥 加载 RepLLaMA 模型...")
        logger.info(f"   基础模型: {base_model_path}")
        logger.info(f"   Adapter: {model_name}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型（8-bit 量化）
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        base_model = AutoModel.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 加载 LoRA Adapter
        adapter_path = "/home/luwa/Documents/models/repllama-v1-7b-lora-passage/castorini/repllama-v1-7b-lora-passage"
        model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = model.merge_and_unload()
        self.model.eval()
        
        logger.info(f"✅ RepLLaMA 模型加载完成")
    
    def _encode_with_template(self, texts: List[str], template: str, batch_size: Optional[int] = None) -> torch.Tensor:
        """使用 template 编码文本"""
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 应用 template
            formatted_texts = [template.format(text=text) for text in batch_texts]
            
            inputs = self.tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 提取最后一个 token 的 embedding
                batch_embeddings = []
                for j in range(len(batch_texts)):
                    seq_len = inputs['attention_mask'][j].sum().item()
                    embedding = outputs.last_hidden_state[j, seq_len - 1]
                    batch_embeddings.append(embedding)
                
                batch_embeddings = torch.stack(batch_embeddings)
                
                # L2 归一化
                if self.normalize_embeddings:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码查询 - 使用 query: template"""
        return self._encode_with_template(texts, "query: {text}</s>", batch_size)
    
    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> torch.Tensor:
        """编码文档 - 使用 passage: template"""
        return self._encode_with_template(texts, "passage: {text}</s>", batch_size)
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return 4096
