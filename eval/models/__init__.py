"""模型模块"""
from .encoder import BaseEncoder, SentenceTransformerEncoder, ModelFactory, DenseRetriever
from .e5_mistral_encoder import E5MistralEncoder
from .repllama_encoder import RepLLaMAEncoder

# 注册 E5-Mistral 编码器
ModelFactory.register("e5-mistral-7b-instruct", E5MistralEncoder)
ModelFactory.register("intfloat/e5-mistral-7b-instruct", E5MistralEncoder)

# 注册 RepLLaMA 编码器
ModelFactory.register("repllama-v1-7b-lora-passage", RepLLaMAEncoder)
ModelFactory.register("castorini/repllama-v1-7b-lora-passage", RepLLaMAEncoder)

__all__ = [
    'BaseEncoder',
    'SentenceTransformerEncoder',
    'E5MistralEncoder',
    'RepLLaMAEncoder',
    'ModelFactory',
    'DenseRetriever'
]
