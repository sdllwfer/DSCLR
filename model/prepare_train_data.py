"""
训练数据准备脚本
支持多种编码器：BGE-large / E5-Mistral-7B / RepLLaMA
"""

import json
import os

# 使用官方 HuggingFace 或镜像站点
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

# 允许在线下载模型（RepLLaMA 需要）
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# 删除可能存在的离线标记
for key in list(os.environ.keys()):
    if 'OFFLINE' in key and 'HF' in key:
        os.environ[key] = "0"

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入后再次确保环境变量正确设置（防止被其他模块覆盖）
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = ""

from eval.models.encoder import DenseRetriever, ModelFactory
from eval.models.e5_mistral_encoder import E5MistralEncoder
from eval.models.repllama_encoder import RepLLaMAEncoder

# 最终确认环境变量
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"


def load_jsonl(path):
    """加载 JSONL 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def encode_with_encoder(
    texts: List[str],
    encoder: Any,
    batch_size: int,
    desc: str = "Encoding",
    mode: str = "document"
) -> torch.Tensor:
    """
    统一的编码接口
    
    Args:
        texts: 待编码文本列表
        encoder: 编码器实例（BGE / E5-Mistral / RepLLaMA）
        batch_size: 批处理大小
        desc: 进度描述
        mode: 编码模式，"query" 或 "document"
    
    Returns:
        编码后的嵌入向量 [num_texts, embedding_dim]
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i + batch_size]
        
        # 根据编码器类型调用不同的编码方法
        if isinstance(encoder, E5MistralEncoder):
            # E5-Mistral 编码器
            if mode == "query":
                batch_embeddings = encoder.encode_queries(batch_texts, batch_size=batch_size)
            else:
                batch_embeddings = encoder.encode_documents(batch_texts, batch_size=batch_size)
        elif isinstance(encoder, RepLLaMAEncoder):
            # RepLLaMA 编码器
            if mode == "query":
                batch_embeddings = encoder.encode_queries(batch_texts, batch_size=batch_size)
            else:
                batch_embeddings = encoder.encode_documents(batch_texts, batch_size=batch_size)
        else:
            # BGE / SentenceTransformer 编码器
            batch_embeddings = encoder.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        
        # 确保是 CPU 上的 tensor
        if isinstance(batch_embeddings, np.ndarray):
            batch_embeddings = torch.from_numpy(batch_embeddings)
        elif batch_embeddings.device != torch.device('cpu'):
            batch_embeddings = batch_embeddings.cpu()
            
        all_embeddings.append(batch_embeddings)
    
    return torch.cat(all_embeddings, dim=0)


def prepare_training_data(
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cuda:0",
    batch_size: int = 32,
    output_dir: str = "dataset/FollowIR_train/embeddings",
    data_dir: str = "dataset/FollowIR_train"
):
    """
    准备训练数据缓存
    
    Args:
        model_name: 模型名称，支持 "BAAI/bge-large-en-v1.5" 或 "intfloat/e5-mistral-7b-instruct"
        device: GPU 设备
        batch_size: 批处理大小
        output_dir: 输出目录
        data_dir: 训练数据目录
    """
    print("="*60)
    print(f"模型: {model_name}")
    print(f"设备: {device}")
    print(f"批次: {batch_size}")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据模型名称确定缓存文件名
    if "mistral" in model_name.lower():
        model_short = "e5-mistral-7b"
        batch_size = min(batch_size, 28)  # E5-Mistral 显存限制
    elif "repllama" in model_name.lower():
        model_short = "repllama-v1-7b"
        batch_size = min(batch_size, 28)  # RepLLaMA 也是大模型，显存限制
    else:
        model_short = "bge-large-en"
    
    cache_filename = f"dsclr_train_embeddings_{model_short}.pt"
    cache_path = os.path.join(output_dir, cache_filename)
    
    # 检查是否已存在缓存
    if os.path.exists(cache_path):
        print(f"⚠️ 缓存已存在: {cache_path}")
        response = input("是否重新生成? (y/n): ")
        if response.lower() != 'y':
            print("使用已有缓存，退出")
            return cache_path
    
    print("\n" + "="*60)
    print("Step 1: 加载数据...")
    print("="*60)
    
    train_data = load_jsonl(os.path.join(data_dir, "train_data_dsclr.jsonl"))
    print(f"训练数据: {len(train_data)} 条")
    
    distilled_queries = load_jsonl(os.path.join(data_dir, "distilled_queries_v4.jsonl"))
    print(f"双流查询: {len(distilled_queries)} 条")
    
    q_plus_list = []
    q_minus_list = []
    for item in tqdm(distilled_queries, desc="解析双流查询"):
        output = json.loads(item['output'])
        q_plus_list.append(output['Q_plus'])
        q_minus_list.append(output['Q_minus'])
    
    print("\n" + "="*60)
    print("Step 2: 加载编码器...")
    print("="*60)
    
    # 根据模型类型创建编码器
    if "mistral" in model_name.lower():
        print(f"加载 E5-Mistral-7B 编码器...")
        encoder = E5MistralEncoder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=True
        )
    elif "repllama" in model_name.lower():
        print(f"加载 RepLLaMA 编码器...")
        # 使用专门的 RepLLaMAEncoder，支持正确的 prompt template 和 last token 提取
        encoder = RepLLaMAEncoder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=True
        )
    else:
        print(f"加载 BGE 编码器...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name, device=device)
        encoder = encoder.half()
    
    print("\n" + "="*60)
    print("Step 3: 编码 Q+ 和 Q- ...")
    print("="*60)
    
    print("\n编码 Q+ (正向查询)...")
    q_plus_embeddings = encode_with_encoder(
        q_plus_list, encoder, batch_size, desc="Q+", mode="query"
    )
    print(f"Q+ embeddings shape: {q_plus_embeddings.shape}")
    
    print("\n编码 Q- (负向查询)...")
    q_minus_embeddings = encode_with_encoder(
        q_minus_list, encoder, batch_size, desc="Q-", mode="query"
    )
    print(f"Q- embeddings shape: {q_minus_embeddings.shape}")
    
    print("\n" + "="*60)
    print("Step 4: 编码文档 (Pos + Neg) ...")
    print("="*60)
    
    pos_docs = []
    neg_docs = []
    for item in train_data:
        pos_docs.extend(item['pos'])
        neg_docs.extend(item['neg'])
    
    print(f"正样本文档: {len(pos_docs)} 个")
    print(f"负样本文档: {len(neg_docs)} 个")
    
    print("\n编码正样本文档...")
    pos_embeddings = encode_with_encoder(
        pos_docs, encoder, batch_size, desc="Pos", mode="document"
    )
    print(f"Pos embeddings shape: {pos_embeddings.shape}")
    
    print("\n编码负样本文档...")
    neg_embeddings = encode_with_encoder(
        neg_docs, encoder, batch_size, desc="Neg", mode="document"
    )
    print(f"Neg embeddings shape: {neg_embeddings.shape}")
    
    print("\n" + "="*60)
    print("Step 5: 保存缓存 ...")
    print("="*60)
    
    cache_data = {
        'q_plus_embeddings': q_plus_embeddings.cpu(),
        'q_minus_embeddings': q_minus_embeddings.cpu(),
        'pos_embeddings': pos_embeddings.cpu(),
        'neg_embeddings': neg_embeddings.cpu(),
        'train_data_idx': [item['idx'] for item in train_data],
        'num_pos_per_query': len(train_data[0]['pos']),
        'num_neg_per_query': len(train_data[0]['neg']),
        'model_name': model_name,
        'embedding_dim': q_plus_embeddings.shape[1]
    }
    
    torch.save(cache_data, cache_path)
    print(f"✅ 缓存已保存: {cache_path}")
    print(f"   文件大小: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB")
    print(f"   嵌入维度: {q_plus_embeddings.shape[1]}")
    
    return cache_path


def main():
    parser = argparse.ArgumentParser(description='准备 DSCLR 训练数据缓存')
    parser.add_argument('--model', type=str, default='BAAI/bge-large-en-v1.5',
                        choices=['BAAI/bge-large-en-v1.5', 'intfloat/e5-mistral-7b-instruct', 'castorini/repllama-v1-7b-lora-passage'],
                        help='编码器模型名称')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU 设备')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--output_dir', type=str, default='dataset/FollowIR_train/embeddings',
                        help='输出目录')
    parser.add_argument('--data_dir', type=str, default='dataset/FollowIR_train',
                        help='训练数据目录')
    
    args = parser.parse_args()
    
    prepare_training_data(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )


if __name__ == "__main__":
    main()
