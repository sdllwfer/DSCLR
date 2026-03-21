import json
import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    DEVICE = "cuda:0"
    BATCH_SIZE = 64
    
    DATA_DIR = "dataset/FollowIR_train"
    CACHE_DIR = "dataset/FollowIR_train/embeddings"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print("="*60)
    print("Step 1: 加载数据...")
    print("="*60)
    
    train_data = load_jsonl(os.path.join(DATA_DIR, "train_data_dsclr.jsonl"))
    print(f"训练数据: {len(train_data)} 条")
    
    distilled_queries = load_jsonl(os.path.join(DATA_DIR, "distilled_queries.jsonl"))
    print(f"双流查询: {len(distilled_queries)} 条")
    
    q_plus_list = []
    q_minus_list = []
    for item in tqdm(distilled_queries, desc="解析双流查询"):
        output = json.loads(item['output'])
        q_plus_list.append(output['Q_plus'])
        q_minus_list.append(output['Q_minus'])
    
    print(f"\n加载模型: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)
    encoder = encoder.half()
    
    print("\n" + "="*60)
    print("Step 2: 编码 Q+ 和 Q- ...")
    print("="*60)
    
    print("\n编码 Q+ (正向查询)...")
    q_plus_embeddings = encoder.encode(
        q_plus_list, batch_size=BATCH_SIZE, 
        show_progress_bar=True, convert_to_tensor=True,
        normalize_embeddings=True
    )
    print(f"Q+ embeddings shape: {q_plus_embeddings.shape}")
    
    print("\n编码 Q- (负向查询)...")
    q_minus_embeddings = encoder.encode(
        q_minus_list, batch_size=BATCH_SIZE,
        show_progress_bar=True, convert_to_tensor=True,
        normalize_embeddings=True
    )
    print(f"Q- embeddings shape: {q_minus_embeddings.shape}")
    
    print("\n" + "="*60)
    print("Step 3: 编码文档 (Pos + Neg) ...")
    print("="*60)
    
    pos_docs = []
    neg_docs = []
    for item in train_data:
        pos_docs.extend(item['pos'])
        neg_docs.extend(item['neg'])
    
    print(f"正样本文档: {len(pos_docs)} 个")
    print(f"负样本文档: {len(neg_docs)} 个")
    
    print("\n编码正样本文档...")
    pos_embeddings = encoder.encode(
        pos_docs, batch_size=BATCH_SIZE,
        show_progress_bar=True, convert_to_tensor=True,
        normalize_embeddings=True
    )
    print(f"Pos embeddings shape: {pos_embeddings.shape}")
    
    print("\n编码负样本文档...")
    neg_embeddings = encoder.encode(
        neg_docs, batch_size=BATCH_SIZE,
        show_progress_bar=True, convert_to_tensor=True,
        normalize_embeddings=True
    )
    print(f"Neg embeddings shape: {neg_embeddings.shape}")
    
    print("\n" + "="*60)
    print("Step 4: 保存缓存 ...")
    print("="*60)
    
    cache_data = {
        'q_plus_embeddings': q_plus_embeddings.cpu(),
        'q_minus_embeddings': q_minus_embeddings.cpu(),
        'pos_embeddings': pos_embeddings.cpu(),
        'neg_embeddings': neg_embeddings.cpu(),
        'train_data_idx': [item['idx'] for item in train_data],
        'num_pos_per_query': len(train_data[0]['pos']),
        'num_neg_per_query': len(train_data[0]['neg'])
    }
    
    cache_path = os.path.join(CACHE_DIR, "dsclr_train_embeddings.pt")
    torch.save(cache_data, cache_path)
    print(f"✅ 缓存已保存: {cache_path}")
    print(f"   文件大小: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
