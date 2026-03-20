"""
FollowIR 训练数据文档向量编码工具
用于将训练数据集中的文档编码为稠密向量
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

import numpy as np
import torch

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentEncoder:
    """文档稠密向量编码器"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda",
        batch_size: int = 128,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        
        self._load_model()
    
    def _load_model(self):
        """加载编码模型"""
        logger.info(f"📥 加载模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model = self.model.half()
        logger.info(f"✅ 模型加载完成 (device: {self.device}, dtype: float16)")
    
    def encode(self, texts: List[str], desc: str = "编码中") -> np.ndarray:
        """编码文本列表"""
        from tqdm import tqdm
        
        all_embeddings = []
        pbar = tqdm(total=len(texts), desc=desc, unit="batch")
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            all_embeddings.append(embeddings.cpu().numpy())
            pbar.update(len(batch_texts))
        
        pbar.close()
        return np.vstack(all_embeddings)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    logger.info(f"📂 加载数据文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")
                continue
    
    logger.info(f"✅ 加载完成: {len(data)} 条记录")
    return data


def encode_documents(
    encoder: DocumentEncoder,
    data: List[Dict[str, Any]],
    output_dir: str,
    prefix: str = "train"
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """编码文档并保存"""
    
    pos_docs = []
    neg_docs = []
    metadata = []
    
    pos_counter = 0
    neg_counter = 0
    
    logger.info("📝 提取文档文本...")
    for idx, item in enumerate(data):
        query = item.get('query', '')
        pos_list = item.get('pos', [])
        neg_list = item.get('neg', [])
        
        for pos_idx, pos_doc in enumerate(pos_list):
            pos_docs.append(pos_doc)
            metadata.append({
                'idx': idx,
                'doc_type': 'pos',
                'sub_idx': pos_idx,
                'query': query,
                'doc_id': f"{prefix}_pos_{pos_counter}"
            })
            pos_counter += 1
        
        for neg_idx, neg_doc in enumerate(neg_list):
            neg_docs.append(neg_doc)
            metadata.append({
                'idx': idx,
                'doc_type': 'neg',
                'sub_idx': neg_idx,
                'query': query,
                'doc_id': f"{prefix}_neg_{neg_counter}"
            })
            neg_counter += 1
    
    pos_count = len(pos_docs)
    neg_count = len(neg_docs)
    total_count = pos_count + neg_count
    
    logger.info(f"📊 文档统计:")
    logger.info(f"   正向文档 (pos): {pos_count}")
    logger.info(f"   负向文档 (neg): {neg_count}")
    logger.info(f"   总计: {total_count}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("🔢 编码正向文档 (pos)...")
    pos_embeddings = encoder.encode(pos_docs, desc=f"编码 pos ({pos_count})")
    pos_path = os.path.join(output_dir, f"{prefix}_pos_embeddings.npy")
    np.save(pos_path, pos_embeddings)
    logger.info(f"💾 正向向量已保存: {pos_path} (shape: {pos_embeddings.shape})")
    
    logger.info("🔢 编码负向文档 (neg)...")
    neg_embeddings = encoder.encode(neg_docs, desc=f"编码 neg ({neg_count})")
    neg_path = os.path.join(output_dir, f"{prefix}_neg_embeddings.npy")
    np.save(neg_path, neg_embeddings)
    logger.info(f"💾 负向向量已保存: {neg_path} (shape: {neg_embeddings.shape})")
    
    all_embeddings = np.concatenate([pos_embeddings, neg_embeddings], axis=0)
    all_path = os.path.join(output_dir, f"{prefix}_all_embeddings.npy")
    np.save(all_path, all_embeddings)
    logger.info(f"💾 全部向量已保存: {all_path} (shape: {all_embeddings.shape})")
    
    metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 元数据已保存: {metadata_path}")
    
    mapping = {
        'pos_count': pos_count,
        'neg_count': neg_count,
        'total_count': total_count,
        'embedding_dim': pos_embeddings.shape[1] if pos_count > 0 else neg_embeddings.shape[1],
        'pos_file': pos_path,
        'neg_file': neg_path,
        'all_file': all_path,
        'metadata_file': metadata_path
    }
    
    return pos_embeddings, neg_embeddings, metadata


def save_metadata_index(
    output_dir: str,
    prefix: str,
    metadata: List[Dict[str, Any]],
    pos_count: int,
    neg_count: int,
    embedding_dim: int
):
    """保存索引元数据文件"""
    
    index_info = {
        'prefix': prefix,
        'pos_count': pos_count,
        'neg_count': neg_count,
        'total_count': pos_count + neg_count,
        'embedding_dim': embedding_dim,
        'embedding_dtype': 'float16',
        'files': {
            'pos_embeddings': f"{prefix}_pos_embeddings.npy",
            'neg_embeddings': f"{prefix}_neg_embeddings.npy",
            'all_embeddings': f"{prefix}_all_embeddings.npy",
            'metadata': f"{prefix}_metadata.json"
        },
        'doc_id_to_idx': {
            f"{prefix}_pos_{i}": i for i in range(pos_count)
        } | {
            f"{prefix}_neg_{i}": pos_count + i for i in range(neg_count)
        }
    }
    
    index_path = os.path.join(output_dir, f"{prefix}_index_info.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 索引信息已保存: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="FollowIR 训练数据文档向量编码工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python encode_documents.py --input /path/to/train.jsonl --output /path/to/output
    python encode_documents.py -i data.jsonl -o embeddings/ --model BAAI/bge-base-en-v1.5 -b 256
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="输入 JSONL 文件路径"
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="编码模型名称 (default: BAAI/bge-large-en-v1.5)"
    )
    
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=128,
        help="批处理大小 (default: 128)"
    )
    
    parser.add_argument(
        '-d', '--device',
        type=str,
        default="cuda",
        help="计算设备: cuda 或 cpu (default: cuda)"
    )
    
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help="不进行向量归一化"
    )
    
    parser.add_argument(
        '-p', '--prefix',
        type=str,
        default="train",
        help="输出文件前缀 (default: train)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("FollowIR 文档向量编码工具")
    logger.info("=" * 60)
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"编码模型: {args.model}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"计算设备: {args.device}")
    logger.info("=" * 60)
    
    try:
        encoder = DocumentEncoder(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            normalize=not args.no_normalize
        )
        
        data = load_jsonl(args.input)
        
        if len(data) == 0:
            logger.error("数据文件为空")
            sys.exit(1)
        
        pos_emb, neg_emb, metadata = encode_documents(
            encoder=encoder,
            data=data,
            output_dir=args.output,
            prefix=args.prefix
        )
        
        save_metadata_index(
            output_dir=args.output,
            prefix=args.prefix,
            metadata=metadata,
            pos_count=len(pos_emb),
            neg_count=len(neg_emb),
            embedding_dim=pos_emb.shape[1] if len(pos_emb) > 0 else neg_emb.shape[1]
        )
        
        logger.info("=" * 60)
        logger.info("✅ 编码完成!")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"文件错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"编码过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
