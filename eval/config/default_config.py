"""
FollowIR 评测系统配置文件
支持自定义评估参数、模型配置和输出设置
"""

EVAL_CONFIG = {
    "tasks": [
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval", 
        "News21InstructionRetrieval"
    ],
    
    "model_defaults": {
        "batch_size": 64,
        "device": "cuda",
        "normalize_embeddings": True,
        "max_seq_length": 512
    },
    
    "evaluation": {
        "top_k": [5, 10, 20],
        "save_trec": True,
        "save_debug_info": False,
        "compute_pmr": True,
        "compute_ndcg": True
    },
    
    "output": {
        "base_dir": "/home/luwa/Documents/DSCLR/evaluation",
        "save_formats": ["json", "trec"],
        "generate_report": True
    }
}


MODEL_REGISTRY = {
    "bge-large-en-v1.5": {
        "type": "sentence_transformer",
        "model_name": "BAAI/bge-large-en-v1.5",
        "embedding_dim": 1024
    },
    "bge-base-en-v1.5": {
        "type": "sentence_transformer", 
        "model_name": "BAAI/bge-base-en-v1.5",
        "embedding_dim": 768
    },
    "bge-small-en-v1.5": {
        "type": "sentence_transformer",
        "model_name": "BAAI/bge-small-en-v1.5", 
        "embedding_dim": 384
    },
    "e5-base-v2": {
        "type": "sentence_transformer",
        "model_name": "intfloat/e5-base-v2",
        "embedding_dim": 768
    },
    "gte-base": {
        "type": "sentence_transformer",
        "model_name": "Alibaba-NLP/gte-base-en-v1.5",
        "embedding_dim": 768
    }
}


METRIC_REGISTRY = {
    "p-MRR": {
        "display_name": "p-MRR",
        "description": "Permutation-aware Mean Reciprocal Rank",
        "higher_is_better": True
    },
    "ndcg_at_k": {
        "display_name": "nDCG@{k}",
        "description": "Normalized Discounted Cumulative Gain",
        "higher_is_better": True
    },
    "map": {
        "display_name": "MAP", 
        "description": "Mean Average Precision",
        "higher_is_better": True
    },
    "recall_at_k": {
        "display_name": "Recall@{k}",
        "description": "Recall at k",
        "higher_is_better": True
    }
}
