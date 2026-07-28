"""
Timing script for TRIX reranking with embedding cache

Measures per-query reranking time and reports mean/std statistics.
Simulates the actual reranking process with pre-computed embeddings.

Usage:
  cd /home/luwa/Documents/DSCLR-remote && python scripts/time_rerank.py \
    --dataset core17 \
    --encoder BAAI/bge-large-en-v1.5 \
    --device cuda \
    --num_reps 100
"""

import os
import sys
import json
import time
import logging
import argparse
import statistics
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FollowIR dataset mapping
FOLLOWIR_DATASETS = {
    "core17": "jhu-clsp/core17-instructions-mteb",
    "robust04": "jhu-clsp/robust04-instructions-mteb",
    "news21": "jhu-clsp/news21-instructions-mteb",
}


def load_followir_corpus(dataset_name: str):
    """Load FollowIR corpus from HuggingFace."""
    import datasets
    dataset_path = FOLLOWIR_DATASETS.get(dataset_name)
    if not dataset_path:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Loading corpus from {dataset_path}...")
    ds = datasets.load_dataset(dataset_path, 'corpus')
    split = 'corpus' if 'corpus' in ds else list(ds.keys())[0]
    
    corpus = {}
    for item in tqdm(ds[split], desc="Loading corpus"):
        doc_id = str(item.get('_id', item.get('doc_id', '')))
        text = item.get('text', '')
        corpus[doc_id] = text
    
    logger.info(f"Loaded {len(corpus)} documents")
    return corpus


def load_followir_queries_with_dual(dataset_name: str, dual_queries_path: str):
    """Load FollowIR queries and dual queries."""
    import datasets
    dataset_path = FOLLOWIR_DATASETS.get(dataset_name)
    
    # Load original queries
    logger.info(f"Loading queries from {dataset_path}...")
    ds_q = datasets.load_dataset(dataset_path, 'queries')
    q_split = 'queries' if 'queries' in ds_q else list(ds_q.keys())[0]
    
    queries = {}
    for q in tqdm(ds_q[q_split], desc="Loading queries"):
        qid = str(q.get('_id', q.get('id', '')))
        text = q.get('text', '')
        queries[qid] = text
    
    # Load dual queries
    dual_data = {}
    if dual_queries_path and os.path.exists(dual_queries_path):
        with open(dual_queries_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["qid"]
                dual_data[qid] = item
        logger.info(f"Loaded {len(dual_data)} dual queries")
    
    return queries, dual_data


class RerankTimer:
    """Timer for TRIX reranking with embedding cache."""
    
    def __init__(
        self,
        encoder_name: str,
        device: str = "cuda",
        alpha: float = 0.5,
        beta: float = 1.0,
        delta: float = 0.02,
        t_safety: float = 20.0,
    ):
        self.encoder_name = encoder_name
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.t_safety = t_safety
        
        # Load encoder
        logger.info(f"Loading encoder: {encoder_name}")
        from eval.models import ModelFactory
        self.encoder = ModelFactory.create(
            model_name=encoder_name,
            device=device,
            batch_size=64,
            normalize_embeddings=True,
        )
        logger.info(f"Encoder loaded on {device}")
    
    def encode_corpus(self, corpus: Dict[str, str]) -> Tuple[torch.Tensor, List[str]]:
        """Encode corpus documents."""
        doc_ids = list(corpus.keys())
        texts = [corpus[did] for did in doc_ids]
        
        logger.info(f"Encoding {len(doc_ids)} documents...")
        embeddings = self.encoder.encode_documents(texts)
        
        logger.info(f"Corpus embeddings shape: {embeddings.shape}")
        return embeddings, doc_ids
    
    def encode_queries(self, queries: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Encode queries."""
        qids = list(queries.keys())
        texts = [queries[qid] for qid in qids]
        
        logger.info(f"Encoding {len(qids)} queries...")
        embeddings = self.encoder.encode_queries(texts)
        
        q_emb = {qid: embeddings[i] for i, qid in enumerate(qids)}
        return q_emb
    
    def time_rerank(
        self,
        corpus_emb: torch.Tensor,
        q_base_emb: torch.Tensor,
        q_plus_emb: torch.Tensor,
        q_neg_emb: torch.Tensor,
        num_reps: int = 100,
    ) -> Tuple[float, float, float]:
        """
        Time the reranking operation for a single query.
        
        Returns:
            mean_time: mean time in milliseconds
            std_time: std deviation in milliseconds
            single_time: single run time in milliseconds
        """
        # Move to device
        corpus_emb = corpus_emb.to(self.device)
        q_base_emb = q_base_emb.to(self.device)
        q_plus_emb = q_plus_emb.to(self.device)
        q_neg_emb = q_neg_emb.to(self.device)
        
        num_docs = corpus_emb.shape[0]
        
        # Warmup (GPU kernel compilation)
        for _ in range(10):
            with torch.no_grad():
                s_base = torch.matmul(corpus_emb, q_base_emb)
                s_plus = torch.matmul(corpus_emb, q_plus_emb)
                s_neg = torch.matmul(corpus_emb, q_neg_emb)
                cos_qbase_qneg = F.cosine_similarity(
                    q_base_emb.unsqueeze(0), q_neg_emb.unsqueeze(0)
                ).item()
                _ = torch.topk(s_base, min(1000, num_docs))
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Time multiple runs
        times = []
        for _ in range(num_reps):
            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                # Step 1: Compute similarities (3 matrix-vector products)
                s_base = torch.matmul(corpus_emb, q_base_emb)
                s_plus = torch.matmul(corpus_emb, q_plus_emb)
                s_neg = torch.matmul(corpus_emb, q_neg_emb)
                
                # Step 2: Compute cosine similarity
                cos_qbase_qneg = F.cosine_similarity(
                    q_base_emb.unsqueeze(0), q_neg_emb.unsqueeze(0)
                ).item()
                
                # Step 3: Compute safety and penalty
                has_neg = q_neg_emb.norm().item() > 1e-6
                
                if has_neg:
                    tau = cos_qbase_qneg + self.delta
                    overflow = s_neg - tau
                    smooth_penalty = F.softplus(overflow)
                    safety = 1.0 - torch.sigmoid(overflow * self.t_safety)
                    
                    # Step 4: Compute final score
                    s_final = s_base + self.beta * s_plus * safety - self.alpha * smooth_penalty
                else:
                    s_final = s_base + self.beta * s_plus
                
                # Step 5: Top-k retrieval
                top_k = 1000
                _, _ = torch.topk(s_final, min(top_k, num_docs))
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)
        
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        single_time = times[0]
        
        return mean_time, std_time, single_time


def main():
    parser = argparse.ArgumentParser(description="Time TRIX reranking with embedding cache")
    parser.add_argument("--dataset", type=str, default="core17",
                        choices=["core17", "robust04", "news21"])
    parser.add_argument("--encoder", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--dual_queries_path", type=str, 
                        default="dataset/FollowIR/dual_queries_v6/core17_TSC_BALANCED_t01.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--num_reps", type=int, default=100,
                        help="Number of repetitions for timing each query")
    parser.add_argument("--cache_embeddings", type=str, default="",
                        help="Path to cached corpus embeddings (optional)")
    parser.add_argument("--output_file", type=str, default="logs/rerank_timing.json")
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == "cuda":
        try:
            torch.cuda._lazy_init()
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                args.device = "cpu"
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}, falling back to CPU")
            args.device = "cpu"
    
    # Initialize timer and encoder
    timer = RerankTimer(
        encoder_name=args.encoder,
        device=args.device,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
    )
    
    # Load corpus
    corpus = load_followir_corpus(args.dataset)
    
    # Load queries and dual queries
    queries, dual_queries = load_followir_queries_with_dual(
        args.dataset, args.dual_queries_path
    )
    
    # Encode corpus (this simulates "Emb. cache" - embeddings are pre-computed)
    corpus_emb, doc_ids = timer.encode_corpus(corpus)
    num_docs = len(doc_ids)
    
    # Encode all queries (Q_base)
    q_base_embs = timer.encode_queries(queries)
    
    # Time each query
    logger.info(f"Timing reranking for {len(dual_queries)} queries...")
    results = []
    
    for i, (qid, dq) in enumerate(tqdm(dual_queries.items(), desc="Timing queries")):
        # Get Q_base embedding
        q_base_emb = q_base_embs.get(qid)
        if q_base_emb is None:
            # Try OG version
            og_qid = qid.replace('-changed', '-og')
            q_base_emb = q_base_embs.get(og_qid)
        
        if q_base_emb is None:
            logger.warning(f"Query embedding not found for {qid}, skipping")
            continue
        
        # Get Q_plus and Q_neg from dual query
        q_plus_text = dq.get("Q_plus", "")
        q_neg_text = dq.get("Q_minus", "[NONE]")
        
        # Encode Q_plus and Q_neg
        q_plus_emb = timer.encoder.encode_queries([q_plus_text])[0]
        
        if q_neg_text == "[NONE]" or not q_neg_text.strip():
            # Zero embedding for [NONE]
            q_neg_emb = torch.zeros_like(q_plus_emb)
        else:
            q_neg_emb = timer.encoder.encode_queries([q_neg_text])[0]
        
        # Time the reranking
        mean_ms, std_ms, single_ms = timer.time_rerank(
            corpus_emb, q_base_emb, q_plus_emb, q_neg_emb,
            num_reps=args.num_reps
        )
        
        results.append({
            "qid": qid,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "single_ms": single_ms,
        })
    
    # Compute overall statistics
    all_means = [r["mean_ms"] for r in results]
    all_singles = [r["single_ms"] for r in results]
    
    overall_mean = statistics.mean(all_means)
    overall_std = statistics.stdev(all_means)
    overall_min = min(all_means)
    overall_max = max(all_means)
    
    # Print results
    print("\n" + "="*70)
    print("TRIX RERANKING TIMING RESULTS (with Emb. cache)")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Encoder: {args.encoder}")
    print(f"Number of queries: {len(results)}")
    print(f"Number of documents: {num_docs}")
    print(f"Repetitions per query: {args.num_reps}")
    print("-"*70)
    print(f"Mean reranking time: {overall_mean:.2f} ms/query")
    print(f"Std deviation: {overall_std:.2f} ms")
    print(f"Min time: {overall_min:.2f} ms")
    print(f"Max time: {overall_max:.2f} ms")
    print(f"Throughput: {1000/overall_mean:.1f} queries/second")
    print("="*70)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output = {
        "statistics": {
            "dataset": args.dataset,
            "encoder": args.encoder,
            "num_queries": len(results),
            "num_documents": num_docs,
            "repetitions": args.num_reps,
            "mean_ms": overall_mean,
            "std_ms": overall_std,
            "min_ms": overall_min,
            "max_ms": overall_max,
            "throughput_qps": 1000/overall_mean,
        },
        "per_query_results": results,
    }
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()