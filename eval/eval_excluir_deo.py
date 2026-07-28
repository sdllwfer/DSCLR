"""
DEO: Direct Embedding Optimization on ExcluIR Benchmark

Reproduction of:
  DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval
  (Lee et al., 2026, arXiv:2603.09185)

Method:
  1. Query Decomposition: LLM decomposes query into positive & negative sub-queries
  2. Direct Embedding Optimization: Optimize query embedding via contrastive loss
     L(e_u) = λ_p/K * Σ||e_u - e_pi||² - λ_n/M * Σ||e_u - e_nj||² + λ_o * ||e_u - e_o||²
  3. Retrieval: Use optimized embedding for retrieval

ExcluIR Metrics:
  - R@1, R@5, R@10: Recall of positive document
  - MRR@10: Mean Reciprocal Rank of positive document
  - ΔR@1 = R@1(pos) - R@1(neg)
  - ΔMRR@10 = MRR@10(pos) - MRR@10(neg)
  - RR: Proportion of queries where positive doc ranks higher than negative doc

Usage:
  python -m eval.eval_excluir_deo \
    --model_name BAAI/bge-large-en-v1.5 \
    --data_dir dataset/ExcluIR \
    --dual_queries_path dataset/ExcluIR/dual_queries/dual_queries_excluir.jsonl \
    --device cuda
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ExcluIR Metrics (same as eval_excluir_trace.py)
# ============================================================

def compute_recall(result_list: List[List[int]], right_list: List[List[int]],
                   recall_k: List[int] = [1, 5, 10]) -> List[float]:
    """Compute Recall@K (ExcluIR official implementation)"""
    recall_values = [0.0] * len(recall_k)
    for i in range(len(result_list)):
        retrieved = set(result_list[i])
        relevant = set(right_list[i])
        if not relevant:
            continue
        for j, k in enumerate(recall_k):
            hit = len(retrieved & set(result_list[i][:k]) & relevant)
            recall_values[j] += hit / len(relevant)
    return [r / len(result_list) for r in recall_values]


def compute_mrr(result_list: List[List[int]], right_list: List[List[int]]) -> float:
    """Compute MRR@10"""
    mrr = 0.0
    for i in range(len(result_list)):
        relevant = set(right_list[i])
        for rank, doc_idx in enumerate(result_list[i][:10]):
            if doc_idx in relevant:
                mrr += 1.0 / (rank + 1)
                break
    return mrr / len(result_list)


def compute_right_rank(result_list: List[List[int]],
                       neg_indices: List[int],
                       pos_indices: List[int]) -> float:
    """Compute RR (Right Rank): proportion of queries where positive ranks higher than negative"""
    right_count = 0
    for i in range(len(result_list)):
        neg_idx = neg_indices[i]
        pos_idx = pos_indices[i]
        result = result_list[i]

        pos_rank = None
        neg_rank = None
        for rank, doc_idx in enumerate(result):
            if doc_idx == pos_idx:
                pos_rank = rank
            if doc_idx == neg_idx:
                neg_rank = rank

        if pos_rank is not None and neg_rank is not None:
            if pos_rank < neg_rank:
                right_count += 1
        elif pos_rank is not None and neg_rank is None:
            right_count += 1

    return right_count / len(result_list)


def evaluate_excluir(result_list: List[List[int]],
                     neg_indices: List[int],
                     pos_indices: List[int]) -> Dict[str, float]:
    """Compute all ExcluIR metrics"""
    right_list_pos = [[pos_indices[i]] for i in range(len(pos_indices))]
    right_list_neg = [[neg_indices[i]] for i in range(len(neg_indices))]

    recall_pos = compute_recall(result_list, right_list_pos)
    recall_neg = compute_recall(result_list, right_list_neg)
    mrr_pos = compute_mrr(result_list, right_list_pos)
    mrr_neg = compute_mrr(result_list, right_list_neg)
    rr = compute_right_rank(result_list, neg_indices, pos_indices)

    metrics = {
        "R@1": round(recall_pos[0] * 100, 2),
        "R@5": round(recall_pos[1] * 100, 2),
        "R@10": round(recall_pos[2] * 100, 2),
        "MRR@10": round(mrr_pos * 100, 2),
        "R@1_neg": round(recall_neg[0] * 100, 2),
        "MRR@10_neg": round(mrr_neg * 100, 2),
        "delta_R@1": round(recall_pos[0] * 100, 2) - round(recall_neg[0] * 100, 2),
        "delta_MRR@10": round(mrr_pos * 100, 2) - round(mrr_neg * 100, 2),
        "RR": round(rr * 100, 2),
    }
    return metrics


# ============================================================
# ExcluIR Data Loading
# ============================================================

def load_excluir_data(data_dir: str) -> Tuple[List[str], List[Dict]]:
    """Load ExcluIR data"""
    corpus_path = os.path.join(data_dir, "corpus.json")
    queries_path = os.path.join(data_dir, "test_manual_final.json")

    logger.info(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info(f"Loaded {len(corpus)} documents")

    logger.info(f"Loading queries from {queries_path}...")
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries")

    return corpus, queries


def load_dual_queries(dual_queries_path: str) -> Dict[int, Dict[str, Any]]:
    """Load dual queries (q_plus, q_minus) for DEO decomposition"""
    dual_data = {}
    with open(dual_queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            qid = item["qid"]
            dual_data[qid] = item
    logger.info(f"Loaded {len(dual_data)} dual queries")
    return dual_data


# ============================================================
# DEO: Direct Embedding Optimization (Core)
# ============================================================

def deo_optimize_embedding(
    e_o: torch.Tensor,       # Original query embedding [d]
    e_pos_list: List[torch.Tensor],  # Positive sub-query embeddings [K, d]
    e_neg_list: List[torch.Tensor],  # Negative sub-query embeddings [M, d]
    lambda_p: float = 1.0,   # Positive attraction weight
    lambda_n: float = 1.0,   # Negative repulsion weight
    lambda_o: float = 0.2,   # Original consistency weight
    n_steps: int = 20,       # Number of optimization steps
    lr: float = 0.01,        # Learning rate
) -> torch.Tensor:
    """
    DEO: Direct Embedding Optimization
    
    Minimizes:
      L(e_u) = λ_p/K * Σ||e_u - e_pi||² - λ_n/M * Σ||e_u - e_nj||² + λ_o * ||e_u - e_o||²
    
    Args:
        e_o: Original query embedding
        e_pos_list: List of positive sub-query embeddings
        e_neg_list: List of negative sub-query embeddings
        lambda_p: Weight for positive attraction term
        lambda_n: Weight for negative repulsion term
        lambda_o: Weight for original consistency term
        n_steps: Number of Adam optimization steps
        lr: Learning rate for Adam optimizer
    
    Returns:
        e_u: Optimized query embedding
    """
    K = len(e_pos_list)
    M = len(e_neg_list)
    
    if K == 0 and M == 0:
        return e_o.clone()
    
    # Stack positive and negative embeddings
    e_pos = torch.stack(e_pos_list) if K > 0 else torch.empty(0, device=e_o.device)
    e_neg = torch.stack(e_neg_list) if M > 0 else torch.empty(0, device=e_o.device)
    
    # Initialize learnable embedding
    e_u = e_o.clone().detach().requires_grad_(True)
    
    # Adam optimizer (as specified in DEO paper)
    optimizer = torch.optim.Adam([e_u], lr=lr)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Positive attraction: pull e_u toward positive embeddings
        loss_pos = torch.tensor(0.0, device=e_o.device)
        if K > 0:
            loss_pos = (lambda_p / K) * torch.sum(torch.norm(e_u.unsqueeze(0) - e_pos, p=2, dim=-1) ** 2)
        
        # Negative repulsion: push e_u away from negative embeddings
        # Note: negative sign means minimizing this term maximizes distance
        loss_neg = torch.tensor(0.0, device=e_o.device)
        if M > 0:
            loss_neg = -(lambda_n / M) * torch.sum(torch.norm(e_u.unsqueeze(0) - e_neg, p=2, dim=-1) ** 2)
        
        # Original consistency: keep e_u close to original embedding
        loss_orig = lambda_o * torch.norm(e_u - e_o, p=2) ** 2
        
        # Total loss
        loss = loss_pos + loss_neg + loss_orig
        
        loss.backward()
        optimizer.step()
    
    return e_u.detach()


# ============================================================
# DEO Query Decomposition (LLM-based)
# ============================================================

DEO_DECOMPOSE_PROMPT = """You are a query decomposition assistant. Given a user query that may contain negation or exclusion, decompose it into positive and negative sub-queries.

Positive sub-queries should capture what the user WANTS to find (inclusion intent).
Negative sub-queries should capture what the user DOES NOT want (exclusion intent).

Input Query: {query}

Respond in the following JSON format:
{{
    "positive_queries": ["query1 about what to include", "query2 about what to include"],
    "negative_queries": ["query1 about what to exclude", "query2 about what to exclude"]
}}

Important:
- Positive queries should expand on the inclusion aspects of the original query
- Negative queries should explicitly capture the exclusion aspects
- If there is no exclusion/negation in the query, negative_queries should be an empty list
- Provide 1-3 positive queries and 0-3 negative queries"""


def decompose_query_with_llm(
    query: str,
    llm_model=None,
    llm_tokenizer=None,
    device: str = "cuda",
) -> Tuple[List[str], List[str]]:
    """
    Decompose a query using a local LLM (Qwen3-4B or similar).
    
    Returns:
        positive_queries: List of positive sub-queries
        negative_queries: List of negative sub-queries
    """
    if llm_model is None:
        # Fallback: simple heuristic decomposition
        return heuristic_decompose(query)
    
    prompt = DEO_DECOMPOSE_PROMPT.format(query=query)
    
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Apply chat template
        text = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = llm_tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = llm_tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Parse JSON response
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            pos = result.get("positive_queries", [])
            neg = result.get("negative_queries", [])
            return pos, neg
    except Exception as e:
        logger.warning(f"LLM decomposition failed: {e}")
    
    return heuristic_decompose(query)


def heuristic_decompose(query: str) -> Tuple[List[str], List[str]]:
    """Simple heuristic decomposition as fallback"""
    # Common exclusion patterns
    exclusion_patterns = [
        "except", "excluding", "besides", "other than", "not including",
        "without", "but not", "apart from", "aside from", "rather than",
        "instead of", "don't", "do not", "does not", "not"
    ]
    
    has_exclusion = any(pat in query.lower() for pat in exclusion_patterns)
    
    if not has_exclusion:
        return [query], []
    
    # Return the query itself as positive, empty negative
    # (This is a minimal fallback; LLM decomposition is preferred)
    return [query], []


# ============================================================
# Main Evaluation
# ============================================================

def run_excluir_deo(
    model_name: str = "BAAI/bge-large-en-v1.5",
    data_dir: str = "dataset/ExcluIR",
    dual_queries_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    # DEO hyperparameters
    lambda_p: float = 1.0,
    lambda_n: float = 1.0,
    lambda_o: float = 0.2,
    n_steps: int = 20,
    lr: float = 0.01,
    # LLM decomposition
    use_llm_decompose: bool = False,
    llm_name: str = "Qwen/Qwen3-4B",
    # General
    device: str = "auto",
    batch_size: int = 64,
    top_k: int = 10,
    use_cache: bool = True,
):
    """Run DEO evaluation on ExcluIR"""
    if output_dir is None:
        output_dir = f"evaluation/excluir_deo/{model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # Device setup
    if device == "auto":
        try:
            torch.cuda._lazy_init()
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Load data
    corpus, queries = load_excluir_data(data_dir)
    n_docs = len(corpus)

    # Load dual queries for DEO decomposition
    dual_data = {}
    has_dual = False
    if dual_queries_path and os.path.exists(dual_queries_path):
        dual_data = load_dual_queries(dual_queries_path)
        has_dual = len(dual_data) > 0

    # Load LLM for decomposition if requested
    llm_model = None
    llm_tokenizer = None
    if use_llm_decompose:
        logger.info(f"Loading LLM for decomposition: {llm_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info("LLM loaded for query decomposition")

    # Load encoder
    from eval.models import ModelFactory
    logger.info(f"Loading encoder: {model_name}")
    encoder = ModelFactory.create(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=True,
    )

    # Encode corpus
    cache_dir = os.path.join(data_dir, "embeddings")
    model_short_name = model_name.split("/")[-1].replace("-", "_")
    cache_file = os.path.join(cache_dir, f"{model_short_name}_corpus_embeddings.npy")

    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading cached corpus embeddings from {cache_file}")
        doc_embeddings = np.load(cache_file)
        doc_embeddings = torch.tensor(doc_embeddings, device=device, dtype=torch.float16)
    else:
        logger.info(f"Encoding {n_docs} documents...")
        doc_embeddings = encoder.encode_documents(corpus, batch_size=batch_size)
        doc_embeddings = doc_embeddings.to(device=device, dtype=torch.float16)
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_file, doc_embeddings.cpu().float().numpy())
            logger.info(f"Cached corpus embeddings to {cache_file}")

    # Normalize
    doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=-1)
    logger.info(f"Corpus embeddings shape: {doc_embeddings.shape}")

    # Prepare query data
    query_texts = []
    q_plus_texts = []
    q_minus_texts = []
    neg_indices_list = []
    pos_indices_list = []
    has_neg_flags = []

    for i, q in enumerate(queries):
        query_text = q.get("ExcluQ", q.get("RQ_rewrite", ""))
        index = q.get("index", q.get("corpus_sub_index", []))

        if not query_text or len(index) < 2:
            continue

        query_texts.append(query_text)
        neg_index = index[0]
        pos_index = index[1] if len(index) > 1 else index[0]
        neg_indices_list.append(neg_index)
        pos_indices_list.append(pos_index)

        # Get positive and negative sub-queries
        if use_llm_decompose and llm_model is not None:
            pos_queries, neg_queries = decompose_query_with_llm(
                query_text, llm_model, llm_tokenizer, device
            )
            # Use the first positive query (or original if empty)
            q_plus = pos_queries[0] if pos_queries else query_text
            q_minus = neg_queries[0] if neg_queries else ""
            has_neg = len(neg_queries) > 0 and q_minus.strip() != ""
        elif has_dual and i in dual_data:
            d = dual_data[i]
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")
            if not q_minus or q_minus.strip().upper() in ("[NONE]", "NONE", "NULL", "N/A", ""):
                q_minus = ""
                has_neg = False
            else:
                has_neg = True
        else:
            q_plus = query_text
            q_minus = ""
            has_neg = False

        q_plus_texts.append(q_plus if q_plus else query_text)
        q_minus_texts.append(q_minus)
        has_neg_flags.append(has_neg)

    n_queries = len(query_texts)
    logger.info(f"Processing {n_queries} queries (dual queries: {has_dual}, LLM decompose: {use_llm_decompose})")
    logger.info(f"Queries with negation: {sum(has_neg_flags)}/{n_queries}")

    # Encode queries
    logger.info("Encoding Q_full queries...")
    q_full_emb = encoder.encode_queries(query_texts, batch_size=batch_size)
    q_full_emb = F.normalize(q_full_emb.to(device).float(), p=2, dim=-1)

    logger.info("Encoding Q_plus queries...")
    q_pos_emb = encoder.encode_queries(q_plus_texts, batch_size=batch_size)
    q_pos_emb = F.normalize(q_pos_emb.to(device).float(), p=2, dim=-1)

    # Encode Q_minus
    q_minus_emb = None
    if any(has_neg_flags):
        non_empty_minus = [(i, q_minus_texts[i]) for i in range(n_queries) if has_neg_flags[i]]
        minus_indices = [x[0] for x in non_empty_minus]
        minus_texts = [x[1] for x in non_empty_minus]

        logger.info(f"Encoding {len(minus_texts)} Q_minus queries...")
        q_neg_emb_all = encoder.encode_queries(minus_texts, batch_size=batch_size)
        q_neg_emb_all = F.normalize(q_neg_emb_all.to(device).float(), p=2, dim=-1)

        # Build full Q_minus embedding matrix
        q_minus_emb = torch.zeros_like(q_full_emb)
        for j, orig_idx in enumerate(minus_indices):
            q_minus_emb[orig_idx] = q_neg_emb_all[j]

    # ---- Baseline (no DEO) ----
    logger.info("Computing baseline results (no DEO)...")
    S_full = torch.matmul(q_full_emb, doc_embeddings.T)  # [n_queries, n_docs]

    baseline_results = []
    for i in range(n_queries):
        scores = S_full[i]
        top_k_indices = torch.topk(scores, min(top_k, n_docs)).indices.cpu().tolist()
        baseline_results.append(top_k_indices)

    baseline_metrics = evaluate_excluir(baseline_results, neg_indices_list, pos_indices_list)
    logger.info(f"Baseline metrics: {baseline_metrics}")

    # ---- DEO Optimization ----
    logger.info(f"Running DEO optimization (λ_p={lambda_p}, λ_n={lambda_n}, λ_o={lambda_o}, steps={n_steps}, lr={lr})...")
    
    deo_results = []
    deo_time_start = time.time()
    
    for i in range(n_queries):
        e_o = q_full_emb[i]  # Original query embedding
        
        if has_neg_flags[i]:
            # Positive sub-query embedding
            e_pos = [q_pos_emb[i]]
            # Negative sub-query embedding
            e_neg = [q_minus_emb[i]] if q_minus_emb is not None else []
            
            # DEO optimization
            e_u = deo_optimize_embedding(
                e_o, e_pos, e_neg,
                lambda_p=lambda_p,
                lambda_n=lambda_n,
                lambda_o=lambda_o,
                n_steps=n_steps,
                lr=lr,
            )
        else:
            # No negation → use original embedding
            e_u = e_o
        
        # Compute similarity scores with optimized embedding
        scores = torch.matmul(e_u, doc_embeddings.T)
        top_k_indices = torch.topk(scores, min(top_k, n_docs)).indices.cpu().tolist()
        deo_results.append(top_k_indices)
        
        if (i + 1) % 500 == 0 or i == n_queries - 1:
            elapsed = time.time() - deo_time_start
            avg_time = elapsed / (i + 1)
            logger.info(f"  [{i+1}/{n_queries}] avg time per query: {avg_time*1000:.2f}ms")

    deo_metrics = evaluate_excluir(deo_results, neg_indices_list, pos_indices_list)
    logger.info(f"DEO metrics: {deo_metrics}")

    # ---- Also run DEO with different hyperparameter configurations ----
    configs = [
        {"lambda_p": 1.0, "lambda_n": 1.0, "lambda_o": 0.2, "n_steps": 20, "label": "DEO(default)"},
        {"lambda_p": 1.0, "lambda_n": 1.0, "lambda_o": 1.0, "n_steps": 20, "label": "DEO(λ_o=1.0)"},
        {"lambda_p": 0.2, "lambda_n": 1.0, "lambda_o": 1.0, "n_steps": 20, "label": "DEO(λ_p=0.2)"},
        {"lambda_p": 1.0, "lambda_n": 1.0, "lambda_o": 0.2, "n_steps": 50, "label": "DEO(steps=50)"},
    ]

    all_results = []
    for cfg in configs:
        label = cfg.pop("label")
        logger.info(f"Running {label}...")
        
        config_results = []
        config_start = time.time()
        
        for i in range(n_queries):
            e_o = q_full_emb[i]
            
            if has_neg_flags[i]:
                e_pos = [q_pos_emb[i]]
                e_neg = [q_minus_emb[i]] if q_minus_emb is not None else []
                
                e_u = deo_optimize_embedding(
                    e_o, e_pos, e_neg,
                    **cfg,
                    lr=lr,
                )
            else:
                e_u = e_o
            
            scores = torch.matmul(e_u, doc_embeddings.T)
            top_k_indices = torch.topk(scores, min(top_k, n_docs)).indices.cpu().tolist()
            config_results.append(top_k_indices)
        
        config_metrics = evaluate_excluir(config_results, neg_indices_list, pos_indices_list)
        config_time = time.time() - config_start
        
        logger.info(f"  {label}: {config_metrics} (time: {config_time:.1f}s)")
        
        all_results.append({
            "config": label,
            "params": {**cfg, "lr": lr},
            "metrics": config_metrics,
            "time": config_time,
        })

    # Summary
    logger.info("=" * 70)
    logger.info("DEO on ExcluIR - Evaluation Complete")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Queries: {n_queries}, Documents: {n_docs}")
    logger.info(f"  Queries with negation: {sum(has_neg_flags)}/{n_queries}")
    logger.info(f"  Baseline: {baseline_metrics}")
    for r in all_results:
        logger.info(f"  {r['config']}: {r['metrics']}")
    logger.info("=" * 70)

    # Compute improvements
    improvements = {}
    for r in all_results:
        label = r["config"]
        improvements[label] = {
            "delta_R@1": round(r["metrics"]["delta_R@1"] - baseline_metrics["delta_R@1"], 2),
            "delta_MRR@10": round(r["metrics"]["delta_MRR@10"] - baseline_metrics["delta_MRR@10"], 2),
            "delta_RR": round(r["metrics"]["RR"] - baseline_metrics["RR"], 2),
        }

    # Save results
    result_data = {
        "method": "DEO",
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_queries": n_queries,
        "n_docs": n_docs,
        "n_queries_with_negation": sum(has_neg_flags),
        "has_dual_queries": has_dual,
        "use_llm_decompose": use_llm_decompose,
        "baseline_metrics": baseline_metrics,
        "all_configs": all_results,
        "improvements_over_baseline": improvements,
    }

    result_path = os.path.join(output_dir, "excluir_deo_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {result_path}")

    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEO evaluation on ExcluIR")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--data_dir", type=str, default="dataset/ExcluIR")
    parser.add_argument("--dual_queries_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    # DEO hyperparameters
    parser.add_argument("--lambda_p", type=float, default=1.0, help="Positive attraction weight")
    parser.add_argument("--lambda_n", type=float, default=1.0, help="Negative repulsion weight")
    parser.add_argument("--lambda_o", type=float, default=0.2, help="Original consistency weight")
    parser.add_argument("--n_steps", type=int, default=20, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    # LLM decomposition
    parser.add_argument("--use_llm_decompose", action="store_true", help="Use local LLM for query decomposition")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-4B", help="LLM for decomposition")
    # General
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--use_cache", type=str, default="true")

    args = parser.parse_args()
    use_cache = args.use_cache.lower() == "true"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = run_excluir_deo(
        model_name=args.model_name,
        data_dir=args.data_dir,
        dual_queries_path=args.dual_queries_path,
        output_dir=args.output_dir,
        lambda_p=args.lambda_p,
        lambda_n=args.lambda_n,
        lambda_o=args.lambda_o,
        n_steps=args.n_steps,
        lr=args.lr,
        use_llm_decompose=args.use_llm_decompose,
        llm_name=args.llm_name,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        use_cache=use_cache,
    )

    print(f"\n{'='*70}")
    print(f"DEO on ExcluIR - Results Summary")
    print(f"{'='*70}")
    print(f"\nBaseline: {result['baseline_metrics']}")
    for r in result["all_configs"]:
        print(f"{r['config']:25s}: {r['metrics']}")
    print(f"\nImprovements over baseline:")
    for label, imp in result["improvements_over_baseline"].items():
        print(f"  {label:25s}: ΔR@1={imp['delta_R@1']:+.2f}, ΔMRR@10={imp['delta_MRR@10']:+.2f}, ΔRR={imp['delta_RR']:+.2f}")
