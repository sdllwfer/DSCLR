"""
RepLLaMA 在 ExcluIR 基准上的评测脚本

ExcluIR: Exclusionary Neural Information Retrieval (https://arxiv.org/abs/2404.17288)

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.eval_excluir_repllama \
    --model_name samaya-ai/RepLLaMA-reproduced \
    --data_dir dataset/ExcluIR \
    --dual_queries_path dataset/ExcluIR/dual_queries/dual_queries_excluir.jsonl \
    --output_dir results/excluir_repllama \
    --device cuda
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ExcluIR Metrics
# ============================================================

def compute_recall(result_list: List[List[int]], right_list: List[List[int]],
                   recall_k: List[int] = [1, 5, 10]) -> List[float]:
    """计算 Recall@K (ExcluIR 原始实现)"""
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
    """计算 MRR@10 (ExcluIR 原始实现)"""
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
    """计算 RR (Right Rank): 正例排名高于负例的查询比例

    - 如果正例和负例都在结果中: 正例排名更高则正确
    - 如果只有正例在结果中(负例被排除): 正确
    - 其他情况: 不正确
    """
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
            # 两者都在结果中，正例排名更高则正确
            if pos_rank < neg_rank:
                right_count += 1
        elif pos_rank is not None and neg_rank is None:
            # 只有正例在结果中，负例被排除 → 正确
            right_count += 1
        # 其他情况: 只有负例/两者都不在 → 不正确

    return right_count / len(result_list)


def evaluate_excluir(result_list: List[List[int]],
                     neg_indices: List[int],
                     pos_indices: List[int]) -> Dict[str, float]:
    """计算所有 ExcluIR 指标"""
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
    """加载 ExcluIR 数据

    Returns:
        corpus: 文档文本列表
        queries: 查询列表，每个包含 ExcluQ 和 index
    """
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
    """加载 dual queries (q_plus, q_minus)

    Returns:
        {query_index: {"q_plus": ..., "q_minus": ...}}
    """
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
# Main Evaluation
# ============================================================

def run_excluir_repllama(
    model_name: str = "samaya-ai/RepLLaMA-reproduced",
    data_dir: str = "dataset/ExcluIR",
    dual_queries_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    top_k: int = 10,
    batch_size: int = 32,
    use_cache: bool = True,
    device: str = "cuda",
):
    """运行 RepLLaMA 在 ExcluIR 上的评测"""
    if output_dir is None:
        output_dir = "results/excluir_repllama"
    os.makedirs(output_dir, exist_ok=True)

    # Device setup
    try:
        torch.cuda._lazy_init()
    except Exception:
        pass

    # Load data
    corpus, queries = load_excluir_data(data_dir)
    n_docs = len(corpus)

    # Load dual queries if provided
    dual_data = {}
    has_dual = False
    if dual_queries_path and os.path.exists(dual_queries_path):
        dual_data = load_dual_queries(dual_queries_path)
        has_dual = len(dual_data) > 0

    # Load RepLLaMA encoder
    from eval.models.repllama_encoder import RepLLaMAEncoder

    logger.info(f"Loading RepLLaMA encoder: {model_name}")
    encoder = RepLLaMAEncoder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=True,
    )

    # Encode corpus with cache
    cache_dir = os.path.join(data_dir, "embeddings")
    model_short_name = model_name.split("/")[-1].replace("-", "_")
    cache_file = os.path.join(cache_dir, f"{model_short_name}_corpus_embeddings.pt")

    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading cached corpus embeddings from {cache_file}")
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        doc_embeddings = cache["embeddings"]
        logger.info(f"Cached: {n_docs} docs, shape={doc_embeddings.shape}")
    else:
        logger.info(f"Encoding {n_docs} documents with RepLLaMA...")
        doc_embeddings = encoder.encode_documents(corpus, batch_size=batch_size)
        if doc_embeddings.dim() == 2:
            doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=1)
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save({"embeddings": doc_embeddings.cpu()}, cache_file)
            logger.info(f"Cached corpus embeddings to {cache_file}")

    doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=-1).to(device)
    logger.info(f"Corpus embeddings shape: {doc_embeddings.shape}")

    # Prepare query texts
    query_texts = []
    q_plus_texts = []
    q_minus_texts = []
    neg_indices = []
    pos_indices = []
    has_neg_flags = []

    for i, q in enumerate(queries):
        # Support both ExcluQ and RQ_rewrite field names
        query_text = q.get("ExcluQ", q.get("RQ_rewrite", ""))
        index = q.get("index", q.get("corpus_sub_index", []))

        if not query_text or len(index) < 2:
            continue

        query_texts.append(query_text)
        neg_indices.append(index[0])
        pos_indices.append(index[1] if len(index) > 1 else index[0])

        if has_dual and i in dual_data:
            d = dual_data[i]
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")
            if not q_minus or q_minus.strip().upper() in ("[NONE]", "NONE", "NULL", "N/A", ""):
                q_minus = ""
                has_neg_flags.append(False)
            else:
                has_neg_flags.append(True)
            q_plus_texts.append(q_plus if q_plus else query_text)
            q_minus_texts.append(q_minus)
        else:
            q_plus_texts.append(query_text)
            q_minus_texts.append("")
            has_neg_flags.append(False)

    n_queries = len(query_texts)
    logger.info(f"Processing {n_queries} queries (has Q_minus: {sum(has_neg_flags)})")

    # Encode queries
    logger.info("Encoding Q_full (original) queries...")
    q_full_emb = encoder.encode_queries(query_texts, batch_size=batch_size)
    q_full_emb = F.normalize(q_full_emb.to(device).float(), p=2, dim=-1)

    logger.info("Encoding Q_plus queries...")
    q_pos_emb = encoder.encode_queries(q_plus_texts, batch_size=batch_size)
    q_pos_emb = F.normalize(q_pos_emb.to(device).float(), p=2, dim=-1)

    # Encode Q_minus (only non-empty ones)
    q_minus_emb = None
    if any(has_neg_flags):
        non_empty_minus = [(i, q_minus_texts[i]) for i in range(n_queries) if has_neg_flags[i]]
        minus_indices = [x[0] for x in non_empty_minus]
        minus_texts = [x[1] for x in non_empty_minus]

        logger.info(f"Encoding {len(minus_texts)} Q_minus queries...")
        q_neg_emb_all = encoder.encode_queries(minus_texts, batch_size=batch_size)
        q_neg_emb_all = F.normalize(q_neg_emb_all.to(device).float(), p=2, dim=-1)

        # Build full Q_minus embedding matrix (zeros for queries without Q_minus)
        q_minus_emb = torch.zeros_like(q_full_emb)
        for j, orig_idx in enumerate(minus_indices):
            q_minus_emb[orig_idx] = q_neg_emb_all[j]

    # Compute similarity scores
    logger.info("Computing similarity scores...")
    S_full = torch.matmul(q_full_emb, doc_embeddings.T)  # [n_queries, n_docs]
    S_pos = torch.matmul(q_pos_emb, doc_embeddings.T)    # [n_queries, n_docs]
    S_neg = torch.zeros_like(S_full)
    if q_minus_emb is not None:
        S_neg = torch.matmul(q_minus_emb, doc_embeddings.T)

    # ---- Baseline (no exclusion) ----
    logger.info("Computing baseline results (original query only)...")
    baseline_results = []
    for i in range(n_queries):
        scores = S_full[i]
        top_k_indices = torch.topk(scores, min(top_k, n_docs)).indices.cpu().tolist()
        baseline_results.append(top_k_indices)

    baseline_metrics = evaluate_excluir(baseline_results, neg_indices, pos_indices)
    logger.info(f"Baseline metrics: {baseline_metrics}")

    # ---- Q_plus only (reward, no penalty) ----
    logger.info("Computing Q_plus only results (S_full + S_pos, no penalty)...")
    qplus_results = []
    for i in range(n_queries):
        scores = S_full[i] + S_pos[i]  # Simple additive combination
        top_k_indices = torch.topk(scores, min(top_k, n_docs)).indices.cpu().tolist()
        qplus_results.append(top_k_indices)

    qplus_metrics = evaluate_excluir(qplus_results, neg_indices, pos_indices)
    logger.info(f"Q_plus only metrics: {qplus_metrics}")

    # ---- DeIR-Dual V2 (full) ----
    # Parameters from negconstraint_eval.py
    ALPHA = 1.0
    BETA = 1.5
    DELTA = 0.05
    T_SAFETY = 20.0

    logger.info("Computing DeIR-Dual V2 results (full scoring)...")
    deir_results = []
    for i in range(n_queries):
        s_full = S_full[i]
        s_pos = S_pos[i]
        s_neg = S_neg[i]
        has_neg = has_neg_flags[i]

        if not has_neg:
            # No exclusion signal, use baseline
            s_final = s_full
        else:
            # DeIR-Dual V2 scoring
            tau = 0.0 + DELTA  # cos(Q_full, Q_neg) is approximated as 0
            overflow = s_neg - tau
            smooth_penalty = F.softplus(overflow)
            raw_penalty = ALPHA * smooth_penalty
            safety = 1.0 - torch.sigmoid((s_neg - tau) * T_SAFETY)
            s_final = s_full + BETA * s_pos * safety - raw_penalty

        top_k_indices = torch.topk(s_final, min(top_k, n_docs)).indices.cpu().tolist()
        deir_results.append(top_k_indices)

    deir_metrics = evaluate_excluir(deir_results, neg_indices, pos_indices)
    logger.info(f"DeIR-Dual V2 metrics: {deir_metrics}")

    # Summary
    print("\n" + "=" * 80)
    print(f"ExcluIR EVALUATION SUMMARY (RepLLaMA: {model_name})")
    print("=" * 80)
    print(f"{'Method':<35} {'R@1':>8} {'R@10':>8} {'MRR@10':>8} {'delta_R@1':>10} {'RR':>8}")
    print("-" * 80)
    print(f"{'Baseline (original query)':<35} {baseline_metrics['R@1']:>8.2f} {baseline_metrics['R@10']:>8.2f} {baseline_metrics['MRR@10']:>8.2f} {baseline_metrics['delta_R@1']:>10.2f} {baseline_metrics['RR']:>8.2f}")
    print(f"{'Q_plus only (no penalty)':<35} {qplus_metrics['R@1']:>8.2f} {qplus_metrics['R@10']:>8.2f} {qplus_metrics['MRR@10']:>8.2f} {qplus_metrics['delta_R@1']:>10.2f} {qplus_metrics['RR']:>8.2f}")
    print(f"{'DeIR-Dual V2 (full)':<35} {deir_metrics['R@1']:>8.2f} {deir_metrics['R@10']:>8.2f} {deir_metrics['MRR@10']:>8.2f} {deir_metrics['delta_R@1']:>10.2f} {deir_metrics['RR']:>8.2f}")
    print("-" * 80)

    # Save results
    result_data = {
        "encoder": "repllama",
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_queries": n_queries,
        "n_docs": n_docs,
        "has_dual_queries": has_dual,
        "q_minus_rate": f"{sum(has_neg_flags)}/{n_queries}",
        "baseline_metrics": baseline_metrics,
        "qplus_only_metrics": qplus_metrics,
        "deir_dual_v2_metrics": deir_metrics,
        "params": {
            "alpha": ALPHA,
            "beta": BETA,
            "delta": DELTA,
            "t_safety": T_SAFETY,
            "top_k": top_k,
        },
    }

    result_path = os.path.join(output_dir, "metrics_summary.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {result_path}")

    # Cleanup
    del encoder
    torch.cuda.empty_cache()

    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RepLLaMA evaluation on ExcluIR")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--data_dir", type=str, default="dataset/ExcluIR")
    parser.add_argument("--dual_queries_path", type=str,
                        default="dataset/ExcluIR/dual_queries/dual_queries_excluir.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/excluir_repllama")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = run_excluir_repllama(
        model_name=args.model_name,
        data_dir=args.data_dir,
        dual_queries_path=args.dual_queries_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\nBaseline: {result['baseline_metrics']}")
    print(f"Q_plus only: {result['qplus_only_metrics']}")
    print(f"DeIR-Dual V2: {result['deir_dual_v2_metrics']}")