"""
TRACE on BEIR with Original Queries (No Neutral Instruction)

Evaluates TRACE on BEIR datasets using original query text as q_base,
without appending neutral task instructions.

Key design:
  - q_base = original query text (with BGE query prefix, NOT neutral instruction)
  - q_plus / q_minus loaded from existing CONSERVATIVE dual_queries JSONL files
  - Full TRACE scoring including exclusion channel (when q_minus != [NONE])
  - V8.6 per-query lambda/tau grid search
  - Reports nDCG@10 and MAP@100 for baseline and TRACE

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \\
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.run_trace_beir \\
    --dataset scifact \\
    --model_name BAAI/bge-large-en-v1.5 \\
    --dual_queries_path dataset/BEIR/dual_queries/scifact_CONSERVATIVE_t01.jsonl \\
    --output_dir results/beir_trace_original/scifact \\
    --device cuda
"""

import os, sys, json, argparse, time, logging

# Handle --gpus argument before importing torch
_pre_argv = sys.argv[:]
_pre_gpus = None
for i, a in enumerate(_pre_argv):
    if a == "--gpus" and i + 1 < len(_pre_argv):
        _pre_gpus = _pre_argv[i + 1]
    elif a.startswith("--gpus="):
        _pre_gpus = a.split("=", 1)[1]
if _pre_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_gpus

import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force online mode for BEIR datasets
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.pop("HF_ENDPOINT", None)

try:
    import huggingface_hub.constants as _hf_const
    _hf_const.HF_HUB_OFFLINE = False
except Exception:
    pass

import datasets
try:
    datasets.config.HF_DATASETS_OFFLINE = False
except Exception:
    pass

import pytrec_eval
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from eval.models.encoder import ModelFactory
from eval.engine_trace import TRACEEvaluator, robust_standardize, _mad, fit_huber_regression

BEIR_DATASET_MAP = {
    "nq": "BeIR/nq",
    "hotpotqa": "BeIR/hotpotqa",
    "quora": "BeIR/quora",
    "fiqa": "BeIR/fiqa",
    "arguana": "BeIR/arguana",
    "scifact": "BeIR/scifact",
    "nfcorpus": "BeIR/nfcorpus",
    "trec-covid": "BeIR/trec-covid",
}

DEFAULT_DUAL_QUERIES_MAP = {
    "nq": "dataset/BEIR/dual_queries/nq_CONSERVATIVE_t01.jsonl",
    "hotpotqa": "dataset/BEIR/dual_queries/hotpotqa_CONSERVATIVE_t01.jsonl",
    "quora": "dataset/BEIR/dual_queries/quora_CONSERVATIVE_t01.jsonl",
    "fiqa": "dataset/BEIR/dual_queries/fiqa_CONSERVATIVE_t01.jsonl",
    "arguana": "dataset/BEIR/dual_queries/arguana_CONSERVATIVE_t01.jsonl",
    "scifact": "dataset/BEIR/dual_queries/scifact_CONSERVATIVE_t01.jsonl",
    "nfcorpus": "dataset/BEIR/dual_queries/nfcorpus_CONSERVATIVE_t01.jsonl",
    "trec-covid": "dataset/BEIR/dual_queries/trec-covid_CONSERVATIVE_t01.jsonl",
}


def load_beir_dataset(dataset_name):
    """Load BEIR dataset from HuggingFace."""
    hf_name = BEIR_DATASET_MAP.get(dataset_name, dataset_name)

    from datasets import load_dataset

    corpus = {}
    queries = {}
    qrels = {}

    logger.info(f"Loading corpus from {hf_name}...")
    corpus_ds = load_dataset(hf_name, "corpus")
    for doc in corpus_ds['corpus']:
        text = doc.get('text', '') or ''
        title = doc.get('title', '') or ''
        corpus[str(doc['_id'])] = f"{title} {text}".strip() if title else text

    logger.info(f"Loading queries from {hf_name}...")
    queries_ds = load_dataset(hf_name, "queries")
    for q in queries_ds['queries']:
        queries[str(q['_id'])] = q.get('text', '')

    logger.info(f"Loading qrels from {hf_name}...")
    qrels_hf_name = hf_name + "-qrels"
    try:
        qrels_ds = load_dataset(qrels_hf_name)
        split_name = None
        for s in ['test', 'train', 'validation']:
            if s in qrels_ds:
                split_name = s
                break
        if split_name is None:
            split_name = list(qrels_ds.keys())[0]
        for item in qrels_ds[split_name]:
            qid = str(item['query-id'])
            did = str(item['corpus-id'])
            score = int(item['score'])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score
    except Exception as e:
        logger.error(f"Failed to load qrels from {qrels_hf_name}: {e}")

    return corpus, queries, qrels


def load_dual_queries(dual_path):
    """Load dual queries for BEIR datasets."""
    dual_data = {}
    if dual_path and os.path.exists(dual_path):
        with open(dual_path) as f:
            for line in f:
                item = json.loads(line)
                dual_data[item['qid']] = item
    return dual_data


def is_none_query(q):
    if not q:
        return True
    t = str(q).strip().upper()
    return t in ("[NONE]", "NONE", "NULL", "N/A", "")


def compute_baseline_metrics(q_embs, doc_embs, eval_qids, doc_ids, qrels, K=100):
    """Compute baseline top-K scores and evaluate with pytrec_eval."""
    n_docs = doc_embs.shape[0]
    n_queries = q_embs.shape[0]

    if n_docs > 100000:
        batch_size = 50000
        topk_scores = torch.full((n_queries, K), float('-inf'), device='cpu')
        topk_indices = torch.full((n_queries, K), -1, dtype=torch.long, device='cpu')
        for start in range(0, n_docs, batch_size):
            end = min(start + batch_size, n_docs)
            doc_batch = doc_embs[start:end].to('cuda')
            q_batch = q_embs.to('cuda')
            S_batch = torch.matmul(q_batch, doc_batch.T).float().cpu()
            del doc_batch
            torch.cuda.empty_cache()
            local_k = min(K, S_batch.shape[1])
            local_scores, local_idx = torch.topk(S_batch, k=local_k, dim=1)
            local_indices = local_idx + start
            del S_batch
            merged_scores = torch.cat([topk_scores, local_scores], dim=1)
            merged_indices = torch.cat([topk_indices, local_indices], dim=1)
            topk_scores, merge_idx = torch.topk(merged_scores, k=K, dim=1)
            topk_indices = torch.gather(merged_indices, 1, merge_idx)
            del merged_scores, merged_indices, local_scores, local_idx, local_indices
    else:
        S = torch.matmul(q_embs.to('cuda'), doc_embs.to('cuda').T).float().cpu()
        topk_scores, topk_indices = torch.topk(S, k=min(K, S.shape[1]), dim=1)

    run_data = {}
    for i, qid in enumerate(eval_qids):
        run_data[qid] = {}
        for j in range(topk_scores.shape[1]):
            idx = topk_indices[i, j].item()
            if idx < 0:
                continue
            did = doc_ids[idx]
            run_data[qid][did] = topk_scores[i, j].item()

    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: qrels[qid] for qid in eval_qids if qid in qrels},
        {'ndcg_cut_10', 'map_cut_100'}
    )
    results = evaluator.evaluate(run_data)
    metrics = {}
    for m in ['ndcg_cut_10', 'map_cut_100']:
        vals = [results[qid][m] for qid in eval_qids if qid in results]
        metrics[m] = np.mean(vals) if vals else 0.0
    return metrics, topk_scores, topk_indices


def main():
    parser = argparse.ArgumentParser(description="TRACE on BEIR with Original Queries")
    parser.add_argument("--dataset", type=str, required=True, choices=list(BEIR_DATASET_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--dual_queries_path", type=str, default="",
                        help="Path to dual queries JSONL. Defaults to dataset/BEIR/dual_queries/{dataset}_CONSERVATIVE_t01.jsonl")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g. '4'). Sets CUDA_VISIBLE_DEVICES before torch init.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top-K candidates for TRACE scoring")
    args = parser.parse_args()

    # Resolve dual queries path
    if not args.dual_queries_path:
        args.dual_queries_path = DEFAULT_DUAL_QUERIES_MAP.get(args.dataset, "")
        if not args.dual_queries_path:
            logger.error(f"No default dual queries path for dataset: {args.dataset}")
            return

    logger.info(f"Loading BEIR dataset: {args.dataset}")
    corpus, queries, qrels = load_beir_dataset(args.dataset)
    eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))
    logger.info(f"Corpus: {len(corpus)}, Queries: {len(queries)}, Eval: {len(eval_qids)}")

    # Load dual queries
    dual_data = load_dual_queries(args.dual_queries_path)
    logger.info(f"Loaded {len(dual_data)} dual queries from {args.dual_queries_path}")

    # Create encoder
    encoder = ModelFactory.create(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True
    )

    # Encode corpus
    doc_ids = sorted(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]

    # Try to load cached embeddings
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset", "BEIR", "embeddings", args.dataset
    )
    doc_embs = None

    if "bge" in args.model_name.lower():
        cache_path = os.path.join(cache_dir, f"{args.dataset}_bge_large_en_v1.5_corpus.pt")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached BGE embeddings from {cache_path}...")
            cached = torch.load(cache_path, map_location='cpu')
            if isinstance(cached, dict) and 'embeddings' in cached:
                doc_embs = F.normalize(cached['embeddings'].float(), p=2, dim=1)
                if 'doc_ids' in cached:
                    doc_ids = [str(d) for d in cached['doc_ids']]
            elif isinstance(cached, torch.Tensor):
                doc_embs = F.normalize(cached.float(), p=2, dim=1)
            logger.info(f"Loaded cached embeddings: {doc_embs.shape}")

    if doc_embs is None:
        logger.info(f"Encoding {len(doc_texts)} documents...")
        doc_embs = F.normalize(
            encoder.encode_documents(doc_texts, batch_size=args.batch_size).float(), p=2, dim=1
        ).to('cuda')

    # BGE query prefix (NOT a neutral instruction — this is the model's required query prefix)
    q_prefix = ""
    if "bge" in args.model_name.lower():
        q_prefix = "Represent this sentence for searching relevant passages: "
    elif "e5" in args.model_name.lower() and "mistral" in args.model_name.lower():
        q_prefix = "Instruct: "

    # ============================================================
    # Baseline: original query text as q_base (with BGE prefix)
    # ============================================================
    logger.info("Computing baseline (original query, no TRACE)...")
    q_base_list = [queries[qid] for qid in eval_qids]
    q_base_prefixed = [q_prefix + q for q in q_base_list] if q_prefix else q_base_list
    q_base_embs = F.normalize(
        encoder.encode_queries(q_base_prefixed, batch_size=args.batch_size).float(), p=2, dim=1
    )
    baseline_metrics, topk_scores_base, topk_indices_base = compute_baseline_metrics(
        q_base_embs, doc_embs, eval_qids, doc_ids, qrels, args.top_k
    )
    logger.info(f"Baseline: nDCG@10={baseline_metrics['ndcg_cut_10']:.4f}, "
                f"MAP@100={baseline_metrics['map_cut_100']:.4f}")

    # ============================================================
    # Encode q_plus and q_minus
    # ============================================================
    q_plus_list = []
    q_minus_list = []
    has_neg_mask = []

    for qid in eval_qids:
        d = dual_data.get(qid, {})
        q_plus = d.get('q_plus', '')
        if not q_plus or is_none_query(q_plus):
            q_plus = queries[qid]  # fallback to original query
        q_plus_list.append(q_plus)

        q_minus = d.get('q_minus', '')
        if is_none_query(q_minus):
            q_minus = ""
            has_neg_mask.append(0.0)
        else:
            has_neg_mask.append(1.0)
        q_minus_list.append(q_minus)

    # Encode q_plus
    logger.info("Encoding q_plus...")
    q_plus_prefixed = [q_prefix + q for q in q_plus_list] if q_prefix else q_plus_list
    q_plus_embs = F.normalize(
        encoder.encode_queries(q_plus_prefixed, batch_size=args.batch_size).float(), p=2, dim=1
    )

    # Encode q_minus (only for queries that have exclusion)
    neg_qids = [i for i, h in enumerate(has_neg_mask) if h > 0]
    if neg_qids:
        logger.info(f"Encoding q_minus for {len(neg_qids)} queries with exclusion...")
        neg_texts = [q_minus_list[i] for i in neg_qids]
        neg_prefixed = [q_prefix + q for q in neg_texts] if q_prefix else neg_texts
        neg_embs = F.normalize(
            encoder.encode_queries(neg_prefixed, batch_size=args.batch_size).float(), p=2, dim=1
        )
    else:
        logger.info("No queries with exclusion — skipping q_minus encoding.")
        neg_embs = None

    # ============================================================
    # Compute S_full, S_pos, S_neg on top-K candidates
    # ============================================================
    K = args.top_k
    n_queries = len(eval_qids)

    logger.info(f"Computing TRACE scores on top-{K} candidates...")

    # Gather document embeddings for top-K candidates
    S_full_topk = torch.zeros(n_queries, K)
    S_pos_topk = torch.zeros(n_queries, K)
    S_neg_topk = torch.zeros(n_queries, K)

    for i in range(n_queries):
        indices = topk_indices_base[i]
        valid_mask = indices >= 0
        if valid_mask.sum() == 0:
            continue
        valid_indices = indices[valid_mask]

        doc_emb_selected = doc_embs[valid_indices].to('cuda')
        k = valid_mask.sum().item()

        s_full = torch.matmul(q_base_embs[i].unsqueeze(0).to('cuda'), doc_emb_selected.T).squeeze(0)
        S_full_topk[i, :k] = s_full.float().cpu()

        s_pos = torch.matmul(q_plus_embs[i].unsqueeze(0).to('cuda'), doc_emb_selected.T).squeeze(0)
        S_pos_topk[i, :k] = s_pos.float().cpu()

        if has_neg_mask[i] > 0 and neg_embs is not None:
            neg_idx = neg_qids.index(i)
            s_neg = torch.matmul(neg_embs[neg_idx].unsqueeze(0).to('cuda'), doc_emb_selected.T).squeeze(0)
            S_neg_topk[i, :k] = s_neg.float().cpu()

        del doc_emb_selected

    torch.cuda.empty_cache()

    # ============================================================
    # Grid search over lambda_boundary and tau_decay
    # ============================================================
    lambda_list = [0.5, 1.0, 1.5, 2.0]
    tau_list = [0.1, 0.2, 0.5, 1.0]
    total_trials = len(lambda_list) * len(tau_list)

    # Create a TRACEEvaluator instance for trace_score_query
    trace_evaluator = TRACEEvaluator(
        model_name=args.model_name,
        task_name=f"beir_{args.dataset}",
        output_dir=args.output_dir or f"results/beir_trace_original/{args.dataset}",
        dual_queries_path=args.dual_queries_path,
        device=args.device,
    )

    best_metrics = None
    best_params = None
    all_results = []

    logger.info(f"Grid search: {total_trials} combinations (lambda x tau)")
    trial_idx = 0

    for lam in lambda_list:
        for tau_d in tau_list:
            trial_idx += 1
            trace_evaluator.lambda_boundary = lam
            trace_evaluator.tau_decay = tau_d

            # Apply TRACE scoring per query
            run_trace = {}
            for i, qid in enumerate(eval_qids):
                valid_mask = topk_indices_base[i] >= 0
                k = valid_mask.sum().item()
                if k == 0:
                    run_trace[qid] = {}
                    continue

                s_f = S_full_topk[i, :k]
                s_p = S_pos_topk[i, :k]
                s_n = S_neg_topk[i, :k]
                has_neg = bool(has_neg_mask[i] > 0)

                result = trace_evaluator.trace_score_query(s_f, s_p, s_n, has_neg)
                s_final = result.s_final

                # Build run entry sorted by s_final descending
                scored_pairs = []
                for j in range(k):
                    idx = topk_indices_base[i, j].item()
                    if idx < 0:
                        continue
                    did = doc_ids[idx]
                    scored_pairs.append((did, s_final[j].item()))
                scored_pairs.sort(key=lambda x: x[1], reverse=True)
                run_trace[qid] = dict(scored_pairs)

            # Evaluate
            evaluator = pytrec_eval.RelevanceEvaluator(
                {qid: qrels[qid] for qid in eval_qids if qid in qrels},
                {'ndcg_cut_10', 'map_cut_100'}
            )
            trace_results = evaluator.evaluate(run_trace)
            trace_metrics = {}
            for m in ['ndcg_cut_10', 'map_cut_100']:
                vals = [trace_results[qid][m] for qid in eval_qids if qid in trace_results]
                trace_metrics[m] = np.mean(vals) if vals else 0.0

            logger.info(
                "[%d/%d] lambda=%.1f, tau=%.2f: nDCG@10=%.4f, MAP@100=%.4f",
                trial_idx, total_trials,
                lam, tau_d,
                trace_metrics['ndcg_cut_10'], trace_metrics['map_cut_100'],
            )

            all_results.append({
                "lambda_boundary": lam,
                "tau_decay": tau_d,
                "ndcg_cut_10": trace_metrics['ndcg_cut_10'],
                "map_cut_100": trace_metrics['map_cut_100'],
            })

            # Select best by nDCG@10 (tie-break by MAP@100)
            if (best_metrics is None
                or trace_metrics['ndcg_cut_10'] > best_metrics['ndcg_cut_10']
                or (trace_metrics['ndcg_cut_10'] == best_metrics['ndcg_cut_10']
                    and trace_metrics['map_cut_100'] > best_metrics['map_cut_100'])):
                best_metrics = trace_metrics
                best_params = {"lambda_boundary": lam, "tau_decay": tau_d}

    # ============================================================
    # Save results
    # ============================================================
    output = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "dual_queries_path": args.dual_queries_path,
        "q_base_mode": "original_query_no_neutral_instruction",
        "top_k": args.top_k,
        "baseline": baseline_metrics,
        "trace_best": {
            "params": best_params,
            "metrics": best_metrics,
        },
        "delta_trace_vs_baseline": {
            "ndcg_cut_10": best_metrics['ndcg_cut_10'] - baseline_metrics['ndcg_cut_10'],
            "map_cut_100": best_metrics['map_cut_100'] - baseline_metrics['map_cut_100'],
        },
        "grid_search_results": all_results,
    }

    output_dir = args.output_dir or f"results/beir_trace_original/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics_summary.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print(f"TRACE on BEIR ({args.dataset}, {args.model_name})")
    print(f"q_base = original query text (no neutral instruction)")
    print("=" * 80)
    print(f"{'Condition':<40} {'nDCG@10':>10} {'MAP@100':>10}")
    print("-" * 80)
    print(f"{'Baseline (original query)':<40} {baseline_metrics['ndcg_cut_10']:>10.4f} {baseline_metrics['map_cut_100']:>10.4f}")
    print(f"{'TRACE (best grid)':<40} {best_metrics['ndcg_cut_10']:>10.4f} {best_metrics['map_cut_100']:>10.4f}")
    print("-" * 80)
    print(f"{'Δ TRACE vs Baseline':<40} "
          f"{best_metrics['ndcg_cut_10']-baseline_metrics['ndcg_cut_10']:>+10.4f} "
          f"{best_metrics['map_cut_100']-baseline_metrics['map_cut_100']:>+10.4f}")
    print(f"\nBest params: lambda_boundary={best_params['lambda_boundary']}, tau_decay={best_params['tau_decay']}")

    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
