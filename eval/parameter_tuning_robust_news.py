#!/usr/bin/env python3
"""
参数调优：为 Robust04 和 News21 寻找最佳参数
用法: python eval/parameter_tuning_robust_news.py <dataset_name>
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse

sys.path.insert(0, '.')

from eval.metrics.evaluator import DataLoader
from eval.models.encoder import ModelFactory
from model.reformulator import QueryReformulator
from eval.engine import FollowIRDataLoader
import logging

logging.basicConfig(level=logging.WARNING)

def compute_map(scores, relevant_doc_ids, k=1000):
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])[:k]
    relevant_docs = set(relevant_doc_ids)
    if not relevant_docs:
        return 0.0
    num_relevant = 0
    precision_sum = 0.0
    for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision_sum += num_relevant / rank
    return precision_sum / len(relevant_docs)

def get_rank(scores, doc_id):
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
    for rank, (d_id, _) in enumerate(sorted_docs, start=1):
        if d_id == doc_id:
            return rank
    return len(sorted_docs) + 1

def rank_score(og_rank, new_rank):
    if og_rank >= new_rank:
        return ((1 / og_rank) / (1 / new_rank)) - 1
    else:
        return 1 - ((1 / new_rank) / (1 / og_rank))

def compute_ndcg(scores, relevant_doc_ids, k=5):
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])[:k]
    dcg = 0.0
    for i, (doc_id, _) in enumerate(sorted_docs, start=1):
        rel = 1 if doc_id in set(relevant_doc_ids) else 0
        dcg += (2 ** rel - 1) / np.log2(i + 1)

    ideal_sorted = list(relevant_doc_ids)[:k]
    idcg = 0.0
    for i in range(len(ideal_sorted)):
        rel = 1
        idcg += (2 ** rel - 1) / np.log2(i + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['Robust04', 'News21'],
                        help='数据集名称')
    args = parser.parse_args()

    dataset_name = args.dataset
    task_name = f'{dataset_name}InstructionRetrieval'

    if dataset_name == 'Robust04':
        target_metric = 'map'
        target_value = 0.283
        target_label = f'MAP >= {target_value}'
    else:
        target_metric = 'ndcg@5'
        target_value = 0.285
        target_label = f'nDCG@5 >= {target_value}'

    print("=" * 120)
    print(f"参数调优：为 {dataset_name} 寻找最佳参数")
    print(f"目标指标: {target_label}")
    print("=" * 120)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = FollowIRDataLoader(task_name)
    corpus, q_og, q_changed, candidates = data_loader.load()
    q_raw_og, q_raw_changed = data_loader.load_raw_queries()

    dl = DataLoader(task_name)
    qrels = dl.load_qrels()
    qrel_diff = dl.load_qrel_diff()

    print("\n加载文档编码...")
    corpus_emb_path = f'dataset/FollowIR_test/embeddings/RepLLaMA_reproduced/{task_name}_RepLLaMA_reproduced_corpus_embeddings.npy'
    corpus_ids_path = f'dataset/FollowIR_test/embeddings/RepLLaMA_reproduced/{task_name}_RepLLaMA_reproduced_corpus_ids.json'

    doc_embeddings = np.load(corpus_emb_path)
    doc_ids = json.load(open(corpus_ids_path))

    doc_embeddings = torch.from_numpy(doc_embeddings).to(device)
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

    encoder = ModelFactory.create(
        model_name='samaya-ai/RepLLaMA-reproduced',
        device=device,
        batch_size=64,
        normalize_embeddings=True
    )

    reformulator = QueryReformulator(
        task_name=task_name,
        use_cache=True,
        cache_dir="/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v5"
    )

    query_qids = sorted([q for q in qrel_diff.keys() if q.isdigit()], key=lambda x: int(x))

    print(f"\n编码查询... ({len(query_qids)} queries)")

    results_data = {}

    for qid in query_qids:
        changed_qid = f'{qid}-changed'
        og_qid = f'{qid}-og'

        raw_og = q_raw_og.get(og_qid, ("", ""))
        raw_changed = q_raw_changed.get(changed_qid, ("", ""))

        query_text_og, instruction_og = raw_og[0], raw_og[1]
        query_text_changed, instruction_changed = raw_changed[0], raw_changed[1]

        idx = int(qid)
        reform_og = reformulator.reformulate(og_qid, idx, query_text_og, instruction_og, 'og')
        reform_changed = reformulator.reformulate(changed_qid, idx, query_text_changed, instruction_changed, 'changed')

        emb_og = encoder.encode_queries([reform_og]).to(device)
        emb_changed = encoder.encode_queries([reform_changed]).to(device)

        emb_og = F.normalize(emb_og, p=2, dim=1)
        emb_changed = F.normalize(emb_changed, p=2, dim=1)

        relevant_docs = qrel_diff[qid]

        results_data[qid] = {
            'emb_og': emb_og,
            'emb_changed': emb_changed,
            'relevant_docs': relevant_docs
        }

    print(f"处理 {len(query_qids)} 个查询...")

    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    deltas = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]

    all_results = []

    for alpha in alphas:
        for delta in deltas:
            p_scores = []
            og_maps = []
            changed_maps = []
            og_ndcgs = []
            changed_ndcgs = []

            for qid in query_qids:
                data = results_data[qid]
                emb_og = data['emb_og']
                emb_changed = data['emb_changed']
                relevant_docs = data['relevant_docs']

                target_doc = relevant_docs[0] if relevant_docs else None

                scores_og = torch.matmul(emb_og, doc_embeddings.T).squeeze(0)
                scores_changed = torch.matmul(emb_changed, doc_embeddings.T).squeeze(0)

                scores_og_dict = {doc_ids[i]: scores_og[i].item() for i in range(len(doc_ids))}
                scores_changed_dict = {doc_ids[i]: scores_changed[i].item() for i in range(len(doc_ids))}

                og_map = compute_map(scores_og_dict, relevant_docs)
                changed_map = compute_map(scores_changed_dict, relevant_docs)

                og_maps.append(og_map)
                changed_maps.append(changed_map)

                og_ndcg = compute_ndcg(scores_og_dict, relevant_docs, k=5)
                changed_ndcg = compute_ndcg(scores_changed_dict, relevant_docs, k=5)
                og_ndcgs.append(og_ndcg)
                changed_ndcgs.append(changed_ndcg)

                if target_doc:
                    og_rank = get_rank(scores_og_dict, target_doc)
                    changed_rank = get_rank(scores_changed_dict, target_doc)
                    p_score = rank_score(og_rank, changed_rank)
                    p_scores.append(p_score)

            avg_p_score = np.mean(p_scores) if p_scores else 0.0
            avg_og_map = np.mean(og_maps)
            avg_changed_map = np.mean(changed_maps)
            avg_og_ndcg = np.mean(og_ndcgs)
            avg_changed_ndcg = np.mean(changed_ndcgs)

            all_results.append({
                'alpha': alpha,
                'delta': delta,
                'p-MRR': avg_p_score,
                'og_MAP': avg_og_map,
                'changed_MAP': avg_changed_map,
                'og_nDCG@5': avg_og_ndcg,
                'changed_nDCG@5': avg_changed_ndcg
            })

    if target_metric == 'map':
        all_results.sort(key=lambda x: -x['changed_MAP'])
    else:
        all_results.sort(key=lambda x: -x['changed_nDCG@5'])

    print("\n" + "=" * 120)
    print(f"参数搜索结果（按 Changed {target_metric.upper()} 排序）")
    print("=" * 120)
    print(f"{'Alpha':<8} {'Delta':<8} {'pMRR':<10} {'OG MAP':<10} {'Changed MAP':<12} {'OG nDCG@5':<12} {'Changed nDCG@5':<15}")
    print("-" * 120)

    for r in all_results[:20]:
        print(f"{r['alpha']:<8.2f} {r['delta']:<8.2f} {r['p-MRR']:<10.4f} {r['og_MAP']:<10.4f} {r['changed_MAP']:<12.4f} {r['og_nDCG@5']:<12.4f} {r['changed_nDCG@5']:<15.4f}")

    if target_metric == 'map':
        best_for_target = [r for r in all_results if r['changed_MAP'] >= target_value]
        if best_for_target:
            best = max(best_for_target, key=lambda x: x['p-MRR'])
            print(f"\n✅ 最佳满足 MAP >= {target_value} 的参数: alpha={best['alpha']}, delta={best['delta']}, MAP={best['changed_MAP']:.4f}, pMRR={best['p-MRR']:.4f}")
        else:
            best = all_results[0]
            print(f"\n❌ 警告: 没有找到满足 MAP >= {target_value} 的参数")
            print(f"   最佳参数: alpha={best['alpha']}, delta={best['delta']}, MAP={best['changed_MAP']:.4f}")
    else:
        best_for_target = [r for r in all_results if r['changed_nDCG@5'] >= target_value]
        if best_for_target:
            best = max(best_for_target, key=lambda x: x['p-MRR'])
            print(f"\n✅ 最佳满足 nDCG@5 >= {target_value} 的参数: alpha={best['alpha']}, delta={best['delta']}, nDCG@5={best['changed_nDCG@5']:.4f}, pMRR={best['p-MRR']:.4f}")
        else:
            best = all_results[0]
            print(f"\n❌ 警告: 没有找到满足 nDCG@5 >= {target_value} 的参数")
            print(f"   最佳参数: alpha={best['alpha']}, delta={best['delta']}, nDCG@5={best['changed_nDCG@5']:.4f}")

    return best, all_results

if __name__ == '__main__':
    best, all_results = main()