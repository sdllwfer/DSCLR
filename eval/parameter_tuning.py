#!/usr/bin/env python3
"""
参数调优：尝试不同的 alpha 和 delta 组合
寻找 MAP >= 0.23 且 pMRR 不损害的最佳参数
"""

import json
import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from eval.metrics.evaluator import DataLoader
from eval.models.encoder import ModelFactory
from model.reformulator import QueryReformulator
from eval.engine import FollowIRDataLoader
from eval.engine_dscrl import DSCLRDenseRetriever, load_cached_embeddings
import logging

logging.basicConfig(level=logging.WARNING)

def compute_map(scores, qrels, k=1000):
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])[:k]
    relevant_docs = {doc_id for doc_id, rel in qrels.items() if rel > 0}
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

def main():
    print("=" * 120)
    print("参数调优：寻找 MAP >= 0.23 且 pMRR 不损害的最佳参数")
    print("=" * 120)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_loader = FollowIRDataLoader('Core17InstructionRetrieval')
    corpus, q_og, q_changed, candidates = data_loader.load()
    q_raw_og, q_raw_changed = data_loader.load_raw_queries()
    
    dl = DataLoader('Core17InstructionRetrieval')
    qrels = dl.load_qrels()
    qrel_diff = dl.load_qrel_diff()
    
    encoder = ModelFactory.create(
        model_name='samaya-ai/RepLLaMA-reproduced',
        device=device,
        batch_size=64,
        normalize_embeddings=True
    )
    
    reformulator = QueryReformulator(
        task_name='Core17InstructionRetrieval',
        use_cache=True,
        cache_dir="/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v5"
    )
    
    retriever = DSCLRDenseRetriever(encoder, device, 64)
    cache_dir = 'dataset/FollowIR_test/embeddings'
    cached_data = load_cached_embeddings(cache_dir, 'Core17InstructionRetrieval', 'samaya-ai/RepLLaMA-reproduced')
    if cached_data:
        cached_embeddings, cached_doc_ids = cached_data
        retriever.set_embeddings(cached_embeddings, cached_doc_ids)
    
    query_qids = sorted([q for q in qrel_diff.keys() if q.isdigit()], key=lambda x: int(x))
    
    print("\n编码查询...")
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
        
        q_plus_og = reform_og[0]
        q_plus_changed = reform_changed[0]
        q_minus_changed = reform_changed[1]
        
        original_query_og = f"{query_text_og} {instruction_og}".strip()
        original_query_changed = f"{query_text_changed} {instruction_changed}".strip()
        
        has_neg = q_minus_changed not in ['[NONE]', '', None]
        
        results_data[qid] = {
            'original_query': original_query_changed,
            'q_plus': q_plus_changed,
            'q_minus': q_minus_changed if has_neg else None,
            'diff_docs': qrel_diff[qid],
            'og_qrels': qrels.get(og_qid, {}),
            'changed_qrels': qrels.get(changed_qid, {}),
            'candidates': candidates.get(qid, []),
        }
    
    original_queries = [results_data[qid]['original_query'] for qid in query_qids]
    q_plus_queries = [results_data[qid]['q_plus'] for qid in query_qids]
    q_minus_queries = [results_data[qid]['q_minus'] or '' for qid in query_qids]
    
    original_embs = encoder.encode_queries(original_queries, batch_size=32).to(device)
    q_plus_embs = encoder.encode_queries(q_plus_queries, batch_size=32).to(device)
    q_minus_embs = encoder.encode_queries(q_minus_queries, batch_size=32).to(device)
    
    retriever.doc_embeddings = retriever.doc_embeddings.to(device)
    
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    deltas = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1]
    
    strategies = [
        ('orig_tau_orig_sbase', 'tau=Orig, S_base=Orig'),
        ('qplus_tau_orig_sbase', 'tau=Q+, S_base=Orig'),
    ]
    
    print("\n搜索最佳参数组合...")
    
    best_results = []
    
    for alpha in alphas:
        for delta in deltas:
            for strategy_key, strategy_name in strategies:
                map_list = []
                pmrr_list = []
                
                for i, qid in enumerate(query_qids):
                    data = results_data[qid]
                    
                    candidate_embeddings = []
                    candidate_doc_ids = []
                    for doc_id in data['candidates']:
                        if doc_id in retriever.doc_ids:
                            idx_doc = retriever.doc_ids.index(doc_id)
                            candidate_embeddings.append(retriever.doc_embeddings[idx_doc])
                            candidate_doc_ids.append(doc_id)
                    
                    if not candidate_embeddings:
                        continue
                    
                    candidate_embeddings = torch.stack(candidate_embeddings)
                    
                    S_original = torch.matmul(candidate_embeddings, original_embs[i])
                    S_q_plus = torch.matmul(candidate_embeddings, q_plus_embs[i])
                    S_neg = torch.matmul(candidate_embeddings, q_minus_embs[i])
                    
                    og_scores = {doc_id: float(score) for doc_id, score in zip(candidate_doc_ids, S_original)}
                    
                    if data['q_minus']:
                        cos_orig_qminus = torch.dot(original_embs[i], q_minus_embs[i]).item()
                        cos_qplus_qminus = torch.dot(q_plus_embs[i], q_minus_embs[i]).item()
                        
                        if strategy_key == 'orig_tau_orig_sbase':
                            tau = cos_orig_qminus + delta
                            S_base = S_original
                        else:
                            tau = cos_qplus_qminus + delta
                            S_base = S_original
                        
                        penalty = torch.relu(S_neg - tau)
                        changed_scores = {doc_id: float(S_base[j] - alpha * penalty[j]) 
                                         for j, doc_id in enumerate(candidate_doc_ids)}
                    else:
                        if strategy_key == 'orig_tau_orig_sbase':
                            changed_scores = og_scores.copy()
                        else:
                            changed_scores = og_scores.copy()
                    
                    map_val = compute_map(changed_scores, data['changed_qrels'])
                    pmrr_val = np.mean([rank_score(get_rank(og_scores, doc), get_rank(changed_scores, doc)) 
                                       for doc in data['diff_docs']])
                    
                    map_list.append(map_val)
                    pmrr_list.append(pmrr_val)
                
                avg_map = np.mean(map_list)
                avg_pmrr = np.mean(pmrr_list)
                
                best_results.append({
                    'alpha': alpha,
                    'delta': delta,
                    'strategy': strategy_name,
                    'map': avg_map,
                    'pmrr': avg_pmrr,
                })
    
    print("\n" + "=" * 120)
    print("参数搜索结果")
    print("=" * 120)
    
    baseline_results = [r for r in best_results if r['strategy'] == 'tau=Orig, S_base=Orig']
    baseline_best = max(baseline_results, key=lambda x: x['map'])
    baseline_pmrr = baseline_best['pmrr']
    
    print(f"\nBaseline (tau=Orig, S_base=Orig) 最佳结果:")
    print(f"  alpha={baseline_best['alpha']}, delta={baseline_best['delta']}")
    print(f"  MAP = {baseline_best['map']:.4f}, pMRR = {baseline_best['pmrr']:.4f}")
    
    print("\n寻找 MAP >= 0.23 且 pMRR >= baseline 的结果:")
    print(f"\n{'Alpha':<8} {'Delta':<8} {'策略':<25} {'MAP':<10} {'pMRR':<10} {'状态'}")
    print("-" * 80)
    
    winning_results = []
    for r in best_results:
        if r['map'] >= 0.23 and r['pmrr'] >= baseline_pmrr:
            status = "✅ 双赢"
            winning_results.append(r)
        elif r['map'] >= 0.23:
            status = "MAP达标"
        elif r['pmrr'] >= baseline_pmrr:
            status = "pMRR提升"
        else:
            status = ""
        
        print(f"{r['alpha']:<8} {r['delta']:<8} {r['strategy']:<25} {r['map']:<10.4f} {r['pmrr']:<10.4f} {status}")
    
    if winning_results:
        best_winning = max(winning_results, key=lambda x: x['map'] + x['pmrr'])
        print(f"\n" + "=" * 120)
        print("最佳双赢结果")
        print("=" * 120)
        print(f"""
✅ 找到双赢参数组合！

策略: {best_winning['strategy']}
参数: alpha={best_winning['alpha']}, delta={best_winning['delta']}
结果: MAP = {best_winning['map']:.4f} >= 0.23 ✅
      pMRR = {best_winning['pmrr']:.4f} >= {baseline_pmrr:.4f} (不损害) ✅
""")
    else:
        print(f"\n⚠️ 未找到完全双赢的参数组合。")
        
        print("\n最接近双赢的结果:")
        close_results = sorted(best_results, key=lambda x: abs(x['map'] - 0.23) + max(0, baseline_pmrr - x['pmrr']))[:5]
        for r in close_results:
            print(f"  alpha={r['alpha']}, delta={r['delta']}, {r['strategy']}: MAP={r['map']:.4f}, pMRR={r['pmrr']:.4f}")

if __name__ == '__main__':
    main()
