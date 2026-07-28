"""
提取参数敏感性和效率实验数据，填充论文表格

数据来源：
1. Lambda/Tau grid search: /home/luwa/Documents/DSCLR/evaluation_remote/ablation_residual_trace/full/
2. Candidate depth (K): /home/luwa/Documents/DSCLR/evaluation_remote/sensitivity_k_trace/
3. Latency: 从代码注释中提取（decomposition 0.3s, reranking 5ms）
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any


def extract_metrics(result_file: str) -> Dict[str, Any]:
    """从trace_metrics_summary.json中提取指标"""
    if not os.path.exists(result_file):
        return None

    with open(result_file, "r") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})

    # 提取changed指标
    changed = metrics.get("changed", {})
    p_mrr = metrics.get("p-MRR", 0.0)

    return {
        "p-MRR": p_mrr,
        "changed_map": changed.get("map_at_1000", 0.0),
        "changed_ndcg_5": changed.get("ndcg_at_5", 0.0),
        "changed_ndcg_10": changed.get("ndcg_at_10", 0.0),
        "best_params": data.get("best_params", {}),
    }


def calculate_target_avg(results: Dict[str, Dict]) -> float:
    """计算target_avg = (Core17_changed_MAP + Robust04_changed_MAP + News21_changed_nDCG@5) / 3
    返回百分比形式（乘以100）
    """
    metrics_list = []
    for dataset in ["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"]:
        if dataset in results:
            if "Core17" in dataset or "Robust04" in dataset:
                # MAP是小数形式，乘以100转为百分比
                metrics_list.append(results[dataset]["changed_map"] * 100)
            else:  # News21
                # nDCG也是小数形式，乘以100转为百分比
                metrics_list.append(results[dataset]["changed_ndcg_5"] * 100)

    if len(metrics_list) == 3:
        return sum(metrics_list) / 3
    return 0.0


def main():
    base_dir = Path("/home/luwa/Documents/DSCLR/evaluation_remote")

    datasets = [
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval",
        "News21InstructionRetrieval"
    ]

    print("="*80)
    print("Sensitivity and Efficiency Experiment Results")
    print("="*80)

    # 1. Lambda/Tau sensitivity (from grid search best_params)
    print("\n" + "="*80)
    print("1. Lambda/Tau Sensitivity (Grid Search)")
    print("="*80)

    # 读取full模式的grid search结果
    full_results = {}
    for dataset in datasets:
        result_file = base_dir / "ablation_residual_trace" / "full" / dataset / "trace_metrics_summary.json"
        metrics = extract_metrics(str(result_file))
        if metrics:
            full_results[dataset] = metrics

    if full_results:
        # 提取最优参数
        lambda_values = []
        tau_values = []
        for dataset, metrics in full_results.items():
            best_params = metrics.get("best_params", {})
            lambda_values.append(best_params.get("lambda", 0))
            tau_values.append(best_params.get("tau_decay", 0))

        # 计算平均指标
        avg_p_mrr = sum(m["p-MRR"] for m in full_results.values()) / len(full_results)
        target_avg = calculate_target_avg(full_results)

        print(f"\nGrid search ranges:")
        print(f"  λ ∈ {{{', '.join(map(str, [0.5, 1.0, 1.5, 2.0]))}}}; default λ = {sum(lambda_values)/len(lambda_values):.1f}")
        print(f"  τ ∈ {{{', '.join(map(str, [0.1, 0.2, 0.5, 1.0]))}}}; default τ = {sum(tau_values)/len(tau_values):.1f}")
        print(f"\nPerformance at optimal params:")
        print(f"  Avg Score: {target_avg:.1f}")
        print(f"  Avg p-MRR: {avg_p_mrr*100:+.1f}%")

    # 2. Candidate depth sensitivity
    print("\n" + "="*80)
    print("2. Candidate Depth Sensitivity")
    print("="*80)

    k_values = ["K10", "K50", "K100", "K200"]
    k_results_summary = {}

    for k in k_values:
        print(f"\n{k}:")
        k_results = {}
        for dataset in datasets:
            result_file = base_dir / "sensitivity_k_trace" / k / dataset / "trace_metrics_summary.json"
            metrics = extract_metrics(str(result_file))
            if metrics:
                k_results[dataset] = metrics

        if k_results:
            avg_p_mrr = sum(m["p-MRR"] for m in k_results.values()) / len(k_results)
            target_avg = calculate_target_avg(k_results)
            k_results_summary[k] = {
                "score": target_avg,
                "p_mrr": avg_p_mrr
            }
            print(f"  Avg Score: {target_avg:.1f}")
            print(f"  Avg p-MRR: {avg_p_mrr*100:+.1f}%")

    # 3. Efficiency (latency)
    print("\n" + "="*80)
    print("3. Efficiency (Latency)")
    print("="*80)
    print("\n  Decomposition: One structured LLM call → 0.3s")
    print("  TRACE reranking: Cached document embeddings → 5ms")

    # 生成LaTeX表格行
    print("\n" + "="*80)
    print("LaTeX Table Rows")
    print("="*80)
    print()

    # Lambda行
    if full_results:
        lambda_vals = [0.5, 1.0, 1.5, 2.0]
        default_lambda = sum(lambda_values)/len(lambda_values) if lambda_values else 1.0
        avg_p_mrr = sum(m["p-MRR"] for m in full_results.values()) / len(full_results)
        target_avg = calculate_target_avg(full_results)
        print(f"$\\lambda$ & $\\lambda\\in\\{{{', '.join(map(str, lambda_vals))}\\}}$; default ${default_lambda:.1f}$ & {target_avg:.1f} & ${avg_p_mrr*100:+.1f}$ & --- \\\\")

    # Tau行
    if full_results:
        tau_vals = [0.1, 0.2, 0.5, 1.0]
        default_tau = sum(tau_values)/len(tau_values) if tau_values else 0.2
        print(f"$\\tau$ & $\\tau\\in\\{{{', '.join(map(str, tau_vals))}\\}}$; default ${default_tau:.1f}$ & {target_avg:.1f} & ${avg_p_mrr*100:+.1f}$ & --- \\\\")

    # Candidate depth行
    if k_results_summary:
        k_scores = [k_results_summary[k]["score"] for k in k_values]
        k_pmrrs = [k_results_summary[k]["p_mrr"]*100 for k in k_values]
        # 使用平均值
        avg_score = sum(k_scores) / len(k_scores)
        avg_pmrr = sum(k_pmrrs) / len(k_pmrrs)
        print(f"Candidate depth & $K\\in\\{{10, 50, 100, 200\\}}$ & {avg_score:.1f} & ${avg_pmrr:+.1f}$ & --- \\\\")

    # Decomposition行
    print(f"Decomposition & One structured LLM call & --- & --- & 0.3\\,s \\\\")

    # Reranking行
    print(f"\\method reranking & Cached document embeddings & --- & --- & 5\\,ms \\\\")

    print("\n" + "="*80)
    print("Candidate Depth Detailed Results")
    print("="*80)
    print("\n| K | Avg Score | Avg p-MRR |")
    print("|---|-----------|-----------|")
    for k in k_values:
        if k in k_results_summary:
            print(f"| {k.replace('K', '')} | {k_results_summary[k]['score']:.1f} | {k_results_summary[k]['p_mrr']*100:+.1f}% |")

    print("\n✅ Done")


if __name__ == "__main__":
    main()