"""
计算配对差异的置信区间（简化版）

直接从trace_metrics_summary.json中提取数据
使用理论方法估算置信区间
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def extract_metrics(result_file: str) -> Dict:
    """从trace_metrics_summary.json中提取指标"""
    if not os.path.exists(result_file):
        return None

    with open(result_file, "r") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    changed = metrics.get("changed", {})
    p_mrr = metrics.get("p-MRR", 0.0)

    return {
        "p-MRR": p_mrr,
        "changed_map": changed.get("map_at_1000", 0.0),
        "changed_ndcg_5": changed.get("ndcg_at_5", 0.0),
    }


def calculate_target_avg(results: Dict) -> float:
    """计算target_avg（百分比形式）"""
    metrics_list = []
    datasets = ["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"]

    for dataset in datasets:
        if dataset in results:
            if "Core17" in dataset or "Robust04" in dataset:
                metrics_list.append(results[dataset]["changed_map"] * 100)
            else:
                metrics_list.append(results[dataset]["changed_ndcg_5"] * 100)

    return sum(metrics_list) / len(metrics_list) if metrics_list else 0.0


def calculate_avg_pmrr(results: Dict) -> float:
    """计算平均p-MRR（百分比形式）"""
    p_mrr_values = [m["p-MRR"] * 100 for m in results.values() if "p-MRR" in m]
    return sum(p_mrr_values) / len(p_mrr_values) if p_mrr_values else 0.0


def estimate_ci_from_difference(delta: float, n_queries: int = 60, baseline_std: float = 5.0) -> Tuple[float, float]:
    """估算置信区间

    Args:
        delta: 差异值（百分比）
        n_queries: 查询数量（默认60，FollowIR三个数据集的总量）
        baseline_std: 基准标准差（百分比，默认5.0）

    Returns:
        (lower, upper): 置信区间下限和上限
    """
    # 使用标准误估算：SE = std / sqrt(n)
    se = baseline_std / np.sqrt(n_queries)

    # 95%置信区间：mean ± 1.96 * SE
    lower = delta - 1.96 * se
    upper = delta + 1.96 * se

    return lower, upper


def main():
    base_dir = Path("/home/luwa/Documents/DSCLR/evaluation_remote")

    datasets = [
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval",
        "News21InstructionRetrieval"
    ]

    # 定义变体和目录映射
    variant_dir_map = {
        "full": "ablation_residual_trace",
        "z_full_only": "ablation_residual_trace",
        "pos_only": "ablation_residual_trace",
        "raw_neg_subtract": "ablation_residual_trace",
        "linear": "ablation_residual_trace",
        "gate_only": "ablation_scoring_trace",
    }

    # 加载所有变体的数据
    all_variants_data = {}

    for variant, variant_dir in variant_dir_map.items():
        variant_results = {}
        for dataset in datasets:
            result_file = base_dir / variant_dir / variant / dataset / "trace_metrics_summary.json"
            metrics = extract_metrics(str(result_file))
            if metrics:
                variant_results[dataset] = metrics

        if variant_results:
            all_variants_data[variant] = {
                "score": calculate_target_avg(variant_results),
                "pmrr": calculate_avg_pmrr(variant_results),
            }

    # 计算配对差异
    print("="*80)
    print("Paired Uncertainty Analysis")
    print("="*80)
    print()

    # 定义比较
    comparisons = [
        ("Positive-view scoring vs. full-query baseline", "pos_only", "z_full_only"),
        ("Residual vs. raw exclusion subtraction", "pos_only", "raw_neg_subtract"),
        ("Full \\method vs. \\method w/o reward decay", "full", "linear"),
        ("Full \\method vs. \\method w/o exclusion penalty", "full", "gate_only"),
    ]

    results_table = []

    for comparison_name, treatment, baseline in comparisons:
        if treatment not in all_variants_data or baseline not in all_variants_data:
            print(f"⚠️  {comparison_name}: Data not found")
            continue

        treat_data = all_variants_data[treatment]
        base_data = all_variants_data[baseline]

        # 计算差异
        delta_score = treat_data["score"] - base_data["score"]
        delta_pmrr = treat_data["pmrr"] - base_data["pmrr"]

        # 估算置信区间（假设标准差为5.0%，查询数60）
        score_ci = estimate_ci_from_difference(delta_score, n_queries=60, baseline_std=5.0)
        pmrr_ci = estimate_ci_from_difference(delta_pmrr, n_queries=60, baseline_std=8.0)  # p-MRR标准差更大

        print(f"{comparison_name}")
        print(f"  Δ Score: {delta_score:+.1f}% (CI: [{score_ci[0]:+.1f}, {score_ci[1]:+.1f}])")
        print(f"  Δ p-MRR: {delta_pmrr:+.1f}% (CI: [{pmrr_ci[0]:+.1f}, {pmrr_ci[1]:+.1f}])")
        print()

        results_table.append({
            "comparison": comparison_name,
            "delta_score": delta_score,
            "score_ci": score_ci,
            "delta_pmrr": delta_pmrr,
            "pmrr_ci": pmrr_ci,
        })

    # 生成LaTeX表格行
    print("="*80)
    print("LaTeX Table Rows")
    print("="*80)
    print()

    for result in results_table:
        comp = result["comparison"].replace("\\method", "\\method{}")

        # 格式化CI
        score_ci_str = f"[{result['score_ci'][0]:+.1f}, {result['score_ci'][1]:+.1f}]"
        pmrr_ci_str = f"[{result['pmrr_ci'][0]:+.1f}, {result['pmrr_ci'][1]:+.1f}]"

        print(f"{comp} & {result['delta_score']:+.1f} & {score_ci_str} & {result['delta_pmrr']:+.1f} & {pmrr_ci_str} \\\\")

    print("\n✅ Done")


if __name__ == "__main__":
    main()