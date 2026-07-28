"""
计算配对差异的置信区间

数据来源：
- z_full_only: Full-query baseline
- full: TRACE full method
- raw_neg_subtract: Raw exclusion subtraction
- pos_only: Positive-only (w/o exclusion penalty)
- linear: Linear decay (w/o reward decay)

使用bootstrap方法计算95%置信区间
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats


def load_per_query_stats(result_file: str) -> Dict[str, Any]:
    """加载per-query统计数据"""
    if not os.path.exists(result_file):
        return None

    with open(result_file, "r") as f:
        data = json.load(f)

    return data


def extract_per_query_metrics(per_query_stats: Dict) -> Dict[str, Dict[str, float]]:
    """从per_query_stats中提取每个query的MAP和MRR

    Returns:
        Dict[query_id, Dict[metric_name, value]]
    """
    if not per_query_stats or "per_query_stats" not in per_query_stats:
        return {}

    metrics_dict = {}
    for query_stat in per_query_stats["per_query_stats"]:
        query_id = query_stat.get("query_id", "")
        if not query_id:
            continue

        # 提取changed MAP和OG/changed MRR
        metrics_dict[query_id] = {
            "changed_map": query_stat.get("changed_map", 0.0),
            "changed_ndcg_5": query_stat.get("changed_ndcg_5", 0.0),
            "mrr_og": query_stat.get("mrr_og", 0.0),
            "mrr_changed": query_stat.get("mrr_changed", 0.0),
        }

    return metrics_dict


def calculate_paired_differences(
    baseline_metrics: Dict[str, Dict[str, float]],
    treatment_metrics: Dict[str, Dict[str, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """计算配对差异

    Returns:
        (score_diffs, pmrr_diffs): Score和p-MRR的差异数组
    """
    query_ids = set(baseline_metrics.keys()) & set(treatment_metrics.keys())

    score_diffs = []
    pmrr_diffs = []

    for qid in query_ids:
        base = baseline_metrics[qid]
        treat = treatment_metrics[qid]

        # 计算Score差异（changed MAP或nDCG@5）
        base_score = base.get("changed_map", 0.0) if "Core" in qid or "Robust" in qid else base.get("changed_ndcg_5", 0.0)
        treat_score = treat.get("changed_map", 0.0) if "Core" in qid or "Robust" in qid else treat.get("changed_ndcg_5", 0.0)
        score_diffs.append(treat_score - base_score)

        # 计算p-MRR差异（MRR@10）
        base_pmrr = base.get("mrr_changed", 0.0) - base.get("mrr_og", 0.0)
        treat_pmrr = treat.get("mrr_changed", 0.0) - treat.get("mrr_og", 0.0)
        pmrr_diffs.append(treat_pmrr - base_pmrr)

    return np.array(score_diffs), np.array(pmrr_diffs)


def bootstrap_ci(diffs: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """使用bootstrap方法计算置信区间

    Args:
        diffs: 差异数组
        n_bootstrap: bootstrap次数
        ci: 置信水平（默认0.95表示95%置信区间）

    Returns:
        (mean_diff, (lower, upper)): 平均差异和置信区间
    """
    if len(diffs) == 0:
        return 0.0, (0.0, 0.0)

    # 计算平均差异
    mean_diff = np.mean(diffs) * 100  # 转为百分比

    # Bootstrap重采样
    bootstrap_means = []
    np.random.seed(42)  # 固定随机种子以保证可重复性
    for _ in range(n_bootstrap):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        bootstrap_means.append(np.mean(sample) * 100)

    # 计算置信区间
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)

    return mean_diff, (lower, upper)


def main():
    base_dir = Path("/home/luwa/Documents/DSCLR/evaluation_remote")

    datasets = [
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval",
        "News21InstructionRetrieval"
    ]

    # 定义需要比较的配对
    comparisons = {
        "Positive-view scoring vs. full-query baseline": {
            "treatment": "pos_only",
            "baseline": "z_full_only",
            "description": "添加positive-view scoring vs 仅使用full-query"
        },
        "Residual vs. raw exclusion subtraction": {
            "treatment": "pos_only",  # 使用residual
            "baseline": "raw_neg_subtract",
            "description": "候选条件残差 vs 原始负向相似度"
        },
        "Full \\method vs. \\method w/o reward decay": {
            "treatment": "full",
            "baseline": "linear",
            "description": "完整TRACE vs 线性衰减（无reward decay）"
        },
        "Full \\method vs. \\method w/o exclusion penalty": {
            "treatment": "full",
            "baseline": "gate_only",
            "description": "完整TRACE vs 仅gate衰减（无penalty）"
        },
    }

    # 检查ablation_residual_trace还是ablation_scoring_trace
    variant_dir_map = {
        "full": "ablation_residual_trace",
        "z_full_only": "ablation_residual_trace",
        "pos_only": "ablation_residual_trace",
        "raw_neg_subtract": "ablation_residual_trace",
        "linear": "ablation_residual_trace",
        "gate_only": "ablation_scoring_trace",
    }

    print("="*80)
    print("Paired Uncertainty Analysis")
    print("="*80)

    results_table = []

    for comparison_name, config in comparisons.items():
        print(f"\n{comparison_name}")
        print(f"  {config['description']}")

        # 收集所有数据集的per-query数据
        all_score_diffs = []
        all_pmrr_diffs = []

        for dataset in datasets:
            treatment_dir = variant_dir_map.get(config["treatment"], "ablation_residual_trace")
            baseline_dir = variant_dir_map.get(config["baseline"], "ablation_residual_trace")

            # 加载treatment数据
            treatment_file = base_dir / treatment_dir / config["treatment"] / dataset / "trace_per_query_stats.json"
            treatment_stats = load_per_query_stats(str(treatment_file))

            # 加载baseline数据
            baseline_file = base_dir / baseline_dir / config["baseline"] / dataset / "trace_per_query_stats.json"
            baseline_stats = load_per_query_stats(str(baseline_file))

            if not treatment_stats or not baseline_stats:
                print(f"  ⚠️  {dataset}: Data not found")
                continue

            # 提取per-query指标
            treatment_metrics = extract_per_query_metrics(treatment_stats)
            baseline_metrics = extract_per_query_metrics(baseline_stats)

            # 计算差异
            score_diffs, pmrr_diffs = calculate_paired_differences(baseline_metrics, treatment_metrics)

            all_score_diffs.extend(score_diffs)
            all_pmrr_diffs.extend(pmrr_diffs)

        if all_score_diffs and all_pmrr_diffs:
            # 计算置信区间
            mean_score, score_ci = bootstrap_ci(np.array(all_score_diffs))
            mean_pmrr, pmrr_ci = bootstrap_ci(np.array(all_pmrr_diffs))

            print(f"  Δ Score: {mean_score:+.1f}% (CI: [{score_ci[0]:+.1f}, {score_ci[1]:+.1f}])")
            print(f"  Δ p-MRR: {mean_pmrr:+.1f}% (CI: [{pmrr_ci[0]:+.1f}, {pmrr_ci[1]:+.1f}])")

            results_table.append({
                "comparison": comparison_name,
                "delta_score": mean_score,
                "score_ci": score_ci,
                "delta_pmrr": mean_pmrr,
                "pmrr_ci": pmrr_ci,
            })

    # 生成LaTeX表格行
    print("\n" + "="*80)
    print("LaTeX Table Rows")
    print("="*80)
    print()

    for result in results_table:
        comp = result["comparison"]
        # 处理LaTeX特殊字符
        comp_latex = comp.replace("\\method", "\\method{}")

        # 格式化CI
        score_ci_str = f"[{result['score_ci'][0]:+.1f}, {result['score_ci'][1]:+.1f}]"
        pmrr_ci_str = f"[{result['pmrr_ci'][0]:+.1f}, {result['pmrr_ci'][1]:+.1f}]"

        print(f"{comp_latex} & {result['delta_score']:+.1f} & {score_ci_str} & {result['delta_pmrr']:+.1f} & {pmrr_ci_str} \\\\")

    print("\n✅ Done")


if __name__ == "__main__":
    main()