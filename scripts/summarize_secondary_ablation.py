"""
汇总TRACE次要设计选择消融实验结果

Usage:
    cd /home/luwa/Documents/DSCLR-remote && \
    /home/luwa/.conda/envs/dsclr/bin/python -m scripts.summarize_secondary_ablation
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
    }


def main():
    base_dir = Path("results/trace_secondary_ablation")

    if not base_dir.exists():
        print(f"❌ Results directory not found: {base_dir}")
        return

    # 定义变体
    variants = {
        "default": "Default TRIX (baseline)",
        "ols_fit": "OLS fit - Replace Huber regression",
        "mean_std_scaling": "Mean/std scaling - Replace median/MAD scaling",
        "uncentered_residual": "Uncentered residual - Omit residual recentering",
    }

    datasets = [
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval",
        "News21InstructionRetrieval"
    ]

    # 收集所有结果
    all_results = {}

    for variant, description in variants.items():
        print(f"\n{'='*80}")
        print(f"Variant: {variant}")
        print(f"Description: {description}")
        print(f"{'='*80}")

        variant_results = {}

        for dataset in datasets:
            result_file = base_dir / variant / dataset / "trace_metrics_summary.json"

            if not result_file.exists():
                print(f"  ⚠️  {dataset}: Results not found")
                continue

            metrics = extract_metrics(str(result_file))

            if metrics is None:
                print(f"  ⚠️  {dataset}: Could not extract metrics")
                continue

            # 提取关键指标
            p_mrr = metrics.get("p-MRR", 0.0)

            if "Core17" in dataset or "Robust04" in dataset:
                main_metric = metrics.get("changed_map", 0.0)
                metric_name = "changed_MAP@1000"
            else:  # News21
                main_metric = metrics.get("changed_ndcg_5", 0.0)
                metric_name = "changed_nDCG@5"

            variant_results[dataset] = {
                "metric": main_metric,
                "metric_name": metric_name,
                "p-MRR": p_mrr,
            }

            print(f"  {dataset}:")
            print(f"    {metric_name}: {main_metric:.2f}")
            print(f"    p-MRR: {p_mrr:+.1f}")

        all_results[variant] = variant_results

    # 计算Score (target_avg)
    print(f"\n{'='*80}")
    print("Summary: Score and p-MRR")
    print(f"{'='*80}\n")

    summary_table = []

    for variant, description in variants.items():
        if variant not in all_results:
            continue

        datasets_dict = all_results[variant]

        # 计算target_avg
        metrics_list = []
        pmrr_list = []

        for dataset in datasets:
            if dataset in datasets_dict:
                metrics_list.append(datasets_dict[dataset]["metric"])
                pmrr_list.append(datasets_dict[dataset]["p-MRR"])

        if len(metrics_list) == 3:
            target_avg = sum(metrics_list) / 3
            avg_pmrr = sum(pmrr_list) / len(pmrr_list) if pmrr_list else 0

            print(f"{variant}:")
            print(f"  Score (target_avg): {target_avg:.1f}")
            print(f"  Avg p-MRR: {avg_pmrr:+.1f}")

            summary_table.append({
                "variant": variant,
                "description": description,
                "score": target_avg,
                "p_mrr": avg_pmrr,
                "datasets": datasets_dict,
            })
        else:
            print(f"{variant}: Incomplete results (missing datasets)")
            print(f"  Available: {list(datasets_dict.keys())}")

    # 生成LaTeX表格行
    print(f"\n{'='*80}")
    print("LaTeX Table Rows")
    print(f"{'='*80}\n")

    for entry in summary_table:
        variant = entry["variant"]
        desc = entry["description"]
        score = entry["score"]
        pmrr = entry["p_mrr"]

        # 生成LaTeX表格行
        if variant == "default":
            latex_row = f"\\method{{}} & None & {score:.1f} & $+{pmrr:.1f}$ \\\\"
        else:
            variant_label = variant.replace("_", " ").title()
            latex_row = f"{variant_label} & {desc.split(' - ')[1] if ' - ' in desc else desc} & {score:.1f} & $+{pmrr:.1f}$ \\\\"

        print(latex_row)

    # 保存汇总结果
    output_file = base_dir / "ablation_summary.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary_table": summary_table,
            "all_results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Summary saved to: {output_file}")


if __name__ == "__main__":
    main()