"""
提取已有的次要设计选择消融实验结果

结果位置：/home/luwa/Documents/DSCLR/evaluation_remote/ablation_diagnostic_trace/
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
    base_dir = Path("/home/luwa/Documents/DSCLR/evaluation_remote/ablation_diagnostic_trace")

    if not base_dir.exists():
        print(f"❌ Results directory not found: {base_dir}")
        return

    # 定义变体及其描述
    variants = {
        "trace_baseline": "Default TRIX (baseline)",
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

    print("="*80)
    print("Extracting Secondary Design Choices Ablation Results")
    print("="*80)

    for variant, description in variants.items():
        print(f"\n{'-'*80}")
        print(f"Variant: {variant}")
        print(f"Description: {description}")
        print(f"{'-'*80}")

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

            # 打印指标（百分比形式）
            metric_pct = main_metric * 100
            pmrr_pct = p_mrr * 100
            print(f"  {dataset}:")
            print(f"    {metric_name}: {metric_pct:.1f}")
            print(f"    p-MRR: {pmrr_pct:+.1f}%")

        all_results[variant] = variant_results

    # 计算Score (target_avg)和汇总表格
    print(f"\n{'='*80}")
    print("Summary Table: Score and p-MRR")
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
                # 将百分比转换回小数进行计算
                metric_val = datasets_dict[dataset]["metric"]
                pmrr_val = datasets_dict[dataset]["p-MRR"]
                metrics_list.append(metric_val)
                pmrr_list.append(pmrr_val)

        if len(metrics_list) == 3:
            # target_avg = (Core17_changed_MAP + Robust04_changed_MAP + News21_changed_nDCG@5) / 3
            target_avg = sum(metrics_list) / 3
            avg_pmrr = sum(pmrr_list) / len(pmrr_list) if pmrr_list else 0

            # 转换为百分比显示
            score_pct = target_avg * 100
            pmrr_pct = avg_pmrr * 100

            print(f"{variant}:")
            print(f"  Score (target_avg): {score_pct:.1f}")
            print(f"  Avg p-MRR: {pmrr_pct:+.1f}%")

            summary_table.append({
                "variant": variant,
                "description": description,
                "score": score_pct,
                "p_mrr": pmrr_pct,
                "datasets": datasets_dict,
            })
        else:
            print(f"{variant}: Incomplete results (missing datasets)")
            print(f"  Available: {list(datasets_dict.keys())}")

    # 生成LaTeX表格行
    print(f"\n{'='*80}")
    print("LaTeX Table Rows (for paper_trace_aaai27.tex)")
    print(f"{'='*80}\n")

    for entry in summary_table:
        variant = entry["variant"]
        desc = entry["description"]
        score = entry["score"]
        pmrr = entry["p_mrr"]

        # 生成LaTeX表格行
        if variant == "trace_baseline":
            latex_row = f"\\method{{}} & None & {score:.1f} & ${pmrr:+.1f}$ \\\\"
        else:
            # 从描述中提取改变
            if "Replace" in desc:
                change = desc.split("Replace ")[1]
            elif "Omit" in desc:
                change = "Omit residual recentering"
            else:
                change = desc

            # 格式化p-MRR
            pmrr_str = f"${pmrr:+.1f}$"
            latex_row = f"{variant.replace('_', ' ').title()} & {change} & {score:.1f} & {pmrr_str} \\\\"

        print(latex_row)

    # 保存汇总结果
    output_file = Path("results/trace_secondary_ablation_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "summary_table": summary_table,
            "all_results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Summary saved to: {output_file}")

    # 打印完整的表格数据
    print(f"\n{'='*80}")
    print("Complete Table Data")
    print(f"{'='*80}\n")

    print("| Variant | Change from TRIX | Score | p-MRR |")
    print("|---------|------------------|-------|-------|")
    for entry in summary_table:
        variant = entry["variant"]
        score = entry["score"]
        pmrr = entry["p_mrr"]

        if variant == "trace_baseline":
            change = "None"
        else:
            change = variants[variant].split(" - ")[1] if " - " in variants[variant] else variants[variant]

        print(f"| {variant.replace('_', ' ').title()} | {change} | {score:.1f} | {pmrr:+.1f}% |")


if __name__ == "__main__":
    main()