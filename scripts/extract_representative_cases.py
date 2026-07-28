"""
从TRACE实验结果中提取代表性案例

案例选择标准：
- Success: p-MRR正向、ranking变化明显、TRACE机制有效
- Failure: 语义纠缠导致的失败案例

数据来源：
- /home/luwa/Documents/DSCLR/evaluation_remote/ablation_residual_trace/full/
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any


def load_per_query_stats(result_file: str) -> Dict:
    """加载per-query统计数据"""
    if not os.path.exists(result_file):
        return None

    with open(result_file, "r") as f:
        data = json.load(f)

    return data


def extract_case_info(per_query_stats: Dict, query_id: str) -> Dict:
    """提取单个query的详细信息"""
    if not per_query_stats or "per_query_stats" not in per_query_stats:
        return None

    for query_stat in per_query_stats["per_query_stats"]:
        if query_stat.get("query_id") == query_id:
            return query_stat

    return None


def main():
    base_dir = Path("/home/luwa/Documents/DSCLR/evaluation_remote/ablation_residual_trace/full")

    # 读取Core17的per-query数据
    result_file = base_dir / "Core17InstructionRetrieval" / "trace_per_query_stats.json"
    per_query_data = load_per_query_stats(str(result_file))

    if not per_query_data:
        print("❌ Failed to load per-query data")
        return

    print("="*80)
    print("Representative TRIX Cases")
    print("="*80)
    print()

    # 查找Success案例：p-MRR正向，ranking变化明显
    success_candidates = []
    failure_candidates = []

    # per_query_data可能是list或dict
    if isinstance(per_query_data, list):
        query_stats_list = per_query_data
    elif isinstance(per_query_data, dict) and "per_query_stats" in per_query_data:
        query_stats_list = per_query_data["per_query_stats"]
    else:
        print("❌ Unexpected data format")
        return

    for query_stat in query_stats_list:
        query_id = query_stat.get("query_id", "")

        # 计算MRR变化
        mrr_og = query_stat.get("mrr_og", 0.0)
        mrr_changed = query_stat.get("mrr_changed", 0.0)
        mrr_diff = mrr_changed - mrr_og

        # 提取关键信息
        case_info = {
            "query_id": query_id,
            "mrr_og": mrr_og,
            "mrr_changed": mrr_changed,
            "mrr_diff": mrr_diff,
            "changed_map": query_stat.get("changed_map", 0.0),
            "z_full_mean": query_stat.get("z_full_mean", 0.0),
            "z_pos_mean": query_stat.get("z_pos_mean", 0.0),
            "z_neg_mean": query_stat.get("z_neg_mean", 0.0),
            "residual_mean": query_stat.get("residual_mean", 0.0),
        }

        # Success: MRR提升>0.1，changed_MAP>0.5
        if mrr_diff > 0.1 and case_info["changed_map"] > 0.5:
            success_candidates.append(case_info)

        # Failure: MRR下降>0.1，或语义纠缠（z_neg接近z_pos）
        if mrr_diff < -0.1 or (abs(case_info["z_neg_mean"] - case_info["z_pos_mean"]) < 0.1):
            failure_candidates.append(case_info)

    # 选择最佳案例
    if success_candidates:
        # 选择MRR提升最大的案例
        success_case = max(success_candidates, key=lambda x: x["mrr_diff"])
        print("✅ Success Case:")
        print(f"  Query ID: {success_case['query_id']}")
        print(f"  MRR change: {success_case['mrr_og']:.4f} → {success_case['mrr_changed']:.4f} (Δ={success_case['mrr_diff']:+.4f})")
        print(f"  Changed MAP: {success_case['changed_map']:.4f}")
        print(f"  z_full mean: {success_case['z_full_mean']:.4f}")
        print(f"  z_pos mean: {success_case['z_pos_mean']:.4f}")
        print(f"  z_neg mean: {success_case['z_neg_mean']:.4f}")
        print(f"  residual mean: {success_case['residual_mean']:.4f}")
    else:
        print("⚠️  No success case found")

    print()

    if failure_candidates:
        # 选择MRR下降最大的案例
        failure_case = min(failure_candidates, key=lambda x: x["mrr_diff"])
        print("❌ Failure Case:")
        print(f"  Query ID: {failure_case['query_id']}")
        print(f"  MRR change: {failure_case['mrr_og']:.4f} → {failure_case['mrr_changed']:.4f} (Δ={failure_case['mrr_diff']:+.4f})")
        print(f"  Changed MAP: {failure_case['changed_map']:.4f}")
        print(f"  z_full mean: {failure_case['z_full_mean']:.4f}")
        print(f"  z_pos mean: {failure_case['z_pos_mean']:.4f}")
        print(f"  z_neg mean: {failure_case['z_neg_mean']:.4f}")
        print(f"  residual mean: {failure_case['residual_mean']:.4f}")
    else:
        print("⚠️  No failure case found")

    # 生成LaTeX表格内容（简化版）
    print()
    print("="*80)
    print("LaTeX Table Content (Simplified)")
    print("="*80)
    print()

    if success_candidates:
        success_case = max(success_candidates, key=lambda x: x["mrr_diff"])
        print("Success &")
        print(f"  Query: [Core17 {success_case['query_id']}] &")
        print(f"  z_full={success_case['z_full_mean']:.3f}, z_pos={success_case['z_pos_mean']:.3f}, z_neg={success_case['z_neg_mean']:.3f} &")
        print(f"  MRR: {success_case['mrr_og']:.3f}→{success_case['mrr_changed']:.3f} (+{success_case['mrr_diff']*100:.1f}\\%) \\\\")

    print()

    if failure_candidates:
        failure_case = min(failure_candidates, key=lambda x: x["mrr_diff"])
        print("Failure &")
        print(f"  Query: [Core17 {failure_case['query_id']}] &")
        print(f"  z_full={failure_case['z_full_mean']:.3f}, z_pos={failure_case['z_pos_mean']:.3f}, z_neg={failure_case['z_neg_mean']:.3f} &")
        print(f"  MRR: {failure_case['mrr_og']:.3f}→{failure_case['mrr_changed']:.3f} ({failure_case['mrr_diff']*100:.1f}\\%) \\\\")

    print("\n⚠️  注意：完整案例需要提取query text, instruction, dual_queries等详细信息")


if __name__ == "__main__":
    main()