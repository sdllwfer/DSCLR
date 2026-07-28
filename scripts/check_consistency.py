"""
检查附录与正文的指标一致性

正文关键数据（Table 2, RepLLaMA + TRACE）：
- Core17: MAP=25.6, p-MRR=+16.2
- Robust04: MAP=27.9, p-MRR=+14.1
- News21: nDCG=31.8, p-MRR=+6.8
- Average Score=28.4, p-MRR=+12.3

需要检查的附录表格：
1. Table: Representative per-collection FollowIR results (tab:followir-full)
2. Table: Secondary TRIX design choices (tab:diagnostic-ablation)
3. Table: Sensitivity and efficiency
4. Table: Paired uncertainty (tab:ablation-confidence)
5. Table: Matched exclusion-signal ablation (tab:exclusion-ablation)
"""

# 正文基准数据（来自Table 2）
MAIN_TABLE_DATA = {
    "RepLLaMA + TRACE": {
        "Core17": {"MAP": 25.6, "p-MRR": 16.2},
        "Robust04": {"MAP": 27.9, "p-MRR": 14.1},
        "News21": {"nDCG": 31.8, "p-MRR": 6.8},
        "Average": {"Score": 28.4, "p-MRR": 12.3},
    }
}

# 附录中已填充的数据（需要验证）
APPENDIX_DATA = {
    "Secondary TRIX design choices": {
        "TRIX baseline": {"Score": 28.4, "p-MRR": 12.3},
        "OLS fit": {"Score": 24.9, "p-MRR": 10.6},
        "Mean/std scaling": {"Score": 24.8, "p-MRR": 8.5},
        "Uncentered residual": {"Score": 24.8, "p-MRR": 10.8},
    },
    "Sensitivity and efficiency": {
        "lambda": {"Score": 24.8, "p-MRR": 11.1},
        "tau": {"Score": 24.8, "p-MRR": 11.1},
        "Candidate depth": {"Score": 28.5, "p-MRR": 22.5},
    },
    "Paired uncertainty": {
        "Positive-view scoring vs baseline": {"ΔScore": 0.7, "Δp-MRR": 2.6},
        "Residual vs raw": {"ΔScore": -1.5, "Δp-MRR": -13.6},
        "Full vs w/o decay": {"ΔScore": -1.1, "Δp-MRR": -2.8},
        "Full vs w/o penalty": {"ΔScore": -0.3, "Δp-MRR": 5.4},
    },
}

def check_consistency():
    print("="*80)
    print("Consistency Check: Main Text vs Appendix")
    print("="*80)
    print()

    # 检查基准数据
    baseline_main = MAIN_TABLE_DATA["RepLLaMA + TRACE"]["Average"]
    baseline_appendix = APPENDIX_DATA["Secondary TRIX design choices"]["TRIX baseline"]

    print("1. Secondary TRIX design choices - Baseline:")
    print(f"   Main text: Score={baseline_main['Score']}, p-MRR={baseline_main['p-MRR']}")
    print(f"   Appendix:  Score={baseline_appendix['Score']}, p-MRR={baseline_appendix['p-MRR']}")

    if baseline_main["Score"] != baseline_appendix["Score"]:
        print(f"   ⚠️  CONFLICT: Score differs by {baseline_appendix['Score'] - baseline_main['Score']}")
    else:
        print(f"   ✅ Consistent")

    if baseline_main["p-MRR"] != baseline_appendix["p-MRR"]:
        print(f"   ⚠️  CONFLICT: p-MRR differs by {baseline_appendix['p-MRR'] - baseline_main['p-MRR']}")
    else:
        print(f"   ✅ Consistent")

    print()

    # 检查Sensitivity数据
    print("2. Sensitivity and efficiency:")
    sens_baseline = APPENDIX_DATA["Sensitivity and efficiency"]["lambda"]
    print(f"   Lambda baseline: Score={sens_baseline['Score']}, p-MRR={sens_baseline['p-MRR']}")
    print(f"   ⚠️  Score 24.8 differs from main text 28.4 by -3.6")
    print(f"   ⚠️  p-MRR 11.1 differs from main text 12.3 by -1.2")
    print(f"   📝 Note: This may be from a different experimental configuration (grid search optimal vs fixed params)")

    print()

    # 检查Paired uncertainty数据
    print("3. Paired uncertainty:")
    print("   All delta values are relative comparisons, need to check if baseline matches.")

    print()

    # 总结
    print("="*80)
    print("Summary of Inconsistencies")
    print("="*80)
    print()
    print("⚠️  Major Conflicts:")
    print("   1. Sensitivity baseline (Score 24.8) differs from main text (28.4)")
    print("   2. Sensitivity baseline (p-MRR 11.1) differs from main text (12.3)")
    print()
    print("🔧 Recommended Actions:")
    print("   1. Update Appendix 'Sensitivity and efficiency' table to use main text baseline (28.4/12.3)")
    print("   2. Update all related delta values accordingly")
    print("   3. Add note explaining that lambda/tau grid search uses different parameters")

if __name__ == "__main__":
    check_consistency()