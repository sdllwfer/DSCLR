"""
TRACE Secondary Design Choices Ablation Experiment

运行三个消融实验来验证TRIX框架的次要设计选择：
1. OLS fit vs Huber regression
2. Mean/std scaling vs median/MAD scaling
3. Uncentered residual vs centered residual

Usage:
    cd /home/luwa/Documents/DSCLR-remote && \
    /home/luwa/.conda/envs/dsclr/bin/python -m scripts.run_secondary_design_ablation
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_trace_variant(
    model_name: str,
    task_name: str,
    dual_queries_path: str,
    output_dir: str,
    variant_name: str,
    regression_mode: str = "huber",
    normalization_mode: str = "median_mad",
    uncentered_residual: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    运行单个TRACE变体的评测
    """
    from eval.engine_trace import TRACEEvaluator
    from eval.metrics.evaluator import FollowIREvaluator, DataLoader
    import torch
    import torch.nn.functional as F

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating variant: {variant_name}")
    logger.info(f"  regression_mode={regression_mode}")
    logger.info(f"  normalization_mode={normalization_mode}")
    logger.info(f"  uncentered_residual={uncentered_residual}")
    logger.info(f"{'='*80}\n")

    # 创建输出目录
    variant_output_dir = Path(output_dir) / variant_name
    variant_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # 加载FollowIR数据
        data_loader = DataLoader(task_name)
        queries_og, queries_changed = data_loader.load_queries()
        qrels = data_loader.load_qrels()
        changed_qrels = data_loader.load_qrel_diff()
        candidates = data_loader.load_candidates()

        # 加载模型
        from eval.models.encoder import ModelFactory
        model = ModelFactory.load_model(model_name, device=device)

        # 加载dual queries
        dual_queries = {}
        with open(dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                dual_queries[item["qid"]] = item

        # 编码查询
        all_queries_og = [queries_og[qid] for qid in queries_og]
        all_queries_changed = [queries_changed[qid] for qid in queries_changed]

        q_og_embs = model.encode(all_queries_og, batch_size=64, device=device)
        q_changed_embs = model.encode(all_queries_changed, batch_size=64, device=device)

        # 编码dual queries
        qids_dual = list(dual_queries.keys())
        q_base_texts = [dual_queries[qid]["q_base"] for qid in qids_dual]
        q_plus_texts = [dual_queries[qid]["q_plus"] for qid in qids_dual]
        q_minus_texts = [dual_queries[qid].get("q_minus", "") for qid in qids_dual]

        q_base_embs = model.encode(q_base_texts, batch_size=64, device=device)
        q_plus_embs = model.encode(q_plus_texts, batch_size=64, device=device)
        q_minus_embs = model.encode(q_minus_texts, batch_size=64, device=device)

        # 编码文档
        all_docs = []
        doc_id_to_idx = {}
        for qid, doc_ids in candidates.items():
            for did in doc_ids:
                if did not in doc_id_to_idx:
                    doc_id_to_idx[did] = len(all_docs)
                    all_docs.append(did)

        # 这里需要从MTEB加载文档内容，简化处理使用预缓存的嵌入
        # 为了简化，假设已经有缓存的文档嵌入
        # 实际应该从 FollowIR_test/embeddings 加载

        # 初始化TRACEEvaluator
        trace_evaluator = TRACEEvaluator(
            model_name=model_name,
            task_name=task_name,
            output_dir=str(variant_output_dir),
            dual_queries_path=dual_queries_path,
            huber_delta=1.345,
            lambda_boundary=1.0,
            tau_decay=0.2,
            regression_mode=regression_mode,
            normalization_mode=normalization_mode,
            uncentered_residual=uncentered_residual,
            device=device,
        )

        # 为og和changed查询构建结果字典
        results_og = {}
        results_changed = {}

        # 对每个查询计算TRACE得分
        # (这里简化处理，实际需要加载文档嵌入并计算相似度)

        # 使用FollowIREvaluator计算指标
        followir_evaluator = FollowIREvaluator(task_name)
        # metrics = followir_evaluator.evaluate(results_og, results_changed)

        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed_time:.2f}s")

        # 暂时返回模拟结果用于调试
        logger.warning("⚠️ Simplified evaluation - returning placeholder results")
        return {
            "variant": variant_name,
            "config": {
                "regression_mode": regression_mode,
                "normalization_mode": normalization_mode,
                "uncentered_residual": uncentered_residual,
            },
            "results": {
                "changed_MAP@1000": "N/A",
                "changed_nDCG@5": "N/A",
                "p-MRR": "N/A",
            },
            "elapsed_time": elapsed_time,
            "status": "success_placeholder",
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "variant": variant_name,
            "config": {
                "regression_mode": regression_mode,
                "normalization_mode": normalization_mode,
                "uncentered_residual": uncentered_residual,
            },
            "error": str(e),
            "status": "failed",
        }


def main():
    parser = argparse.ArgumentParser(description="TRACE Secondary Design Choices Ablation")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced",
                        help="Model name or path")
    parser.add_argument("--output_base", type=str, default="results/trace_secondary_ablation",
                        help="Base output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--datasets", nargs="+",
                        default=["Core17InstructionRetrieval",
                                 "Robust04InstructionRetrieval",
                                 "News21InstructionRetrieval"],
                        help="Datasets to evaluate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 定义消融实验配置
    ablation_configs = [
        {
            "name": "default",
            "description": "Default TRIX (baseline)",
            "regression_mode": "huber",
            "normalization_mode": "median_mad",
            "uncentered_residual": False,
        },
        {
            "name": "ols_fit",
            "description": "OLS regression instead of Huber",
            "regression_mode": "ols",
            "normalization_mode": "median_mad",
            "uncentered_residual": False,
        },
        {
            "name": "mean_std_scaling",
            "description": "Mean/std scaling instead of median/MAD",
            "regression_mode": "huber",
            "normalization_mode": "mean_std",
            "uncentered_residual": False,
        },
        {
            "name": "uncentered_residual",
            "description": "Skip residual recentering",
            "regression_mode": "huber",
            "normalization_mode": "median_mad",
            "uncentered_residual": True,
        },
    ]

    # 运行所有数据集的实验
    all_results = {}

    for dataset in args.datasets:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"{'#'*80}\n")

        dataset_results = []

        # dual_queries路径
        dual_queries_path = f"dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_{dataset}.jsonl"

        if not os.path.exists(dual_queries_path):
            logger.warning(f"Dual queries file not found: {dual_queries_path}")
            logger.warning(f"Skipping dataset: {dataset}")
            continue

        for config in ablation_configs:
            result = evaluate_trace_variant(
                model_name=args.model_name,
                task_name=dataset,
                dual_queries_path=dual_queries_path,
                output_dir=os.path.join(args.output_base, dataset),
                variant_name=config["name"],
                regression_mode=config["regression_mode"],
                normalization_mode=config["normalization_mode"],
                uncentered_residual=config["uncentered_residual"],
                device=args.device,
            )
            result["description"] = config["description"]
            dataset_results.append(result)

            # 保存中间结果
            result_file = os.path.join(
                args.output_base, dataset, f"{config['name']}_result.json"
            )
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved result to {result_file}")

        all_results[dataset] = dataset_results

    # 生成汇总报告
    logger.info("\n" + "="*80)
    logger.info("ABLATION SUMMARY")
    logger.info("="*80 + "\n")

    summary_data = []
    for dataset, results in all_results.items():
        logger.info(f"\n{dataset}:")
        logger.info("-" * 80)

        for result in results:
            if result["status"] == "success":
                # 从results中提取关键指标
                r = result["results"]
                logger.info(f"\n  {result['variant']} ({result['description']}):")

                # 提取changed_MAP或changed_nDCG和p-MRR
                if "Core17" in dataset or "Robust04" in dataset:
                    metric_key = "changed_MAP@1000"
                else:  # News21
                    metric_key = "changed_nDCG@5"

                metric_val = r.get(metric_key, "N/A")
                pmrr = r.get("p-MRR", "N/A")

                logger.info(f"    {metric_key}: {metric_val}")
                logger.info(f"    p-MRR: {pmrr}")

                summary_data.append({
                    "dataset": dataset,
                    "variant": result["variant"],
                    "metric": metric_val,
                    "p-MRR": pmrr,
                })
            else:
                logger.info(f"\n  {result['variant']}: FAILED - {result.get('error', 'Unknown error')}")

    # 计算Score (target_avg)
    # Score = (Core17_changed_MAP@1000 + Robust04_changed_MAP@1000 + News21_changed_nDCG@5) / 3
    logger.info("\n" + "="*80)
    logger.info("TARGET AVG SCORE SUMMARY")
    logger.info("="*80 + "\n")

    # 按variant分组计算平均分
    from collections import defaultdict
    variant_scores = defaultdict(dict)

    for item in summary_data:
        variant_scores[item["variant"]][item["dataset"]] = {
            "metric": item["metric"],
            "p-MRR": item["p-MRR"],
        }

    for variant, datasets_dict in variant_scores.items():
        # 计算target_avg
        metrics = []
        pmrrs = []

        for dataset in ["Core17InstructionRetrieval",
                        "Robust04InstructionRetrieval",
                        "News21InstructionRetrieval"]:
            if dataset in datasets_dict:
                metric_val = datasets_dict[dataset]["metric"]
                pmrr_val = datasets_dict[dataset]["p-MRR"]

                if isinstance(metric_val, (int, float)):
                    metrics.append(metric_val)
                if isinstance(pmrr_val, (int, float)):
                    pmrrs.append(pmrr_val)

        if len(metrics) == 3:
            target_avg = sum(metrics) / 3
            avg_pmrr = sum(pmrrs) / len(pmrrs) if pmrrs else None

            logger.info(f"{variant}:")
            logger.info(f"  Score (target_avg): {target_avg:.1f}")
            logger.info(f"  Avg p-MRR: {avg_pmrr:.1f}" if avg_pmrr is not None else "  Avg p-MRR: N/A")

    # 保存完整结果
    final_output = os.path.join(args.output_base, "all_ablation_results.json")
    with open(final_output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nFinal results saved to {final_output}")


if __name__ == "__main__":
    main()