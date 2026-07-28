#!/usr/bin/env python3
"""
Score-p-MRR Pareto Experiment for TRIX Paper
Follows the protocol in: paper/模拟评审/TRIX_简单线性基线与Pareto实验方案.md

This script implements:
1. Validation of existing ablation results
2. Parameter grid search for four methods
3. Leave-one-dataset-out parameter selection
4. Pareto frontier generation
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Parameter Spaces (from protocol)
# ============================================================

def get_positive_linear_params() -> List[float]:
    """alpha ∈ {0, 0.5, 1.0, 1.5, 2.0}"""
    return [0, 0.5, 1.0, 1.5, 2.0]


def get_raw_linear_params() -> List[Tuple[float, float]]:
    """(alpha, beta) ∈ {0,0.5,1.0,1.5,2.0} × {0,0.25,0.5,1.0,2.0}"""
    alphas = [0, 0.5, 1.0, 1.5, 2.0]
    betas = [0, 0.25, 0.5, 1.0, 2.0]
    return list(product(alphas, betas))


def get_residual_linear_params() -> List[Tuple[float, float]]:
    """Same as raw linear"""
    return get_raw_linear_params()


def get_trix_params() -> List[Tuple[float, float]]:
    """(lambda, tau) ∈ {0,0.25,0.5,1.0,1.5} × {0.05,0.1,0.2,0.5,1.0}"""
    lambdas = [0, 0.25, 0.5, 1.0, 1.5]
    taus = [0.05, 0.1, 0.2, 0.5, 1.0]
    return list(product(lambdas, taus))


# ============================================================
# Scoring Functions
# ============================================================

def score_positive_linear(z_full: np.ndarray, p: np.ndarray, alpha: float) -> np.ndarray:
    """Score = z_full + alpha * p"""
    return z_full + alpha * p


def score_raw_linear(z_full: np.ndarray, p: np.ndarray, z_neg: np.ndarray, 
                      alpha: float, beta: float) -> np.ndarray:
    """Score = z_full + alpha * p - beta * z_neg"""
    return z_full + alpha * p - beta * z_neg


def score_residual_linear(z_full: np.ndarray, p: np.ndarray, r: np.ndarray,
                           alpha: float, beta: float) -> np.ndarray:
    """Score = z_full + alpha * p - beta * r"""
    return z_full + alpha * p - beta * r


def score_trix(z_full: np.ndarray, p: np.ndarray, r: np.ndarray,
               lambda_boundary: float, tau: float) -> np.ndarray:
    """
    Score = z_full + p * g - h
    where:
        h(d) = [r(d) - lambda]_+
        g(d) = exp(-h(d) / tau)
    """
    h = np.maximum(r - lambda_boundary, 0)
    g = np.exp(-h / tau)
    return z_full + p * g - h


# ============================================================
# Evaluation Functions
# ============================================================

def compute_score(ranking: List[str], qrels: Dict[str, float], k: int = 10) -> float:
    """Compute NDCG@10 as Score"""
    # Simplified implementation
    # In practice, you would use proper NDCG implementation
    dcg = 0.0
    for i, doc_id in enumerate(ranking[:k]):
        if doc_id in qrels:
            dcg += qrels[doc_id] / np.log2(i + 2)
    
    # Ideal DCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_p_mrr(ranking: List[str], preferred_docs: List[str], 
                  excluded_docs: List[str], k: int = 1000) -> float:
    """Compute preference-MRR"""
    # Find ranks of preferred and excluded docs
    pref_ranks = []
    excl_ranks = []
    
    for i, doc_id in enumerate(ranking[:k]):
        if doc_id in preferred_docs:
            pref_ranks.append(i + 1)
        if doc_id in excluded_docs:
            excl_ranks.append(i + 1)
    
    if not pref_ranks or not excl_ranks:
        return 0.0
    
    # Simplified p-MRR calculation
    # In practice, you would use the proper FollowIR p-MRR implementation
    pref_rank = pref_ranks[0]  # Take first preferred doc
    excl_rank = excl_ranks[0]  # Take first excluded doc
    
    if pref_rank < excl_rank:
        return 1.0 / pref_rank
    else:
        return -1.0 / excl_rank


# ============================================================
# Parameter Selection
# ============================================================

def select_score_optimal(results: pd.DataFrame, method: str) -> pd.DataFrame:
    """Select Score-optimal configuration for each fold"""
    selected = []
    
    for fold in results['fold'].unique():
        fold_data = results[(results['fold'] == fold) & 
                            (results['method'] == method) &
                            (results['split'] == 'development')]
        
        if fold_data.empty:
            continue
        
        # Sort by Score (descending), then p-MRR (descending)
        fold_data = fold_data.sort_values(['score', 'p_mrr'], ascending=[False, False])
        
        # Take top configuration
        best = fold_data.iloc[0].copy()
        best['selection_type'] = 'score-optimal'
        selected.append(best)
    
    return pd.DataFrame(selected)


def select_pmrr_optimal(results: pd.DataFrame, method: str) -> pd.DataFrame:
    """Select p-MRR-optimal configuration for each fold"""
    selected = []
    
    for fold in results['fold'].unique():
        fold_data = results[(results['fold'] == fold) & 
                            (results['method'] == method) &
                            (results['split'] == 'development')]
        
        if fold_data.empty:
            continue
        
        # Sort by p-MRR (descending), then Score (descending)
        fold_data = fold_data.sort_values(['p_mrr', 'score'], ascending=[False, False])
        
        # Take top configuration
        best = fold_data.iloc[0].copy()
        best['selection_type'] = 'pmrr-optimal'
        selected.append(best)
    
    return pd.DataFrame(selected)


def find_pareto_frontier(results: pd.DataFrame) -> pd.DataFrame:
    """Find Pareto-optimal configurations"""
    pareto_points = []
    
    for i, row in results.iterrows():
        # Check if this point is dominated by any other point
        dominated = False
        
        for j, other in results.iterrows():
            if (other['score'] >= row['score'] and 
                other['p_mrr'] >= row['p_mrr'] and
                (other['score'] > row['score'] or other['p_mrr'] > row['p_mrr'])):
                dominated = True
                break
        
        if not dominated:
            pareto_points.append(row)
    
    return pd.DataFrame(pareto_points)


# ============================================================
# Main Experiment Functions
# ============================================================

def validate_existing_results(data_dir: str):
    """Validate that existing results can reproduce paper anchors"""
    logger.info("Validating existing results...")
    
    datasets = ['core17', 'news21', 'robust04']
    
    for dataset in datasets:
        base_path = f"{data_dir}/ablation_{dataset}"
        
        if not os.path.exists(base_path):
            logger.warning(f"Dataset {dataset} not found at {base_path}")
            continue
        
        # Check base_only
        base_only_path = f"{base_path}/base_only/metrics_summary.json"
        if os.path.exists(base_only_path):
            with open(base_only_path) as f:
                data = json.load(f)
            logger.info(f"{dataset} base_only: p-MRR={data['metrics']['p-MRR']:.3f}")
        
        # Check full
        full_path = f"{base_path}/full/metrics_summary.json"
        if os.path.exists(full_path):
            with open(full_path) as f:
                data = json.load(f)
            logger.info(f"{dataset} full: p-MRR={data['metrics']['p-MRR']:.3f}")


def run_parameter_grid(data_dir: str, output_dir: str):
    """Run full parameter grid search"""
    logger.info("Running parameter grid search...")
    
    # This is a placeholder for the actual implementation
    # In practice, you would:
    # 1. Load cached z_full, p, z_neg, r for each document
    # 2. For each parameter combination, compute final scores
    # 3. Re-rank and evaluate
    # 4. Store results
    
    logger.warning("Parameter grid search not yet implemented. See script for framework.")
    logger.info("Required steps:")
    logger.info("1. Load cached scores (z_full, p, z_neg, r)")
    logger.info("2. For each method and parameter combination:")
    logger.info("   - Compute final scores")
    logger.info("   - Re-rank candidates")
    logger.info("   - Evaluate Score and p-MRR")
    logger.info("3. Save results to CSV")


def generate_pareto_plot(results_path: str, output_path: str):
    """Generate Score-p-MRR Pareto plot"""
    logger.info(f"Generating Pareto plot...")
    
    # Load results
    df = pd.read_csv(results_path)
    
    # Filter to held-out results
    held_out = df[df['split'] == 'held_out']
    
    # Find Pareto frontier for each method
    methods = ['Raw linear', 'Residual linear', 'TRIX']
    colors = {'Raw linear': 'blue', 'Residual linear': 'green', 'TRIX': 'red'}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for method in methods:
        method_data = held_out[held_out['method'] == method]
        
        if method_data.empty:
            continue
        
        # Plot all points
        ax.scatter(method_data['score'], method_data['p_mrr'], 
                   c=colors[method], alpha=0.5, label=method, s=50)
        
        # Find and plot Pareto frontier
        pareto = find_pareto_frontier(method_data)
        if not pareto.empty:
            pareto = pareto.sort_values('score')
            ax.plot(pareto['score'], pareto['p_mrr'], 
                    c=colors[method], linewidth=2, linestyle='-')
    
    ax.set_xlabel('Score (NDCG@10)', fontsize=14)
    ax.set_ylabel('p-MRR', fontsize=14)
    ax.set_title('Score-p-MRR Pareto Frontier', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Pareto plot saved to {output_path}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Score-p-MRR Pareto Experiment")
    parser.add_argument("--data_dir", type=str, default="/home/luwa/Documents/DSCLR-remote/results",
                        help="Directory containing cached results")
    parser.add_argument("--output_dir", type=str, default="/home/luwa/Documents/DSCLR-remote/paper/wp05_pareto",
                        help="Output directory for experiment results")
    parser.add_argument("--mode", type=str, default="validate",
                        choices=["validate", "grid", "plot", "all"],
                        help="Experiment mode")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ["validate", "all"]:
        validate_existing_results(args.data_dir)
    
    if args.mode in ["grid", "all"]:
        run_parameter_grid(args.data_dir, args.output_dir)
    
    if args.mode in ["plot", "all"]:
        # Placeholder for plot generation
        logger.info("Plot generation requires grid search results first")


if __name__ == "__main__":
    main()