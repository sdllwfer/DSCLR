#!/usr/bin/env python3
"""
Statistical Significance Tests for TRIX Paper
Follows the protocol in: paper/模拟评审/TRACE_统计显著性实验与报告方案.md
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Statistical Methods
# ============================================================

def paired_bootstrap_ci(delta_values: np.ndarray,
                         n_bootstrap: int = 10000,
                         seed: int = 42,
                         alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute 95% confidence interval using paired bootstrap.
    
    Args:
        delta_values: Array of paired differences (TRIX - Base) for each query/cluster
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility
        alpha: Significance level (0.05 for 95% CI)
    
    Returns:
        (ci_low, ci_high) in the same scale as delta_values
    """
    np.random.seed(seed)
    n = len(delta_values)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        resampled_deltas = delta_values[indices]
        bootstrap_means.append(np.mean(resampled_deltas))
    
    bootstrap_means = np.array(bootstrap_means)
    ci_low = np.percentile(bootstrap_means, 100 * (alpha / 2))
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return ci_low, ci_high


def paired_randomization_test(delta_values: np.ndarray,
                               n_permutations: int = 10000,
                               seed: int = 43) -> float:
    """
    Two-sided paired randomization test (sign-flip test).
    
    Args:
        delta_values: Array of paired differences (TRIX - Base) for each query/cluster
        n_permutations: Number of randomization iterations
        seed: Random seed for reproducibility
    
    Returns:
        p-value (Monte Carlo estimate)
    """
    np.random.seed(seed)
    n = len(delta_values)
    observed_mean = np.mean(delta_values)
    
    # Count how many permuted means have absolute value >= observed
    count_extreme = 0
    
    for _ in range(n_permutations):
        # Randomly flip signs with probability 0.5
        signs = np.random.choice([-1, 1], size=n)
        permuted_deltas = delta_values * signs
        permuted_mean = np.mean(permuted_deltas)
        
        if abs(permuted_mean) >= abs(observed_mean):
            count_extreme += 1
    
    # Add 1 to numerator and denominator (Monte Carlo correction)
    p_value = (1 + count_extreme) / (1 + n_permutations)
    
    return p_value


def holm_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of raw p-values
    
    Returns:
        List of adjusted p-values (Holm corrected)
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = np.array(p_values)[sorted_indices]
    
    # Apply Holm correction
    adjusted_pvalues = np.zeros(n)
    cumulative_max = 0
    
    for i in range(n):
        rank = i + 1
        adjusted = sorted_pvalues[i] * (n - rank + 1)
        # Ensure monotonicity (adjusted p-values are non-decreasing)
        cumulative_max = max(cumulative_max, adjusted)
        adjusted_pvalues[i] = min(cumulative_max, 1.0)
    
    # Restore original order
    holm_pvalues = np.zeros(n)
    holm_pvalues[sorted_indices] = adjusted_pvalues
    
    return holm_pvalues.tolist()


# ============================================================
# Data Loading Functions
# ============================================================

def load_followir_per_query(data_dir: str,
                            backbone: str,
                            dataset: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Load per-query results for FollowIR dataset.
    
    Returns:
        (base_results, trix_results): Dicts mapping query_id to metric value
    """
    # Try different possible paths
    possible_paths = [
        f"{data_dir}/followir/{backbone.lower()}/{dataset}",
        f"{data_dir}/followir_{backbone.lower()}_{dataset.lower()}",
        f"{data_dir}/bge_v86_kappa10_{dataset.lower()}",
        f"{data_dir}/e5mistral_v86_kappa10_{dataset.lower()}",
        f"{data_dir}/repllama_v86_kappa10_{dataset.lower()}",
    ]
    
    base_results = {}
    trix_results = {}
    
    for base_path in possible_paths:
        base_file = f"{base_path}/base_only/per_query_stats.json"
        trix_file = f"{base_path}/full/per_query_stats.json"
        
        if os.path.exists(base_file) and os.path.exists(trrix_file):
            with open(base_file) as f:
                base_data = json.load(f)
            with open(trrix_file) as f:
                trix_data = json.load(f)
            
            # Extract metric values
            for qid, stats in base_data.items():
                base_results[qid] = stats.get('score', stats.get('ndcg', 0))
            
            for qid, stats in trix_data.items():
                trix_results[qid] = stats.get('score', stats.get('ndcg', 0))
            
            return base_results, trix_results
    
    logger.warning(f"Could not find FollowIR data for {backbone}/{dataset}")
    return base_results, trix_results


def load_negconstraint_per_query(data_dir: str,
                                  backbone: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Load per-query results for NegConstraint.
    
    Returns:
        (base_results, trix_results): Dicts mapping query_id to {metric: value}
    """
    # Try different possible paths
    possible_paths = [
        f"{data_dir}/negconstraint/{backbone.lower()}",
        f"{data_dir}/negconstraint_{backbone.lower()}",
    ]
    
    base_results = {}
    trix_results = {}
    
    for base_path in possible_paths:
        base_file = f"{base_path}/baseline/metrics_summary.json"
        trix_file = f"{base_path}/deir_dual_v2/metrics_summary.json"
        
        if os.path.exists(base_file) and os.path.exists(trrix_file):
            with open(base_file) as f:
                base_data = json.load(f)
            with open(trrix_file) as f:
                trix_data = json.load(f)
            
            # Extract per-query metrics
            for qid in base_data.get('per_query_map', {}):
                base_results[qid] = {
                    'map': base_data['per_query_map'][qid],
                    'ndcg': base_data['per_query_ndcg'][qid]
                }
                trix_results[qid] = {
                    'map': trix_data['per_query_map'][qid],
                    'ndcg': trix_data['per_query_ndcg'][qid]
                }
            
            return base_results, trix_results
    
    logger.warning(f"Could not find NegConstraint data for {backbone}")
    return base_results, trix_results


def load_excluir_per_query(data_dir: str,
                            backbone: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Load per-query results for ExcluIR.
    
    Returns:
        (base_results, trix_results): Dicts mapping request_id to {metric: value}
    """
    # Try different possible paths
    possible_paths = [
        f"{data_dir}/excluir_{backbone.lower()}",
        f"{data_dir}/excluir_repllama",
    ]
    
    base_results = {}
    trix_results = {}
    
    for base_path in possible_paths:
        result_file = f"{base_path}/metrics_summary.json"
        
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            
            # Extract per-request metrics
            for rid, metrics in data.get('per_request_metrics', {}).items():
                base_results[rid] = metrics.get('baseline', {})
                trix_results[rid] = metrics.get('trix', {})
            
            return base_results, trix_results
    
    logger.warning(f"Could not find ExcluIR data for {backbone}")
    return base_results, trix_results


def load_beir_per_query(data_dir: str,
                        backbone: str,
                        dataset: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Load per-query results for BEIR dataset.
    
    Returns:
        (base_results, trix_results): Dicts mapping query_id to {metric: value}
    """
    # Try different possible paths
    possible_paths = [
        f"{data_dir}/beir/{backbone.lower()}/{dataset}",
        f"{data_dir}/beir_{backbone.lower()}_{dataset.lower()}",
    ]
    
    base_results = {}
    trix_results = {}
    
    for base_path in possible_paths:
        base_file = f"{base_path}/baseline/per_query_stats.json"
        trix_file = f"{base_path}/trix/per_query_stats.json"
        
        if os.path.exists(base_file) and os.path.exists(trrix_file):
            with open(base_file) as f:
                base_data = json.load(f)
            with open(trix_file) as f:
                trix_data = json.load(f)
            
            for qid, stats in base_data.items():
                base_results[qid] = {
                    'ndcg': stats.get('ndcg@10', 0),
                    'map': stats.get('map@100', 0)
                }
            
            for qid, stats in trix_data.items():
                trix_results[qid] = {
                    'ndcg': stats.get('ndcg@10', 0),
                    'map': stats.get('map@100', 0)
                }
            
            return base_results, trix_results
    
    logger.warning(f"Could not find BEIR data for {backbone}/{dataset}")
    return base_results, trix_results


# ============================================================
# Main Testing Functions
# ============================================================

def test_followir_main(data_dir: str) -> pd.DataFrame:
    """
    Test FollowIR main table: 4 backbones x 2 metrics (Score, p-MRR).
    """
    results = []
    
    backbones = ['BM25', 'E5-Mistral', 'BGE-large', 'RepLLaMA']
    datasets = ['Core17', 'Robust04', 'News21']
    metrics = ['score', 'p-mrr']
    
    for backbone in backbones:
        for metric in metrics:
            # Aggregate across three datasets
            all_base_values = []
            all_trix_values = []
            
            for dataset in datasets:
                base, trix = load_followir_per_query(data_dir, backbone, dataset)
                
                if not base:
                    continue
                
                for qid in base:
                    all_base_values.append(base[qid])
                    all_trix_values.append(trix[qid])
            
            if not all_base_values:
                continue
            
            # Compute paired differences
            delta_values = np.array(all_trix_values) - np.array(all_base_values)
            
            # Statistical tests
            base_mean = np.mean(all_base_values) * 100
            trix_mean = np.mean(all_trix_values) * 100
            delta_mean = np.mean(delta_values) * 100
            ci_low, ci_high = paired_bootstrap_ci(delta_values)
            ci_low_pct = ci_low * 100
            ci_high_pct = ci_high * 100
            raw_p = paired_randomization_test(delta_values)
            
            results.append({
                'benchmark': 'FollowIR',
                'dataset': 'Aggregate',
                'backbone': backbone,
                'metric': metric,
                'base': round(base_mean, 2),
                'trix': round(trix_mean, 2),
                'delta': round(delta_mean, 2),
                'ci_low': round(ci_low_pct, 2),
                'ci_high': round(ci_high_pct, 2),
                'raw_p': raw_p
            })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Apply Holm correction within this family (8 tests)
    df['holm_p'] = holm_correction(df['raw_p'].tolist())
    df['significant'] = df['holm_p'] < 0.05
    
    return df


def test_negconstraint_main(data_dir: str) -> pd.DataFrame:
    """
    Test NegConstraint: 4 backbones x 2 metrics (MAP, nDCG@10).
    """
    results = []
    
    backbones = ['BGE-large', 'E5-Mistral', 'RepLLaMA', 'GritLM-7B']
    metrics = ['map', 'ndcg']
    
    for backbone in backbones:
        base, trix = load_negconstraint_per_query(data_dir, backbone)
        
        if not base:
            continue
        
        for metric in metrics:
            base_values = [base[qid][metric] for qid in base]
            trix_values = [trix[qid][metric] for qid in trix]
            
            delta_values = np.array(trix_values) - np.array(base_values)
            
            base_mean = np.mean(base_values) * 100
            trix_mean = np.mean(trix_values) * 100
            delta_mean = np.mean(delta_values) * 100
            ci_low, ci_high = paired_bootstrap_ci(delta_values)
            ci_low_pct = ci_low * 100
            ci_high_pct = ci_high * 100
            raw_p = paired_randomization_test(delta_values)
            
            results.append({
                'benchmark': 'NegConstraint',
                'dataset': '-',
                'backbone': backbone,
                'metric': metric,
                'base': round(base_mean, 2),
                'trix': round(trix_mean, 2),
                'delta': round(delta_mean, 2),
                'ci_low': round(ci_low_pct, 2),
                'ci_high': round(ci_high_pct, 2),
                'raw_p': raw_p
            })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df['holm_p'] = holm_correction(df['raw_p'].tolist())
    df['significant'] = df['holm_p'] < 0.05
    
    return df


def test_excluir_main(data_dir: str) -> pd.DataFrame:
    """
    Test ExcluIR: 4 backbones x 2 metrics (MRR, RR).
    """
    results = []
    
    backbones = ['BGE-large', 'BGE-M3', 'RepLLaMA', 'E5-Mistral']
    metrics = ['mrr', 'rr']
    
    for backbone in backbones:
        base, trix = load_excluir_per_query(data_dir, backbone)
        
        if not base:
            continue
        
        for metric in metrics:
            base_values = [base[rid].get(metric, 0) for rid in base]
            trix_values = [trix[rid].get(metric, 0) for rid in trix]
            
            delta_values = np.array(trix_values) - np.array(base_values)
            
            base_mean = np.mean(base_values) * 100
            trix_mean = np.mean(trix_values) * 100
            delta_mean = np.mean(delta_values) * 100
            ci_low, ci_high = paired_bootstrap_ci(delta_values)
            ci_low_pct = ci_low * 100
            ci_high_pct = ci_high * 100
            raw_p = paired_randomization_test(delta_values)
            
            results.append({
                'benchmark': 'ExcluIR',
                'dataset': '-',
                'backbone': backbone,
                'metric': metric,
                'base': round(base_mean, 2),
                'trix': round(trix_mean, 2),
                'delta': round(delta_mean, 2),
                'ci_low': round(ci_low_pct, 2),
                'ci_high': round(ci_high_pct, 2),
                'raw_p': raw_p
            })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df['holm_p'] = holm_correction(df['raw_p'].tolist())
    df['significant'] = df['holm_p'] < 0.05
    
    return df


def test_beir_main(data_dir: str) -> pd.DataFrame:
    """
    Test BEIR main table: BGE-large macro-average across 8 datasets.
    """
    results = []
    
    backbone = 'BGE-large'
    datasets = ['trec-covid', 'arguana', 'webis-touche2020', 'fiqa',
                'scidocs', 'nfcorpus', 'quora', 'dbpedia-entity']
    metrics = ['ndcg', 'map']
    
    # Stratified bootstrap for macro-average
    dataset_results = {metric: {'base': [], 'trix': []} for metric in metrics}
    
    for dataset in datasets:
        base, trix = load_beir_per_query(data_dir, backbone, dataset)
        
        if not base:
            continue
        
        for metric in metrics:
            base_values = [base[qid][metric] for qid in base]
            trix_values = [trix[qid][metric] for qid in trix]
            
            dataset_results[metric]['base'].append(base_values)
            dataset_results[metric]['trix'].append(trix_values)
    
    for metric in metrics:
        # Compute macro-average using stratified bootstrap
        n_datasets = len(dataset_results[metric]['base'])
        if n_datasets == 0:
            continue
        
        # For macro-average, we need to compute mean for each dataset first
        # Then apply stratified bootstrap
        n_bootstrap = 10000
        np.random.seed(42)
        
        bootstrap_deltas = []
        
        for _ in range(n_bootstrap):
            # Resample datasets
            dataset_indices = np.random.choice(n_datasets, size=n_datasets, replace=True)
            
            # For each resampled dataset, compute mean
            macro_base = 0
            macro_trix = 0
            
            for idx in dataset_indices:
                base_vals = dataset_results[metric]['base'][idx]
                trix_vals = dataset_results[metric]['trix'][idx]
                
                # Resample queries within dataset
                n_queries = len(base_vals)
                query_indices = np.random.choice(n_queries, size=n_queries, replace=True)
                
                base_mean = np.mean([base_vals[i] for i in query_indices])
                trix_mean = np.mean([trix_vals[i] for i in query_indices])
                
                macro_base += base_mean
                macro_trix += trix_mean
            
            macro_base /= n_datasets
            macro_trix /= n_datasets
            
            bootstrap_deltas.append(macro_trix - macro_base)
        
        bootstrap_deltas = np.array(bootstrap_deltas)
        
        # Compute statistics
        observed_base = np.mean([np.mean(vals) for vals in dataset_results[metric]['base']])
        observed_trix = np.mean([np.mean(vals) for vals in dataset_results[metric]['trix']])
        delta_mean = np.mean(bootstrap_deltas) * 100
        
        ci_low = np.percentile(bootstrap_deltas, 2.5) * 100
        ci_high = np.percentile(bootstrap_deltas, 97.5) * 100
        
        # Randomization test (sign-flip within each dataset)
        np.random.seed(43)
        n_permutations = 10000
        observed_delta = observed_trix - observed_base
        count_extreme = 0
        
        for _ in range(n_permutations):
            permuted_delta = 0
            
            for idx in range(n_datasets):
                base_vals = dataset_results[metric]['base'][idx]
                trix_vals = dataset_results[metric]['trix'][idx]
                
                delta_vals = np.array(trix_vals) - np.array(base_vals)
                
                # Sign-flip
                signs = np.random.choice([-1, 1], size=len(delta_vals))
                permuted_vals = delta_vals * signs
                
                permuted_delta += np.mean(permuted_vals)
            
            permuted_delta /= n_datasets
            
            if abs(permuted_delta) >= abs(observed_delta):
                count_extreme += 1
        
        raw_p = (1 + count_extreme) / (1 + n_permutations)
        
        results.append({
            'benchmark': 'BEIR',
            'dataset': 'Macro-average',
            'backbone': backbone,
            'metric': metric,
            'base': round(observed_base * 100, 2),
            'trix': round(observed_trix * 100, 2),
            'delta': round(delta_mean, 2),
            'ci_low': round(ci_low, 2),
            'ci_high': round(ci_high, 2),
            'raw_p': raw_p
        })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df['holm_p'] = holm_correction(df['raw_p'].tolist())
    df['significant'] = df['holm_p'] < 0.05
    
    return df


def run_all_tests(data_dir: str, output_dir: str):
    """
    Run all statistical significance tests and save results.
    """
    logger.info("Starting statistical significance tests...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tests for each benchmark
    all_results = []
    
    logger.info("Testing FollowIR main table...")
    df_followir = test_followir_main(data_dir)
    if not df_followir.empty:
        all_results.append(df_followir)
        logger.info(f"  FollowIR: {len(df_followir)} tests")
    
    logger.info("Testing NegConstraint...")
    df_negconstraint = test_negconstraint_main(data_dir)
    if not df_negconstraint.empty:
        all_results.append(df_negconstraint)
        logger.info(f"  NegConstraint: {len(df_negconstraint)} tests")
    
    logger.info("Testing ExcluIR...")
    df_excluir = test_excluir_main(data_dir)
    if not df_excluir.empty:
        all_results.append(df_excluir)
        logger.info(f"  ExcluIR: {len(df_excluir)} tests")
    
    logger.info("Testing BEIR...")
    df_beir = test_beir_main(data_dir)
    if not df_beir.empty:
        all_results.append(df_beir)
        logger.info(f"  BEIR: {len(df_beir)} tests")
    
    if not all_results:
        logger.error("No results generated!")
        return
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    output_file = f"{output_dir}/statistical_significance_results.csv"
    df_all.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("Statistical Significance Test Results")
    print("="*80)
    print(df_all.to_string(index=False))
    print("="*80)
    
    # Count significant results
    n_significant = df_all['significant'].sum()
    print(f"\nSignificant results: {n_significant}/{len(df_all)}")
    
    # Save detailed JSON
    output_json = f"{output_dir}/statistical_significance_results.json"
    df_all.to_json(output_json, orient='records', indent=2)
    logger.info(f"Detailed results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Significance Tests for TRIX Paper")
    parser.add_argument("--data_dir", type=str, default="/home/luwa/Documents/DSCLR-remote/results",
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="/home/luwa/Documents/DSCLR-remote/paper/statistical_tests",
                        help="Output directory for test results")
    
    args = parser.parse_args()
    
    run_all_tests(args.data_dir, args.output_dir)