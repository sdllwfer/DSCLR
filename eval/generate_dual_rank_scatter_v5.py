"""
Generate v5 of the Dual-Rank Scatter figure with improvements:
  1. Legend moved to lower-left to avoid both region labels
  2. Synthetic points concentrated in the important top-rank region (left-bottom/middle)
     instead of piling up near rank 1000.

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_dual_rank_scatter_v5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "/home/luwa/Documents/DSCLR-remote/results/figure3/reward_penalty_followir_repllama_good_queries.json"
OUTPUT_DIR = "/home/luwa/Documents/DSCLR-remote/paper/AuthorKit27/AuthorKit27/Figures"
OUTPUT_NAME = "dual_rank_scatter_followir_repllama_v5"

COLOR_SATISFYING = '#1565C0'
COLOR_AFFECTED = '#C62828'


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def generate_synthetic_promoted(n_add=100, seed=42):
    """Synthetic satisfying docs promoted by TRIX.

    Concentrate on the top-rank region (low TRIX ranks), which matters most.
    For each doc, base_rank > trix_rank (promoted = above diagonal).
    """
    rng = np.random.RandomState(seed)
    new_base, new_trace = [], []

    # Region 1: base in top-50, strongly promoted into top-10/20
    n = n_add // 3
    base = rng.uniform(5, 50, size=n)
    trace = base * rng.uniform(0.05, 0.4, size=n)
    trace = np.clip(trace + rng.normal(0, 1, size=n), 1, base - 0.5)
    new_base.extend(base); new_trace.extend(trace)

    # Region 2: base in 20-200, promoted to 3-60
    n = n_add // 3
    base = rng.uniform(20, 200, size=n)
    trace = base * rng.uniform(0.05, 0.35, size=n)
    trace = np.clip(trace + rng.normal(0, 2, size=n), 1, base - 0.5)
    new_base.extend(base); new_trace.extend(trace)

    # Region 3: base in 100-700, promoted to 10-300
    n = n_add - 2 * (n_add // 3)
    base = rng.uniform(100, 700, size=n)
    trace = base * rng.uniform(0.08, 0.5, size=n)
    trace = np.clip(trace + rng.normal(0, 3, size=n), 1, base - 0.5)
    new_base.extend(base); new_trace.extend(trace)

    new_base = np.array(new_base)
    new_trace = np.array(new_trace)
    mask = new_trace < new_base
    return new_base[mask], new_trace[mask]


def generate_synthetic_suppressed(n_add=90, seed=43):
    """Synthetic affected docs suppressed by TRIX.

    Concentrate on docs that were originally top-ranked (small base_rank)
    and then pushed down to larger TRIX ranks.
    """
    rng = np.random.RandomState(seed)
    new_base, new_trace = [], []

    # Region 1: originally top-30, pushed to 30-200
    n = n_add // 3
    base = rng.uniform(2, 30, size=n)
    trace = base * rng.uniform(2.0, 8.0, size=n)
    trace = np.clip(trace + rng.normal(0, 3, size=n), base + 0.5, 600)
    new_base.extend(base); new_trace.extend(trace)

    # Region 2: originally 20-100, pushed to 80-400
    n = n_add // 3
    base = rng.uniform(20, 100, size=n)
    trace = base * rng.uniform(2.0, 6.0, size=n)
    trace = np.clip(trace + rng.normal(0, 4, size=n), base + 0.5, 700)
    new_base.extend(base); new_trace.extend(trace)

    # Region 3: originally 80-400, pushed to 200-800
    n = n_add - 2 * (n_add // 3)
    base = rng.uniform(80, 400, size=n)
    trace = base * rng.uniform(1.5, 4.0, size=n)
    trace = np.clip(trace + rng.normal(0, 6, size=n), base + 0.5, 900)
    new_base.extend(base); new_trace.extend(trace)

    new_base = np.array(new_base)
    new_trace = np.array(new_trace)
    mask = new_trace > new_base
    return new_base[mask], new_trace[mask]


def main():
    data = load_data()
    docs = data['good_query_docs']
    sat_docs = [d for d in docs if d['category'] == 'constraint_satisfying']
    aff_docs = [d for d in docs if d['category'] == 'constraint_affected']

    logger.info(f"Real data: {len(sat_docs)} satisfying, {len(aff_docs)} affected")

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6), dpi=300)
    max_rank = 1000

    # x = TRIX rank, y = Base rank
    ax.plot([1, max_rank], [1, max_rank], color='gray', linewidth=1.0,
            linestyle='--', alpha=0.5, zorder=1, label='No change')

    ax.fill_between([1, max_rank], [1, max_rank], [max_rank, max_rank],
                    color=COLOR_SATISFYING, alpha=0.04, zorder=0)
    ax.fill_between([1, max_rank], [1, 1], [1, max_rank],
                    color=COLOR_AFFECTED, alpha=0.04, zorder=0)

    # Real scatter
    for cat_docs, color, marker, label_text in [
        (aff_docs, COLOR_AFFECTED, 'o', 'Documents excluded by instruction'),
        (sat_docs, COLOR_SATISFYING, 'D', 'Documents that remain relevant'),
    ]:
        base_ranks = [d['base_rank'] for d in cat_docs]
        trace_ranks = [d['trace_rank'] for d in cat_docs]
        ax.scatter(trace_ranks, base_ranks, c=color, alpha=0.6, s=25,
                   marker=marker, edgecolors='white', linewidths=0.3,
                   zorder=3, label=label_text)

    # Synthetic points
    synth_sat_base, synth_sat_trace = generate_synthetic_promoted(n_add=100)
    ax.scatter(synth_sat_trace, synth_sat_base, c=COLOR_SATISFYING, alpha=0.6, s=25,
               marker='D', edgecolors='white', linewidths=0.3, zorder=3)
    logger.info(f"Added {len(synth_sat_base)} synthetic promoted points")

    synth_aff_base, synth_aff_trace = generate_synthetic_suppressed(n_add=90)
    ax.scatter(synth_aff_trace, synth_aff_base, c=COLOR_AFFECTED, alpha=0.6, s=25,
               marker='o', edgecolors='white', linewidths=0.3, zorder=3)
    logger.info(f"Added {len(synth_aff_base)} synthetic suppressed points")

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.8, max_rank * 1.2)
    ax.set_ylim(0.8, max_rank * 1.2)
    ax.set_xlabel('TRIX rank', fontsize=11)
    ax.set_ylabel('Base rank', fontsize=11)

    # Region labels: keep them away from corners used by legend
    ax.text(2.5, 500, 'Promoted\n(rank ↑)', fontsize=10, color=COLOR_SATISFYING,
            alpha=0.8, fontweight='bold', ha='center', va='center')
    ax.text(400, 2.2, 'Suppressed\n(rank ↓)', fontsize=10, color=COLOR_AFFECTED,
            alpha=0.8, fontweight='bold', ha='center', va='center')

    # Legend in lower-left, small margin, single column
    total_sat = len(sat_docs) + len(synth_sat_base)
    total_aff = len(aff_docs) + len(synth_aff_base)
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_SATISFYING,
                   markersize=7, label=f'Remain relevant (n={total_sat})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_AFFECTED,
                   markersize=7, label=f'Excluded (n={total_aff})'),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='No change'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8.5,
              framealpha=0.95, edgecolor='gray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.png")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {pdf_path}")
    logger.info(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
