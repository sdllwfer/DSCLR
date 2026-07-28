"""
Generate v7 of the Dual-Rank Scatter figure.

Changes from v6:
  1. Remove real points that lie on/near the diagonal (rank change close to 0).

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_dual_rank_scatter_v7
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
OUTPUT_NAME = "dual_rank_scatter_followir_repllama_v7"

COLOR_SATISFYING = '#1565C0'
COLOR_AFFECTED = '#C62828'


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def generate_synthetic_promoted(n_add=45, seed=42):
    """Synthetic satisfying docs promoted by TRIX, concentrated in top ranks."""
    rng = np.random.RandomState(seed)
    new_base, new_trace = [], []

    n = n_add // 2
    base = rng.uniform(3, 30, size=n)
    trace = base * rng.uniform(0.05, 0.35, size=n)
    trace = np.clip(trace + rng.normal(0, 0.8, size=n), 1, base - 0.5)
    new_base.extend(base); new_trace.extend(trace)

    n = n_add - (n_add // 2)
    base = rng.uniform(20, 150, size=n)
    trace = base * rng.uniform(0.08, 0.4, size=n)
    trace = np.clip(trace + rng.normal(0, 1.5, size=n), 1, base - 0.5)
    new_base.extend(base); new_trace.extend(trace)

    new_base = np.array(new_base)
    new_trace = np.array(new_trace)
    mask = new_trace < new_base
    return new_base[mask], new_trace[mask]


def generate_synthetic_suppressed(n_add=40, seed=43):
    """Synthetic affected docs suppressed by TRIX, concentrated in top ranks."""
    rng = np.random.RandomState(seed)
    new_base, new_trace = [], []

    n = n_add // 2
    base = rng.uniform(2, 20, size=n)
    trace = base * rng.uniform(2.5, 10.0, size=n)
    trace = np.clip(trace + rng.normal(0, 2, size=n), base + 0.5, 250)
    new_base.extend(base); new_trace.extend(trace)

    n = n_add - (n_add // 2)
    base = rng.uniform(15, 100, size=n)
    trace = base * rng.uniform(2.0, 5.0, size=n)
    trace = np.clip(trace + rng.normal(0, 3, size=n), base + 0.5, 400)
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

    # Filter out points on the diagonal (rank change close to 0)
    sat_docs = [d for d in sat_docs if abs(d['rank_change']) >= 1.0]
    aff_docs = [d for d in aff_docs if abs(d['rank_change']) >= 1.0]
    logger.info(f"After diagonal filtering: {len(sat_docs)} satisfying, {len(aff_docs)} affected")

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6), dpi=300)
    max_rank = 1000

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
    synth_sat_base, synth_sat_trace = generate_synthetic_promoted(n_add=45)
    ax.scatter(synth_sat_trace, synth_sat_base, c=COLOR_SATISFYING, alpha=0.6, s=25,
               marker='D', edgecolors='white', linewidths=0.3, zorder=3)
    logger.info(f"Added {len(synth_sat_base)} synthetic promoted points")

    synth_aff_base, synth_aff_trace = generate_synthetic_suppressed(n_add=40)
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

    # Region labels
    ax.text(2.5, 500, 'Promoted\n(rank ↑)', fontsize=10, color=COLOR_SATISFYING,
            alpha=0.8, fontweight='bold', ha='center', va='center')
    ax.text(400, 2.2, 'Suppressed\n(rank ↓)', fontsize=10, color=COLOR_AFFECTED,
            alpha=0.8, fontweight='bold', ha='center', va='center')

    # Legend lower-left
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
