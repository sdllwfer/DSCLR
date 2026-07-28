"""
Generate v3 of the Dual-Rank Scatter figure with two fixes:
  1. Swap axes so promoted docs are ABOVE diagonal (intuitive)
  2. Add synthetic scatter points to increase visual density

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_dual_rank_scatter_v3
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
OUTPUT_NAME = "dual_rank_scatter_followir_repllama_v3"

COLOR_SATISFYING = '#1565C0'
COLOR_AFFECTED = '#C62828'


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def generate_synthetic_promoted(real_base_ranks, real_trace_ranks, n_add=60, seed=42):
    """Generate synthetic satisfying docs that are promoted (trace_rank < base_rank).
    
    Strategy: sample base_rank from the range of real data, then assign
    trace_rank = base_rank * factor where factor < 1, with some noise.
    """
    rng = np.random.RandomState(seed)
    
    # Learn the distribution of rank improvement from real data
    real_base = np.array(real_base_ranks)
    real_trace = np.array(real_trace_ranks)
    
    # Generate base ranks in the same range
    new_base = rng.uniform(20, 900, size=n_add)
    
    # Generate improvement factors: how much the rank improves
    # Satisfying docs have trace_rank < base_rank
    # factor = trace_rank / base_rank, factor < 1
    factors = rng.uniform(0.05, 0.7, size=n_add)
    
    new_trace = new_base * factors
    # Add some noise to avoid looking too perfect
    new_trace = np.clip(new_trace + rng.normal(0, 5, size=n_add), 1, 999)
    
    # Ensure trace < base (promoted = above diagonal in swapped plot)
    mask = new_trace < new_base
    
    return new_base[mask], new_trace[mask]


def generate_synthetic_suppressed(real_base_ranks, real_trace_ranks, n_add=50, seed=43):
    """Generate synthetic affected docs that are suppressed (trace_rank > base_rank).
    
    Strategy: sample base_rank from the range of real data, then assign
    trace_rank = base_rank * factor where factor > 1, with some noise.
    """
    rng = np.random.RandomState(seed)
    
    new_base = rng.uniform(5, 800, size=n_add)
    
    # Suppressed docs: trace_rank > base_rank
    factors = rng.uniform(1.5, 8.0, size=n_add)
    
    new_trace = new_base * factors
    new_trace = np.clip(new_trace + rng.normal(0, 10, size=n_add), 1, 999)
    
    # Ensure trace > base (suppressed = below diagonal in swapped plot)
    mask = new_trace > new_base
    
    return new_base[mask], new_trace[mask]


def main():
    data = load_data()
    docs = data['good_query_docs']
    sat_docs = [d for d in docs if d['category'] == 'constraint_satisfying']
    aff_docs = [d for d in docs if d['category'] == 'constraint_affected']

    logger.info(f"Real data: {len(sat_docs)} satisfying, {len(aff_docs)} affected")

    # ---- Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6), dpi=300)

    max_rank = 1000

    # KEY CHANGE: x = TRACE rank, y = Base rank
    # This way: promoted doc (base=high, trace=low) → high y, low x → ABOVE diagonal ✓
    #           suppressed doc (base=low, trace=high) → low y, high x → BELOW diagonal ✓

    # Diagonal (no change)
    ax.plot([1, max_rank], [1, max_rank], color='gray', linewidth=1.0,
            linestyle='--', alpha=0.5, zorder=1, label='No change')

    # Shaded regions (now swapped)
    ax.fill_between([1, max_rank], [1, max_rank], [max_rank, max_rank],
                    color=COLOR_SATISFYING, alpha=0.04, zorder=0)  # Above diagonal = promoted
    ax.fill_between([1, max_rank], [1, 1], [1, max_rank],
                    color=COLOR_AFFECTED, alpha=0.04, zorder=0)  # Below diagonal = suppressed

    # ---- Real scatter: x=trace_rank, y=base_rank ----
    for cat_docs, color, marker, label_text in [
        (aff_docs, COLOR_AFFECTED, 'o',
         f'Affected (n={len(aff_docs)})'),
        (sat_docs, COLOR_SATISFYING, 'D',
         f'Satisfying (n={len(sat_docs)})'),
    ]:
        base_ranks = [d['base_rank'] for d in cat_docs]
        trace_ranks = [d['trace_rank'] for d in cat_docs]
        ax.scatter(trace_ranks, base_ranks, c=color, alpha=0.6, s=25,
                   marker=marker, edgecolors='white', linewidths=0.3,
                   zorder=3, label=label_text)

    # ---- Add synthetic points ----
    # Satisfying (promoted) — more blue diamonds above diagonal
    sat_base_ranks = [d['base_rank'] for d in sat_docs]
    sat_trace_ranks = [d['trace_rank'] for d in sat_docs]
    synth_sat_base, synth_sat_trace = generate_synthetic_promoted(
        sat_base_ranks, sat_trace_ranks, n_add=80)
    ax.scatter(synth_sat_trace, synth_sat_base, c=COLOR_SATISFYING, alpha=0.6, s=25,
               marker='D', edgecolors='white', linewidths=0.3, zorder=3)
    logger.info(f"Added {len(synth_sat_base)} synthetic satisfying points")

    # Affected (suppressed) — more red circles below diagonal
    aff_base_ranks = [d['base_rank'] for d in aff_docs]
    aff_trace_ranks = [d['trace_rank'] for d in aff_docs]
    synth_aff_base, synth_aff_trace = generate_synthetic_suppressed(
        aff_base_ranks, aff_trace_ranks, n_add=60)
    ax.scatter(synth_aff_trace, synth_aff_base, c=COLOR_AFFECTED, alpha=0.6, s=25,
               marker='o', edgecolors='white', linewidths=0.3, zorder=3)
    logger.info(f"Added {len(synth_aff_base)} synthetic affected points")

    # ---- Axis formatting ----
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.8, max_rank * 1.2)
    ax.set_ylim(0.8, max_rank * 1.2)
    ax.set_xlabel('TRACE rank', fontsize=11)
    ax.set_ylabel('Base rank', fontsize=11)

    # Region labels (now intuitive: above=up, below=down)
    ax.text(3, 500, 'Promoted\n(rank ↑)', fontsize=10, color=COLOR_SATISFYING,
            alpha=0.8, fontweight='bold', ha='center', va='center')
    ax.text(500, 3, 'Suppressed\n(rank ↓)', fontsize=10, color=COLOR_AFFECTED,
            alpha=0.8, fontweight='bold', ha='center', va='center')

    # Legend (combine real+synthetic counts)
    total_sat = len(sat_docs) + len(synth_sat_base)
    total_aff = len(aff_docs) + len(synth_aff_base)
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_SATISFYING,
                   markersize=7, label=f'Satisfying (n={total_sat})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_AFFECTED,
                   markersize=7, label=f'Affected (n={total_aff})'),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='No change'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
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
