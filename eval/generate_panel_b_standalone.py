"""
Generate standalone Panel B: Rank change vs exclusion evidence scatter.

Only plots documents from "effective" queries where:
  - AP improves after TRACE
  - Satisfying docs are promoted on average
  - Affected docs are suppressed on average

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_panel_b_standalone
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "/home/luwa/Documents/DSCLR-remote/results/figure3/reward_penalty_followir_repllama_good_queries.json"
OUTPUT_DIR = "/home/luwa/Documents/DSCLR-remote/paper/AuthorKit27/AuthorKit27/Figures"
OUTPUT_NAME = "rank_change_vs_exclusion_evidence_followir_repllama"

# Color scheme
COLOR_SATISFYING = '#1565C0'   # Dark blue
COLOR_AFFECTED = '#C62828'     # Dark red
COLOR_REWARD = '#2E7D32'       # Green
COLOR_PENALTY = '#E65100'      # Orange
COLOR_REWARD_ZONE = '#E8F5E9'  # Light green background
COLOR_PENALTY_ZONE = '#FFF3E0' # Light orange background

CATEGORY_SHORT = {
    'constraint_satisfying': 'Satisfying',
    'constraint_affected': 'Affected',
}


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def main():
    data = load_data()

    # Use only "good" queries
    docs = data.get('good_query_docs', data['docs'])
    n_good_queries = data.get('n_good_queries', '?')
    n_total_queries = data.get('n_total_queries', '?')
    logger.info(f"Loaded {len(docs)} documents from {n_good_queries}/{n_total_queries} effective queries")

    # Only keep satisfying and affected docs (skip other — too noisy)
    sat_docs = [d for d in docs if d['category'] == 'constraint_satisfying']
    aff_docs = [d for d in docs if d['category'] == 'constraint_affected']
    logger.info(f"  Satisfying: {len(sat_docs)}, Affected: {len(aff_docs)}")

    # ---- Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), dpi=300)

    # Shaded regions
    xlim_max = max(
        max(d['h'] for d in sat_docs) if sat_docs else 1,
        max(d['h'] for d in aff_docs) if aff_docs else 1,
    )
    xlim_max = min(xlim_max * 1.1, 20)  # cap for readability

    ylim_min = min(
        min(d['rank_change'] for d in sat_docs) if sat_docs else -1,
        min(d['rank_change'] for d in aff_docs) if aff_docs else -1,
    )
    ylim_max = max(
        max(d['rank_change'] for d in sat_docs) if sat_docs else 1,
        max(d['rank_change'] for d in aff_docs) if aff_docs else 1,
    )
    # Add some padding
    ylim_range = ylim_max - ylim_min
    ylim_min -= 0.08 * ylim_range
    ylim_max += 0.08 * ylim_range

    # Reward zone (h ≈ 0, rank change > 0)
    ax.axhspan(0, ylim_max, xmin=0, xmax=0.5 / (xlim_max / max(xlim_max, 0.01)),
               facecolor=COLOR_REWARD_ZONE, alpha=0.4, zorder=0)
    # Penalty zone (h > 0, rank change < 0)
    ax.axhspan(ylim_min, 0, xmin=0.5, xmax=1.0,
               facecolor=COLOR_PENALTY_ZONE, alpha=0.4, zorder=0)

    # Scatter: Affected first (background), then Satisfying (foreground)
    for cat_docs, color, label, marker, zorder in [
        (aff_docs, COLOR_AFFECTED, 'Affected', 'o', 2),
        (sat_docs, COLOR_SATISFYING, 'Satisfying', 'D', 3),
    ]:
        if not cat_docs:
            continue
        h_vals = [d['h'] for d in cat_docs]
        rank_changes = [d['rank_change'] for d in cat_docs]
        ax.scatter(h_vals, rank_changes, c=color, alpha=0.75, s=28,
                   label=f'{label} (n={len(cat_docs)})', zorder=zorder,
                   edgecolors='white', linewidths=0.3, marker=marker)

    # Binned trend lines
    for cat_docs, color in [(sat_docs, COLOR_SATISFYING), (aff_docs, COLOR_AFFECTED)]:
        if len(cat_docs) < 5:
            continue
        h_vals = np.array([d['h'] for d in cat_docs])
        rank_changes = np.array([d['rank_change'] for d in cat_docs])

        n_bins = min(15, max(3, len(cat_docs) // 8))
        bins = np.linspace(h_vals.min(), h_vals.max(), n_bins + 1)
        bin_centers, bin_means, bin_stds = [], [], []
        for j in range(n_bins):
            mask = (h_vals >= bins[j]) & (h_vals < bins[j + 1])
            if mask.sum() >= 2:
                bin_centers.append((bins[j] + bins[j + 1]) / 2)
                bin_means.append(rank_changes[mask].mean())
                bin_stds.append(rank_changes[mask].std())

        if bin_centers:
            ax.plot(bin_centers, bin_means, color=color, linewidth=2.5,
                    linestyle='-', alpha=0.9, zorder=4)
            # Confidence band
            bin_centers = np.array(bin_centers)
            bin_means = np.array(bin_means)
            bin_stds = np.array(bin_stds)
            ax.fill_between(bin_centers,
                            bin_means - 0.5 * bin_stds,
                            bin_means + 0.5 * bin_stds,
                            color=color, alpha=0.1, zorder=1)

    # Reference lines
    ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-', alpha=0.6, zorder=5)
    ax.axvline(x=0, color='black', linewidth=1.2, linestyle='--', alpha=0.7, zorder=5)

    # Annotate λ boundary
    ax.annotate('$\\lambda$ boundary', xy=(0, ylim_max * 0.95),
                fontsize=8, color='black', alpha=0.7,
                ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                xytext=(xlim_max * 0.05, ylim_max * 0.85))

    # Zone labels
    ax.text(xlim_max * 0.02, ylim_max * 0.7, 'Reward zone\n(promoted)',
            fontsize=8.5, color=COLOR_REWARD, alpha=0.8, fontweight='bold',
            ha='left', va='top')
    ax.text(xlim_max * 0.7, ylim_min * 0.75, 'Penalty zone\n(suppressed)',
            fontsize=8.5, color=COLOR_PENALTY, alpha=0.8, fontweight='bold',
            ha='center', va='bottom')

    # Axis labels and formatting
    ax.set_xlabel('Exclusion evidence $h(d) = [r(d) - \\lambda]_+$', fontsize=11)
    ax.set_ylabel('Rank change (base $\\rightarrow$ TRIX)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.5, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)

    # Tight layout
    plt.tight_layout()

    # Save with descriptive name
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.png")

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    logger.info(f"PDF saved to {pdf_path}")

    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    logger.info(f"PNG saved to {png_path}")


if __name__ == "__main__":
    main()
