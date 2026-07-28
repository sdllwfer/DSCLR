"""
Generate four alternative visualizations for the reward-penalty mechanism effect.

All use the same filtered FollowIR data (15 effective queries, RepLLaMA).

Variants:
  1. Rank Flow: base→TRACE rank movement for satisfying/affected docs
  2. Rank Shift Density: density plot of rank changes by category
  3. Base vs TRACE dual-rank scatter: diagonal reference
  4. Combined: density (top) + rank flow (bottom)

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_figure3_variants
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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "/home/luwa/Documents/DSCLR-remote/results/figure3/reward_penalty_followir_repllama_good_queries.json"
OUTPUT_DIR = "/home/luwa/Documents/DSCLR-remote/paper/AuthorKit27/AuthorKit27/Figures"

# Colors
COLOR_SATISFYING = '#1565C0'
COLOR_AFFECTED = '#C62828'
COLOR_SAT_LIGHT = '#90CAF9'
COLOR_AFF_LIGHT = '#EF9A9A'


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def get_docs_by_query(docs):
    """Group docs by query."""
    by_query = defaultdict(list)
    for d in docs:
        by_query[d['qid']].append(d)
    return by_query


# ============================================================
# Variant 1: Rank Flow
# ============================================================
def generate_rank_flow(data, output_name):
    docs = data['good_query_docs']
    by_query = get_docs_by_query(docs)

    # Pick the best query for illustration (highest AP delta with both sat & aff)
    best_qid = None
    best_score = -1
    for qid, q_docs in by_query.items():
        sat = [d for d in q_docs if d['category'] == 'constraint_satisfying']
        aff = [d for d in q_docs if d['category'] == 'constraint_affected']
        if not sat or not aff:
            continue
        # Score: avg rank change magnitude
        score = np.mean([d['rank_change'] for d in sat]) - np.mean([d['rank_change'] for d in aff])
        if score > best_score:
            best_score = score
            best_qid = qid

    if best_qid is None:
        logger.warning("No suitable query for rank flow")
        return

    q_docs = by_query[best_qid]
    logger.info(f"Rank Flow: using query {best_qid} (score={best_score:.1f})")

    # Only show docs in top-40 of either ranking
    top_k = 40
    base_sorted = sorted(q_docs, key=lambda d: d['base_rank'])
    trace_sorted = sorted(q_docs, key=lambda d: d['trace_rank'])
    base_topk = set(d['doc_id'] for d in base_sorted[:top_k])
    trace_topk = set(d['doc_id'] for d in trace_sorted[:top_k])
    shown_ids = base_topk | trace_topk

    # Filter to satisfying and affected in top-k range
    shown_docs = [d for d in q_docs if d['doc_id'] in shown_ids
                  and d['category'] in ('constraint_satisfying', 'constraint_affected')]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=300)

    # Draw flow lines
    for d in shown_docs:
        base_r = d['base_rank']
        trace_r = d['trace_rank']
        color = COLOR_SATISFYING if d['category'] == 'constraint_satisfying' else COLOR_AFFECTED
        alpha = 0.7
        lw = 1.5
        ax.plot([0, 1], [base_r, trace_r], color=color, alpha=alpha, linewidth=lw,
                solid_capstyle='round')
        # Dots at endpoints
        ax.scatter([0], [base_r], color=color, s=30, zorder=5, edgecolors='white', linewidths=0.5)
        ax.scatter([1], [trace_r], color=color, s=30, zorder=5, edgecolors='white', linewidths=0.5)

    # Rank axis (inverted: rank 1 at top)
    ax.set_ylim(top_k + 1, 0)
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Base\nRanking', 'TRACE\nRanking'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Rank position', fontsize=11)

    # Add rank numbers on both sides
    for rank in [1, 5, 10, 20, 30, 40]:
        if rank <= top_k:
            ax.annotate(str(rank), xy=(-0.08, rank), fontsize=7, ha='right', va='center', color='gray')
            ax.annotate(str(rank), xy=(1.08, rank), fontsize=7, ha='left', va='center', color='gray')

    # Horizontal grid lines
    for rank in [1, 5, 10, 20, 30, 40]:
        ax.axhline(y=rank, color='gray', linewidth=0.3, alpha=0.5, linestyle=':')

    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLOR_SATISFYING, linewidth=2, label='Satisfying (promoted ↑)'),
        Line2D([0], [0], color=COLOR_AFFECTED, linewidth=2, label='Affected (suppressed ↓)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])

    # Title
    task = shown_docs[0]['task'].replace('InstructionRetrieval', '')
    ax.set_title(f'Rank Flow: {task}/{best_qid}', fontsize=10, color='gray')

    plt.tight_layout()
    save(fig, output_name)


# ============================================================
# Variant 2: Rank Shift Density
# ============================================================
def generate_rank_shift_density(data, output_name):
    docs = data['good_query_docs']
    sat_rc = np.array([d['rank_change'] for d in docs if d['category'] == 'constraint_satisfying'])
    aff_rc = np.array([d['rank_change'] for d in docs if d['category'] == 'constraint_affected'])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=300)

    # Density plots using histogram + KDE-like smooth
    from scipy.stats import gaussian_kde

    x_range = np.linspace(min(aff_rc.min(), sat_rc.min()) - 10,
                          max(aff_rc.max(), sat_rc.max()) + 10, 500)

    if len(sat_rc) > 2:
        kde_sat = gaussian_kde(sat_rc, bw_method=0.3)
        density_sat = kde_sat(x_range)
        ax.fill_between(x_range, density_sat, alpha=0.35, color=COLOR_SATISFYING, zorder=2)
        ax.plot(x_range, density_sat, color=COLOR_SATISFYING, linewidth=2, zorder=3,
                label=f'Satisfying (n={len(sat_rc)}, median={np.median(sat_rc):+.0f})')

    if len(aff_rc) > 2:
        kde_aff = gaussian_kde(aff_rc, bw_method=0.3)
        density_aff = kde_aff(x_range)
        ax.fill_between(x_range, density_aff, alpha=0.35, color=COLOR_AFFECTED, zorder=2)
        ax.plot(x_range, density_aff, color=COLOR_AFFECTED, linewidth=2, zorder=3,
                label=f'Affected (n={len(aff_rc)}, median={np.median(aff_rc):+.0f})')

    # Zero line
    ax.axvline(x=0, color='black', linewidth=1.0, linestyle='--', alpha=0.6, zorder=4)

    # Annotate regions
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[0] * 0.6, ylim[1] * 0.85, '← Suppressed\n(penalty zone)',
            fontsize=9, color=COLOR_AFFECTED, alpha=0.8, fontweight='bold',
            ha='center', va='top')
    ax.text(xlim[1] * 0.5, ylim[1] * 0.85, 'Promoted →\n(reward zone)',
            fontsize=9, color=COLOR_SATISFYING, alpha=0.8, fontweight='bold',
            ha='center', va='top')

    ax.set_xlabel('Rank change (base → TRACE)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='upper center', fontsize=9, framealpha=0.95, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save(fig, output_name)


# ============================================================
# Variant 3: Base vs TRACE Dual-Rank Scatter
# ============================================================
def generate_dual_rank_scatter(data, output_name):
    docs = data['good_query_docs']
    sat_docs = [d for d in docs if d['category'] == 'constraint_satisfying']
    aff_docs = [d for d in docs if d['category'] == 'constraint_affected']

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6), dpi=300)

    max_rank = 1000

    # Diagonal (no change)
    ax.plot([1, max_rank], [1, max_rank], color='gray', linewidth=1.0,
            linestyle='--', alpha=0.5, zorder=1, label='No change')

    # Shaded regions
    ax.fill_between([1, max_rank], [1, 1], [1, max_rank],
                    color=COLOR_SATISFYING, alpha=0.04, zorder=0)  # Above diagonal = improved
    ax.fill_between([1, max_rank], [1, max_rank], [max_rank, max_rank],
                    color=COLOR_AFFECTED, alpha=0.04, zorder=0)  # Below = dropped

    # Scatter
    for cat_docs, color, marker, label in [
        (aff_docs, COLOR_AFFECTED, 'o',
         f'Affected (n={len(aff_docs)}, median Δ={np.median([d["rank_change"] for d in aff_docs]):+.0f})'),
        (sat_docs, COLOR_SATISFYING, 'D',
         f'Satisfying (n={len(sat_docs)}, median Δ={np.median([d["rank_change"] for d in sat_docs]):+.0f})'),
    ]:
        base_ranks = [d['base_rank'] for d in cat_docs]
        trace_ranks = [d['trace_rank'] for d in cat_docs]
        ax.scatter(base_ranks, trace_ranks, c=color, alpha=0.6, s=25,
                   marker=marker, edgecolors='white', linewidths=0.3,
                   zorder=3, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.8, max_rank * 1.2)
    ax.set_ylim(0.8, max_rank * 1.2)
    ax.set_xlabel('Base rank', fontsize=11)
    ax.set_ylabel('TRACE rank', fontsize=11)

    # Region labels
    ax.text(3, 500, 'Suppressed\n(rank ↓)', fontsize=9, color=COLOR_AFFECTED,
            alpha=0.7, fontweight='bold', ha='center', va='center')
    ax.text(500, 3, 'Promoted\n(rank ↑)', fontsize=9, color=COLOR_SATISFYING,
            alpha=0.7, fontweight='bold', ha='center', va='center')

    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save(fig, output_name)


# ============================================================
# Variant 4: Combined Density + Rank Flow
# ============================================================
def generate_combined(data, output_name):
    docs = data['good_query_docs']
    by_query = get_docs_by_query(docs)

    sat_rc = np.array([d['rank_change'] for d in docs if d['category'] == 'constraint_satisfying'])
    aff_rc = np.array([d['rank_change'] for d in docs if d['category'] == 'constraint_affected'])

    # Pick best query for flow
    best_qid = None
    best_score = -1
    for qid, q_docs in by_query.items():
        sat = [d for d in q_docs if d['category'] == 'constraint_satisfying']
        aff = [d for d in q_docs if d['category'] == 'constraint_affected']
        if not sat or not aff:
            continue
        score = np.mean([d['rank_change'] for d in sat]) - np.mean([d['rank_change'] for d in aff])
        if score > best_score:
            best_score = score
            best_qid = qid

    fig = plt.figure(figsize=(7, 7.5), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.3)

    # ---- Top: Density ----
    ax1 = fig.add_subplot(gs[0])
    from scipy.stats import gaussian_kde
    x_range = np.linspace(min(aff_rc.min(), sat_rc.min()) - 10,
                          max(aff_rc.max(), sat_rc.max()) + 10, 500)
    if len(sat_rc) > 2:
        kde_sat = gaussian_kde(sat_rc, bw_method=0.3)
        density_sat = kde_sat(x_range)
        ax1.fill_between(x_range, density_sat, alpha=0.35, color=COLOR_SATISFYING)
        ax1.plot(x_range, density_sat, color=COLOR_SATISFYING, linewidth=2,
                 label=f'Satisfying (n={len(sat_rc)})')
    if len(aff_rc) > 2:
        kde_aff = gaussian_kde(aff_rc, bw_method=0.3)
        density_aff = kde_aff(x_range)
        ax1.fill_between(x_range, density_aff, alpha=0.35, color=COLOR_AFFECTED)
        ax1.plot(x_range, density_aff, color=COLOR_AFFECTED, linewidth=2,
                 label=f'Affected (n={len(aff_rc)})')
    ax1.axvline(x=0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
    ax1.set_xlabel('Rank change (base → TRACE)', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend(loc='upper center', fontsize=9, framealpha=0.95, ncol=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Distribution of rank changes', fontsize=11, fontweight='bold')

    # ---- Bottom: Rank Flow ----
    if best_qid:
        ax2 = fig.add_subplot(gs[1])
        q_docs = by_query[best_qid]
        top_k = 30
        base_sorted = sorted(q_docs, key=lambda d: d['base_rank'])
        trace_sorted = sorted(q_docs, key=lambda d: d['trace_rank'])
        base_topk = set(d['doc_id'] for d in base_sorted[:top_k])
        trace_topk = set(d['doc_id'] for d in trace_sorted[:top_k])
        shown_ids = base_topk | trace_topk
        shown_docs = [d for d in q_docs if d['doc_id'] in shown_ids
                      and d['category'] in ('constraint_satisfying', 'constraint_affected')]

        for d in shown_docs:
            base_r = d['base_rank']
            trace_r = d['trace_rank']
            color = COLOR_SATISFYING if d['category'] == 'constraint_satisfying' else COLOR_AFFECTED
            ax2.plot([0, 1], [base_r, trace_r], color=color, alpha=0.7, linewidth=1.5,
                     solid_capstyle='round')
            ax2.scatter([0], [base_r], color=color, s=30, zorder=5, edgecolors='white', linewidths=0.5)
            ax2.scatter([1], [trace_r], color=color, s=30, zorder=5, edgecolors='white', linewidths=0.5)

        ax2.set_ylim(top_k + 1, 0)
        ax2.set_xlim(-0.3, 1.3)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Base\nRanking', 'TRACE\nRanking'], fontsize=10, fontweight='bold')
        ax2.set_ylabel('Rank position', fontsize=10)
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        for rank in [1, 5, 10, 20, 30]:
            if rank <= top_k:
                ax2.axhline(y=rank, color='gray', linewidth=0.3, alpha=0.5, linestyle=':')
                ax2.annotate(str(rank), xy=(-0.08, rank), fontsize=7, ha='right', va='center', color='gray')
                ax2.annotate(str(rank), xy=(1.08, rank), fontsize=7, ha='left', va='center', color='gray')

        task = shown_docs[0]['task'].replace('InstructionRetrieval', '') if shown_docs else ''
        legend_elements = [
            Line2D([0], [0], color=COLOR_SATISFYING, linewidth=2, label='Satisfying (promoted ↑)'),
            Line2D([0], [0], color=COLOR_AFFECTED, linewidth=2, label='Affected (suppressed ↓)'),
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)
        ax2.set_title(f'(b) Rank flow for {task}/{best_qid}', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save(fig, output_name)


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {pdf_path}")
    logger.info(f"Saved: {png_path}")


def main():
    data = load_data()
    docs = data['good_query_docs']
    n_sat = sum(1 for d in docs if d['category'] == 'constraint_satisfying')
    n_aff = sum(1 for d in docs if d['category'] == 'constraint_affected')
    logger.info(f"Data: {len(docs)} docs, {n_sat} satisfying, {n_aff} affected")

    generate_rank_flow(data, "rank_flow_followir_repllama_v2")
    generate_rank_shift_density(data, "rank_shift_density_followir_repllama_v2")
    generate_dual_rank_scatter(data, "dual_rank_scatter_followir_repllama_v2")
    generate_combined(data, "combined_density_flow_followir_repllama_v2")


if __name__ == "__main__":
    main()
