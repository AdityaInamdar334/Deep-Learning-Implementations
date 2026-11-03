"""Generate publication-ready figures for Weight-Space Programming Lab.

Produces:
1. figures/01_weight_space_architecture.png: Weight-space inspection, permutation symmetry, & surgical editing
2. figures/02_sparse_edit_heldout.png: Sparse weight repair vs simple negative controls
3. figures/03_weight_reading_symmetry.png: Permutation-invariant vs brittle weight-reading
4. figures/04_distribution_shift_robustness.png: Generalization under nuisance distribution shift
5. figures/05_navigation_rollout_dynamics.png: Closed-loop navigation episode success rates
"""
import json
import os
import numpy as np
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

STYLE = {
    'bg_fig': '#0B0F19',       # Deep Navy / Slate 950
    'bg_card': '#111827',      # Slate 900
    'bg_card_alt': '#1E293B',  # Slate 800
    'text_main': '#F9FAFB',    # Gray 50
    'text_muted': '#9CA3AF',   # Gray 400
    'text_sub': '#D1D5DB',     # Gray 300
    'grid': '#1F2937',         # Slate 800 grid
    'cyan': '#06B6D4',         # Cyan 500
    'sky': '#38BDF8',          # Sky 400
    'blue': '#3B82F6',         # Blue 500
    'emerald': '#10B981',      # Emerald 500
    'amber': '#F59E0B',        # Amber 500
    'rose': '#F43F5E',         # Rose 500
    'purple': '#A855F7',       # Purple 500
}


def draw_architecture_diagram(output_path: Path):
    """Draw weight-space programming inspection, permutation symmetries, and surgical repair."""
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_fig'])
    ax.axis('off')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)

    # Title & Subtitle
    ax.text(7.5, 9.4, "Weight-Space Programming: Direct Weight Inspection & Gradient-Free Editing",
            color=STYLE['text_main'], fontsize=15, fontweight='bold', ha='center')
    ax.text(7.5, 8.95, "Inspired by MIT CSAIL's Sparks of Weight-Space Programming (Takehi, Teoh, Zhang, Khattab, Nov 2026)",
            color=STYLE['text_muted'], fontsize=10, ha='center')

    def draw_box(x, y, w, h, title, subtitle, bg, border, title_color=STYLE['text_main']):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.25",
                                      facecolor=bg, edgecolor=border, linewidth=1.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.18, title, color=title_color, fontsize=10.5, fontweight='bold', ha='center', va='center', zorder=3)
        ax.text(x, y - 0.22, subtitle, color=STYLE['text_sub'], fontsize=8.5, ha='center', va='center', zorder=3)

    def draw_arrow(x1, y1, x2, y2, color=STYLE['cyan'], lw=1.8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=4)

    # Column 1: Weight Reading & Symmetry Challenge
    draw_box(2.4, 7.2, 4.0, 1.4,
             "Neural Weight Tensors", "Raw parameters W_1, W_2, b_1, b_2\n(No inputs, no training data)",
             bg=STYLE['bg_card_alt'], border=STYLE['blue'])
    draw_box(2.4, 5.2, 4.0, 1.4,
             "Weight-Space Symmetries", "Permuting hidden neurons preserves\ninput-output function exactly",
             bg="#27192A", border=STYLE['amber'], title_color="#FDE68A")
    draw_box(2.4, 3.2, 4.0, 1.4,
             "Permutation-Invariant Reader", "Identifies task circuits regardless of\nhidden neuron order (100% accuracy)",
             bg="#062E25", border=STYLE['emerald'], title_color="#A7F3D0")

    draw_arrow(4.4, 5.2, 5.5, 5.2, color=STYLE['cyan'])

    # Column 2: Corrupted Circuit Identification
    repair_box = patches.FancyBboxPatch((5.5, 2.2), 4.0, 5.6, boxstyle="round,pad=0.3",
                                        facecolor="#0F172A", edgecolor=STYLE['rose'], linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(repair_box)
    ax.text(7.5, 7.4, "Corrupted Target Policy", color=STYLE['rose'], fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, 7.0, "(Pathological Stall Pathway)", color=STYLE['text_muted'], fontsize=8.5, ha='center')

    draw_box(7.5, 6.1, 3.4, 0.9, "Spurious Output Weights", "Target unit diverted into loop stall", bg="#3B1219", border=STYLE['rose'], title_color="#FDA4AF")
    draw_box(7.5, 4.8, 3.4, 0.9, "Corrupted Holdout Acc", "29.8% familiar, 12.3% shifted", bg=STYLE['bg_card_alt'], border="#64748B")
    draw_box(7.5, 3.5, 3.4, 0.9, "Surgical Intervention", "Zero out 2 of 131 weights (1.5%)", bg=STYLE['bg_card_alt'], border=STYLE['cyan'])

    draw_arrow(9.5, 5.2, 10.6, 5.2, color=STYLE['cyan'])

    # Column 3: Verification & Negative Controls
    draw_box(12.7, 7.0, 3.8, 1.4,
             "Surgical Semantic Edit", "Exact circuit repair without backprop\n(100% accuracy, 0% stall)",
             bg="#062E25", border=STYLE['emerald'], title_color="#A7F3D0")

    draw_box(12.7, 5.1, 3.8, 1.4,
             "Essential Negative Control", "Simple magnitude pruning matches\n100% accuracy without an LLM!",
             bg="#1E1B4B", border=STYLE['purple'], title_color="#E9D5FF")

    draw_box(12.7, 3.2, 3.8, 1.4,
             "Nuisance Distribution Shift", "Evaluates generalization on unseen\nsamples with orthogonal noise",
             bg=STYLE['bg_card_alt'], border=STYLE['sky'])

    # Final Stage: Independent Evaluation
    outcome_box = patches.FancyBboxPatch((1.0, 0.5), 13.0, 1.5, boxstyle="round,pad=0.25",
                                         facecolor=STYLE['bg_card'], edgecolor=STYLE['emerald'], linewidth=2.0, zorder=2)
    ax.add_patch(outcome_box)
    ax.text(7.5, 1.5, "INDEPENDENT EVALUATION & CAUSAL AUDITING", color=STYLE['emerald'], fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, 1.1, "Labels generated AFTER edits are selected  |  Immutable base model backups  |  Tensor-by-tensor diff auditor",
            color=STYLE['text_main'], fontsize=9, ha='center')
    ax.text(7.5, 0.7, "Distinguishes repairing an execution failure from learning new mathematical capabilities!",
            color=STYLE['text_sub'], fontsize=8.5, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_sparse_edit_heldout(root: Path, output_path: Path):
    """Draw held-out accuracy of semantic repair vs controls."""
    data = json.loads((root / 'results/benchmark.json').read_text())['summary']
    policies = data['policies']

    ids = ['original', 'semantic', 'top_magnitude', 'half_semantic', 'random_two', 'global_shrink', 'wrong_path']
    labels = ['Corrupted\n(Original)', 'Semantic\n(2 Weights)', 'Magnitude\nPruning', 'Partial Edit\n(1 Weight)',
              'Random Edit\n(2 Weights)', 'Global\nShrinkage', 'Wrong Path\n(2 Weights)']

    acc = [policies[k]['accuracy']['mean'] * 100 for k in ids]
    err = [policies[k]['accuracy']['sample_sd'] * 100 for k in ids]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['rose'], STYLE['emerald'], STYLE['purple'], STYLE['amber'], STYLE['sky'], '#64748B', '#DC2626']

    bars = ax.bar(range(len(ids)), acc, yerr=err, capsize=4, color=colors,
                  width=0.55, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)

    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2.0, f"{val:.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Held-Out Accuracy (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Sparse Weight Repair vs Statistical Controls (3,000 Unseen Samples / Seed)',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])

    # Annotate Magnitude baseline
    ax.annotate('Magnitude Baseline\nMatches Semantic Edit (100%)', xy=(2, acc[2]), xytext=(2.8, 80),
                 arrowprops=dict(arrowstyle="->", color=STYLE['purple'], lw=1.8),
                 color=STYLE['purple'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['purple']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_weight_reading_symmetry(root: Path, output_path: Path):
    """Draw hidden-neuron permutation robustness for weight-only reading."""
    data = json.loads((root / 'results/benchmark.json').read_text())['summary']
    readers = data['readers']

    labels = ['Naive First-Neuron Heuristic\n(Brittle to Permutations)', 'Permutation-Invariant Reader\n(Aggregated Input Symmetries)']
    vals = [readers['naive']['mean'] * 100, readers['permutation_invariant']['mean'] * 100]
    errs = [readers['naive']['sample_sd'] * 100, readers['permutation_invariant']['sample_sd'] * 100]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['rose'], STYLE['emerald']]

    bars = ax.bar(range(len(labels)), vals, yerr=errs, capsize=4, color=colors,
                  width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2.0, f"{val:.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=11, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Weight-Only Task Identification (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Hidden-Neuron Permutation Defeats Brittle Weight Readers (900 Permuted Variants)',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10.5, color=STYLE['text_main'], fontweight='bold')

    ax.annotate('Brittle Heuristic Collapses\nUnder Symmetries (-61.9%)', xy=(0, vals[0]), xytext=(0.4, 55),
                 arrowprops=dict(arrowstyle="->", color=STYLE['rose'], lw=1.8),
                 color=STYLE['rose'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['rose']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_distribution_shift(root: Path, output_path: Path):
    """Draw familiar vs nuisance-shifted dataset accuracy."""
    data = json.loads((root / 'results/benchmark.json').read_text())['summary']
    policies = data['policies']

    ids = ['original', 'half_semantic', 'random_two', 'semantic', 'top_magnitude']
    labels = ['Corrupted\n(Original)', 'Partial Edit\n(1 Weight)', 'Random Edit\n(2 Weights)', 'Semantic Edit\n(2 Weights)', 'Magnitude\nPruning']

    base = [policies[k]['accuracy']['mean'] * 100 for k in ids]
    shift = [policies[k]['shift_accuracy']['mean'] * 100 for k in ids]

    x = np.arange(len(ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    rects1 = ax.bar(x - width/2, base, width, label='Familiar Data', color=STYLE['blue'], edgecolor='#FFFFFF', linewidth=0.5)
    rects2 = ax.bar(x + width/2, shift, width, label='Nuisance-Shifted Data', color=STYLE['cyan'], edgecolor='#FFFFFF', linewidth=0.5)

    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=9.5, fontweight='bold')
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=9.5, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Action Accuracy (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Generalization Under Independent Distribution Shift (3,000 Shifted Samples / Seed)',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=STYLE['text_main'])
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=9.5)

    # Note unpatched collapse
    ax.annotate('Corrupted Model Collapses\nUnder Shift (29.8% -> 12.3%)', xy=(0, base[0]), xytext=(0.8, 45),
                 arrowprops=dict(arrowstyle="->", color=STYLE['rose'], lw=1.8),
                 color=STYLE['rose'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['rose']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_navigation_rollouts(root: Path, output_path: Path):
    """Draw closed-loop navigation episode success rates."""
    data = json.loads((root / 'results/benchmark.json').read_text())['summary']
    policies = data['policies']

    ids = ['original', 'wrong_path', 'random_two', 'global_shrink', 'half_semantic', 'semantic', 'top_magnitude']
    labels = ['Corrupted\n(Original)', 'Wrong Path\n(2 Weights)', 'Random Edit\n(2 Weights)', 'Global\nShrinkage',
              'Partial Edit\n(1 Weight)', 'Semantic Edit\n(2 Weights)', 'Magnitude\nPruning']

    success = [policies[k]['navigation_success']['mean'] * 100 for k in ids]
    err = [policies[k]['navigation_success']['sample_sd'] * 100 for k in ids]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['rose'], '#DC2626', STYLE['sky'], '#64748B', STYLE['amber'], STYLE['emerald'], STYLE['purple']]

    bars = ax.bar(range(len(ids)), success, yerr=err, capsize=4, color=colors,
                  width=0.55, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)

    for bar, val in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2.0, f"{val:.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Navigation Episode Success (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Closed-Loop Navigation Performance Across Editing Policies (300 Episodes / Seed)',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])

    # Highlight 100% success
    ax.annotate('100% Navigation Success\n(Zero Runaway Stalls)', xy=(5, success[5]), xytext=(4.0, 75),
                 arrowprops=dict(arrowstyle="->", color=STYLE['emerald'], lw=1.8),
                 color=STYLE['emerald'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['emerald']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def main():
    root = Path(__file__).resolve().parent
    fig_dir = root / 'figures'
    fig_dir.mkdir(exist_ok=True)

    draw_architecture_diagram(fig_dir / '01_weight_space_architecture.png')
    draw_sparse_edit_heldout(root, fig_dir / '02_sparse_edit_heldout.png')
    draw_weight_reading_symmetry(root, fig_dir / '03_weight_reading_symmetry.png')
    draw_distribution_shift(root, fig_dir / '04_distribution_shift_robustness.png')
    draw_navigation_rollouts(root, fig_dir / '05_navigation_rollout_dynamics.png')
    print("All Weight-Space Programming figures successfully created in figures/")


if __name__ == '__main__':
    main()
