"""Generate publication-ready figures for Primer EZ Transformer Lab.

Produces:
1. figures/01_primer_ez_architecture.png: Vanilla Transformer vs Primer EZ architecture & DConv block
2. figures/02_accuracy_vs_shift.png: In-distribution vs structural distribution shift accuracy
3. figures/03_quality_vs_shift.png: Negative log-likelihood (NLL) across 9 configurations
4. figures/04_quality_vs_wall_time.png: Time-to-quality efficiency & 390-step equal-time control
5. figures/05_conv_ablation_transfer.png: Q vs K vs V projection convolution transfer dynamics
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
    """Draw side-by-side comparison of Vanilla Transformer vs Primer EZ architecture."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_fig'])
    ax2.set_facecolor(STYLE['bg_fig'])
    ax1.axis('off')
    ax2.axis('off')

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 13)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 13)

    # Headers
    ax1.text(5, 12.3, "Vanilla Transformer Block",
             color=STYLE['sky'], fontsize=14, fontweight='bold', ha='center')
    ax1.text(5, 11.7, "Standard multi-head attention + ReLU / GELU feed-forward network",
             color=STYLE['text_muted'], fontsize=9.5, ha='center')

    ax2.text(5, 12.3, "Primer EZ Transformer Block",
             color=STYLE['emerald'], fontsize=14, fontweight='bold', ha='center')
    ax2.text(5, 11.7, "Multi-DConv-Head Attention + Squared ReLU FFN (NeurIPS 2021)",
             color=STYLE['text_muted'], fontsize=9.5, ha='center')

    def draw_box(ax, x, y, w, h, title, subtitle, bg, border, title_color=STYLE['text_main']):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.25",
                                      facecolor=bg, edgecolor=border, linewidth=1.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.18, title, color=title_color, fontsize=10.5, fontweight='bold', ha='center', va='center', zorder=3)
        ax.text(x, y - 0.22, subtitle, color=STYLE['text_sub'], fontsize=8.5, ha='center', va='center', zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2, color=STYLE['cyan']):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0), zorder=4)

    # --- LEFT: VANILLA TRANSFORMER ---
    draw_box(ax1, 5, 10.3, 8.4, 1.2,
             "Input Token Embeddings + Positional Encoding", "Sequence tensor: [Batch, Length, D_model]",
             bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(ax1, 5, 9.6, 5, 8.8, color="#64748B")

    draw_box(ax1, 5, 8.1, 8.4, 1.2,
             "Linear Projections (Q, K, V)", "Direct linear transformations without local temporal mixing",
             bg=STYLE['bg_card_alt'], border=STYLE['blue'])

    draw_arrow(ax1, 5, 7.4, 5, 6.6, color=STYLE['blue'])

    draw_box(ax1, 5, 5.9, 8.4, 1.2,
             "Standard Causal Self-Attention", "Softmax(Q K^T / sqrt(d_k)) V with autoregressive lower-triangular mask",
             bg=STYLE['bg_card'], border=STYLE['blue'])

    draw_arrow(ax1, 5, 5.2, 5, 4.4, color="#64748B")

    draw_box(ax1, 5, 3.7, 8.4, 1.2,
             "Feed-Forward Network (ReLU / GELU)", "Linear -> max(0, x) -> Linear  |  Flat first-derivative saturation",
             bg=STYLE['bg_card_alt'], border="#64748B")

    outcome_left = patches.FancyBboxPatch((0.8, 0.6), 8.4, 2.1, boxstyle="round,pad=0.3",
                                         facecolor="#1A202C", edgecolor=STYLE['sky'], linewidth=2.0, zorder=2)
    ax1.add_patch(outcome_left)
    ax1.text(5, 2.1, "TRAINING CONVERGENCE & COMPUTE", color=STYLE['sky'], fontsize=11, fontweight='bold', ha='center')
    ax1.text(5, 1.55, "In-Distribution Acc: 34.3%  |  Loss: 1.615 NLL", color=STYLE['text_main'], fontsize=11, fontweight='bold', ha='center')
    ax1.text(5, 1.0, "Slow convergence; requires 4x more training tokens on repetitive motifs", color=STYLE['text_sub'], fontsize=9, ha='center')

    # --- RIGHT: PRIMER EZ TRANSFORMER ---
    draw_box(ax2, 5, 10.3, 8.4, 1.2,
             "Input Token Embeddings + Positional Encoding", "Sequence tensor: [Batch, Length, D_model]",
             bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(ax2, 5, 9.6, 5, 8.8, color="#64748B")

    draw_box(ax2, 5, 8.1, 8.4, 1.2,
             "Multi-DConv-Head Attention (MDHA)", "Causal width-3 depthwise conv: tokens [t-2, t-1, t] mixed before attention",
             bg="#062E25", border=STYLE['emerald'], title_color="#6EE7B7")

    draw_arrow(ax2, 5, 7.4, 5, 6.6, color=STYLE['emerald'])

    draw_box(ax2, 5, 5.9, 8.4, 1.2,
             "Causal Self-Attention on Convolved Q, K, V", "Softmax(Q_conv K_conv^T / sqrt(d_k)) V_conv with causal mask",
             bg=STYLE['bg_card'], border=STYLE['emerald'])

    draw_arrow(ax2, 5, 5.2, 5, 4.4, color=STYLE['purple'])

    draw_box(ax2, 5, 3.7, 8.4, 1.2,
             "Squared ReLU FFN: max(0, x)^2", "Non-linear gradient scaling: 2 * max(0, x) accelerates early optimization",
             bg="#1E1B4B", border=STYLE['purple'], title_color="#E9D5FF")

    outcome_right = patches.FancyBboxPatch((0.8, 0.6), 8.4, 2.1, boxstyle="round,pad=0.3",
                                          facecolor="#0B231D", edgecolor=STYLE['emerald'], linewidth=2.0, zorder=2)
    ax2.add_patch(outcome_right)
    ax2.text(5, 2.1, "TRAINING CONVERGENCE & COMPUTE", color=STYLE['emerald'], fontsize=11, fontweight='bold', ha='center')
    ax2.text(5, 1.55, "In-Distribution Acc: 95.7%  |  Loss: 0.251 NLL", color="#A7F3D0", fontsize=11, fontweight='bold', ha='center')
    ax2.text(5, 1.0, "Rapid learning; 4.2x compute efficiency reported in original Google research", color=STYLE['text_sub'], fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_accuracy_vs_shift(root: Path, output_path: Path):
    """Draw in-distribution vs structural distribution shift accuracy across 9 configurations."""
    report = json.loads((root / 'results/benchmark.json').read_text())
    summary = report['summary']
    order = ['baseline_relu', 'gelu_baseline', 'squared_relu_only', 'dconv_only',
             'primer_ez', 'primer_per_head', 'q_only', 'k_only', 'v_only']
    labels = ['ReLU\nBaseline', 'GELU\nBaseline', 'Squared\nReLU Only', 'QKV Conv\nOnly',
              'Primer EZ\n(Full)', 'Primer\nPer-Head', 'Q-Only\nConv', 'K-Only\nConv', 'V-Only\nConv']

    x = np.arange(len(order))
    width = 0.38

    id_acc = [summary[n]['id_accuracy']['mean'] * 100 for n in order]
    ood_acc = [summary[n]['ood_accuracy']['mean'] * 100 for n in order]

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    rects1 = ax.bar(x - width/2, id_acc, width, label='In-Distribution (Period 4)',
                    color=STYLE['emerald'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.92)
    rects2 = ax.bar(x + width/2, ood_acc, width, label='Structural Shift (Period 4 -> 6)',
                    color=STYLE['rose'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.92)

    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=8.5, fontweight='bold')
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=8.5, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Next-Token Accuracy (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('In-Distribution Excellence vs Structural Distribution Shift (3 Seeds x 260 Steps)',
                 color=STYLE['text_main'], fontsize=13, fontweight='bold', pad=14)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=10, loc='upper left')

    # Highlight K-Only generalization
    ax.annotate('K-Only Conv Robustness!\n(77.2% OOD Accuracy)', xy=(7 + width/2, ood_acc[7]), xytext=(7.0, 95),
                 arrowprops=dict(arrowstyle="->", color=STYLE['cyan'], lw=1.8),
                 color=STYLE['cyan'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['cyan']))

    # Highlight Primer EZ OOD Collapse
    ax.annotate('Primer EZ OOD Collapse\n(95.7% -> 7.5%!)', xy=(4 + width/2, ood_acc[4]), xytext=(3.6, 50),
                 arrowprops=dict(arrowstyle="->", color=STYLE['amber'], lw=1.8),
                 color=STYLE['amber'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['amber']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_quality_vs_shift(root: Path, output_path: Path):
    """Draw next-token negative log-likelihood (NLL) across 9 configurations."""
    report = json.loads((root / 'results/benchmark.json').read_text())
    summary = report['summary']
    order = ['baseline_relu', 'gelu_baseline', 'squared_relu_only', 'dconv_only',
             'primer_ez', 'primer_per_head', 'q_only', 'k_only', 'v_only']
    labels = ['ReLU\nBaseline', 'GELU\nBaseline', 'Squared\nReLU Only', 'QKV Conv\nOnly',
              'Primer EZ\n(Full)', 'Primer\nPer-Head', 'Q-Only\nConv', 'K-Only\nConv', 'V-Only\nConv']

    x = np.arange(len(order))
    width = 0.38

    id_nll = [summary[n]['id_nll']['mean'] for n in order]
    ood_nll = [summary[n]['ood_nll']['mean'] for n in order]

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    rects1 = ax.bar(x - width/2, id_nll, width, label='In-Distribution NLL (Lower is Better)',
                    color=STYLE['sky'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.92)
    rects2 = ax.bar(x + width/2, ood_nll, width, label='Shifted Period NLL (Period 4 -> 6)',
                    color=STYLE['rose'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.92)

    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.1, f"{r.get_height():.2f}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=8.5, fontweight='bold')
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.1, f"{r.get_height():.2f}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=8.5, fontweight='bold')

    ax.set_ylim(0, 7.2)
    ax.set_ylabel('Next-Token Negative Log-Likelihood (NLL)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Cross-Entropy Loss: In-Distribution Optimization vs OOD Fragility',
                 color=STYLE['text_main'], fontsize=13, fontweight='bold', pad=14)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_quality_vs_wall_time(root: Path, output_path: Path):
    """Draw in-distribution loss vs wall-clock CPU training time, including equal-time 390-step control."""
    report = json.loads((root / 'results/benchmark.json').read_text())
    summary = report['summary']
    order = ['baseline_relu', 'gelu_baseline', 'squared_relu_only', 'dconv_only',
             'primer_ez', 'primer_per_head', 'q_only', 'k_only', 'v_only']
    labels = ['ReLU (260 steps)', 'GELU Baseline', 'Squared ReLU Only', 'QKV Conv Only',
              'Primer EZ (Full)', 'Primer Per-Head', 'Q-Only Conv', 'K-Only Conv', 'V-Only Conv']

    times = [summary[n]['time_seconds']['mean'] for n in order]
    nlls = [summary[n]['id_nll']['mean'] for n in order]

    extra = json.loads((root / 'results/baseline_extra_steps.json').read_text())['summary']['baseline_relu']
    extra_time = extra['time_seconds']['mean']
    extra_nll = extra['id_nll']['mean']

    fig, ax = plt.subplots(figsize=(12, 6.0))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    scatter_colors = [STYLE['rose'], '#64748B', STYLE['purple'], STYLE['sky'],
                      STYLE['emerald'], STYLE['cyan'], STYLE['amber'], STYLE['blue'], '#E11D48']

    for t, n, l, c in zip(times, nlls, labels, scatter_colors):
        ax.scatter(t, n, color=c, s=180, edgecolor='#FFFFFF', linewidth=1.2, zorder=3, label=l)
        y_off = 0.06 if 'QKV' not in l and 'Primer EZ' not in l else -0.09
        ax.text(t, n + y_off, l, color=STYLE['text_main'], fontsize=9, fontweight='bold', ha='center', zorder=4)

    # Plot 390-step equal-time control
    ax.scatter([extra_time], [extra_nll], color='#F43F5E', marker='X', s=240, edgecolor='#FFFFFF', linewidth=1.5, zorder=5,
               label='ReLU Baseline (390 steps - Equal Wall Clock)')
    ax.text(extra_time, extra_nll + 0.07, 'ReLU Baseline\n(390 steps - Equal Time)', color='#FDA4AF',
            fontsize=9.5, fontweight='bold', ha='center', zorder=6)

    ax.set_xlim(1.8, 3.6)
    ax.set_ylim(0.1, 1.8)
    ax.set_xlabel('Mean Wall-Clock Training Time on Test CPU (Seconds)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_ylabel('In-Distribution Next-Token NLL (Lower is Better)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Compute Efficiency: Time-to-Quality vs Raw Step Speed',
                 color=STYLE['text_main'], fontsize=13, fontweight='bold', pad=14)
    ax.grid(True, linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=9, loc='upper right')

    # Annotate that training longer does not bridge the architectural gap
    ax.annotate('Equal-Time ReLU Control Stagnates\n(1.497 NLL vs Primer EZ 0.251 NLL)', xy=(extra_time, extra_nll), xytext=(2.6, 1.05),
                 arrowprops=dict(arrowstyle="->", color=STYLE['amber'], lw=1.8),
                 color=STYLE['amber'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['amber']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_conv_ablation_transfer(root: Path, output_path: Path):
    """Draw projection convolution ablation: Q vs K vs V vs QKV transfer dynamics."""
    report = json.loads((root / 'results/benchmark.json').read_text())
    summary = report['summary']

    keys = ['dconv_only', 'q_only', 'k_only', 'v_only', 'baseline_relu']
    labels = ['All Projections\n(Q + K + V)', 'Query Only\n(Q-Only)', 'Key Only\n(K-Only)', 'Value Only\n(V-Only)', 'Vanilla\n(No Conv)']

    id_acc = [summary[k]['id_accuracy']['mean'] * 100 for k in keys]
    ood_acc = [summary[k]['ood_accuracy']['mean'] * 100 for k in keys]

    x = np.arange(len(keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    rects1 = ax.bar(x - width/2, id_acc, width, label='In-Distribution Accuracy',
                    color=STYLE['emerald'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.9)
    rects2 = ax.bar(x + width/2, ood_acc, width, label='Shifted OOD Accuracy',
                    color=STYLE['purple'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.9)

    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Accuracy (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Dissecting Multi-DConv-Head Attention: Where Does Generalization Reside?',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=STYLE['text_main'])
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=10, loc='upper right')

    ax.annotate('Key Convolution Preserves Transfer!\n(77.2% OOD accuracy)', xy=(2 + width/2, ood_acc[2]), xytext=(1.8, 92),
                 arrowprops=dict(arrowstyle="->", color=STYLE['cyan'], lw=1.8),
                 color=STYLE['cyan'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['cyan']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def main():
    root = Path(__file__).resolve().parent
    fig_dir = root / 'figures'
    fig_dir.mkdir(exist_ok=True)

    draw_architecture_diagram(fig_dir / '01_primer_ez_architecture.png')
    draw_accuracy_vs_shift(root, fig_dir / '02_accuracy_vs_shift.png')
    draw_quality_vs_shift(root, fig_dir / '03_quality_vs_shift.png')
    draw_quality_vs_wall_time(root, fig_dir / '04_quality_vs_wall_time.png')
    draw_conv_ablation_transfer(root, fig_dir / '05_conv_ablation_transfer.png')
    print("All Primer EZ figures successfully created in figures/")


if __name__ == '__main__':
    main()
