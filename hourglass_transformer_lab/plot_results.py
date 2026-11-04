"""Generate publication-ready figures for Hourglass Transformer Lab.

Produces:
1. figures/01_hourglass_architecture.png: Hierarchical U-shaped Hourglass vs Vanilla Transformer
2. figures/02_causal_shortening_mechanics.png: The causal shift (k-1) principle & future-leakage prevention
3. figures/03_familiar_vs_shift.png: Familiar period accuracy vs structural distribution shift
4. figures/04_training_wall_clock.png: Wall-clock CPU training time (theoretical vs observed speedup)
5. figures/05_length_extrapolation.png: Length generalization (Length 48 vs Length 64)
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
    """Draw side-by-side comparison of Vanilla Transformer vs Hourglass Hierarchical Transformer."""
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
    ax1.text(5, 12.3, "Standard Causal Transformer (4 Blocks)",
             color=STYLE['sky'], fontsize=14, fontweight='bold', ha='center')
    ax1.text(5, 11.7, "Full token resolution maintained through all layers (O(T^2) attention cost)",
             color=STYLE['text_muted'], fontsize=9.5, ha='center')

    ax2.text(5, 12.3, "Hierarchical Hourglass Transformer",
             color=STYLE['emerald'], fontsize=14, fontweight='bold', ha='center')
    ax2.text(5, 11.7, "Causal shortening (factor k) reduces middle attention complexity to O((T/k)^2)",
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
             "Input Tokens [Batch, T=48, D=32]", "Full resolution embeddings",
             bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(ax1, 5, 9.6, 5, 8.8, color=STYLE['blue'])

    draw_box(ax1, 5, 8.0, 8.4, 1.3,
             "Transformer Block 1 & 2 (Resolution T=48)", "Full causal attention matrix: 48 x 48 = 2,304 pairs",
             bg=STYLE['bg_card_alt'], border=STYLE['blue'])

    draw_arrow(ax1, 5, 7.2, 5, 6.4, color=STYLE['blue'])

    draw_box(ax1, 5, 5.6, 8.4, 1.3,
             "Transformer Block 3 & 4 (Resolution T=48)", "Full causal attention matrix: 48 x 48 = 2,304 pairs",
             bg=STYLE['bg_card_alt'], border=STYLE['blue'])

    draw_arrow(ax1, 5, 4.8, 5, 4.0, color="#64748B")

    draw_box(ax1, 5, 3.3, 8.4, 1.1,
             "Next-Token Projection Head", "Logits: [Batch, T=48, Vocab=32]",
             bg=STYLE['bg_card'], border="#475569")

    outcome_left = patches.FancyBboxPatch((0.8, 0.6), 8.4, 2.0, boxstyle="round,pad=0.3",
                                         facecolor="#1A202C", edgecolor=STYLE['sky'], linewidth=2.0, zorder=2)
    ax1.add_patch(outcome_left)
    ax1.text(5, 2.0, "ATTENTION COMPLEXITY & RUNTIME", color=STYLE['sky'], fontsize=11, fontweight='bold', ha='center')
    ax1.text(5, 1.5, "Attention Pairs: 4 x (48^2) = 9,216 pairs", color=STYLE['text_main'], fontsize=10.5, fontweight='bold', ha='center')
    ax1.text(5, 1.0, "Training Time: 1.20s  |  Familiar Acc: 13.3%", color=STYLE['text_sub'], fontsize=9, ha='center')

    # --- RIGHT: HOURGLASS TRANSFORMER ---
    draw_box(ax2, 5, 10.3, 8.4, 1.1,
             "Input Tokens [Batch, T=48, D=32]", "High-resolution causal embedding",
             bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(ax2, 5, 9.6, 5, 8.9, color=STYLE['emerald'])

    draw_box(ax2, 5, 8.3, 8.4, 1.0,
             "Block 1: Full Resolution [T=48]", "Extracts high-frequency local features",
             bg=STYLE['bg_card_alt'], border=STYLE['emerald'])

    draw_arrow(ax2, 5, 7.7, 5, 7.1, color=STYLE['cyan'])

    draw_box(ax2, 5, 6.5, 8.4, 1.0,
             "Causal Shortening: Shift Right (k-1) -> Pool (factor k=3)", "Compresses 48 tokens -> 16 coarse tokens with ZERO future leakage",
             bg="#062E25", border=STYLE['cyan'], title_color="#6EE7B7")

    draw_arrow(ax2, 5, 5.9, 5, 5.3, color=STYLE['purple'])

    draw_box(ax2, 5, 4.6, 8.4, 1.2,
             "Middle Blocks 2 & 3: Coarse Resolution [T/3 = 16]", "Attention on 16 x 16 = 256 pairs (9x fewer attention comparisons!)",
             bg="#1E1B4B", border=STYLE['purple'], title_color="#E9D5FF")

    draw_arrow(ax2, 5, 3.9, 5, 3.3, color=STYLE['emerald'])

    draw_box(ax2, 5, 2.7, 8.4, 1.0,
             "Upsample (Linear / Repeat) + Skip Connection from Block 1", "Restores resolution T=48; adds residual fine details",
             bg=STYLE['bg_card_alt'], border=STYLE['amber'], title_color="#FDE68A")

    outcome_right = patches.FancyBboxPatch((0.8, 0.4), 8.4, 1.7, boxstyle="round,pad=0.3",
                                          facecolor="#0B231D", edgecolor=STYLE['emerald'], linewidth=2.0, zorder=2)
    ax2.add_patch(outcome_right)
    ax2.text(5, 1.6, "ATTENTION COMPLEXITY & RUNTIME", color=STYLE['emerald'], fontsize=11, fontweight='bold', ha='center')
    ax2.text(5, 1.15, "Middle Attention: 9x fewer pairs (256 vs 2,304)", color="#A7F3D0", fontsize=10.5, fontweight='bold', ha='center')
    ax2.text(5, 0.7, "Training Time: 0.78s - 0.98s  |  Familiar Acc: 55.8%", color=STYLE['text_sub'], fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_causal_shortening_mechanics(output_path: Path):
    """Draw detailed diagram of the causal shift (k-1) principle."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.0))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    T = 12
    k = 3

    # Panel 1: Unshifted (UNSAFE - Future Leakage)
    ax1.set_xlim(-2.0, T + 0.5)
    ax1.set_ylim(-0.2, 3.4)
    ax1.axis('off')
    ax1.text(T/2, 3.1, "UNSAFE: Unshifted Pooling Leaks Future Tokens (Causal Diff = 0.760!)",
             color=STYLE['rose'], fontsize=12, fontweight='bold', ha='center')

    for i in range(T):
        ax1.add_patch(patches.Rectangle((i, 2.2), 0.85, 0.55, facecolor=STYLE['bg_card_alt'], edgecolor="#64748B", lw=1))
        ax1.text(i + 0.42, 2.47, str(i), ha='center', va='center', color=STYLE['text_main'], fontsize=9, fontweight='bold')
        ax1.add_patch(patches.Rectangle((i, 0.2), 0.85, 0.55, facecolor=STYLE['bg_card_alt'], edgecolor="#64748B", lw=1))
        ax1.text(i + 0.42, 0.47, str(i), ha='center', va='center', color=STYLE['text_main'], fontsize=9, fontweight='bold')

    for g in range(T // k):
        xx = g * 3 + 0.35
        ax1.add_patch(patches.Rectangle((xx, 1.2), 2.3, 0.55, facecolor="#3B1219", edgecolor=STYLE['rose'], lw=1.5))
        ax1.text(xx + 1.15, 1.47, f"Group {g} (Tokens {g*3}-{g*3+2})", ha='center', va='center', color="#FDA4AF", fontsize=8.5, fontweight='bold')

        # Connect inputs to group
        for idx in range(g * 3, g * 3 + 3):
            ax1.annotate("", xy=(xx + 1.15, 1.75), xytext=(idx + 0.42, 2.2),
                         arrowprops=dict(arrowstyle="->", color=STYLE['rose'], lw=1.0))
        # Broadcast group to outputs
        for idx in range(g * 3, g * 3 + 3):
            ax1.annotate("", xy=(idx + 0.42, 0.75), xytext=(xx + 1.15, 1.2),
                         arrowprops=dict(arrowstyle="->", color=STYLE['rose'], lw=1.0))

    ax1.text(-1.0, 2.47, "Input Tokens", ha='right', va='center', color=STYLE['text_muted'], fontsize=9.5, fontweight='bold')
    ax1.text(-1.0, 1.47, "Coarse Groups", ha='right', va='center', color=STYLE['rose'], fontsize=9.5, fontweight='bold')
    ax1.text(-1.0, 0.47, "Broadcast Out", ha='right', va='center', color=STYLE['text_muted'], fontsize=9.5, fontweight='bold')

    # Panel 2: Safe Shifted (k-1)
    ax2.set_xlim(-2.0, T + 0.5)
    ax2.set_ylim(-0.2, 3.4)
    ax2.axis('off')
    ax2.text(T/2, 3.1, "SAFE: Right-Shift by (k - 1) Restores Exact Autoregressive Causality (Causal Diff = 0.000)",
             color=STYLE['emerald'], fontsize=12, fontweight='bold', ha='center')

    for i in range(T):
        ax2.add_patch(patches.Rectangle((i, 2.2), 0.85, 0.55, facecolor=STYLE['bg_card_alt'], edgecolor="#64748B", lw=1))
        ax2.text(i + 0.42, 2.47, str(i), ha='center', va='center', color=STYLE['text_main'], fontsize=9, fontweight='bold')
        ax2.add_patch(patches.Rectangle((i, 0.2), 0.85, 0.55, facecolor=STYLE['bg_card_alt'], edgecolor="#64748B", lw=1))
        ax2.text(i + 0.42, 0.47, str(i), ha='center', va='center', color=STYLE['text_main'], fontsize=9, fontweight='bold')

    for g in range(T // k):
        xx = g * 3 + 0.35
        ax2.add_patch(patches.Rectangle((xx, 1.2), 2.3, 0.55, facecolor="#062E25", edgecolor=STYLE['emerald'], lw=1.5))
        ax2.text(xx + 1.15, 1.47, f"Causal Group {g}", ha='center', va='center', color="#A7F3D0", fontsize=8.5, fontweight='bold')

        first = g * 3
        # Causal inputs entering group g after shift k-1: indices [first - 2, first]
        for idx in range(max(0, first - 2), first + 1):
            ax2.annotate("", xy=(xx + 1.15, 1.75), xytext=(idx + 0.42, 2.2),
                         arrowprops=dict(arrowstyle="->", color=STYLE['emerald'], lw=1.0))
        # Broadcast to outputs
        for idx in range(first, first + 3):
            ax2.annotate("", xy=(idx + 0.42, 0.75), xytext=(xx + 1.15, 1.2),
                         arrowprops=dict(arrowstyle="->", color=STYLE['cyan'], lw=1.0))

    ax2.text(-1.0, 2.47, "Input Tokens", ha='right', va='center', color=STYLE['text_muted'], fontsize=9.5, fontweight='bold')
    ax2.text(-1.0, 1.47, "Shifted Groups", ha='right', va='center', color=STYLE['emerald'], fontsize=9.5, fontweight='bold')
    ax2.text(-1.0, 0.47, "Causal Out", ha='right', va='center', color=STYLE['text_muted'], fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_familiar_vs_shift(root: Path, output_path: Path):
    """Draw familiar period accuracy vs changed period structural shift."""
    report = json.loads((root / 'results/benchmark.json').read_text())
    summary = report['summary']
    names = ['vanilla_4', 'hourglass_k2', 'hourglass_k3', 'hourglass_k4',
             'hourglass_linear_k3', 'hourglass_factor_dropout', 'hourglass_no_skip', 'hourglass_unshifted_UNSAFE']
    labels = ['Vanilla 4L', 'Hourglass k=2', 'Hourglass k=3', 'Hourglass k=4',
              'Linear k=3', 'Factor Dropout', 'No Skip', 'UNSAFE No Shift']

    x = np.arange(len(names))
    width = 0.38

    fam = [summary[n]['familiar_accuracy']['mean'] for n in names]
    shift = [summary[n]['shifted_accuracy']['mean'] for n in names]

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    rects1 = ax.bar(x - width/2, fam, width, label='Familiar Period (4-Token Motif)',
                    color=STYLE['emerald'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.92)
    rects2 = ax.bar(x + width/2, shift, width, label='Changed Period (7-Token Motif Shift)',
                    color=STYLE['rose'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.92)

    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.0, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=8.5, fontweight='bold')
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.0, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=8.5, fontweight='bold')

    ax.set_ylim(0, 65)
    ax.set_ylabel('Copy-Region Accuracy (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Copy-Region Next-Token Accuracy: Familiar vs Changed Period (3 Seeds x 95 Steps)',
                 color=STYLE['text_main'], fontsize=13, fontweight='bold', pad=14)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=10, loc='upper left')

    # Highlight learned linear surge and shift drop
    ax.annotate('Learned Linear Resampler Surges!\n(55.8% Familiar Accuracy)', xy=(4 - width/2, fam[4]), xytext=(3.5, 60),
                 arrowprops=dict(arrowstyle="->", color=STYLE['cyan'], lw=1.8),
                 color=STYLE['cyan'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['cyan']))

    ax.annotate('Overfitting to Period 4\n(Drops to 4.4% on Period 7!)', xy=(4 + width/2, shift[4]), xytext=(4.8, 25),
                 arrowprops=dict(arrowstyle="->", color=STYLE['amber'], lw=1.8),
                 color=STYLE['amber'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['amber']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_training_wall_clock(root: Path, output_path: Path):
    """Draw measured CPU training wall-clock time across configurations."""
    report = json.loads((root / 'results/benchmark.json').read_text())
    summary = report['summary']
    names = ['vanilla_4', 'hourglass_linear_k3', 'hourglass_factor_dropout',
             'hourglass_k2', 'hourglass_k4', 'hourglass_k3', 'hourglass_no_skip']
    labels = ['Vanilla 4L (Full Res)', 'Linear k=3 (Learned)', 'Factor Dropout',
              'Hourglass k=2', 'Hourglass k=4', 'Hourglass k=3', 'Hourglass No Skip']

    times = [summary[n]['train_seconds']['mean'] for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['rose'], STYLE['purple'], STYLE['sky'], STYLE['cyan'], STYLE['blue'], STYLE['emerald'], '#64748B']

    bars = ax.barh(labels, times, color=colors, height=0.55, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars, times):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.3f} s",
                ha='left', va='center', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax.set_xlim(0, 1.45)
    ax.set_xlabel('Measured Training Wall-Clock Time, 95 Steps (Seconds, Local CPU)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Training Speedup: Theoretical 9x Attention Savings vs 35% Wall-Clock Gain',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.invert_yaxis()

    ax.annotate('35% Overall Wall-Clock Speedup\n(0.778s vs 1.195s)', xy=(times[5], 5), xytext=(1.0, 4.0),
                 arrowprops=dict(arrowstyle="->", color=STYLE['emerald'], lw=1.8),
                 color=STYLE['emerald'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['emerald']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_length_extrapolation(root: Path, output_path: Path):
    """Draw length generalization: Length 48 vs Length 64 under the same underlying rule."""
    report = json.loads((root / 'results/benchmark.json').read_text())
    summary = report['summary']
    names = ['vanilla_4', 'hourglass_k2', 'hourglass_k3', 'hourglass_k4',
             'hourglass_linear_k3', 'hourglass_factor_dropout', 'hourglass_no_skip']
    labels = ['Vanilla 4L', 'Hourglass k=2', 'Hourglass k=3', 'Hourglass k=4',
              'Linear k=3', 'Factor Dropout', 'No Skip']

    x = np.arange(len(names))
    width = 0.38

    fam = [summary[n]['familiar_accuracy']['mean'] for n in names]
    longer = [summary[n]['longer_accuracy']['mean'] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    rects1 = ax.bar(x - width/2, fam, width, label='Training Length (48 Tokens)',
                    color=STYLE['blue'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.9)
    rects2 = ax.bar(x + width/2, longer, width, label='Extrapolated Length (64 Tokens)',
                    color=STYLE['purple'], edgecolor='#FFFFFF', linewidth=0.5, alpha=0.9)

    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.0, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=9, fontweight='bold')
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.0, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=9, fontweight='bold')

    ax.set_ylim(0, 65)
    ax.set_ylabel('Copy-Region Accuracy (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Length Extrapolation: Changing Sequence Length != Changing the Underlying Rule',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=10, loc='upper left')

    ax.annotate('Linear Resampler Extrapolates!\n(54.6% on Len 64 vs 55.8% on Len 48)', xy=(4 + width/2, longer[4]), xytext=(3.8, 60),
                 arrowprops=dict(arrowstyle="->", color=STYLE['purple'], lw=1.8),
                 color=STYLE['purple'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['purple']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def main():
    root = Path(__file__).resolve().parent
    fig_dir = root / 'figures'
    fig_dir.mkdir(exist_ok=True)

    draw_architecture_diagram(fig_dir / '01_hourglass_architecture.png')
    draw_causal_shortening_mechanics(fig_dir / '02_causal_shortening_mechanics.png')
    draw_familiar_vs_shift(root, fig_dir / '03_familiar_vs_shift.png')
    draw_training_wall_clock(root, fig_dir / '04_training_wall_clock.png')
    draw_length_extrapolation(root, fig_dir / '05_length_extrapolation.png')
    print("All Hourglass Transformer figures successfully created in figures/")


if __name__ == '__main__':
    main()
