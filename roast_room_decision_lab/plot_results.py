"""Generate publication-ready figures for Roast Room Decision Lab.

Produces:
1. figures/01_council_architecture.png: Multi-agent council architecture & ruin veto cap
2. figures/02_oracle_choice_accuracy.png: Option selection accuracy across 7 policies
3. figures/03_disagreement_vs_truth.png: The conformity trap: disagreement spread vs accuracy
4. figures/04_ruin_veto_safety.png: Unconditional GO on catastrophic risk options
5. figures/05_regret_vs_cost_frontier.png: Utility regret vs computational token investment
"""
import json
import os
import numpy as np
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
from pathlib import Path
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
    """Draw multi-agent decision council architecture with relevance weighting and ruin cap."""
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_fig'])
    ax.axis('off')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)

    # Title & Subtitle
    ax.text(7.5, 9.4, "Roast Room Decision Council Architecture & Ruin Veto Cap",
            color=STYLE['text_main'], fontsize=15, fontweight='bold', ha='center')
    ax.text(7.5, 8.95, "Independent specialists, relevance-weighted aggregation, and non-compensatory risk safeguards",
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

    # Column 1: Input Candidate Decisions
    draw_box(2.2, 7.3, 3.8, 1.3,
             "Strategic Decision Choice", "3 Competing Options (A, B, C)",
             bg=STYLE['bg_card_alt'], border=STYLE['blue'])
    draw_box(2.2, 5.5, 3.8, 1.4,
             "7 Evaluation Dimensions", "Unit Economics, Churn, Runway,\nMoat, Execution, Market, Regulatory",
             bg=STYLE['bg_card_alt'], border=STYLE['purple'], title_color="#E9D5FF")
    draw_box(2.2, 3.6, 3.8, 1.3,
             "Relevance Weighting", "High (3x), Medium (2x), Low (1x)",
             bg=STYLE['bg_card'], border=STYLE['cyan'])

    draw_arrow(4.1, 5.5, 5.3, 5.5, color=STYLE['cyan'])

    # Column 2: Specialist Council Pool
    council_box = patches.FancyBboxPatch((5.3, 2.4), 3.8, 5.4, boxstyle="round,pad=0.3",
                                         facecolor="#0F172A", edgecolor=STYLE['cyan'], linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(council_box)
    ax.text(7.2, 7.4, "Specialist Council (7 Seats)", color=STYLE['cyan'], fontsize=11, fontweight='bold', ha='center')
    ax.text(7.2, 7.0, "(Independent Dimension Cards)", color=STYLE['text_muted'], fontsize=8.5, ha='center')

    draw_box(7.2, 6.2, 3.2, 0.85, "Financial Analyst", "Evaluates Unit Economics & Cash", bg=STYLE['bg_card_alt'], border="#64748B")
    draw_box(7.2, 5.15, 3.2, 0.85, "Market Strategist", "Evaluates Demand & Moat", bg=STYLE['bg_card_alt'], border="#64748B")
    draw_box(7.2, 4.1, 3.2, 0.85, "Risk Officer", "Monitors Fatal Ruin Flags", bg="#3B1219", border=STYLE['rose'], title_color="#FDA4AF")
    draw_box(7.2, 3.05, 3.2, 0.85, "Challenger / Seat 4-7", "Domain Counter-Arguments", bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(9.1, 5.5, 10.3, 5.5, color=STYLE['cyan'])

    # Column 3: Aggregation & Risk Veto
    draw_box(12.3, 6.8, 4.0, 1.3,
             "The Chair: Weighted Sum", "Computes Score = sum(w_i * s_i) / sum(w_i)",
             bg="#1E1B4B", border=STYLE['purple'], title_color="#E9D5FF")

    draw_arrow(12.3, 6.0, 12.3, 5.1, color=STYLE['amber'])

    draw_box(12.3, 4.3, 4.0, 1.4,
             "The Ruin Veto Cap", "Non-compensatory: If fatal ruin flagged,\nVerdict CANNOT be GO (Capped at RESHAPE)",
             bg="#27192A", border=STYLE['amber'], title_color="#FDE68A")

    draw_arrow(12.3, 3.4, 12.3, 2.4, color=STYLE['emerald'])

    # Final Stage: Decision Output
    outcome_box = patches.FancyBboxPatch((1.0, 0.5), 13.0, 1.6, boxstyle="round,pad=0.25",
                                         facecolor=STYLE['bg_card'], edgecolor=STYLE['emerald'], linewidth=2.0, zorder=2)
    ax.add_patch(outcome_box)
    ax.text(7.5, 1.6, "FINAL VERDICT & DISSENTING RECORD", color=STYLE['emerald'], fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, 1.15, "Option Choice: Argmax(Weighted Utility)  |  Verdict: GO, GO WITH CONDITIONS, or RESHAPE",
            color=STYLE['text_main'], fontsize=9, ha='center')
    ax.text(7.5, 0.75, "Ruin Veto reduces unsafe approvals on catastrophic options from 7.7% -> 0.0%!",
            color=STYLE['text_sub'], fontsize=8.5, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_choice_accuracy(root: Path, output_path: Path):
    """Draw oracle choice accuracy across all 7 policies."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    means = data['means']

    keys = ['solo_generalist', 'quick_four', 'seven_independent',
            'seven_correlated', 'debate_no_evidence',
            'audited_specialists', 'audited_with_ruin_cap']
    labels = ['Solo\nGeneralist', 'Quick 4\nSpecialists', '7 Seats\nIndependent',
              '7 Seats\nCorrelated', 'Debate\n(No Evidence)', 'Audited\nSpecialists', 'Audited +\nRuin Cap']

    acc = [means[p]['oracle_choice_accuracy']['mean'] * 100 for p in keys]
    err = [means[p]['oracle_choice_accuracy']['sample_std'] * 100 for p in keys]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['rose'], STYLE['sky'], STYLE['emerald'],
              STYLE['amber'], '#DC2626', STYLE['cyan'], STYLE['cyan']]

    bars = ax.bar(range(len(keys)), acc, yerr=err, capsize=4, color=colors,
                  width=0.55, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)

    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2.0, f"{val:.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax.set_ylim(0, 105)
    ax.set_ylabel('Oracle Choice Accuracy (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Option Selection Accuracy: Solo vs Councils vs Correlated Bias (5 Seeds x 600 Cases)',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])

    # Annotate correlated bias drop
    ax.annotate('Correlated Error\nCuts Accuracy (-21.5%)', xy=(3, acc[3]), xytext=(2.2, 45),
                 arrowprops=dict(arrowstyle="->", color=STYLE['amber'], lw=1.8),
                 color=STYLE['amber'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['amber']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_disagreement_vs_truth(root: Path, output_path: Path):
    """Draw the conformity trap: disagreement spread vs choice accuracy."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    means = data['means']

    keys = ['seven_independent', 'seven_correlated', 'debate_no_evidence', 'audited_specialists']
    labels = ['7 Independent\nSpecialists', '7 Correlated\nSpecialists', 'Debate\n(No Evidence)', 'Audited\nSpecialists']

    spread = [means[p]['mean_spread']['mean'] for p in keys]
    acc = [means[p]['oracle_choice_accuracy']['mean'] * 100 for p in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['emerald'], STYLE['amber'], STYLE['rose'], STYLE['cyan']]

    # Panel 1: Advisor Score Spread
    bars1 = ax1.bar(labels, spread, color=colors, width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars1, spread):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.15, f"{val:.2f}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10.5, fontweight='bold')

    ax1.set_ylim(0, 7.5)
    ax1.set_ylabel('Advisor Score Spread (0-10 Scale)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('Advisor Disagreement Spread (Winning Option)', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])

    # Annotate artificial consensus
    ax1.annotate('Debate Cuts Spread\nin Half (-52% Spread)', xy=(2, spread[2]), xytext=(1.4, 4.5),
                 arrowprops=dict(arrowstyle="->", color=STYLE['rose'], lw=1.8),
                 color=STYLE['rose'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['rose']))

    # Panel 2: Oracle Choice Accuracy
    bars2 = ax2.bar(labels, acc, color=colors, width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars2, acc):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.8, f"{val:.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10.5, fontweight='bold')

    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Oracle Choice Accuracy (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_title('Truth Accuracy: Agreement != Truth!', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])

    ax2.annotate('Accuracy Does NOT Improve!\n(59.7% vs 60.2%)', xy=(2, acc[2]), xytext=(1.4, 38),
                 arrowprops=dict(arrowstyle="->", color=STYLE['amber'], lw=1.8),
                 color=STYLE['amber'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['amber']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_ruin_veto_safety(root: Path, output_path: Path):
    """Draw unsafe unconditional approvals on catastrophic options."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    means = data['means']

    keys = ['solo_generalist', 'quick_four', 'seven_correlated', 'debate_no_evidence', 'audited_specialists', 'audited_with_ruin_cap']
    labels = ['Solo\nGeneralist', 'Quick 4\nSpecialists', '7 Correlated\nSpecialists', 'Debate\n(No Evidence)', 'Audited\n(No Cap)', 'Audited +\nRuin Cap']

    unsafe = [means[p]['unsafe_go_rate_per_case']['mean'] * 100 for p in keys]
    err = [means[p]['unsafe_go_rate_per_case']['sample_std'] * 100 for p in keys]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['rose'], STYLE['amber'], STYLE['rose'], STYLE['amber'], STYLE['rose'], STYLE['emerald']]

    bars = ax.bar(range(len(keys)), unsafe, yerr=err, capsize=4, color=colors,
                  width=0.52, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)

    for bar, val in zip(bars, unsafe):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.25, f"{val:.2f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax.set_ylim(0, 10.5)
    ax.set_ylabel('Unconditional GO on Ruin-Flagged Option (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Catastrophic Risk Failure: Unconditional Approval of Ruinous Decisions',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])

    # Annotate zero unsafe GO
    ax.annotate('100% Elimination of Unsafe GO!\n(Capped at RESHAPE)', xy=(5, unsafe[5]), xytext=(4.0, 5.0),
                 arrowprops=dict(arrowstyle="->", color=STYLE['emerald'], lw=1.8),
                 color=STYLE['emerald'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['emerald']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_regret_vs_cost(root: Path, output_path: Path):
    """Draw utility regret vs fictional token budget across policies."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    means = data['means']

    keys = ['solo_generalist', 'quick_four', 'seven_independent', 'seven_correlated',
            'debate_no_evidence', 'audited_specialists']
    labels = ['Solo (1 Call)', 'Quick 4 (4 Calls)', '7 Independent (7 Calls)',
              '7 Correlated (7 Calls)', 'Debate (21 Calls)', 'Audited (14 Calls)']

    regret = [means[p]['mean_regret']['mean'] for p in keys]
    tokens = [means[p]['mean_proxy_tokens']['mean'] for p in keys]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['rose'], STYLE['sky'], STYLE['emerald'], STYLE['amber'], '#DC2626', STYLE['cyan']]

    for c, r, t, l in zip(colors, regret, tokens, labels):
        ax.scatter(t, r, color=c, s=220, edgecolor='#FFFFFF', linewidth=1.5, zorder=3, label=l)
        y_offset = 0.035 if 'Debate' not in l else -0.045
        ax.text(t, r + y_offset, l, color=STYLE['text_main'], fontsize=9.5, fontweight='bold', ha='center', zorder=4)

    ax.set_xlim(0, 16000)
    ax.set_ylim(-0.02, 0.72)
    ax.set_xlabel('Fictional Proxy Tokens (Linear Cost Accounting)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Utility Regret (Lower is Better)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Decision Quality vs Compute Cost: Better Evidence Trumps Pure Debate',
                 color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=9, loc='upper right')

    ax.annotate('Debate Burns 3x Tokens\nWithout Lowering Regret!', xy=(tokens[4], regret[4]), xytext=(10500, 0.28),
                 arrowprops=dict(arrowstyle="->", color=STYLE['rose'], lw=1.8),
                 color=STYLE['rose'], fontsize=9.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['rose']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def main():
    root = Path(__file__).resolve().parent
    fig_dir = root / 'figures'
    fig_dir.mkdir(exist_ok=True)

    draw_architecture_diagram(fig_dir / '01_council_architecture.png')
    draw_choice_accuracy(root, fig_dir / '02_oracle_choice_accuracy.png')
    draw_disagreement_vs_truth(root, fig_dir / '03_disagreement_vs_truth.png')
    draw_ruin_veto_safety(root, fig_dir / '04_ruin_veto_safety.png')
    draw_regret_vs_cost(root, fig_dir / '05_regret_vs_cost_frontier.png')
    print("All Roast Room Decision Lab figures successfully created in figures/")


if __name__ == '__main__':
    main()
