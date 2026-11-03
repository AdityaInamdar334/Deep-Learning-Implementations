"""Generate publication-ready figures for Agent Cost Optimization Lab.

Produces:
1. figures/01_prompt_cache_architecture.png: Prefix cache mechanics & breakpoint placement
2. figures/02_cache_simulation_economics.png: 96-job cache simulation and cumulative spend
3. figures/03_workflow_call_reduction.png: Turn-by-turn context compounding and call reduction
4. figures/04_quality_cost_tradeoff.png: Defect recall vs precision vs cost trade-off
5. figures/05_token_cost_distribution.png: Token volume vs dollar spend across billing buckets
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
    """Draw clear diagram of prefix cache layout comparing anti-pattern vs cache-optimized."""
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

    # Header Panel 1: Anti-pattern
    ax1.text(5, 12.3, "Anti-Pattern: Dynamic Data Before Breakpoint",
             color=STYLE['rose'], fontsize=14, fontweight='bold', ha='center')
    ax1.text(5, 11.7, "Dynamic workspace info invalidates the prefix hash for every request",
             color=STYLE['text_muted'], fontsize=9.5, ha='center')

    # Header Panel 2: Optimized
    ax2.text(5, 12.3, "Optimized: Stable System Prefix & Guidelines First",
             color=STYLE['emerald'], fontsize=14, fontweight='bold', ha='center')
    ax2.text(5, 11.7, "Fixed 12,000-token system prompt stays cached across rotating repositories",
             color=STYLE['text_muted'], fontsize=9.5, ha='center')

    def draw_box(ax, x, y, w, h, title, subtitle, bg, border, title_color=STYLE['text_main'], style="round,pad=0.25"):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle=style,
                                      facecolor=bg, edgecolor=border, linewidth=1.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.18, title, color=title_color, fontsize=10.5, fontweight='bold', ha='center', va='center', zorder=3)
        ax.text(x, y - 0.22, subtitle, color=STYLE['text_sub'], fontsize=8.5, ha='center', va='center', zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2, color=STYLE['cyan']):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0), zorder=4)

    # --- LEFT: ANTI-PATTERN FLOW ---
    # Box 1: Dynamic Repo metadata (changes each repo!)
    draw_box(ax1, 5, 10.3, 8.4, 1.2,
             "1. Dynamic Workspace Header (repo: frontend-app)",
             "Changes per repository -> Breaks hash for subsequent tokens",
             bg="#3B1219", border=STYLE['rose'], title_color="#FDA4AF")

    draw_arrow(ax1, 5, 9.6, 5, 8.8, color=STYLE['rose'])

    # Box 2: Static guidelines (12k tokens, but hash is ruined!)
    draw_box(ax1, 5, 8.1, 8.4, 1.3,
             "2. Shared System Guidelines & Rules (12,000 tokens)",
             "Identical text, BUT prefix hash was poisoned by Block 1",
             bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(ax1, 5, 7.35, 5, 6.55, color="#64748B")

    # Breakpoint placement (Wrong!)
    draw_box(ax1, 5, 5.9, 8.4, 1.1,
             "3. Provider Cache Breakpoint [Anthropic / OpenAI]",
             "Breakpoint placed AFTER dynamic data -> Hash unique to each repo",
             bg="#27192A", border=STYLE['amber'], title_color="#FDE68A")

    draw_arrow(ax1, 5, 5.25, 5, 4.45, color="#64748B")

    # Box 4: Code diff & Task
    draw_box(ax1, 5, 3.8, 8.4, 1.1,
             "4. Pull Request Diff & User Task (2,500 tokens)",
             "Evaluated with zero cache hits across 16 rotating repos",
             bg=STYLE['bg_card'], border="#475569")

    # Left Outcome Badge
    outcome_left = patches.FancyBboxPatch((0.8, 0.6), 8.4, 2.2, boxstyle="round,pad=0.3",
                                         facecolor="#1F1315", edgecolor=STYLE['rose'], linewidth=2.0, zorder=2)
    ax1.add_patch(outcome_left)
    ax1.text(5, 2.2, "SIMULATION OUTCOME (96 JOBS)", color=STYLE['rose'], fontsize=11, fontweight='bold', ha='center')
    ax1.text(5, 1.6, "Cache Hits: 0 / 96 (0.0% Hit Rate)", color="#FECDD3", fontsize=11, fontweight='bold', ha='center')
    ax1.text(5, 1.05, "96 full cache writes @ $3.75/M -> Total: $5.6880 ($0.0593 / job)", color=STYLE['text_sub'], fontsize=9, ha='center')

    # --- RIGHT: OPTIMIZED FLOW ---
    # Box 1: Static guidelines first!
    draw_box(ax2, 5, 10.3, 8.4, 1.3,
             "1. Stable System Guidelines & Rules (12,000 tokens)",
             "Bit-identical prefix across ALL repositories and reviewers",
             bg="#062E25", border=STYLE['emerald'], title_color="#6EE7B7")

    draw_arrow(ax2, 5, 9.55, 5, 8.85, color=STYLE['emerald'])

    # Breakpoint placement (Correct!)
    draw_box(ax2, 5, 8.2, 8.4, 1.1,
             "2. Cache Breakpoint Marker (:cache_control)",
             "Locks identical prefix hash into provider 5-minute memory",
             bg="#132B40", border=STYLE['sky'], title_color="#7DD3FC")

    draw_arrow(ax2, 5, 7.55, 5, 6.75, color=STYLE['sky'])

    # Box 3: Dynamic Workspace Header
    draw_box(ax2, 5, 6.1, 8.4, 1.1,
             "3. Dynamic Workspace Header (repo, branch, author)",
             "Dynamic fields pushed SAFELY after the cached prefix boundary",
             bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(ax2, 5, 5.45, 5, 4.65, color="#64748B")

    # Box 4: Code diff & Task
    draw_box(ax2, 5, 4.0, 8.4, 1.1,
             "4. Pull Request Diff & User Task (2,500 tokens)",
             "Only unique delta billed at regular input rate ($3.00/M)",
             bg=STYLE['bg_card'], border="#475569")

    # Right Outcome Badge
    outcome_right = patches.FancyBboxPatch((0.8, 0.6), 8.4, 2.2, boxstyle="round,pad=0.3",
                                          facecolor="#0B231D", edgecolor=STYLE['emerald'], linewidth=2.0, zorder=2)
    ax2.add_patch(outcome_right)
    ax2.text(5, 2.2, "SIMULATION OUTCOME (96 JOBS)", color=STYLE['emerald'], fontsize=11, fontweight='bold', ha='center')
    ax2.text(5, 1.6, "Cache Hits: 95 / 96 (98.96% Hit Rate)", color="#A7F3D0", fontsize=11, fontweight='bold', ha='center')
    ax2.text(5, 1.05, "1 write + 95 cache reads @ $0.30/M -> Total: $1.7550 (-69.1% Cost!)", color=STYLE['text_sub'], fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_cache_simulation(root: Path, output_path: Path):
    """Draw hit rate and cumulative spend over 96 review jobs."""
    data = json.loads((root / 'results/cache_comparison.json').read_text())
    base_events = data['baseline']['events']
    opt_events = data['optimized']['events']

    times_base = [e['at_seconds'] / 60.0 for e in base_events]
    costs_base = [e['usd'] for e in base_events]
    cum_base = np.cumsum(costs_base)

    times_opt = [e['at_seconds'] / 60.0 for e in opt_events]
    costs_opt = [e['usd'] for e in opt_events]
    cum_opt = np.cumsum(costs_opt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    # Panel 1: Bar chart comparison of Total Cost & Hit Rate
    categories = ['Dynamic-Before-Breakpoint\n(Baseline)', 'Stable-Prefix-First\n(Optimized)']
    totals = [data['baseline']['total_usd'], data['optimized']['total_usd']]
    hit_rates = [data['baseline']['first_call_cache_hit_rate'] * 100,
                 data['optimized']['first_call_cache_hit_rate'] * 100]
    bar_colors = [STYLE['rose'], STYLE['emerald']]

    bars = ax1.bar(categories, totals, color=bar_colors, width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val, hr in zip(bars, totals, hit_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                 f"${val:.4f}\n({hr:.1f}% hit rate)",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10.5, fontweight='bold')

    ax1.set_ylim(0, 6.8)
    ax1.set_ylabel('Total Simulated Cost ($ USD)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('96 First-Call Review Jobs (16 Rotating Repos)', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, fontsize=10, color=STYLE['text_main'])

    # Add reduction badge
    ax1.annotate('-69.1% Spend', xy=(1, totals[1]), xytext=(0.5, 4.2),
                 arrowprops=dict(arrowstyle="->", color=STYLE['cyan'], lw=1.8),
                 color=STYLE['cyan'], fontsize=12, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['cyan']))

    # Panel 2: Cumulative spend over time
    ax2.plot(times_base, cum_base, color=STYLE['rose'], lw=2.4, label='Baseline (0% hits: $0.0593 / job)')
    ax2.plot(times_opt, cum_opt, color=STYLE['emerald'], lw=2.4, label='Optimized (98.9% hits: $0.0183 / job)')
    ax2.fill_between(times_base, cum_base, cum_opt, color=STYLE['emerald'], alpha=0.15, label='Saved Spend ($3.93)')

    ax2.set_xlabel('Elapsed Time (Minutes)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Cost ($ USD)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Inference Bill Over 72 Minutes of CI Jobs', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax2.grid(True, linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')
    ax2.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
               labelcolor=STYLE['text_main'], fontsize=9.5, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_workflow_compounding(root: Path, output_path: Path):
    """Draw turn-by-turn context cost growth and call-reduction impact."""
    wf = json.loads((root / 'results/workflow_comparison.json').read_text())
    before = wf['before']['per_call_usd']
    after = wf['after']['per_call_usd']

    turns_b = np.arange(1, len(before) + 1)
    turns_a = np.arange(1, len(after) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    # Panel 1: Per-turn cost escalation as conversation history grows
    ax1.plot(turns_b, before, marker='o', color=STYLE['rose'], lw=2.0, label='16-Call Agent (Baseline)')
    ax1.plot(turns_a, after, marker='s', color=STYLE['sky'], lw=2.0, label='10-Call Agent (Optimized)')

    ax1.set_xlabel('Model Call / Turn Index', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cost per Call ($ USD)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('Turn-by-Turn Cost Escalation (Context Accumulation)', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
               labelcolor=STYLE['text_main'], fontsize=9.5)

    # Highlight why Turn 1 is distinct
    ax1.annotate('Turn 1: Cache Write ($0.054)', xy=(1, before[0]), xytext=(3, 0.068),
                 arrowprops=dict(arrowstyle="->", color=STYLE['amber'], lw=1.5),
                 color=STYLE['amber'], fontsize=8.5, fontweight='bold')

    # Panel 2: Total cost comparison bar with call-step elimination
    labels = ['16-Turn Agent\n(Verbose tool chain)', '10-Turn Agent\n(Pruned unnecessary steps)']
    costs = [wf['before']['usd'], wf['after']['usd']]
    colors = [STYLE['rose'], STYLE['sky']]

    bars = ax2.bar(labels, costs, color=colors, width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars, costs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"${val:.4f}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=11, fontweight='bold')

    ax2.set_ylim(0, 0.8)
    ax2.set_ylabel('Total Agent Run Cost ($ USD)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_title('End-to-End Workflow Cost (50.3% Scenario Reduction)', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=10, color=STYLE['text_main'])

    ax2.annotate('-50.3% Cost\n(6 tool calls pruned)', xy=(1, costs[1]), xytext=(0.5, 0.52),
                 arrowprops=dict(arrowstyle="->", color=STYLE['cyan'], lw=1.8),
                 color=STYLE['cyan'], fontsize=11, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['cyan']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_quality_tradeoff(root: Path, output_path: Path):
    """Draw quality vs cost trade-off exposing the danger of naive cost cutting."""
    q = json.loads((root / 'results/quality_summary.json').read_text())
    base = q['baseline']
    opt = q['optimized']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    # Panel 1: Defect Recall vs Precision
    metrics = ['Defect Recall (%)', 'Review Precision (%)']
    base_vals = [base['recall'] * 100, base['precision'] * 100]
    opt_vals = [opt['recall'] * 100, opt['precision'] * 100]

    x = np.arange(len(metrics))
    width = 0.32

    rects1 = ax1.bar(x - width/2, base_vals, width, label='Baseline Reviewer (High Effort)',
                     color=STYLE['blue'], edgecolor='#FFFFFF', linewidth=0.5)
    rects2 = ax1.bar(x + width/2, opt_vals, width, label='Cheaper Arm (Aggressive Pruning)',
                     color=STYLE['rose'], edgecolor='#FFFFFF', linewidth=0.5)

    for r in rects1:
        ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    for r in rects2:
        ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax1.set_ylim(0, 115)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10.5, color=STYLE['text_main'], fontweight='bold')
    ax1.set_ylabel('Score (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('The Optimization Trap: Precision Rises, Recall Drops!', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
               labelcolor=STYLE['text_main'], fontsize=9.5)

    # Panel 2: Ground Truth Defect Classification (TP, FP, FN)
    classes = ['Caught Bugs (TP)', 'False Alarms (FP)', 'Missed Bugs (FN)']
    base_counts = [base['tp'], base['fp'], base['fn']]
    opt_counts = [opt['tp'], opt['fp'], opt['fn']]

    x2 = np.arange(len(classes))
    rects3 = ax2.bar(x2 - width/2, base_counts, width, label='Baseline',
                     color=STYLE['blue'], edgecolor='#FFFFFF', linewidth=0.5)
    rects4 = ax2.bar(x2 + width/2, opt_counts, width, label='Cheaper Arm',
                     color=STYLE['rose'], edgecolor='#FFFFFF', linewidth=0.5)

    for r in rects3:
        ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.5, f"{int(r.get_height())}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    for r in rects4:
        ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.5, f"{int(r.get_height())}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax2.set_ylim(0, 62)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes, fontsize=10.5, color=STYLE['text_main'], fontweight='bold')
    ax2.set_ylabel('Finding Count', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_title('Review Defect Outcomes (24 Labeled PRs)', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')
    ax2.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
               labelcolor=STYLE['text_main'], fontsize=9.5)

    # Note the FN increase
    ax2.annotate('3.5x More Missed Bugs!\n(4 FN -> 14 FN)', xy=(2 + width/2, opt_counts[2]), xytext=(1.6, 36),
                 arrowprops=dict(arrowstyle="->", color=STYLE['amber'], lw=1.8),
                 color=STYLE['amber'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['amber']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_token_cost_distribution(root: Path, output_path: Path):
    """Draw token volume vs billable dollar spend across the four token buckets."""
    t = json.loads((root / 'results/trace_summary.json').read_text())
    totals = t['token_totals']

    buckets = ['Regular Input\n($3.00/M)', 'Cache Write\n($3.75/M)', 'Cache Read\n($0.30/M)', 'Output\n($15.00/M)']
    tokens = [totals['regular_input'], totals['cache_write'], totals['cache_read'], totals['output']]
    costs = [
        totals['regular_input'] * 3.0 / 1e6,
        totals['cache_write'] * 3.75 / 1e6,
        totals['cache_read'] * 0.30 / 1e6,
        totals['output'] * 15.0 / 1e6,
    ]

    total_tokens = sum(tokens)
    total_cost = sum(costs)

    tok_pcts = [tok / total_tokens * 100 for tok in tokens]
    cost_pcts = [c / total_cost * 100 for c in costs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    colors = [STYLE['blue'], STYLE['amber'], STYLE['emerald'], STYLE['purple']]

    # Donut 1: Token Volume
    wedges1, texts1, autotexts1 = ax1.pie(
        tok_pcts, labels=buckets, autopct='%1.1f%%', startangle=140,
        colors=colors, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor=STYLE['bg_fig'], linewidth=2)
    )
    for t_text in texts1:
        t_text.set_color(STYLE['text_main'])
        t_text.set_fontsize(9.5)
    for at in autotexts1:
        at.set_color('#FFFFFF')
        at.set_fontweight('bold')
        at.set_fontsize(9.5)
    ax1.set_title(f"Token Share by Bucket\n({total_tokens:,} Total Tokens)",
                  color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=14)

    # Donut 2: Dollar Spend
    wedges2, texts2, autotexts2 = ax2.pie(
        cost_pcts, labels=buckets, autopct='%1.1f%%', startangle=140,
        colors=colors, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor=STYLE['bg_fig'], linewidth=2)
    )
    for t_text in texts2:
        t_text.set_color(STYLE['text_main'])
        t_text.set_fontsize(9.5)
    for at in autotexts2:
        at.set_color('#FFFFFF')
        at.set_fontweight('bold')
        at.set_fontsize(9.5)
    ax2.set_title(f"Dollar Share by Bucket\n(${total_cost:.4f} Total Billed USD)",
                  color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def main():
    root = Path(__file__).resolve().parent
    fig_dir = root / 'figures'
    fig_dir.mkdir(exist_ok=True)

    draw_architecture_diagram(fig_dir / '01_prompt_cache_architecture.png')
    draw_cache_simulation(root, fig_dir / '02_cache_simulation_economics.png')
    draw_workflow_compounding(root, fig_dir / '03_workflow_call_reduction.png')
    draw_quality_tradeoff(root, fig_dir / '04_quality_cost_tradeoff.png')
    draw_token_cost_distribution(root, fig_dir / '05_token_cost_distribution.png')
    print("All figures successfully created in figures/")


if __name__ == '__main__':
    main()
