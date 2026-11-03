"""Generate publication-ready figures for Distributed Science Agent Lab.

Produces:
1. figures/01_orchestrator_architecture.png: Distributed scientific agent work graph & dual safeguards
2. figures/02_coverage_and_discovery.png: Claim coverage across work budgets
3. figures/03_parallel_latency_scaling.png: Wall-clock scaling with remote I/O delay vs local CPU overhead
4. figures/04_safeguard_ablations.png: False discovery explosion when removing holdout & BH FDR
5. figures/05_token_evidence_efficiency.png: Proxy cost & token budget vs verified claim yield
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


def draw_orchestration_architecture(output_path: Path):
    """Draw distributed scientific reasoning architecture with dual verification safeguards."""
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_fig'])
    ax.axis('off')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)

    # Title & Subtitle
    ax.text(7.5, 9.4, "Distributed Scientific Agent Architecture & Dual-Safeguard Verification",
            color=STYLE['text_main'], fontsize=15, fontweight='bold', ha='center')
    ax.text(7.5, 8.95, "Inspired by the 1,946-agent whole-brain CRISPR reasoning pipeline (OpenAI / PerturbAI / Scripps)",
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

    # Column 1: Input Dataset & Null Injection
    draw_box(2.2, 7.3, 3.8, 1.3,
             "High-Throughput Dataset", "Wine Dataset (130 samples, 13 features)",
             bg=STYLE['bg_card_alt'], border=STYLE['blue'])
    draw_box(2.2, 5.6, 3.8, 1.3,
             "+ 24 Synthetic Gaussian Nulls", "Known negative controls (zero causal signal)",
             bg="#3B1219", border=STYLE['rose'], title_color="#FDA4AF")
    draw_box(2.2, 3.9, 3.8, 1.3,
             "Stratified Train/Holdout Split", "70% Discovery Split  |  30% Isolated Holdout",
             bg=STYLE['bg_card'], border=STYLE['purple'])

    draw_arrow(4.1, 5.6, 5.3, 5.6, color=STYLE['cyan'])

    # Column 2: Distributed Specialist Workers
    worker_box = patches.FancyBboxPatch((5.3, 2.5), 3.8, 5.2, boxstyle="round,pad=0.3",
                                        facecolor="#0F172A", edgecolor=STYLE['cyan'], linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(worker_box)
    ax.text(7.2, 7.3, "Distributed Investigators Pool", color=STYLE['cyan'], fontsize=11, fontweight='bold', ha='center')
    ax.text(7.2, 6.9, "(Concurrent Workers / Threads)", color=STYLE['text_muted'], fontsize=8.5, ha='center')

    draw_box(7.2, 6.1, 3.2, 0.9, "Specialist Agent 1", "Investigates Feature A (Welch t-test)", bg=STYLE['bg_card_alt'], border="#64748B")
    draw_box(7.2, 5.0, 3.2, 0.9, "Specialist Agent 2", "Investigates Feature B (Welch t-test)", bg=STYLE['bg_card_alt'], border="#64748B")
    draw_box(7.2, 3.9, 3.2, 0.9, "Specialist Agent K...", "Investigates Feature K (Welch t-test)", bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(9.1, 5.6, 10.3, 5.6, color=STYLE['cyan'])

    # Column 3: Dual Safeguards
    draw_box(12.3, 6.8, 4.0, 1.3,
             "Safeguard 1: Benjamini-Hochberg", "Controls False Discovery Rate (FDR <= 0.05)",
             bg="#1E1B4B", border=STYLE['purple'], title_color="#E9D5FF")

    draw_arrow(12.3, 6.0, 12.3, 5.3, color=STYLE['amber'])

    draw_box(12.3, 4.5, 4.0, 1.3,
             "Safeguard 2: Independent Holdout", "Directional Welch test on 30% holdout (p < 0.05)",
             bg="#062E25", border=STYLE['emerald'], title_color="#A7F3D0")

    draw_arrow(12.3, 3.7, 12.3, 2.8, color=STYLE['emerald'])

    # Final Stage: Verified Provenance Atlas
    outcome_box = patches.FancyBboxPatch((1.0, 0.5), 13.0, 1.7, boxstyle="round,pad=0.25",
                                         facecolor=STYLE['bg_card'], edgecolor=STYLE['emerald'], linewidth=2.0, zorder=2)
    ax.add_patch(outcome_box)
    ax.text(7.5, 1.7, "VERIFIED DISCOVERY ATLAS & PROVENANCE REGISTRY", color=STYLE['emerald'], fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, 1.25, "Every accepted claim records: Feature Name | Effect Direction | Sample Size | Raw p | Corrected p | Held-Out p | Worker ID",
            color=STYLE['text_main'], fontsize=9, ha='center')
    ax.text(7.5, 0.85, "100% Testable Precision across 5 benchmark seeds: Zero non-replicating claims or synthetic nulls accepted!",
            color=STYLE['text_sub'], fontsize=8.5, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_coverage_and_discovery(root: Path, output_path: Path):
    """Draw claim coverage and discovery yield across work budgets."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    s = data['summary']

    keys = ['generalist_8', 'specialists_8_verified', 'specialists_all_serial', 'specialists_all_parallel']
    labels = ['Generalist 8\n(Single prompt)', 'Specialists 8\n(Isolated tasks)', 'Serial All 37\n(Full coverage)', 'Parallel All 37\n(6 Workers)']

    accepted = [s[k]['accepted_claims']['mean'] for k in keys]
    replicated = [s[k]['independently_replicated_claims']['mean'] for k in keys]
    tokens = [s[k]['proxy_tokens']['mean'] for k in keys]

    x = np.arange(len(labels))
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    # Panel 1: Accepted vs Replicated Claims
    rects1 = ax1.bar(x - width/2, accepted, width, label='Accepted Claims', color=STYLE['blue'], edgecolor='#FFFFFF', linewidth=0.5)
    rects2 = ax1.bar(x + width/2, replicated, width, label='Independently Replicated', color=STYLE['emerald'], edgecolor='#FFFFFF', linewidth=0.5)

    for r in rects1:
        ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.15, f"{r.get_height():.1f}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    for r in rects2:
        ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.15, f"{r.get_height():.1f}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax1.set_ylim(0, 11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])
    ax1.set_ylabel('Mean Claim Count (5 Seeds)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('Discovery Coverage Across Investigation Budgets', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
               labelcolor=STYLE['text_main'], fontsize=9.5)

    # Panel 2: Token Budget vs Discoveries
    colors = [STYLE['cyan'], STYLE['purple'], STYLE['amber'], STYLE['emerald']]
    bars = ax2.bar(labels, tokens, color=colors, width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars, tokens):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 400, f"{val:,}\ntok",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax2.set_ylim(0, 23000)
    ax2.set_ylabel('Proxy Tokens Consumed', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_title('Workload & Token Investment by Architecture', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5, color=STYLE['text_main'])

    # Annotate Diminishing Returns
    ax2.annotate('+23% Discoveries for +380% Tokens', xy=(3, tokens[3]), xytext=(1.5, 16000),
                 arrowprops=dict(arrowstyle="->", color=STYLE['amber'], lw=1.8),
                 color=STYLE['amber'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['amber']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_parallel_latency(root: Path, output_path: Path):
    """Draw wall-clock scaling comparing remote tool latency vs zero-latency CPU overhead."""
    lat_data = json.loads((root / 'results/benchmark.json').read_text())['summary']
    no_lat_data = json.loads((root / 'results/no_latency_ablation.json').read_text())['summary']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    labels = ['Serial Specialists\n(1 Worker)', 'Parallel Specialists\n(6 Workers)']
    x = range(len(labels))

    # Panel 1: With 20ms Remote Tool I/O Delay
    r_lat = [lat_data['specialists_all_serial']['runtime_seconds']['mean'],
             lat_data['specialists_all_parallel']['runtime_seconds']['mean']]

    bars1 = ax1.bar(labels, r_lat, color=[STYLE['rose'], STYLE['emerald']], width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars1, r_lat):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.3f} s",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=11, fontweight='bold')

    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Wall-Clock Seconds (Mean, 5 Seeds)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('With Remote Tool / LLM Latency (20 ms / task)', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, color=STYLE['text_main'])

    ax1.annotate('5.2x Wall-Clock Speedup!\n(Concurrent I/O Overlap)', xy=(1, r_lat[1]), xytext=(0.5, 0.55),
                 arrowprops=dict(arrowstyle="->", color=STYLE['cyan'], lw=1.8),
                 color=STYLE['cyan'], fontsize=10.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['cyan']))

    # Panel 2: Zero Artificial Delay (Local CPU bound)
    r_no_lat = [no_lat_data['specialists_all_serial']['runtime_seconds']['mean'] * 1000,
                no_lat_data['specialists_all_parallel']['runtime_seconds']['mean'] * 1000]

    bars2 = ax2.bar(labels, r_no_lat, color=[STYLE['emerald'], STYLE['rose']], width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars2, r_no_lat):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2.0, f"{val:.1f} ms",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=11, fontweight='bold')

    ax2.set_ylim(0, 95)
    ax2.set_ylabel('Wall-Clock Milliseconds (Mean, 3 Seeds)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_title('Zero Added Delay (In-Memory CPU Bound)', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, color=STYLE['text_main'])

    ax2.annotate('Parallel is 42% Slower!\n(Thread synchronization overhead)', xy=(1, r_no_lat[1]), xytext=(0.5, 55),
                 arrowprops=dict(arrowstyle="->", color=STYLE['rose'], lw=1.8),
                 color=STYLE['rose'], fontsize=10.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['rose']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_safeguard_ablations(root: Path, output_path: Path):
    """Draw impact of removing holdout verification and multiple testing correction."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    s = data['summary']

    configs = ['Both Guards\n(Full System)', 'No Holdout\nVerification', 'No BH FDR\nCorrection', 'Neither Guard\n(Uncontrolled)']
    keys = ['specialists_all_parallel', 'no_holdout_verification', 'no_multiple_testing_correction', 'neither_safeguard']

    unreplicated = [s[k]['claims_without_replication']['mean'] for k in keys]
    null_claims = [s[k]['known_null_claims']['mean'] for k in keys]
    precision = [s[k]['testable_precision']['mean'] * 100 for k in keys]

    x = np.arange(len(configs))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    # Panel 1: Unverified claims and known synthetic nulls
    rects1 = ax1.bar(x - width/2, unreplicated, width, label='Unreplicated Claims', color=STYLE['amber'], edgecolor='#FFFFFF', linewidth=0.5)
    rects2 = ax1.bar(x + width/2, null_claims, width, label='Known Synthetic Nulls (Noise)', color=STYLE['rose'], edgecolor='#FFFFFF', linewidth=0.5)

    for r in rects1:
        ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.1, f"{r.get_height():.1f}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    for r in rects2:
        ax1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.1, f"{r.get_height():.1f}",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10, fontweight='bold')

    ax1.set_ylim(0, 4.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9.5, color=STYLE['text_main'])
    ax1.set_ylabel('False Claims Accepted (Mean, 5 Seeds)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('False Discoveries Admitted by Removing Safeguards', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
               labelcolor=STYLE['text_main'], fontsize=9.5)

    # Panel 2: Testable Precision (%)
    colors = [STYLE['emerald'], STYLE['amber'], STYLE['emerald'], STYLE['rose']]
    bars2 = ax2.bar(configs, precision, color=colors, width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars2, precision):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10.5, fontweight='bold')

    ax2.set_ylim(0, 115)
    ax2.set_ylabel('Testable Holdout Precision (%)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_title('Precision Collapse Under Uncontrolled Search', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=9.5, color=STYLE['text_main'])

    ax2.annotate('Precision drops to 73.1%!\n(1 in 4 claims is false)', xy=(3, precision[3]), xytext=(2.2, 55),
                 arrowprops=dict(arrowstyle="->", color=STYLE['rose'], lw=1.8),
                 color=STYLE['rose'], fontsize=10.5, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['rose']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_token_efficiency(root: Path, output_path: Path):
    """Draw proxy cost and token efficiency vs verified claim yield."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    s = data['summary']

    configs = ['Generalist 8', 'Specialists 8', 'Full Parallel 37']
    keys = ['generalist_8', 'specialists_8_verified', 'specialists_all_parallel']

    replicated = [s[k]['independently_replicated_claims']['mean'] for k in keys]
    dollars = [s[k]['proxy_dollars']['mean'] * 100 for k in keys]  # in cents

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    scatter_colors = [STYLE['cyan'], STYLE['purple'], STYLE['emerald']]
    for c, r, d, label in zip(scatter_colors, replicated, dollars, configs):
        ax.scatter(d, r, color=c, s=240, edgecolor='#FFFFFF', linewidth=1.5, zorder=3, label=label)
        ax.text(d, r + 0.18, f"{label}\n({r:.1f} claims, {d:.2f}¢)", color=STYLE['text_main'],
                fontsize=9.5, fontweight='bold', ha='center', zorder=4)

    ax.plot(dollars, replicated, color=STYLE['text_muted'], linestyle=':', lw=1.5, zorder=2)

    ax.set_xlim(0.8, 6.5)
    ax.set_ylim(6.0, 9.5)
    ax.set_xlabel('Proxy Cost (Cents USD @ $3.00 / M tokens)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Independently Replicated Claims', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax.set_title('Scientific Return on Compute: Verified Discoveries vs Compute Cost', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax.tick_params(colors=STYLE['text_muted'], which='both')
    ax.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text_main'], fontsize=9.5, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def main():
    root = Path(__file__).resolve().parent
    fig_dir = root / 'figures'
    fig_dir.mkdir(exist_ok=True)

    draw_orchestration_architecture(fig_dir / '01_orchestrator_architecture.png')
    draw_coverage_and_discovery(root, fig_dir / '02_coverage_and_discovery.png')
    draw_parallel_latency(root, fig_dir / '03_parallel_latency_scaling.png')
    draw_safeguard_ablations(root, fig_dir / '04_safeguard_ablations.png')
    draw_token_efficiency(root, fig_dir / '05_token_evidence_efficiency.png')
    print("All figures successfully created in figures/")


if __name__ == '__main__':
    main()
