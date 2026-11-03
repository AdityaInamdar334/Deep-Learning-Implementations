"""Generate publication-ready figures for SpikeGuard Lab.

Produces:
1. figures/01_bounded_agent_architecture.png: Bounded autonomy vs unbounded improvisation
2. figures/02_spike_raster_epochs.png: Synthetic spike raster with marked cue epochs
3. figures/03_evaluator_scorecard.png: 14-point methodological integrity scorecard
4. figures/04_runtime_accuracy_tradeoff.png: Verification cost vs methodological score
5. figures/05_sttc_synchrony_matrix.png: Pairwise STTC spike-timing synchrony matrix
"""
import json
import os
import numpy as np
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spikeguard_lab.data import make_synthetic_recording
from spikeguard_lab.methods import sttc

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
    """Draw architecture comparing unbounded LLM improvisation against bounded scientific agent."""
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

    # Header Panel 1: Unbounded Improvisation
    ax1.text(5, 12.3, "Unbounded Generalist: Methodological Drift",
             color=STYLE['rose'], fontsize=14, fontweight='bold', ha='center')
    ax1.text(5, 11.7, "Larger model writes executable code but silently improvises invalid methods",
             color=STYLE['text_muted'], fontsize=9.5, ha='center')

    # Header Panel 2: Bounded Autonomy
    ax2.text(5, 12.3, "Bounded Scientific Agent: Versioned Tool Contracts",
             color=STYLE['emerald'], fontsize=14, fontweight='bold', ha='center')
    ax2.text(5, 11.7, "Constrained agent invokes validated domain tools and verifies outputs",
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

    # --- LEFT: IMPROVISED FLOW ---
    draw_box(ax1, 5, 10.3, 8.4, 1.2,
             "1. Ambiguous Prompt Interpretation",
             "Confuses 'no-cue' with inter-trial intervals without clarifying",
             bg="#3B1219", border=STYLE['rose'], title_color="#FDA4AF")

    draw_arrow(ax1, 5, 9.6, 5, 8.8, color=STYLE['rose'])

    draw_box(ax1, 5, 8.1, 8.4, 1.2,
             "2. Method Substitution (Ad-hoc Coding)",
             "Substitutes binned Pearson correlation for STTC spike synchrony",
             bg=STYLE['bg_card_alt'], border="#64748B")

    draw_arrow(ax1, 5, 7.4, 5, 6.6, color="#64748B")

    draw_box(ax1, 5, 5.9, 8.4, 1.2,
             "3. Silent Downsampling & Self-Correlation",
             "Coarsens bins (50ms -> 200ms); includes neuron in its own population",
             bg="#27192A", border=STYLE['amber'], title_color="#FDE68A")

    draw_arrow(ax1, 5, 5.2, 5, 4.4, color="#64748B")

    draw_box(ax1, 5, 3.7, 8.4, 1.1,
             "4. Unchecked Self-Declaration",
             "Generates polished summary without provenance or p-values",
             bg=STYLE['bg_card'], border="#475569")

    # Left Outcome Badge
    outcome_left = patches.FancyBboxPatch((0.8, 0.6), 8.4, 2.1, boxstyle="round,pad=0.3",
                                         facecolor="#1F1315", edgecolor=STYLE['rose'], linewidth=2.0, zorder=2)
    ax1.add_patch(outcome_left)
    ax1.text(5, 2.1, "METHODOLOGICAL EVALUATION", color=STYLE['rose'], fontsize=11, fontweight='bold', ha='center')
    ax1.text(5, 1.55, "Score: 2 / 14 Checks Passed (14.3%)", color="#FECDD3", fontsize=11, fontweight='bold', ha='center')
    ax1.text(5, 1.0, "Plausible text & plots, but invalid neuroscience statistics!", color=STYLE['text_sub'], fontsize=9, ha='center')

    # --- RIGHT: BOUNDED FLOW ---
    draw_box(ax2, 5, 10.3, 8.4, 1.2,
             "1. Explicit Condition Validation & Escalation",
             "Enforces cue/no-cue definitions; requests clarification if ambiguous",
             bg="#062E25", border=STYLE['emerald'], title_color="#6EE7B7")

    draw_arrow(ax2, 5, 9.6, 5, 8.8, color=STYLE['emerald'])

    draw_box(ax2, 5, 8.1, 8.4, 1.2,
             "2. Versioned Tool Registry (STTC & Surrogates)",
             "Calls Cutts-Eglen STTC (dt=20ms) + 19 circular-shift null tests",
             bg="#132B40", border=STYLE['sky'], title_color="#7DD3FC")

    draw_arrow(ax2, 5, 7.4, 5, 6.6, color=STYLE['sky'])

    draw_box(ax2, 5, 5.9, 8.4, 1.2,
             "3. Statistical Rigor (BH FDR & Leave-One-Out)",
             "Controls FDR <= 0.05; computes leave-one-out population coupling",
             bg=STYLE['bg_card_alt'], border=STYLE['purple'], title_color="#E9D5FF")

    draw_arrow(ax2, 5, 5.2, 5, 4.4, color=STYLE['purple'])

    draw_box(ax2, 5, 3.7, 8.4, 1.1,
             "4. Independent Reference Evaluation",
             "Audits SHA-256 digests, bin bounds, and numerical reproducibility",
             bg=STYLE['bg_card'], border="#475569")

    # Right Outcome Badge
    outcome_right = patches.FancyBboxPatch((0.8, 0.6), 8.4, 2.1, boxstyle="round,pad=0.3",
                                          facecolor="#0B231D", edgecolor=STYLE['emerald'], linewidth=2.0, zorder=2)
    ax2.add_patch(outcome_right)
    ax2.text(5, 2.1, "METHODOLOGICAL EVALUATION", color=STYLE['emerald'], fontsize=11, fontweight='bold', ha='center')
    ax2.text(5, 1.55, "Score: 14 / 14 Checks Passed (100.0%)", color="#A7F3D0", fontsize=11, fontweight='bold', ha='center')
    ax2.text(5, 1.0, "Verified numerical equivalence against ground-truth reference!", color=STYLE['text_sub'], fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_spike_raster(output_path: Path):
    """Draw multi-channel synthetic spike raster with highlighted cue window."""
    rec = make_synthetic_recording(seed=0, n_units=12, duration_ms=5000.0)
    cue_start, cue_end = rec.metadata['cue_ms']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6.5), gridspec_kw={'height_ratios': [2.8, 1]})
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    # Raster plot
    t_seconds = rec.duration_ms / 1000.0
    for i, spikes in enumerate(rec.spikes):
        s_sec = spikes / 1000.0
        ax1.vlines(s_sec, i + 0.15, i + 0.85, color=STYLE['sky'], linewidth=0.85, alpha=0.85)

    # Highlight cue window
    ax1.axvspan(cue_start / 1000.0, cue_end / 1000.0, color=STYLE['amber'], alpha=0.18,
                label=f"Cue Epoch [{cue_start/1000:.1f}s - {cue_end/1000:.1f}s]")
    ax1.axvspan(0, cue_start / 1000.0, color=STYLE['blue'], alpha=0.08, label="Baseline Pre-Cue")
    ax1.axvspan(cue_end / 1000.0, t_seconds, color=STYLE['purple'], alpha=0.08, label="Post-Cue Epoch")

    ax1.set_xlim(0, t_seconds)
    ax1.set_ylim(-0.2, rec.n_units + 0.2)
    ax1.set_ylabel('Neural Unit ID', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('Synthetic 12-Unit Spike Raster Across Behavioral Epochs', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.2, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.legend(frameon=True, facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['grid'],
               labelcolor=STYLE['text_main'], fontsize=9.5, loc='upper right')

    # Instantaneous population rate
    bin_ms = 50.0
    bins = np.arange(0, rec.duration_ms + bin_ms, bin_ms)
    all_spikes = np.concatenate(rec.spikes)
    counts, _ = np.histogram(all_spikes, bins=bins)
    rate_hz = (counts / (bin_ms / 1000.0)) / rec.n_units
    bin_centers = (bins[:-1] + bin_ms / 2.0) / 1000.0

    ax2.plot(bin_centers, rate_hz, color=STYLE['cyan'], lw=1.6)
    ax2.axvspan(cue_start / 1000.0, cue_end / 1000.0, color=STYLE['amber'], alpha=0.18)
    ax2.set_xlim(0, t_seconds)
    ax2.set_xlabel('Recording Time (Seconds)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_ylabel('Pop. Rate (Hz)', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.2, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_evaluator_scorecard(root: Path, output_path: Path):
    """Draw 14-point methodological integrity scorecard heatmap across policies."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    policies = ['improvised', 'silent_downsample', 'no_verification', 'bounded']
    policy_labels = ['Unbounded\n(Improvised)', 'Silent\nDownsample', 'No Surrogate\nVerification', 'Bounded\n(Full Safeguards)']

    check_names = [
        ('conditions_explicit', '1. Explicit Condition Intervals'),
        ('ambiguity_escalated', '2. Ambiguity Escalation / Request'),
        ('cue_units_preserved', '3. Preserved Cue Units (12/12)'),
        ('cue_spikes_preserved', '4. Preserved Cue Spike Times'),
        ('no_cue_spikes_preserved', '5. Preserved Baseline Spikes'),
        ('all_pairs_present', '6. All Pairs Computed (66/66)'),
        ('sttc_method_named', '7. STTC Method Explicitly Named'),
        ('sttc_matches_reference', '8. STTC Matches Ground Truth'),
        ('surrogate_p_values_present', '9. Circular-Shift Surrogate p-values'),
        ('multiple_testing_q_values_present', '10. Benjamini-Hochberg FDR q-values'),
        ('no_undocumented_rebin', '11. No Undocumented Re-binning'),
        ('leave_one_out_coupling', '12. Leave-One-Out Pop. Coupling'),
        ('pca_matches_reference', '13. PCA Matches Reference Dim'),
        ('provenance_and_caveats', '14. Provenance SHA-256 Recorded'),
    ]

    # Compute pass rate matrix (14 checks x 4 policies)
    matrix = np.zeros((len(check_names), len(policies)))
    for p_idx, p in enumerate(policies):
        p_rows = [r for r in data['rows'] if r['policy'] == p]
        for c_idx, (c_key, _) in enumerate(check_names):
            passes = sum(1 for r in p_rows if r['checks'][c_key])
            matrix[c_idx, p_idx] = passes / len(p_rows)

    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax.set_facecolor(STYLE['bg_card'])

    # Heatmap display
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policy_labels, fontsize=10.5, color=STYLE['text_main'], fontweight='bold')
    ax.set_yticks(range(len(check_names)))
    ax.set_yticklabels([name for _, name in check_names], fontsize=9.5, color=STYLE['text_main'])

    # Annotate cells
    for i in range(len(check_names)):
        for j in range(len(policies)):
            val = matrix[i, j]
            text = "PASS" if val == 1.0 else ("FAIL" if val == 0.0 else f"{val*100:.0f}%")
            color = "#000000" if val > 0.6 else "#FFFFFF"
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9.5, fontweight='bold')

    ax.set_title('14-Point Methodological Integrity Scorecard (5-Seed Benchmark)',
                 color=STYLE['text_main'], fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors=STYLE['text_muted'], which='both')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0% Pass', '50% Pass', '100% Pass'])
    cbar.ax.tick_params(colors=STYLE['text_muted'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_runtime_tradeoff(root: Path, output_path: Path):
    """Draw compute runtime vs methodological score trade-off."""
    data = json.loads((root / 'results/benchmark.json').read_text())
    s = data['summary']

    policies = ['improvised', 'silent_downsample', 'no_verification', 'bounded']
    labels = ['Unbounded\n(Improvised)', 'Silent\nDownsample', 'No Surrogate\nVerification', 'Bounded\n(Full Rigor)']

    scores = [s[p]['mean_score'] for p in policies]
    runtimes = [s[p]['mean_runtime_s'] * 1000.0 for p in policies]  # in milliseconds

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])

    # Panel 1: Check score comparison
    colors = [STYLE['rose'], STYLE['amber'], STYLE['sky'], STYLE['emerald']]
    bars1 = ax1.bar(labels, scores, color=colors, width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.1f} / 14",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10.5, fontweight='bold')

    ax1.axhline(14, color=STYLE['emerald'], linestyle=':', lw=1.2, alpha=0.7)
    ax1.set_ylim(0, 16)
    ax1.set_ylabel('Evaluator Checks Passed (out of 14)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax1.set_title('Independent Evaluation Score by Agent Policy', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax1.tick_params(colors=STYLE['text_muted'], which='both')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=10, color=STYLE['text_main'])

    # Panel 2: Runtime in milliseconds
    bars2 = ax2.bar(labels, runtimes, color=colors, width=0.48, edgecolor='#FFFFFF', linewidth=0.6, alpha=0.9)
    for bar, val in zip(bars2, runtimes):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 6.0, f"{val:.1f} ms",
                 ha='center', va='bottom', color=STYLE['text_main'], fontsize=10.5, fontweight='bold')

    ax2.set_ylim(0, 310)
    ax2.set_ylabel('Execution Time (Milliseconds)', color=STYLE['text_main'], fontsize=11, fontweight='bold')
    ax2.set_title('Computational Cost of Statistical Verification', color=STYLE['text_main'], fontsize=12.5, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.25, color=STYLE['grid'])
    ax2.tick_params(colors=STYLE['text_muted'], which='both')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=10, color=STYLE['text_main'])

    # Annotate why No Verification is faster
    ax2.annotate('18x Faster without\nSurrogate Tests (14ms vs 250ms)', xy=(2, runtimes[2]), xytext=(1.2, 160),
                 arrowprops=dict(arrowstyle="->", color=STYLE['cyan'], lw=1.8),
                 color=STYLE['cyan'], fontsize=10, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=STYLE['bg_card_alt'], edgecolor=STYLE['cyan']))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_sttc_matrix(output_path: Path):
    """Draw pairwise STTC spike-timing tiling coefficient matrices for cue vs baseline."""
    rec = make_synthetic_recording(seed=0, n_units=8, duration_ms=5000.0)
    cue_start, cue_end = rec.metadata['cue_ms']

    # Compute STTC matrix during cue
    cue_rec = rec.window(cue_start, cue_end)
    sttc_cue = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            sttc_cue[i, j] = sttc(cue_rec.spikes[i], cue_rec.spikes[j], cue_rec.duration_ms, delta_ms=20.0)

    # Compute STTC matrix during baseline
    base_rec = rec.window(0.0, cue_start)
    sttc_base = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            sttc_base[i, j] = sttc(base_rec.spikes[i], base_rec.spikes[j], base_rec.duration_ms, delta_ms=20.0)

    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(14, 5.8), gridspec_kw={'width_ratios': [1, 1, 0.05]})
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_card'])
    ax2.set_facecolor(STYLE['bg_card'])
    cax.set_facecolor(STYLE['bg_card'])

    im1 = ax1.imshow(sttc_base, cmap='viridis', vmin=-0.2, vmax=0.8)
    ax1.set_title('Baseline STTC Matrix (Pre-Cue, dt=20ms)', color=STYLE['text_main'], fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Unit ID', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    ax1.set_ylabel('Unit ID', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    ax1.tick_params(colors=STYLE['text_muted'])

    im2 = ax2.imshow(sttc_cue, cmap='viridis', vmin=-0.2, vmax=0.8)
    ax2.set_title('Cue Epoch STTC Matrix (Synchrony Evoked, dt=20ms)', color=STYLE['text_main'], fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel('Unit ID', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    ax2.set_ylabel('Unit ID', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    ax2.tick_params(colors=STYLE['text_muted'])

    # Colorbar in dedicated axis
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label('Spike-Time Tiling Coefficient (STTC)', color=STYLE['text_main'], fontsize=10, fontweight='bold')
    cbar.ax.tick_params(colors=STYLE['text_muted'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def main():
    root = Path(__file__).resolve().parent
    fig_dir = root / 'figures'
    fig_dir.mkdir(exist_ok=True)

    draw_architecture_diagram(fig_dir / '01_bounded_agent_architecture.png')
    draw_spike_raster(fig_dir / '02_spike_raster_epochs.png')
    draw_evaluator_scorecard(root, fig_dir / '03_evaluator_scorecard.png')
    draw_runtime_tradeoff(root, fig_dir / '04_runtime_accuracy_tradeoff.png')
    draw_sttc_matrix(fig_dir / '05_sttc_synchrony_matrix.png')
    print("All SpikeGuard figures successfully created in figures/")


if __name__ == '__main__':
    main()
