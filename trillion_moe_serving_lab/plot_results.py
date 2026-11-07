"""Generate publication-grade dark-slate figures for trillion_moe_serving_lab."""

import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Styling configuration
BG_DARK = "#0e1117"
PANEL_BG = "#161b22"
BORDER_COLOR = "#30363d"
TEXT_COLOR = "#f0f6fc"
MUTED_COLOR = "#8b949e"

COLOR_BLUE = "#58a6ff"       # Primary
COLOR_PURPLE = "#bc8cff"     # Total / Capacity
COLOR_GREEN = "#3fb950"      # Active / Efficient
COLOR_ORANGE = "#f0883e"     # Warning / Gap
COLOR_RED = "#f85149"        # Fallacy / Cost
COLOR_CYAN = "#39c5cf"       # Hardware

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": BORDER_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
    "grid.color": "#21262d",
    "text.color": TEXT_COLOR,
    "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "sans-serif"],
})

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_total_vs_active():
    """Figure 1: Total Stored Parameters vs Active Parameters per Token."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    models = ["Kimi K3.1 Pro", "Mistral Large 4", "Tencent Hy4", "Inkling 2 Small"]
    total_b = [2800, 1050, 770, 594]
    active_b = [104, 52, 49, 17]

    x = np.arange(len(models))
    width = 0.35

    b1 = ax.bar(x - width/2, total_b, width, label="Total Stored Parameters (Memory Budget)",
                color=COLOR_PURPLE, edgecolor=BORDER_COLOR)
    b2 = ax.bar(x + width/2, active_b, width, label="Active Parameters per Token (Compute Budget)",
                color=COLOR_GREEN, edgecolor=BORDER_COLOR)

    ax.set_ylabel("Parameters (Billions)", fontsize=11)
    ax.set_title("The Trillion-Parameter Asymmetry: Stored Capacity vs. Active Arithmetic", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(10, 4500)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, fontsize=10.5)

    # Label ratios
    for i, (tot, act) in enumerate(zip(total_b, active_b)):
        ratio = (act / tot) * 100
        ax.text(i - width/2, tot * 1.15, f"{tot}B", ha="center", fontsize=9.5, fontweight="bold", color=COLOR_PURPLE)
        ax.text(i + width/2, act * 1.15, f"{act}B\n({ratio:.1f}%)", ha="center", fontsize=9, fontweight="bold", color=COLOR_GREEN)

    plt.tight_layout()
    out = FIG_DIR / "01_total_vs_active_parameters.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


def plot_hardware_memory_h100():
    """Figure 2: Weight Footprint vs 80GB H100 Node Boundaries."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    models = ["Inkling 2 Small\n(594B)", "Tencent Hy4\n(770B)", "Mistral Large 4\n(1.05T)", "Kimi K3.1 Pro\n(2.8T)"]
    fp16 = [1188, 1540, 2100, 5600]
    fp8 = [594, 770, 1050, 2800]
    fp4 = [297, 385, 525, 1400]

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, fp16, width, label="16-bit (BF16 / FP16)", color="#f8514988", edgecolor=COLOR_RED)
    ax.bar(x, fp8, width, label="8-bit (FP8)", color="#f0883e88", edgecolor=COLOR_ORANGE)
    ax.bar(x + width, fp4, width, label="4-bit (Idealized FP4)", color="#3fb95088", edgecolor=COLOR_GREEN)

    # Node capacity horizontal lines (8x 80GB H100s = 640GB per node)
    node_configs = [(640, "1 Node (8x H100: 640 GB)", "#58a6ff"),
                    (1280, "2 Nodes (16x H100: 1.28 TB)", "#79c0ff"),
                    (1920, "3 Nodes (24x H100: 1.92 TB)", "#a5d6ff")]
    for cap, label, col in node_configs:
        ax.axhline(cap, color=col, linestyle="--", lw=1.5, alpha=0.85)
        ax.text(3.35, cap + 40, label, color=col, fontsize=9, fontweight="bold")

    ax.set_ylabel("Weight VRAM Footprint (GB)", fontsize=11)
    ax.set_title("Raw Weight Footprint vs. 8-GPU H100 Node Thresholds", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 3200)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, fontsize=10)

    # Annotate Kimi FP4 requirement
    ax.annotate("Kimi FP4: 1.4 TB\nRequires 18+ H100s\n(3 full nodes!)", xy=(3 + width, 1400), xytext=(2.6, 2200),
                arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=2), fontsize=9.5, color=COLOR_RED, fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "02_hardware_memory_footprint_h100.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


def plot_kv_cache_scaling():
    """Figure 3: KV Cache Growth from 4k to 1M Tokens (MHA vs GQA vs MLA)."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    contexts = [4_096, 32_768, 131_072, 262_144, 524_288, 1_048_576]
    ctx_labels = ["4k", "32k", "131k", "262k", "524k", "1M"]
    x = np.arange(len(contexts))

    # For 72-layer transformer, head_dim 128, 2 bytes/element:
    # Standard MHA (64 heads): 2 * 72 * L * 64 * 128 * 2
    # GQA (8 heads): 2 * 72 * L * 8 * 128 * 2
    # MLA (latent 512 + 64 rope): 72 * L * (576) * 2
    mha_gb = [(2 * 72 * c * 64 * 128 * 2) / (1024**3) for c in contexts]
    gqa_gb = [(2 * 72 * c * 8 * 128 * 2) / (1024**3) for c in contexts]
    mla_gb = [(72 * c * 576 * 2) / (1024**3) for c in contexts]

    ax.plot(x, mha_gb, marker="o", color=COLOR_RED, lw=2.5, label="Standard MHA (64 KV Heads)")
    ax.plot(x, gqa_gb, marker="s", color=COLOR_ORANGE, lw=2.5, label="Grouped-Query Attention (8 KV Heads)")
    ax.plot(x, mla_gb, marker="^", color=COLOR_GREEN, lw=2.5, label="Multi-Head Latent Attention (MLA)")

    ax.set_ylabel("KV Cache Footprint per Batch (GB)", fontsize=11)
    ax.set_title("KV Cache VRAM Explosion at 1 Million Tokens", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ctx_labels, fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 4000)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, fontsize=10.5)

    ax.annotate("MHA at 1M:\n2,360 GB VRAM!\n(Completely unservable)", xy=(5, mha_gb[-1]), xytext=(3.8, 1200),
                arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=1.8), fontsize=9, color=COLOR_RED, fontweight="bold")

    ax.annotate("MLA at 1M:\n~79 GB VRAM\n(Fits in 1 GPU cache)", xy=(5, mla_gb[-1]), xytext=(4.2, 8),
                arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1.8), fontsize=9, color=COLOR_GREEN, fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "03_kv_cache_scaling_1m_context.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


def plot_benchmark_version_trap():
    """Figure 4: Artificial Analysis v4.3 vs v5 Intelligence Index & Cross-Version Fallacy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Left: True same-version scores on v5
    models_v5 = ["Claude Opus 5.5\n(Closed)", "Kimi K3.1 Pro\n(Open)", "Tencent Hy4\n(Open)", "Inkling 2 Sm\n(Open)", "Mistral L4\n(Open)"]
    scores_v5 = [58, 46, 35, 31, 26]
    colors_v5 = [COLOR_PURPLE, COLOR_BLUE, COLOR_CYAN, COLOR_GREEN, COLOR_ORANGE]

    ax1.barh(np.arange(len(models_v5)), scores_v5, color=colors_v5, edgecolor=BORDER_COLOR, height=0.55)
    ax1.set_yticks(np.arange(len(models_v5)))
    ax1.set_yticklabels(models_v5, fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Intelligence Index v5 Score", fontsize=11)
    ax1.set_title("Same-Version v5 Evaluation (Oct 2026)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlim(0, 70)
    ax1.grid(axis="x", linestyle="--", alpha=0.3)

    for i, s in enumerate(scores_v5):
        ax1.text(s + 1.2, i, str(s), va="center", fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    # Right: Cross-Version Fallacy Decomposition (Kimi vs Mistral)
    labels = ["v5 Same-Version Gap\n(46 vs 26)", "v4.3 vs v5 Cross Fallacy\n(55 vs 26)", "Artificial Inflation"]
    gaps = [20.0, 29.0, 9.0]
    bar_cols = [COLOR_BLUE, COLOR_RED, COLOR_ORANGE]

    ax2.bar(labels, gaps, color=bar_cols, width=0.5, edgecolor=BORDER_COLOR)
    ax2.set_ylabel("Gap Points", fontsize=11)
    ax2.set_title("The Benchmark Version Trap: Manufacturing a 29-Point Gap", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(0, 35)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    for i, g in enumerate(gaps):
        ax2.text(i, g + 0.8, f"+{g:.1f} pts", ha="center", fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    plt.tight_layout()
    out = FIG_DIR / "04_benchmark_version_trap_gap.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


def plot_task_economics_verbosity():
    """Figure 5: Active Parameters vs Output Token Verbosity in Task Cost."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    models = ["Inkling 2 Small", "Kimi K3.1 Pro", "Claude Opus 5.5", "Tencent Hy4", "Mistral Large 4"]
    active_params = [17, 104, 120, 49, 52]  # Approximate active B
    tokens_k = [154, 140, 125, 115, 110]   # Thousands of tokens per task evaluation

    x = np.arange(len(models))

    ax.scatter(active_params, tokens_k, s=350, c=[COLOR_GREEN, COLOR_BLUE, COLOR_PURPLE, COLOR_CYAN, COLOR_ORANGE],
               edgecolor="#ffffff", linewidth=1.5, zorder=4)

    for i, m in enumerate(models):
        offset_y = 3.5 if i % 2 == 0 else -4.5
        ax.text(active_params[i], tokens_k[i] + offset_y, f"{m}\n({active_params[i]}B act / {tokens_k[i]}k tok)",
                ha="center", fontsize=9.5, fontweight="bold", color=TEXT_COLOR)

    ax.set_xlabel("Active Parameters per Token (Arithmetic FLOPs)", fontsize=11)
    ax.set_ylabel("Average Tokens Generated per Task (Thousands)", fontsize=11)
    ax.set_title("Task Economics: Why Ultra-Sparse Models Can Cost More", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, 140)
    ax.set_ylim(95, 175)

    ax.annotate("High Reasoning Verbosity:\n154k tokens per task\n(Overcomes 17B active sparsity)",
                xy=(17, 154), xytext=(45, 163),
                arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1.8),
                fontsize=9.5, color=COLOR_GREEN, fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "05_task_economics_verbosity_vs_speed.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


if __name__ == "__main__":
    plot_total_vs_active()
    plot_hardware_memory_h100()
    plot_kv_cache_scaling()
    plot_benchmark_version_trap()
    plot_task_economics_verbosity()
    print("All 5 figures successfully generated in figures/")

