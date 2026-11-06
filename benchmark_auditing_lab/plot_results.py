"""Generate publication-grade dark-slate figures for benchmark_auditing_lab."""

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

COLOR_BLUE = "#58a6ff"       # Primary / Clean
COLOR_PURPLE = "#bc8cff"     # Analytical
COLOR_GREEN = "#3fb950"      # Verified
COLOR_ORANGE = "#f0883e"     # Defect / Warning
COLOR_RED = "#f85149"        # Flawed / Error
COLOR_CYAN = "#39c5cf"       # Contrast

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


def plot_measurement_layer_architecture():
    """Figure 1: Measurement Layer Architecture & Defect Surfaces."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.text(50, 95, "The AI Measurement Layer & Benchmark Vulnerability Surface", fontsize=16, fontweight="bold", ha="center", color=TEXT_COLOR)
    ax.text(50, 89, "How subtle evaluation defects distort reported model capabilities without model changes", fontsize=10.5, ha="center", color=MUTED_COLOR)

    boxes = [
        (4, 52, 26, 28, "1. Model Generation", "Candidate Solution\nZero Retraining\nMathematical Reasoning", COLOR_BLUE),
        (37, 52, 26, 28, "2. Measurement Sandbox", "Execution Environment\nTest Assertions\nNumerical Comparison", COLOR_ORANGE),
        (70, 52, 26, 28, "3. Reported Metrics", "Subproblem Pass Rate\nWhole-Problem Credit\nLeaderboard Ranking", COLOR_PURPLE),
        (20, 12, 28, 28, "A. Score-Suppressing Defects", "Undocumented Constraints\nUnreasonable Float Tol\nDefective Ground Truth", COLOR_RED),
        (54, 12, 28, 28, "B. Verified Measurement", "Domain-Expert Audited\nAnalytical Tolerances\nClean Cascades (69-92%)", COLOR_GREEN),
    ]

    for x, y, w, h, title, sub, col in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8", facecolor=PANEL_BG, edgecolor=col, linewidth=2, transform=ax.transData)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 5, title, fontsize=11, fontweight="bold", ha="center", va="center", color=col)
        ax.text(x + w / 2, y + h / 2 - 2, sub, fontsize=9, ha="center", va="center", color=TEXT_COLOR, multialignment="center")

    # Connectors
    ax.annotate("", xy=(37, 66), xytext=(30, 66), arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=2.5))
    ax.annotate("", xy=(70, 66), xytext=(63, 66), arrowprops=dict(arrowstyle="->", color=COLOR_ORANGE, lw=2.5))
    ax.annotate("Defect Injection", xy=(34, 40), xytext=(50, 52), arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=2.5))
    ax.annotate("Audit & Repair", xy=(68, 40), xytext=(50, 52), arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=2.5))
    ax.annotate("", xy=(54, 26), xytext=(48, 26), arrowprops=dict(arrowstyle="->", color=COLOR_CYAN, lw=2.5))

    plt.tight_layout()
    out = FIG_DIR / "01_measurement_layer_architecture.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


def plot_scicode_cascading_recovery():
    """Figure 2: SciCode Subproblem vs Whole-Problem Recovery Curve."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ["Subproblem Accuracy\n(Step-Level)", "Whole-Problem Accuracy\n(Multi-Step Cascade)"]
    orig_low, orig_high = [45, 9], [60, 27]
    verif_low, verif_high = [84, 69], [98, 92]

    x = np.array([0.8, 2.2])
    width = 0.35

    # Original bars
    orig_means = [(l + h) / 2 for l, h in zip(orig_low, orig_high)]
    orig_err = [(h - l) / 2 for l, h in zip(orig_low, orig_high)]
    b1 = ax.bar(x - width/2, orig_means, width, yerr=orig_err, capsize=6,
                label="Original SciCode (Defective)", color="#f8514933", edgecolor=COLOR_RED, linewidth=1.8)

    # Verified bars
    verif_means = [(l + h) / 2 for l, h in zip(verif_low, verif_high)]
    verif_err = [(h - l) / 2 for l, h in zip(verif_low, verif_high)]
    b2 = ax.bar(x + width/2, verif_means, width, yerr=verif_err, capsize=6,
                label="SciCode-Verified (Audited)", color="#3fb95033", edgecolor=COLOR_GREEN, linewidth=1.8)

    ax.set_ylabel("Accuracy Range (%) across 12 Frontier Models", fontsize=11)
    ax.set_title("The Measurement Dividend: SciCode Defect Repair Without Model Retraining", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, fontsize=10.5)

    # Annotations
    ax.annotate("Cascading Multi-Step Collapse\n(9–27% Whole Problem)", xy=(x[1] - width/2, 18), xytext=(x[1] - 0.75, 45),
                arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=1.5), fontsize=9, color=COLOR_RED, fontweight="bold", ha="center")

    ax.annotate("+65 pp Whole-Problem Surge!\n(Zero Retraining Required)", xy=(x[1] + width/2, 80), xytext=(x[1] + 0.45, 104),
                arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1.5), fontsize=9.5, color=COLOR_GREEN, fontweight="bold", ha="center")

    plt.tight_layout()
    out = FIG_DIR / "02_scicode_cascading_defect_recovery.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


def plot_defect_taxonomy():
    """Figure 3: Taxonomy and Breakdown of 192 Score-Suppressing Defects."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    defects = [
        "Unreasonable Numerical Tolerances\n(e.g. 1e-8 on stiff ODEs)",
        "Undocumented Implementation Constraints\n(hidden helper names)",
        "Defective Reference Answer Keys\n(arithmetic / sign errors)",
        "Inconsistent Mathematical Conventions\n(indexing, branch cuts)",
    ]
    counts = [74, 52, 38, 28]  # Sum = 192
    colors = [COLOR_BLUE, COLOR_PURPLE, COLOR_RED, COLOR_ORANGE]

    y_pos = np.arange(len(defects))
    bars = ax.barh(y_pos, counts, align="center", color=colors, edgecolor=BORDER_COLOR, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(defects, fontsize=10.5, fontweight="bold")
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Number of Score-Suppressing Defects (SciCode Domain Audit, N=192)", fontsize=11)
    ax.set_title("Taxonomy of Benchmark Defects Rejecting Valid Solutions", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, count in zip(bars, counts):
        pct = (count / 192) * 100
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f"{count} ({pct:.1f}%)",
                va="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    ax.set_xlim(0, 95)
    plt.tight_layout()
    out = FIG_DIR / "03_defect_taxonomy_distribution.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


def plot_deepquestion_cognitive_collapse():
    """Figure 4: DeepQuestion Cognitive Complexity Decay across Bloom's Taxonomy."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    levels = ["Level 1:\nRecall", "Level 2:\nApplication", "Level 3:\nAnalysis", "Level 4:\nSynthesis & Eval"]
    frontier_acc = [98.0, 78.0, 74.0, 48.0]
    chat_acc = [92.0, 68.0, 60.0, 24.0]
    memorizer_acc = [95.0, 62.0, 42.0, 4.0]

    x = np.arange(len(levels))

    ax.plot(x, frontier_acc, marker="o", lw=2.5, color=COLOR_BLUE, label="Frontier Reasoning Model")
    ax.plot(x, chat_acc, marker="s", lw=2.5, color=COLOR_PURPLE, label="Standard Instruction Model")
    ax.plot(x, memorizer_acc, marker="^", lw=2.5, color=COLOR_ORANGE, linestyle="--", label="Memorizer Model (Surface Pattern)")

    ax.set_ylabel("Evaluation Accuracy (%)", fontsize=11)
    ax.set_title("Cognitive Complexity Degradation (Bloom's Taxonomy)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, fontsize=10)

    ax.annotate("Up to ~70% Collapse!\n(DeepQuestion Finding)", xy=(3, 24), xytext=(2.2, 10),
                arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=2), fontsize=10, fontweight="bold", color=COLOR_RED)

    plt.tight_layout()
    out = FIG_DIR / "04_deepquestion_cognitive_collapse.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


def plot_epoch_benchmark_registry():
    """Figure 5: Epoch AI Benchmark Reviews Classification Registry."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    labels = ["FLAWED\n(Defect >= 20% or Exploit)", "NOT ENOUGH INFO\n(Private / Unverifiable)", "VERIFIED\n(Defect < 20%)"]
    counts = [13, 8, 5]  # Out of 26 benchmarks
    colors = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN]

    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, width=0.5, edgecolor=BORDER_COLOR)

    ax.set_ylabel("Number of Reviewed Benchmarks (N=26)", fontsize=11)
    ax.set_title("Epoch AI Benchmark Review Registry Distribution (Nov 2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 16)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, count in zip(bars, counts):
        pct = (count / 26) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4, f"{count} ({pct:.1f}%)",
                ha="center", fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    plt.tight_layout()
    out = FIG_DIR / "05_epoch_benchmark_registry_audit.png"
    plt.savefig(out, dpi=300, facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Generated {out.name}")


if __name__ == "__main__":
    plot_measurement_layer_architecture()
    plot_scicode_cascading_recovery()
    plot_defect_taxonomy()
    plot_deepquestion_cognitive_collapse()
    plot_epoch_benchmark_registry()
    print("All 5 figures successfully generated in figures/")
