"""Generates publication-quality benchmark plots for Episodes 1-5 of the Inference Server."""
import json
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_episodes_plot():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc"})

    # -------------------------------------------------------------
    # Panel 1: Episode 1 vs Episode 2 Work Complexity & Positions
    # -------------------------------------------------------------
    prompt_lengths = np.array([16, 64, 128, 256, 512])
    output_tokens = 32
    # Uncached: G*P + G*(G-1)/2
    uncached_work = output_tokens * prompt_lengths + (output_tokens * (output_tokens - 1)) // 2
    # KV Cached: P + (G - 1)
    cached_work = prompt_lengths + (output_tokens - 1)

    x = np.arange(len(prompt_lengths))
    width = 0.35

    b1 = ax1.bar(x - width/2, uncached_work, width, label="Episode 1: Uncached Recomputation", color="#c0392b")
    b2 = ax1.bar(x + width/2, cached_work, width, label="Episode 2: Static KV Caching", color="#27ae60")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"P={p}" for p in prompt_lengths])
    ax1.set_ylabel("Token Positions Evaluated (Lower is Better)")
    ax1.set_title("1. Algorithmic Work Explosion (G=32 output tokens)", fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    for i in range(len(prompt_lengths)):
        reduction = (1.0 - cached_work[i] / uncached_work[i]) * 100
        ax1.annotate(f"-{reduction:.1f}%", xy=(x[i] + width/2, cached_work[i]),
                     xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#1e824c")

    # -------------------------------------------------------------
    # Panel 2: Episode 3 Dynamic Batching Throughput Scaling
    # -------------------------------------------------------------
    batch_sizes = [1, 2, 4, 8]
    throughput_vals = [5672.7, 5980.1, 11481.6, 22744.5] # Measured empirical values from fixture benchmark
    
    ax2.plot(batch_sizes, throughput_vals, marker="o", linewidth=2.5, markersize=8, color="#2980b9", label="Measured Throughput")
    ax2.set_xticks(batch_sizes)
    ax2.set_xlabel("Batch Size (Heterogeneous Prompts with Left-Padding)")
    ax2.set_ylabel("Throughput (Tokens / Second)")
    ax2.set_title("2. Dynamic Request Batching Throughput Scaling", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    for b, t in zip(batch_sizes, throughput_vals):
        ax2.annotate(f"{t:.0f} tok/s", xy=(b, t), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#1c5980")

    # -------------------------------------------------------------
    # Panel 3: Episode 4 PagedAttention Virtual Memory Savings
    # -------------------------------------------------------------
    categories = ["Active Tokens", "Paged Allocated", "Contiguous Reserved"]
    slots = [134, 176, 6144] # Measured from paged_attention.py run
    colors = ["#16a085", "#2ecc71", "#e74c3c"]

    bars = ax3.bar(categories, slots, color=colors, width=0.55)
    ax3.set_ylabel("KV Slots Reserved in Memory (Lower is Better)")
    ax3.set_yscale("log")
    ax3.set_title("3. PagedAttention Memory Allocation (6 Concurrent Reqs)", fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, slots):
        ax3.annotate(f"{val} slots", xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    ax3.text(0.5, 0.45, "34.9x Memory Reduction\nZero External Fragmentation",
             transform=ax3.transAxes, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f8f5", edgecolor="#16a085"),
             fontsize=10, fontweight="bold", color="#117a65")

    # -------------------------------------------------------------
    # Panel 4: Episode 5 Continuous vs Static Batching Scheduling
    # -------------------------------------------------------------
    metrics = ["Avg Queue Wait", "Avg E2E Steps", "Total Makespan"]
    cb_metrics = [0.33, 3.00, 8.0]
    sb_metrics = [2.00, 7.50, 11.0]

    x4 = np.arange(len(metrics))
    w4 = 0.35

    ax4.bar(x4 - w4/2, sb_metrics, w4, label="Static Request Batching", color="#95a5a6")
    ax4.bar(x4 + w4/2, cb_metrics, w4, label="Continuous Iteration Batching", color="#8e44ad")

    ax4.set_xticks(x4)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel("Scheduler Steps (Lower is Better)")
    ax4.set_title("4. Continuous vs Static Batching Dynamics", fontweight="bold")
    ax4.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.spines[["top", "right"]].set_visible(False)

    for i in range(len(metrics)):
        diff = ((sb_metrics[i] - cb_metrics[i]) / sb_metrics[i]) * 100
        ax4.annotate(f"-{diff:.1f}%", xy=(x4[i] + w4/2, cb_metrics[i]),
                     xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#5b2c6f")

    fig.suptitle("Build an Inference Server: Episodes 1-5 Performance & Memory Architecture",
                 fontsize=14, fontweight="bold", y=0.995)

    out_path = FIGURES_DIR / "episodes_benchmark.png"
    fig.savefig(out_path, dpi=200)
    print(f"Generated benchmark figure at: {out_path}")

if __name__ == "__main__":
    generate_episodes_plot()

