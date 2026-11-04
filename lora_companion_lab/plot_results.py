"""Generate publication-quality figures for the LoRA Laboratory.

Figures:
1. 01_lora_architecture_and_forward.png: Dual-branch LoRA architecture vs Full Fine-Tuning and zero-latency merge.
2. 02_initialization_and_gradient_dynamics.png: Asymmetric initialization (B=0, A~Init), Step 0 gradients, and dead-lock control.
3. 03_parameter_and_memory_efficiency.png: Parameter reduction and AdamW optimizer memory savings across GPT-2 model sizes and ranks.
4. 04_rank_capacity_convergence.png: Optimization trajectories recovering low-rank ground-truth dynamics across ranks {1, 2, 4, 8, 16}.
5. 05_zero_latency_merge_parity.png: Numerical parity between unmerged forward passes and merged weights across varied input dimensions.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
BG_COLOR = "#0B0F19"
PANEL_COLOR = "#131D31"
BORDER_COLOR = "#22324D"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"
GRID_COLOR = "#1E2C45"

ACCENT_CYAN = "#38BDF8"
ACCENT_EMERALD = "#34D399"
ACCENT_PURPLE = "#A78BFA"
ACCENT_AMBER = "#FBBF24"
ACCENT_ROSE = "#FB7185"
ACCENT_INDIGO = "#818CF8"

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor": BORDER_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": MUTED_TEXT,
    "ytick.color": MUTED_TEXT,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "text.color": TEXT_COLOR
})

OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = Path("results/benchmark.json")


def load_benchmark_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run benchmark.py first.")
    with open(DATA_FILE) as f:
        return json.load(f)


def plot_architecture():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor=BG_COLOR)

    # Left: Training Mode (Dual Branch)
    ax1.set_facecolor(PANEL_COLOR)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Training Phase: Dual-Branch Forward Pass", color=TEXT_COLOR, fontsize=13, weight="bold", pad=15)

    # Input Box
    ax1.add_patch(plt.Rectangle((3.8, 8.8), 2.4, 0.9, facecolor="#1E293B", edgecolor=ACCENT_CYAN, lw=2, zorder=2))
    ax1.text(5.0, 9.25, r"Input $x \in \mathcal{R}^{B \times L \times d_{in}}$", color=TEXT_COLOR, ha="center", va="center", weight="bold", fontsize=10)

    # Base Weight Box (Frozen)
    ax1.add_patch(plt.Rectangle((0.8, 5.0), 3.2, 2.0, facecolor="#1E293B", edgecolor=ACCENT_ROSE, lw=2, zorder=2))
    ax1.text(2.4, 6.3, "Pretrained Base $W_0$\n(Frozen, No Grad)", color=ACCENT_ROSE, ha="center", va="center", weight="bold", fontsize=10)
    ax1.text(2.4, 5.4, "$h_{base} = x W_0$", color=MUTED_TEXT, ha="center", va="center", fontsize=9)

    # LoRA Branch (Trainable)
    # Down-proj A
    ax1.add_patch(plt.Rectangle((6.0, 6.2), 3.2, 1.3, facecolor="#1E293B", edgecolor=ACCENT_PURPLE, lw=2, zorder=2))
    ax1.text(7.6, 7.0, "Down-Projection $A$\n(Trainable: $r \\times d_{in}$)", color=ACCENT_PURPLE, ha="center", va="center", weight="bold", fontsize=9)
    ax1.text(7.6, 6.45, "Init: Kaiming Uniform", color=MUTED_TEXT, ha="center", va="center", fontsize=8)

    # Up-proj B
    ax1.add_patch(plt.Rectangle((6.0, 4.3), 3.2, 1.3, facecolor="#1E293B", edgecolor=ACCENT_EMERALD, lw=2, zorder=2))
    ax1.text(7.6, 5.1, "Up-Projection $B$\n(Trainable: $d_{out} \\times r$)", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=9)
    ax1.text(7.6, 4.55, "Init: Exactly 0", color=MUTED_TEXT, ha="center", va="center", fontsize=8)

    # Scale Box
    ax1.add_patch(plt.Rectangle((6.5, 2.8), 2.2, 0.8, facecolor="#1E293B", edgecolor=ACCENT_AMBER, lw=1.5, zorder=2))
    ax1.text(7.6, 3.2, "Scaling: $\\frac{\\alpha}{r}$", color=ACCENT_AMBER, ha="center", va="center", weight="bold", fontsize=10)

    # Addition Circle
    add_circle = plt.Circle((5.0, 2.0), 0.5, facecolor="#1E293B", edgecolor=TEXT_COLOR, lw=2, zorder=2)
    ax1.add_patch(add_circle)
    ax1.text(5.0, 2.0, "+", color=TEXT_COLOR, ha="center", va="center", weight="bold", fontsize=16)

    # Output Box
    ax1.add_patch(plt.Rectangle((3.6, 0.3), 2.8, 0.9, facecolor="#1E293B", edgecolor=ACCENT_CYAN, lw=2, zorder=2))
    ax1.text(5.0, 0.75, r"Output $h = h_{base} + \frac{\alpha}{r} (x A^T B^T)$", color=TEXT_COLOR, ha="center", va="center", weight="bold", fontsize=9)

    # Arrows
    arrow_kw = dict(arrowstyle="->", color=MUTED_TEXT, lw=1.8)
    ax1.annotate("", xy=(2.4, 7.0), xytext=(4.2, 8.8), arrowprops=arrow_kw)
    ax1.annotate("", xy=(7.6, 7.5), xytext=(5.8, 8.8), arrowprops=arrow_kw)
    ax1.annotate("", xy=(7.6, 5.6), xytext=(7.6, 6.2), arrowprops=arrow_kw)
    ax1.annotate("", xy=(7.6, 3.6), xytext=(7.6, 4.3), arrowprops=arrow_kw)
    ax1.annotate("", xy=(5.0, 2.5), xytext=(2.4, 5.0), arrowprops=arrow_kw)
    ax1.annotate("", xy=(5.5, 2.0), xytext=(6.5, 3.2), arrowprops=arrow_kw)
    ax1.annotate("", xy=(5.0, 1.2), xytext=(5.0, 1.5), arrowprops=arrow_kw)

    # Right: Inference Mode (Merged Single Weight)
    ax2.set_facecolor(PANEL_COLOR)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Inference Phase: Zero-Latency Weight Merge", color=TEXT_COLOR, fontsize=13, weight="bold", pad=15)

    ax2.add_patch(plt.Rectangle((3.8, 8.8), 2.4, 0.9, facecolor="#1E293B", edgecolor=ACCENT_CYAN, lw=2, zorder=2))
    ax2.text(5.0, 9.25, r"Input $x \in \mathcal{R}^{B \times L \times d_{in}}$", color=TEXT_COLOR, ha="center", va="center", weight="bold", fontsize=10)

    # Merged Matrix Box
    ax2.add_patch(plt.Rectangle((2.2, 3.8), 5.6, 3.4, facecolor="#1E293B", edgecolor=ACCENT_EMERALD, lw=2.5, zorder=2))
    ax2.text(5.0, 6.3, "Merged Weight Matrix $W_{eff}$", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=12)
    ax2.text(5.0, 5.4, "$W_{eff} = W_0 + \\frac{\\alpha}{r} (B A)$", color=TEXT_COLOR, ha="center", va="center", weight="bold", fontsize=11)
    ax2.text(5.0, 4.5, "• Exactly 1 standard GEMM\n• 0 extra inference latency\n• 0 extra runtime memory", color=MUTED_TEXT, ha="center", va="center", fontsize=9.5)

    # Output Box
    ax2.add_patch(plt.Rectangle((3.6, 0.8), 2.8, 0.9, facecolor="#1E293B", edgecolor=ACCENT_CYAN, lw=2, zorder=2))
    ax2.text(5.0, 1.25, "Output $h = x W_{eff}^T$", color=TEXT_COLOR, ha="center", va="center", weight="bold", fontsize=10)

    ax2.annotate("", xy=(5.0, 7.2), xytext=(5.0, 8.8), arrowprops=arrow_kw)
    ax2.annotate("", xy=(5.0, 1.7), xytext=(5.0, 3.8), arrowprops=arrow_kw)

    plt.tight_layout()
    out_path = OUT_DIR / "01_lora_architecture_and_forward.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def plot_initialization_and_gradients(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOR)

    # Left: Step 0 Gradient Norms
    grad_data = data["gradient_audit"]
    labels = ["Base $W_0$\n(Frozen)", "Adapter $B$\n(Init = 0)", "Adapter $A$\n(Init ~ Unif)", "Deadlock $A$\n(Both 0)", "Deadlock $B$\n(Both 0)"]
    values = [0.0, grad_data["b_grad_norm_step0"], 0.0, grad_data["deadlock_control"]["a_grad_when_both_zero"], grad_data["deadlock_control"]["b_grad_when_both_zero"]]
    colors = [ACCENT_ROSE, ACCENT_EMERALD, ACCENT_PURPLE, ACCENT_AMBER, ACCENT_AMBER]

    bars = ax1.bar(labels, values, color=colors, width=0.55, edgecolor=BORDER_COLOR, lw=1.2)
    ax1.set_title(r"Step 0 Gradient Magnitude Flow $(\nabla_\theta \mathcal{L})$", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax1.set_ylabel(r"Gradient Norm $\|\nabla\|$", color=TEXT_COLOR, fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * max(values), f"{h:.3f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9.5, weight="bold")

    ax1.annotate("B receives non-zero gradient\nbecause A outputs non-zero features!", xy=(1, values[1]), xytext=(1.2, values[1] * 0.8),
                 arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=1.5), color=ACCENT_CYAN, fontsize=9, weight="bold")

    # Right: Conceptual Diagram of Gradient Backpropagation Route
    ax2.set_facecolor(PANEL_COLOR)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title(r"Why Asymmetric Initialization $(B=0, A \ne 0)$ is Essential", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)

    ax2.add_patch(plt.Rectangle((0.5, 7.2), 9.0, 2.2, facecolor="#1E293B", edgecolor=ACCENT_EMERALD, lw=1.5))
    ax2.text(5.0, 8.7, r"Standard LoRA Init: $B = 0, A \sim \mathrm{Kaiming\ Uniform}$", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=10.5)
    ax2.text(5.0, 7.8, "• Forward at t=0: Delta W = B @ A = 0  --> Pretrained model behavior preserved.\n• Backward at t=0: dL/dB = (grad_h) @ (x @ A)^T != 0, enabling immediate learning.", color=TEXT_COLOR, ha="center", va="center", fontsize=9)

    ax2.add_patch(plt.Rectangle((0.5, 4.0), 9.0, 2.4, facecolor="#1E293B", edgecolor=ACCENT_ROSE, lw=1.5))
    ax2.text(5.0, 5.6, "Deadlock Failure Mode: Initializing Both A=0 and B=0", color=ACCENT_ROSE, ha="center", va="center", weight="bold", fontsize=10.5)
    ax2.text(5.0, 4.6, "• dL/dB = (grad_h) @ (x @ A)^T = 0 (since A = 0)\n• dL/dA = B^T @ (grad_h) @ x = 0 (since B = 0)\n• Result: Deadlock! Zero gradients lock the adapter permanently at zero.", color=TEXT_COLOR, ha="center", va="center", fontsize=9)

    ax2.add_patch(plt.Rectangle((0.5, 0.8), 9.0, 2.4, facecolor="#1E293B", edgecolor=ACCENT_AMBER, lw=1.5))
    ax2.text(5.0, 2.4, "Non-Zero Failure Mode: Initializing Both A != 0 and B != 0", color=ACCENT_AMBER, ha="center", va="center", weight="bold", fontsize=10.5)
    ax2.text(5.0, 1.4, "• Forward at t=0: Delta W = B @ A != 0\n• Pretrained knowledge is immediately perturbed before any training steps!\n• High initial loss spike and catastrophic degradation of pretrained capabilities.", color=TEXT_COLOR, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    out_path = OUT_DIR / "02_initialization_and_gradient_dynamics.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def plot_parameter_efficiency(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOR)

    scaling_data = data["parameter_scaling"]
    models = list(scaling_data.keys())
    ranks = [1, 2, 4, 8, 16, 32, 64]

    colors = [ACCENT_CYAN, ACCENT_EMERALD, ACCENT_PURPLE, ACCENT_AMBER]

    # Left: Trainable % vs Rank across GPT-2 Families
    for idx, model_name in enumerate(models):
        m_info = scaling_data[model_name]
        pcts = [item["trainable_percent"] for item in m_info["rank_scaling"]]
        ax1.plot(ranks, pcts, marker="o", lw=2.2, label=model_name, color=colors[idx])

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ranks)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_title("LoRA Trainable Parameters (% of Total Model)", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax1.set_xlabel("Adapter Rank ($r$)", color=TEXT_COLOR, fontsize=10)
    ax1.set_ylabel("Trainable Parameter Fraction (%)", color=TEXT_COLOR, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    # Annotation for GPT-2 Small r=8
    gpt2_s_r8 = scaling_data["GPT-2 Small"]["rank_scaling"][3]
    ax1.annotate(f"GPT-2 Small ($r=8$)\n{gpt2_s_r8['trainable_percent']}% ({gpt2_s_r8['total_adapter_params']:,} params)\n99.76% Reduction",
                 xy=(8, gpt2_s_r8["trainable_percent"]), xytext=(10, 0.45),
                 arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=1.5),
                 color=ACCENT_CYAN, fontsize=8.5, weight="bold")

    # Right: AdamW Optimizer Memory Comparison (MB)
    x = np.arange(len(models))
    width = 0.35

    full_mem = [scaling_data[m]["rank_scaling"][3]["full_adamw_mem_mb"] for m in models]
    lora_mem = [scaling_data[m]["rank_scaling"][3]["lora_adamw_mem_mb"] for m in models]

    ax2.bar(x - width/2, full_mem, width, label="Full Fine-Tuning (AdamW)", color=ACCENT_ROSE, edgecolor=BORDER_COLOR)
    ax2.bar(x + width/2, lora_mem, width, label="LoRA Rank 8 (AdamW)", color=ACCENT_EMERALD, edgecolor=BORDER_COLOR)

    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("GPT-2 ", "") for m in models], color=TEXT_COLOR, fontsize=9.5)
    ax2.set_yscale("log")
    ax2.set_title("AdamW Optimizer Memory Footprint (MB, Log Scale)", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax2.set_ylabel("Memory (Megabytes)", color=TEXT_COLOR, fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper left", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    for i in range(len(models)):
        reduction = full_mem[i] / max(lora_mem[i], 1e-4)
        ax2.text(x[i] + width/2, lora_mem[i] * 1.5, f"{lora_mem[i]:.1f} MB\n({reduction:.0f}× less)",
                 ha="center", va="bottom", color=ACCENT_EMERALD, fontsize=8, weight="bold")

    plt.tight_layout()
    out_path = OUT_DIR / "03_parameter_and_memory_efficiency.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def plot_rank_convergence(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOR)

    sim_data = data["convergence_simulation"]
    ranks = [1, 2, 4, 8, 16]
    steps = [i * 10 for i in range(len(sim_data["rank_1"]["loss_trajectory"]))]

    colors = [ACCENT_ROSE, ACCENT_AMBER, ACCENT_EMERALD, ACCENT_CYAN, ACCENT_PURPLE]

    # Left: Training Loss Convergence Trajectory
    for idx, r in enumerate(ranks):
        key = f"rank_{r}"
        losses = sim_data[key]["loss_trajectory"]
        lbl = f"Rank {r} (Target rank=4)"
        ax1.plot(steps, losses, marker="o", markersize=4, lw=2.0, label=lbl, color=colors[idx])

    ax1.set_yscale("log")
    ax1.set_title("Convergence Trajectory on True Rank-4 Target Subspace", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax1.set_xlabel("Optimization Step", color=TEXT_COLOR, fontsize=10)
    ax1.set_ylabel("MSE Loss (Log Scale)", color=TEXT_COLOR, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    ax1.annotate("Ranks < 4 cannot span\ntarget subspace (Bottleneck)", xy=(steps[-1], sim_data["rank_1"]["final_loss"]), xytext=(60, 5),
                 arrowprops=dict(arrowstyle="->", color=ACCENT_ROSE, lw=1.5), color=ACCENT_ROSE, fontsize=8.5, weight="bold")

    # Right: Final Residual Loss vs Adapter Rank
    final_losses = [sim_data[f"rank_{r}"]["final_loss"] for r in ranks]
    bar_colors = [ACCENT_ROSE if r < 4 else ACCENT_EMERALD for r in ranks]
    bars = ax2.bar([str(r) for r in ranks], final_losses, color=bar_colors, width=0.55, edgecolor=BORDER_COLOR)
    ax2.set_yscale("log")
    ax2.set_title("Final Residual Loss by Rank (Target Subspace $r^*=4$)", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax2.set_xlabel("Adapter Rank ($r$)", color=TEXT_COLOR, fontsize=10)
    ax2.set_ylabel("Final MSE Loss (Log Scale)", color=TEXT_COLOR, fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h * 1.2, f"{h:.2e}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8.5, weight="bold")

    ax2.axvline(x=2, color=ACCENT_CYAN, linestyle=":", lw=1.8, label="Ground Truth Rank = 4")
    ax2.legend(loc="upper right", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    plt.tight_layout()
    out_path = OUT_DIR / "04_rank_capacity_convergence.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def plot_merge_parity(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOR)

    parity = data["parity_audit"]
    items = parity["standard_linear_discrepancies"]

    labels = [f"B={d['batch_size']}, L={d['seq_len']}" for d in items]
    max_errs = [d["max_abs_error"] for d in items]
    mean_errs = [d["mean_abs_error"] for d in items]

    x = np.arange(len(labels))
    width = 0.38

    # Left: Numerical Parity Errors
    ax1.bar(x - width/2, max_errs, width, label="Max Absolute Error", color=ACCENT_CYAN, edgecolor=BORDER_COLOR)
    ax1.bar(x + width/2, mean_errs, width, label="Mean Absolute Error", color=ACCENT_EMERALD, edgecolor=BORDER_COLOR)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", color=MUTED_TEXT, fontsize=8.5)
    ax1.set_title("Numerical Parity: Unmerged vs Merged Weight Output", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax1.set_ylabel("Absolute Error $|y_{unmerged} - y_{merged}|$", color=TEXT_COLOR, fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    ax1.axhline(1e-5, color=ACCENT_ROSE, linestyle="--", lw=1.5, label="Tolerance Threshold ($10^{-5}$)")
    ax1.legend(loc="upper right", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    # Right: Deployment Architecture Workflow
    ax2.set_facecolor(PANEL_COLOR)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Production Serving Strategy: Dynamic Adapter Hot-Swapping", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)

    # Base Model
    ax2.add_patch(plt.Rectangle((0.5, 3.5), 3.0, 4.0, facecolor="#1E293B", edgecolor=ACCENT_CYAN, lw=2))
    ax2.text(2.0, 5.8, "Single Frozen Base\nGPT-2 Checkpoint\n(500 MB VRAM)", color=ACCENT_CYAN, ha="center", va="center", weight="bold", fontsize=9.5)
    ax2.text(2.0, 4.4, "Shared across all tenants", color=MUTED_TEXT, ha="center", va="center", fontsize=8.5)

    # Adapters
    ax2.add_patch(plt.Rectangle((6.0, 7.0), 3.5, 1.8, facecolor="#1E293B", edgecolor=ACCENT_EMERALD, lw=1.8))
    ax2.text(7.75, 8.0, "Tenant A: Shakespeare Adapter\nRank 8 (1.1 MB checkpoint)", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=8.5)

    ax2.add_patch(plt.Rectangle((6.0, 4.5), 3.5, 1.8, facecolor="#1E293B", edgecolor=ACCENT_PURPLE, lw=1.8))
    ax2.text(7.75, 5.5, "Tenant B: Medical QA Adapter\nRank 8 (1.1 MB checkpoint)", color=ACCENT_PURPLE, ha="center", va="center", weight="bold", fontsize=8.5)

    ax2.add_patch(plt.Rectangle((6.0, 2.0), 3.5, 1.8, facecolor="#1E293B", edgecolor=ACCENT_AMBER, lw=1.8))
    ax2.text(7.75, 3.0, "Tenant C: Code Generation\nRank 8 (1.1 MB checkpoint)", color=ACCENT_AMBER, ha="center", va="center", weight="bold", fontsize=8.5)

    arrow_kw = dict(arrowstyle="<->", color=MUTED_TEXT, lw=1.5)
    ax2.annotate("", xy=(6.0, 7.9), xytext=(3.5, 6.2), arrowprops=arrow_kw)
    ax2.annotate("", xy=(6.0, 5.4), xytext=(3.5, 5.5), arrowprops=arrow_kw)
    ax2.annotate("", xy=(6.0, 2.9), xytext=(3.5, 4.8), arrowprops=arrow_kw)

    ax2.text(4.75, 1.0, "• 1 Shared Base Model in GPU Memory\n• 100s of task-specialized LoRAs swapped on-the-fly\n• Option: Merge offline into base weights for zero latency",
             ha="center", va="center", color=TEXT_COLOR, fontsize=9, weight="bold")

    plt.tight_layout()
    out_path = OUT_DIR / "05_zero_latency_merge_parity.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def main():
    data = load_benchmark_data()
    plot_architecture()
    plot_initialization_and_gradients(data)
    plot_parameter_efficiency(data)
    plot_rank_convergence(data)
    plot_merge_parity(data)
    print("All LoRA figures successfully generated in figures/")


if __name__ == "__main__":
    main()
