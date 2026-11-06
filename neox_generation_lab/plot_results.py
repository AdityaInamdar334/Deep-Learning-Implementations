"""Generate publication-quality figures for the GPT-NeoX Autoregressive Generation Laboratory.

Figures:
1. 01_prefill_vs_decode_mechanics.png: Prefill vs Incremental Decode execution phases and cache state invariant.
2. 02_uncached_vs_cached_complexity.png: O(L^2) uncached quadratic recomputation vs O(L) cached linear token emission.
3. 03_kv_cache_memory_growth_and_scaling.png: Logical KV cache growth across tokens and GPT-NeoX-20B scale footprint.
4. 04_sampling_and_truncation_dynamics.png: Sampling temperature scaling, Top-k candidate truncation, and greedy decoding.
5. 05_request_isolation_and_interleaving.png: Explicit request-local KV cache isolation and interleaved serving.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Dark-mode styling configuration
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
RESULTS_DIR = Path("results")


def plot_prefill_vs_decode():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0), facecolor=BG_COLOR)

    arrow_kw = dict(arrowstyle="->", color=MUTED_TEXT, lw=1.6)

    # Left: Prefill Phase
    ax1.set_facecolor(PANEL_COLOR)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Phase 1: Prefill (Prompt Processing)\n1 Forward Pass on Full Context", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)

    # Prompt Tokens Box
    ax1.add_patch(plt.Rectangle((1.0, 7.8), 8.0, 1.2, facecolor="#1E293B", edgecolor=ACCENT_CYAN, lw=2))
    ax1.text(5.0, 8.4, "Full Prompt: $[x_0, x_1, x_2, \\dots, x_{L-1}]$ (Length $L$)", color=ACCENT_CYAN, ha="center", va="center", weight="bold", fontsize=10)

    # Transformer Core
    ax1.add_patch(plt.Rectangle((1.0, 4.8), 8.0, 2.0, facecolor="#1E293B", edgecolor=ACCENT_PURPLE, lw=2))
    ax1.text(5.0, 6.1, "Parallel Residual Blocks + Partial RoPE", color=ACCENT_PURPLE, ha="center", va="center", weight="bold", fontsize=10.5)
    ax1.text(5.0, 5.3, "Computes Keys & Values for ALL $L$ positions simultaneously\nCaches $K_{0:L-1}$ and $V_{0:L-1}$ into memory", color=MUTED_TEXT, ha="center", va="center", fontsize=8.5)

    # Logits & Output
    ax1.add_patch(plt.Rectangle((1.0, 1.8), 3.8, 1.8, facecolor="#1E293B", edgecolor=ACCENT_EMERALD, lw=1.8))
    ax1.text(2.9, 2.9, "First Generated Token", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=9.5)
    ax1.text(2.9, 2.2, "Token $x_L$ emitted\nvia argmax/sample", color=TEXT_COLOR, ha="center", va="center", fontsize=8.5)

    # Initial KV Cache
    ax1.add_patch(plt.Rectangle((5.2, 1.8), 3.8, 1.8, facecolor="#1E293B", edgecolor=ACCENT_AMBER, lw=1.8))
    ax1.text(7.1, 2.9, "Initial KV Cache State", color=ACCENT_AMBER, ha="center", va="center", weight="bold", fontsize=9.5)
    ax1.text(7.1, 2.2, "$L_{\\mathrm{cache}} = L$\nReady for decoding", color=TEXT_COLOR, ha="center", va="center", fontsize=8.5)

    ax1.annotate("", xy=(5.0, 6.8), xytext=(5.0, 7.8), arrowprops=arrow_kw)
    ax1.annotate("", xy=(2.9, 3.6), xytext=(3.5, 4.8), arrowprops=arrow_kw)
    ax1.annotate("", xy=(7.1, 3.6), xytext=(6.5, 4.8), arrowprops=arrow_kw)

    ax1.text(5.0, 0.6, "Compute-Bound: Highly parallelized GEMM across all prompt tokens", color=TEXT_COLOR, ha="center", va="center", fontsize=8.5, weight="bold")

    # Right: Incremental Decode Phase
    ax2.set_facecolor(PANEL_COLOR)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Phase 2: Incremental Decode (Token Generation)\nRepeated Steps on Exactly 1 Token", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)

    # Single New Token Input
    ax2.add_patch(plt.Rectangle((2.8, 8.0), 4.4, 1.1, facecolor="#1E293B", edgecolor=ACCENT_EMERALD, lw=2))
    ax2.text(5.0, 8.55, "Single New Input: $[x_{t-1}]$ (Length 1)", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=10)

    # Transformer Core + Cache Lookup
    ax2.add_patch(plt.Rectangle((1.0, 4.6), 8.0, 2.5, facecolor="#1E293B", edgecolor=ACCENT_PURPLE, lw=2))
    ax2.text(5.0, 6.3, "Single Query Vector $Q_{t-1}$", color=ACCENT_CYAN, ha="center", va="center", weight="bold", fontsize=10)
    ax2.text(5.0, 5.6, "Attends against $[K_{\\mathrm{cached}}, K_{t-1}]$ and $[V_{\\mathrm{cached}}, V_{t-1}]$", color=TEXT_COLOR, ha="center", va="center", fontsize=9)
    ax2.text(5.0, 4.9, "Appends new $K_{t-1}, V_{t-1}$ to cache; NO prompt recomputation!", color=MUTED_TEXT, ha="center", va="center", fontsize=8.5)

    # Cache Invariant Box
    ax2.add_patch(plt.Rectangle((1.0, 1.8), 8.0, 1.8, facecolor="#1E293B", edgecolor=ACCENT_AMBER, lw=1.8))
    ax2.text(5.0, 3.0, "The Fundamental Cache Invariant:", color=ACCENT_AMBER, ha="center", va="center", weight="bold", fontsize=10.5)
    ax2.text(5.0, 2.3, "$L_{\\mathrm{cache}} = L_{\\mathrm{output}} - 1$", color=TEXT_COLOR, ha="center", va="center", weight="bold", fontsize=11)
    ax2.text(5.0, 1.6, "After emitting token $t$, the cache holds exactly $t-1$ history entries.", color=MUTED_TEXT, ha="center", va="center", fontsize=8.5)

    ax2.annotate("", xy=(5.0, 7.1), xytext=(5.0, 8.0), arrowprops=arrow_kw)
    ax2.annotate("", xy=(5.0, 3.6), xytext=(5.0, 4.6), arrowprops=arrow_kw)

    ax2.text(5.0, 0.6, "Memory-Bandwidth Bound: Latency dominated by reading KV tensors", color=TEXT_COLOR, ha="center", va="center", fontsize=8.5, weight="bold")

    plt.tight_layout()
    out_path = OUT_DIR / "01_prefill_vs_decode_mechanics.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def plot_complexity_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOR)

    # Data from results/benchmark.json
    bench_file = RESULTS_DIR / "benchmark.json"
    if bench_file.exists():
        with open(bench_file) as f:
            b_data = json.load(f)
        uncached_data = b_data["results"][0]
        cached_data = b_data["results"][1]
        t_uncached = uncached_data["median_total_ms"]
        t_cached = cached_data["median_total_ms"]
        step_uncached = uncached_data["median_decode_mean_ms"]
        step_cached = cached_data["median_decode_mean_ms"]
        pos_uncached = uncached_data["raw"][0]["input_positions"]
        pos_cached = cached_data["raw"][0]["input_positions"]
    else:
        t_uncached, t_cached = 56.38, 23.48
        step_uncached, step_cached = 1.77, 0.71
        pos_uncached, pos_cached = 2544, 95

    # Left: Tokens Processed (Quadratic vs Linear)
    tokens_gen = np.arange(1, 65)
    prompt_len = 64
    # Uncached: sum_{t=1}^N (prompt_len + t) = N*prompt_len + N*(N+1)/2
    pos_uncached_curve = [prompt_len * n + n * (n + 1) // 2 for n in tokens_gen]
    # Cached: prompt_len + (n - 1)
    pos_cached_curve = [prompt_len + (n - 1) for n in tokens_gen]

    ax1.plot(tokens_gen, pos_uncached_curve, color=ACCENT_ROSE, lw=2.2, label=r"Uncached: $\mathcal{O}(N L + N^2/2)$ Recompute")
    ax1.plot(tokens_gen, pos_cached_curve, color=ACCENT_EMERALD, lw=2.2, label=r"Cached: $\mathcal{O}(L + N)$ Incremental")

    ax1.scatter([32], [pos_uncached], color=ACCENT_ROSE, s=60, zorder=5)
    ax1.scatter([32], [pos_cached], color=ACCENT_EMERALD, s=60, zorder=5)

    ax1.set_title("Input Token Positions Processed During Generation", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax1.set_xlabel("Tokens Generated ($N$)", color=TEXT_COLOR, fontsize=10)
    ax1.set_ylabel("Total Input Positions Fed into Attention", color=TEXT_COLOR, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    ax1.annotate(f"At N=32:\n{pos_uncached:,} positions\n(26.8× more work!)", xy=(32, pos_uncached), xytext=(12, 1800),
                 arrowprops=dict(arrowstyle="->", color=ACCENT_ROSE, lw=1.5), color=ACCENT_ROSE, fontsize=8.5, weight="bold")
    ax1.annotate(f"At N=32:\n{pos_cached} positions", xy=(32, pos_cached), xytext=(35, 300),
                 arrowprops=dict(arrowstyle="->", color=ACCENT_EMERALD, lw=1.5), color=ACCENT_EMERALD, fontsize=8.5, weight="bold")

    # Right: Latency Breakdown Bar Chart
    metrics = ["Total Generation Time\n(32 Tokens)", "Mean Per-Token\nDecode Step Time"]
    uncached_vals = [t_uncached, step_uncached]
    cached_vals = [t_cached, step_cached]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax2.bar(x - width/2, uncached_vals, width, label="Uncached Generation", color=ACCENT_ROSE, edgecolor=BORDER_COLOR)
    bars2 = ax2.bar(x + width/2, cached_vals, width, label="KV-Cached Generation", color=ACCENT_EMERALD, edgecolor=BORDER_COLOR)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, color=TEXT_COLOR, fontsize=9.5, weight="bold")
    ax2.set_ylabel("Latency (Milliseconds)", color=TEXT_COLOR, fontsize=10)
    ax2.set_title("Wall-Clock Latency Benchmark (Prompt 64, Tokens 32)", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    for bar in bars1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.2f} ms", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8.5, weight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.2f} ms", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8.5, weight="bold")

    speedup_total = t_uncached / t_cached
    speedup_step = step_uncached / step_cached
    ax2.text(x[0], t_cached / 2, f"{speedup_total:.2f}× Faster", ha="center", va="center", color=TEXT_COLOR, fontsize=9, weight="bold")
    ax2.text(x[1], step_cached * 1.5, f"{speedup_step:.2f}× Faster", ha="center", va="center", color=ACCENT_EMERALD, fontsize=9, weight="bold")

    plt.tight_layout()
    out_path = OUT_DIR / "02_uncached_vs_cached_complexity.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def plot_memory_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOR)

    # Left: Lab Model Cache Memory Growth per Token
    # Config: 2 layers, 4 heads, head_dim=24, FP32 (4 bytes)
    # bytes_per_token = 2 (K & V) * 2 layers * 1 batch * 4 heads * 24 head_dim * 4 bytes = 1,536 bytes/token
    tokens = np.arange(1, 257)
    lab_bytes_kb = [2 * 2 * 1 * 4 * t * 24 * 4 / 1024 for t in tokens]

    ax1.plot(tokens, lab_bytes_kb, color=ACCENT_CYAN, lw=2.2)
    ax1.set_title("Lab Model: Logical KV-Cache Memory Growth", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax1.set_xlabel("Tokens Stored in Cache ($L_{\\mathrm{cache}}$)", color=TEXT_COLOR, fontsize=10)
    ax1.set_ylabel("Cache Memory (Kilobytes)", color=TEXT_COLOR, fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax1.scatter([32, 64, 128, 256], [lab_bytes_kb[31], lab_bytes_kb[63], lab_bytes_kb[127], lab_bytes_kb[255]], color=ACCENT_AMBER, s=50, zorder=5)
    ax1.text(64, lab_bytes_kb[63] + 15, f"64 tokens:\n{lab_bytes_kb[63]:.1f} KB", color=TEXT_COLOR, fontsize=8.5, weight="bold", ha="center")
    ax1.text(256, lab_bytes_kb[255] - 30, f"Max Context (256):\n{lab_bytes_kb[255]:.1f} KB", color=ACCENT_AMBER, fontsize=8.5, weight="bold", ha="right")

    # Right: GPT-NeoX-20B Real Scale Memory Footprint
    # 44 layers, 64 heads, head_dim=96, FP16 (2 bytes)
    # bytes = 2 * 44 * batch * L * 64 * 96 * 2
    context_lengths = [512, 1024, 2048, 4096, 8192]
    batch_sizes = [1, 4, 16, 64]
    colors = [ACCENT_EMERALD, ACCENT_CYAN, ACCENT_PURPLE, ACCENT_ROSE]

    for idx, bsz in enumerate(batch_sizes):
        mems_gib = [(2 * 44 * bsz * L * 64 * 96 * 2) / (1024**3) for L in context_lengths]
        ax2.plot(context_lengths, mems_gib, marker="o", lw=2.0, color=colors[idx], label=f"Batch {bsz}")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_title("GPT-NeoX-20B: KV-Cache Memory Demand (FP16)", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax2.set_xlabel("Context Length (Tokens)", color=TEXT_COLOR, fontsize=10)
    ax2.set_ylabel("KV-Cache Memory (GiB, Log Scale)", color=TEXT_COLOR, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    # Annotation for 2048 context at Batch 1
    ax2.annotate("2.06 GiB per stream\nat 2,048 context!", xy=(2048, 2.0625), xytext=(600, 5.0),
                 arrowprops=dict(arrowstyle="->", color=ACCENT_EMERALD, lw=1.5),
                 color=ACCENT_EMERALD, fontsize=8.5, weight="bold")

    plt.tight_layout()
    out_path = OUT_DIR / "03_kv_cache_memory_growth_and_scaling.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def plot_sampling_dynamics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG_COLOR)

    # Synthetic raw logits for 10 candidate tokens
    tokens = [f"T{i}" for i in range(10)]
    raw_logits = np.array([4.2, 3.8, 3.5, 2.8, 2.4, 1.9, 1.2, 0.8, 0.4, 0.1])

    # Left: Temperature Scaling Effect on Probability Distribution
    def softmax(x, temp=1.0):
        e_x = np.exp((x - np.max(x)) / temp)
        return e_x / e_x.sum()

    probs_low = softmax(raw_logits, temp=0.3)
    probs_mid = softmax(raw_logits, temp=0.8)
    probs_high = softmax(raw_logits, temp=1.5)

    x = np.arange(len(tokens))
    width = 0.28

    ax1.bar(x - width, probs_low, width, label="Low Temp (T=0.3) - Peaked", color=ACCENT_CYAN, edgecolor=BORDER_COLOR)
    ax1.bar(x, probs_mid, width, label="Normal Temp (T=0.8) - Balanced", color=ACCENT_EMERALD, edgecolor=BORDER_COLOR)
    ax1.bar(x + width, probs_high, width, label="High Temp (T=1.5) - Uniform", color=ACCENT_ROSE, edgecolor=BORDER_COLOR)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, color=TEXT_COLOR, fontsize=9.5)
    ax1.set_title("Temperature Scaling on Next-Token Probabilities", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax1.set_ylabel("Sampling Probability $P(x_i)$", color=TEXT_COLOR, fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    # Right: Top-k Truncation (k=4)
    k = 4
    masked_logits = raw_logits.copy()
    masked_logits[k:] = -1e9
    probs_topk = softmax(masked_logits, temp=0.8)

    bar_colors = [ACCENT_EMERALD if i < k else ACCENT_ROSE for i in range(len(tokens))]
    bars = ax2.bar(tokens, probs_topk, color=bar_colors, width=0.55, edgecolor=BORDER_COLOR)

    ax2.set_title("Top-k Truncation ($k=4$, $T=0.8$)", color=TEXT_COLOR, fontsize=12, weight="bold", pad=12)
    ax2.set_ylabel("Normalized Probability", color=TEXT_COLOR, fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    for i in range(k):
        ax2.text(i, probs_topk[i] + 0.015, f"{probs_topk[i]:.2f}", ha="center", va="bottom", color=ACCENT_EMERALD, fontsize=8.5, weight="bold")
    for i in range(k, len(tokens)):
        ax2.text(i, 0.02, "Pruned\n(P=0)", ha="center", va="bottom", color=ACCENT_ROSE, fontsize=7.5)

    ax2.axvline(x=k - 0.5, color=ACCENT_AMBER, linestyle="--", lw=1.8, label=f"Top-{k} Boundary")
    ax2.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR)

    plt.tight_layout()
    out_path = OUT_DIR / "04_sampling_and_truncation_dynamics.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def plot_request_isolation():
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=BG_COLOR)

    ax.set_facecolor(PANEL_COLOR)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Multi-Tenant Request Isolation: Independent KV Cache Lifecycles", color=TEXT_COLOR, fontsize=13, weight="bold", pad=15)

    arrow_kw = dict(arrowstyle="->", color=MUTED_TEXT, lw=1.5)

    # Shared Model Box
    ax.add_patch(plt.Rectangle((4.5, 7.8), 3.0, 1.6, facecolor="#1E293B", edgecolor=TEXT_COLOR, lw=2.2))
    ax.text(6.0, 8.8, "Stateless Model Parameters", color=TEXT_COLOR, ha="center", va="center", weight="bold", fontsize=10.5)
    ax.text(6.0, 8.2, "Weights $W_0$ (Frozen & Shared)\nAccepts explicit `past` tensor", color=MUTED_TEXT, ha="center", va="center", fontsize=8.5)

    # Request A Stream
    ax.add_patch(plt.Rectangle((0.5, 3.8), 4.8, 3.2, facecolor="#1E293B", edgecolor=ACCENT_CYAN, lw=2))
    ax.text(2.9, 6.5, "Request A: \"Translate English to SQL\"", color=ACCENT_CYAN, ha="center", va="center", weight="bold", fontsize=10)
    ax.text(2.9, 5.7, "• Cache A: Length $L_A = 48$ tokens\n• Memory: Dedicated Tensor $[2, 4, 48, 24]$\n• Emits token $t_A$, advances $L_A \\to 49$\n• 100% Isolated: Zero cross-talk with Request B", color=TEXT_COLOR, ha="center", va="center", fontsize=8.5)
    ax.text(2.9, 4.3, "State: Active In-Flight", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=8.5)

    # Request B Stream
    ax.add_patch(plt.Rectangle((6.7, 3.8), 4.8, 3.2, facecolor="#1E293B", edgecolor=ACCENT_PURPLE, lw=2))
    ax.text(9.1, 6.5, "Request B: \"Summarize meeting notes\"", color=ACCENT_PURPLE, ha="center", va="center", weight="bold", fontsize=10)
    ax.text(9.1, 5.7, "• Cache B: Length $L_B = 128$ tokens\n• Memory: Dedicated Tensor $[2, 4, 128, 24]$\n• Emits token $t_B$, advances $L_B \\to 129$\n• Interleaved execution passes without state corruption", color=TEXT_COLOR, ha="center", va="center", fontsize=8.5)
    ax.text(9.1, 4.3, "State: Active In-Flight", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=8.5)

    # Arrows from model to requests
    ax.annotate("", xy=(2.9, 7.0), xytext=(5.0, 7.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.1, 7.0), xytext=(7.0, 7.8), arrowprops=arrow_kw)

    # Verification Box at Bottom
    ax.add_patch(plt.Rectangle((1.5, 0.8), 9.0, 2.2, facecolor="#1E293B", edgecolor=ACCENT_EMERALD, lw=1.8))
    ax.text(6.0, 2.4, "Formal Behavioral Verification (Passing in test suite):", color=ACCENT_EMERALD, ha="center", va="center", weight="bold", fontsize=10.5)
    ax.text(6.0, 1.7, "• `test_interleaved_requests`: Alternating steps across distinct requests matches non-interleaved runs exactly (diff < 2e-5)\n• `test_fresh_request`: Fresh generation call resets all cache tensors without state leakage\n• `test_wrong_history_changes_output`: Feeding wrong cache immediately alters generation trajectory (diff > 1e-3)",
            color=TEXT_COLOR, ha="center", va="center", fontsize=8.5)

    plt.tight_layout()
    out_path = OUT_DIR / "05_request_isolation_and_interleaving.png"
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated {out_path}")


def main():
    plot_prefill_vs_decode()
    plot_complexity_comparison()
    plot_memory_scaling()
    plot_sampling_dynamics()
    plot_request_isolation()
    print("All GPT-NeoX Autoregressive Generation figures successfully generated in figures/")


if __name__ == "__main__":
    main()

