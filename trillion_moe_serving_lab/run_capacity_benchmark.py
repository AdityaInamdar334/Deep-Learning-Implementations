"""Automated Capacity Benchmark Runner.

Calculates the complete hardware memory grid and serving limits across
Kimi K3.1 Pro, Mistral Large 4, Tencent Hy4, and Inkling 2 Small.
Exports results to results/results.json and results/REPORT.md.
"""

import json
from pathlib import Path
from moe_capacity_planner import (
    FRONTIER_MODELS,
    HARDWARE_H100,
    HARDWARE_H200,
    calculate_serving_capacity,
    estimate_weight_memory,
    estimate_kv_cache_memory,
)
from benchmark_version_analyzer import (
    INDEX_SNAPSHOTS,
    analyze_version_gap,
    compute_task_economics,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 75)
    print("RUNNING TRILLION-PARAMETER MOE CAPACITY PLANNING BENCHMARK")
    print("=" * 75)

    capacity_matrix = []
    print("\n[1/3] Evaluating Weight & KV-Cache Footprint on NVIDIA H100 (80GB)...")

    # Evaluate across 4 models at FP8 and FP4 for 128k and 1M context
    test_contexts = [8_192, 131_072, 1_048_576]
    test_precisions = [8, 4]

    for model_name, model in FRONTIER_MODELS.items():
        print(f"\nModel: {model_name} (Total: {model.total_params_b}B | Active: {model.active_params_b}B)")
        for prec in test_precisions:
            for ctx in test_contexts:
                plan_h100 = calculate_serving_capacity(model, prec, ctx, HARDWARE_H100)
                plan_h200 = calculate_serving_capacity(model, prec, ctx, HARDWARE_H200)
                capacity_matrix.append({
                    "model": model_name,
                    "precision_bits": prec,
                    "context_tokens": ctx,
                    "weight_gb": plan_h100["weight_memory_gb"],
                    "kv_gb": plan_h100["kv_cache_memory_gb"],
                    "total_needed_gb": plan_h100["total_memory_needed_gb"],
                    "h100_gpus": plan_h100["gpus_required"],
                    "h100_nodes": plan_h100["nodes_required"],
                    "h100_headroom_gb": plan_h100["headroom_gb"],
                    "h200_gpus": plan_h200["gpus_required"],
                    "h200_nodes": plan_h200["nodes_required"],
                })
                if ctx == 1_048_576:
                    print(
                        f"  Precision: {prec}-bit | Context: 1M | Total VRAM: {plan_h100['total_memory_needed_gb']} GB "
                        f"-> Requires {plan_h100['gpus_required']} H100s ({plan_h100['nodes_required']} nodes)"
                    )

    # -------------------------------------------------------------
    # 2. Benchmark Version Fallacy Analysis
    # -------------------------------------------------------------
    print("\n[2/3] Analyzing Artificial Analysis v4.3 vs v5 Methodology Shift...")
    gap_kimi_mistral = analyze_version_gap("Kimi K3.1 Pro", "Mistral Large 4")
    gap_claude_kimi = analyze_version_gap("Claude Opus 5.5 (Closed Baseline)", "Kimi K3.1 Pro")

    print(f"  Claude vs Kimi (v5 Same-Version Gap): {gap_claude_kimi['true_v5_same_version_gap']} pts")
    print(f"  Kimi vs Mistral (v5 Same-Version Gap): {gap_kimi_mistral['true_v5_same_version_gap']} pts")
    print(f"  Kimi vs Mistral (Cross-Version Fallacy: v4.3 vs v5): {gap_kimi_mistral['manufactured_cross_version_gap']} pts "
          f"(+{gap_kimi_mistral['fallacy_inflation_points']} pts artificial inflation!)")

    # -------------------------------------------------------------
    # 3. Task Economics & Verbosity
    # -------------------------------------------------------------
    print("\n[3/3] Modeling End-to-End Task Economics & Token Verbosity...")
    economics = {}
    for name, snap in INDEX_SNAPSHOTS.items():
        eco = compute_task_economics(
            active_params_b=50.0,
            tokens_generated=snap.avg_tokens_per_task,
            gpu_cost_per_hour_usd=2.50,
            active_gpus=8,
            tokens_per_sec=35.0,
        )
        economics[name] = eco
        print(f"  {name:<32}: {snap.avg_tokens_per_task:,} tokens | {eco['task_duration_seconds']/60:.1f} min | ${eco['compute_cost_usd']:.2f}")

    # -------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------
    results_payload = {
        "capacity_matrix": capacity_matrix,
        "version_gap_analysis": {
            "claude_vs_kimi": gap_claude_kimi,
            "kimi_vs_mistral": gap_kimi_mistral,
        },
        "task_economics": economics,
        "index_snapshots": {k: v.__dict__ for k, v in INDEX_SNAPSHOTS.items()},
    }

    results_file = RESULTS_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved raw results to {results_file}")

    # Generate REPORT.md
    report_lines = [
        "# Trillion-Parameter MoE Serving & Capacity Report",
        "",
        "## 1. Hardware Allocation Grid (1 Million Token Context)",
        "",
        "| Model | Total Params | Active Params | Precision | Weight VRAM | KV Cache (1M) | Total Needed | H100 GPUs (Nodes) | H200 GPUs (Nodes) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for row in capacity_matrix:
        if row["context_tokens"] == 1_048_576:
            report_lines.append(
                f"| **{row['model']}** | — | — | {row['precision_bits']}-bit | {row['weight_gb']:.1f} GB | "
                f"{row['kv_gb']:.1f} GB | {row['total_needed_gb']:.1f} GB | **{row['h100_gpus']} GPUs ({row['h100_nodes']}N)** | "
                f"{row['h200_gpus']} GPUs ({row['h200_nodes']}N) |"
            )

    report_lines.extend([
        "",
        "## 2. Artificial Analysis Intelligence Index v5 vs v4.3 Version Analysis",
        "",
        "| Model | v4.3 Score | v5 Score | True Same-Version Gap vs Leader | Public Weights Downloadable |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for name, snap in INDEX_SNAPSHOTS.items():
        v4_str = f"{snap.v4_score:.1f}" if snap.v4_score is not None else "N/A"
        gap_str = f"-{58.0 - snap.v5_score:.1f} pts" if name != "Claude Opus 5.5 (Closed Baseline)" else "Baseline (58.0)"
        report_lines.append(
            f"| **{name}** | {v4_str} | **{snap.v5_score:.1f}** | {gap_str} | {'Yes (Hugging Face)' if snap.is_open_weights_released else 'Hosted / Planned'} |"
        )

    report_lines.extend([
        "",
        "## 3. End-to-End Task Economics & Token Verbosity",
        "",
        "| Model | Avg Tokens / Task | Task Latency | Hosted Cost / Task | Open-Weight Viability |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for name, snap in INDEX_SNAPSHOTS.items():
        report_lines.append(
            f"| **{name}** | {snap.avg_tokens_per_task:,} | {snap.eval_time_minutes:.1f} min | ${snap.cost_per_task_usd:.2f} | "
            f"{'Self-hostable' if snap.is_open_weights_released else 'API / Platform only'} |"
        )

    report_file = RESULTS_DIR / "REPORT.md"
    report_file.write_text("\n".join(report_lines) + "\n")
    print(f"Saved formatted markdown report to {report_file}")


if __name__ == "__main__":
    main()

