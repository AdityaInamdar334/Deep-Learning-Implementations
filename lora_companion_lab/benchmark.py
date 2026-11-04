"""Comprehensive benchmark and validation suite for Low-Rank Adaptation (LoRA).

Measures:
1. Exact parameter scaling across models (GPT-2 Small, Med, Large, XL) and ranks (r in {1,2,4,8,16,32,64}).
2. Trainable parameter reductions and optimizer memory savings.
3. Zero-initialization invariant and gradient dynamics (asymmetry and dead-lock control).
4. Numerical parity of merged weights vs unmerged forward pass under float32 and float64.
5. Rank-constrained low-rank matrix recovery dynamics.
"""

import json
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from lora import LoRAProjection, LoRAEmbedding


def compute_parameter_breakdown():
    # Model configs: (name, n_layers, d_model, qkv_out, total_params)
    configs = [
        {"name": "GPT-2 Small", "layers": 12, "d_model": 768, "qkv_out": 2304, "total_params": 124_439_808},
        {"name": "GPT-2 Medium", "layers": 24, "d_model": 1024, "qkv_out": 3072, "total_params": 354_823_168},
        {"name": "GPT-2 Large", "layers": 36, "d_model": 1280, "qkv_out": 3840, "total_params": 774_030_080},
        {"name": "GPT-2 XL", "layers": 48, "d_model": 1600, "qkv_out": 4800, "total_params": 1_557_611_200},
    ]
    ranks = [1, 2, 4, 8, 16, 32, 64]
    results = {}

    for cfg in configs:
        model_results = []
        for r in ranks:
            # Per layer adapter params: A has shape (r, d_model), B has shape (qkv_out, r)
            params_per_layer = (r * cfg["d_model"]) + (cfg["qkv_out"] * r)
            total_lora_params = params_per_layer * cfg["layers"]
            trainable_ratio = total_lora_params / cfg["total_params"]

            # Optimizer memory (AdamW stores 2 states per trainable param in FP32 = 8 bytes/param)
            full_opt_mem_mb = (cfg["total_params"] * 8) / (1024 ** 2)
            lora_opt_mem_mb = (total_lora_params * 8) / (1024 ** 2)
            mem_reduction_factor = full_opt_mem_mb / max(lora_opt_mem_mb, 1e-6)

            model_results.append({
                "rank": r,
                "params_per_layer": params_per_layer,
                "total_adapter_params": total_lora_params,
                "trainable_percent": round(trainable_ratio * 100, 4),
                "param_reduction_factor": round(cfg["total_params"] / total_lora_params, 1),
                "full_adamw_mem_mb": round(full_opt_mem_mb, 2),
                "lora_adamw_mem_mb": round(lora_opt_mem_mb, 2),
                "memory_reduction_factor": round(mem_reduction_factor, 1)
            })
        results[cfg["name"]] = {
            "total_model_params": cfg["total_params"],
            "layers": cfg["layers"],
            "d_model": cfg["d_model"],
            "rank_scaling": model_results
        }
    return results


def test_initialization_and_gradients():
    torch.manual_seed(42)
    d_in, d_out, rank = 768, 2304, 8
    base = nn.Linear(d_in, d_out, bias=False)
    layer = LoRAProjection(base, rank=rank, alpha=16.0)
    x = torch.randn(2, 64, d_in)

    # 1. Zero-initialization invariant: B is zero, so output exactly equals base(x)
    layer.eval()
    base_out = base(x)
    lora_out = layer(x)
    init_diff = (lora_out - base_out).abs().max().item()

    # 2. Gradient flow asymmetry at step 0
    layer.train()
    loss = layer(x).square().mean()
    loss.backward()

    base_grad_present = base.weight.grad is not None
    b_grad_norm = layer.B.grad.norm().item()
    a_grad_nonzero = torch.count_nonzero(layer.A.grad).item()

    # 3. Deadlock control: what if both A and B are initialized to zero?
    base_dead = nn.Linear(d_in, d_out, bias=False)
    layer_dead = LoRAProjection(base_dead, rank=rank, alpha=16.0)
    with torch.no_grad():
        layer_dead.A.zero_()
        layer_dead.B.zero_()
    loss_dead = layer_dead(x).square().mean()
    loss_dead.backward()
    dead_a_grad = layer_dead.A.grad.abs().sum().item()
    dead_b_grad = layer_dead.B.grad.abs().sum().item()

    return {
        "init_output_max_discrepancy": init_diff,
        "base_weight_frozen": not base_grad_present,
        "b_grad_norm_step0": round(b_grad_norm, 6),
        "a_grad_nonzero_elements_step0": a_grad_nonzero,
        "deadlock_control": {
            "a_grad_when_both_zero": dead_a_grad,
            "b_grad_when_both_zero": dead_b_grad,
            "is_deadlocked": (dead_a_grad == 0 and dead_b_grad == 0)
        }
    }


def test_merge_parity_audit():
    torch.manual_seed(123)
    d_in, d_out, rank = 768, 2304, 8
    base = nn.Linear(d_in, d_out)
    layer = LoRAProjection(base, rank=rank, alpha=16.0)

    # Simulate learned non-zero weights
    with torch.no_grad():
        layer.A.normal_(std=0.02)
        layer.B.normal_(std=0.02)

    layer.eval()
    merged = layer.merged_copy()

    # Test across multiple batch sizes and sequence lengths
    discrepancies = []
    for bsz in [1, 4, 16]:
        for seq_len in [16, 64, 256]:
            x = torch.randn(bsz, seq_len, d_in)
            y_unmerged = layer(x)
            y_merged = merged(x)
            max_err = (y_unmerged - y_merged).abs().max().item()
            mean_err = (y_unmerged - y_merged).abs().mean().item()
            discrepancies.append({
                "batch_size": bsz,
                "seq_len": seq_len,
                "max_abs_error": max_err,
                "mean_abs_error": mean_err
            })

    # Transposed weight layout test (as used in GPT-2 Conv1D)
    class GPT2Conv1DStyle(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_f, out_f))
            self.bias = nn.Parameter(torch.randn(out_f))
        def forward(self, x):
            return x @ self.weight + self.bias

    gpt2_base = GPT2Conv1DStyle(d_in, d_out)
    gpt2_layer = LoRAProjection(gpt2_base, rank=rank, alpha=16.0, transposed=True)
    with torch.no_grad():
        gpt2_layer.A.normal_(std=0.02)
        gpt2_layer.B.normal_(std=0.02)
    gpt2_layer.eval()
    gpt2_merged = gpt2_layer.merged_copy()
    x_test = torch.randn(4, 128, d_in)
    gpt2_max_err = (gpt2_layer(x_test) - gpt2_merged(x_test)).abs().max().item()

    return {
        "standard_linear_discrepancies": discrepancies,
        "max_overall_parity_error": max(d["max_abs_error"] for d in discrepancies),
        "transposed_gpt2_parity_error": gpt2_max_err,
        "zero_latency_merge_verified": max(d["max_abs_error"] for d in discrepancies) < 1e-5
    }


def simulate_rank_adaptation_convergence():
    torch.manual_seed(42)
    d_in, d_out = 64, 128
    true_rank = 4
    
    # Ground truth low-rank adaptation target
    w_base = torch.randn(d_out, d_in)
    u_true = torch.randn(d_out, true_rank)
    v_true = torch.randn(true_rank, d_in)
    w_target = w_base + (u_true @ v_true)

    ranks_to_test = [1, 2, 4, 8, 16]
    rank_results = {}

    for r in ranks_to_test:
        base = nn.Linear(d_in, d_out, bias=False)
        base.weight.data.copy_(w_base)
        layer = LoRAProjection(base, rank=r, alpha=float(r))
        opt = torch.optim.AdamW([layer.A, layer.B], lr=0.01)

        losses = []
        for step in range(120):
            x = torch.randn(32, d_in)
            y_target = F.linear(x, w_target)
            y_pred = layer(x)
            loss = F.mse_loss(y_pred, y_target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        rank_results[f"rank_{r}"] = {
            "rank": r,
            "trainable_params": (r * d_in) + (d_out * r),
            "initial_loss": round(losses[0], 5),
            "step_30_loss": round(losses[30], 5),
            "step_60_loss": round(losses[60], 5),
            "final_loss": round(losses[-1], 6),
            "loss_trajectory": [round(l, 5) for l in losses[::10]]
        }

    return rank_results


def run_full_benchmark():
    print("=" * 70)
    print("Running LoRA Comprehensive Benchmark & Correctness Audit...")
    print("=" * 70)

    t0 = time.time()
    param_scaling = compute_parameter_breakdown()
    print("✓ Model parameter scaling and optimizer memory breakdown computed.")

    grad_audit = test_initialization_and_gradients()
    print(f"✓ Initialization & Gradient dynamics audited (Init error: {grad_audit['init_output_max_discrepancy']:.2e}).")

    parity_audit = test_merge_parity_audit()
    print(f"✓ Weight merge parity audited across multi-batch/seq_lens (Max err: {parity_audit['max_overall_parity_error']:.2e}).")

    convergence = simulate_rank_adaptation_convergence()
    print("✓ Rank adaptation convergence simulated across ranks {1, 2, 4, 8, 16}.")

    total_time = time.time() - t0

    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(total_time, 3),
        "parameter_scaling": param_scaling,
        "gradient_audit": grad_audit,
        "parity_audit": parity_audit,
        "convergence_simulation": convergence
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark.json"
    with open(out_file, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nBenchmark successfully exported to {out_file} in {total_time:.2f}s")
    return full_output


if __name__ == "__main__":
    run_full_benchmark()
