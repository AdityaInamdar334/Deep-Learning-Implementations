"""Reproducible CPU top-k MoE experiment and CSV report."""
import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np
from .core import ToyMoE


def measure(model, x, k, repeats):
    model.forward(x[:2], top_k=k)  # warmup
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        y = model.forward(x, top_k=k)
        times.append((perf_counter() - t0) * 1000)
    return median(times), y


def run(n_experts=16, hidden=64, intermediate=128, tokens=128, top_ks=(1, 2, 4, 8, 16), repeats=3, seed=7):
    model = ToyMoE(n_experts, hidden, intermediate, seed=seed)
    rng = np.random.default_rng(seed + 1)
    x = rng.normal(size=(tokens, hidden)).astype(np.float32)
    all_output = model.dense_reference(x)
    rows = []
    for k in top_ks:
        if k > n_experts:
            continue
        elapsed_ms, y = measure(model, x, k, repeats)
        c = model.cost(k)
        ids, _ = model.routes(x, top_k=k)
        counts = np.bincount(ids.ravel(), minlength=n_experts)
        rows.append({
            'num_experts': n_experts,
            'top_k': k,
            'tokens': tokens,
            'stored_params': c.total_parameters,
            'active_params_per_token': c.active_parameters_per_token,
            'estimated_expert_and_router_MACs_per_token': c.estimated_macs_per_token,
            'weights_MB_float32': round(c.weight_bytes_float32 / 1e6, 4),
            'elapsed_ms_CPU_median': round(elapsed_ms, 3),
            'output_max_abs_delta_vs_all_experts': round(float(np.max(np.abs(y - all_output))), 8),
            'router_max_to_min_load': round(float(counts.max() / max(counts.min(), 1)), 3),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experts', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--intermediate', type=int, default=128)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--top-k', default='1,2,4,8,16')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--csv', default='results/moe_benchmark.csv')
    a = p.parse_args()
    ks = [int(k) for k in a.top_k.split(',')]
    if min(ks) < 1 or a.repeats < 1 or a.tokens < 1:
        raise ValueError('Top-k, repeats and tokens must be positive')
    rows = run(a.experts, a.hidden, a.intermediate, a.tokens, ks, a.repeats, a.seed)
    for r in rows:
        print(f"top_k={r['top_k']:2} stored={r['stored_params']:7} active={r['active_params_per_token']:7} "
              f"MACs/token={r['estimated_expert_and_router_MACs_per_token']:7} "
              f"CPU ms={r['elapsed_ms_CPU_median']:8.2f} "
              f"delta={r['output_max_abs_delta_vs_all_experts']:.5f}")
    dest = Path(a.csv)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'CSV: {dest}')
    print('NOTE: Toy CPU dispatch overhead and random routing are NOT production GPU MoE throughput.')


if __name__ == '__main__':
    main()
