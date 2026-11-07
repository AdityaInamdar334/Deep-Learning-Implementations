"""Weight capacity estimator, not a deployment certification."""
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

# Published rounded architecture numbers; no guarantee of model availability.
MODELS = {
    'Mistral Large 4': (1050.0, 52.0),
    'Kimi K3.1 Pro': (2800.0, 104.0),
    'Tencent Hy4': (770.0, 49.0),
    'Inkling 2 Small': (594.0, 17.0),
}


@dataclass(frozen=True)
class Capacity:
    raw_weight_gb: float
    effective_weight_gb: float
    extra_memory_gb: float
    required_gb: float
    usable_gb_per_gpu: float
    min_gpus: int
    min_nodes: int


def estimate(params_b, bits=4, gpu_gb=80, gpus_per_node=8,
             weight_overhead_pct=0, reserve_gb_per_gpu=0, extra_gb=0):
    vals = (params_b, bits, gpu_gb, gpus_per_node, weight_overhead_pct, reserve_gb_per_gpu, extra_gb)
    if params_b <= 0 or bits <= 0 or gpu_gb <= 0 or gpus_per_node < 1:
        raise ValueError('Parameters, bits, GPU memory, and GPUs per node must be positive')
    if weight_overhead_pct < 0 or reserve_gb_per_gpu < 0 or extra_gb < 0:
        raise ValueError('Overhead, reserve and extra memory cannot be negative')
    usable = gpu_gb - reserve_gb_per_gpu
    if usable <= 0:
        raise ValueError('GPU reserve must be smaller than physical GPU memory')
    # 1B params at 8 bits is 1 decimal GB.
    raw = params_b * bits / 8.0
    effective = raw * (1 + weight_overhead_pct / 100)
    required = effective + extra_gb
    gpus = math.ceil(required / usable)
    nodes = math.ceil(gpus / gpus_per_node)
    return Capacity(raw, effective, extra_gb, required, usable, gpus, nodes)


def conventional_kv_gb(layers, kv_heads, head_dim, tokens, batch=1, bits=16):
    """A conventional full-attention K+V tensor estimate, decimal GB.

    It does NOT model MLA, KV latent compression, windowed/sparse eviction,
    index caches, paged KV fragmentation, mixed layer architectures, etc.
    """
    if min(layers, kv_heads, head_dim, tokens, batch, bits) < 1:
        raise ValueError('KV settings must be positive')
    return (2 * layers * kv_heads * head_dim * tokens * batch * bits / 8) / 1e9


def main():
    p = argparse.ArgumentParser(description='Estimate minimum GPU weight capacity (NOT serving hardware).')
    p.add_argument('--bits', type=float, default=4)
    p.add_argument('--gpu-gb', type=float, default=80)
    p.add_argument('--gpus-per-node', type=int, default=8)
    p.add_argument('--weight-overhead-pct', type=float, default=0)
    p.add_argument('--reserve-gb-per-gpu', type=float, default=0)
    p.add_argument('--extra-gb', type=float, default=0, help='User-supplied cache and buffers, across the model replica')
    p.add_argument('--csv', default=None, help='Optional destination for CSV report')
    a = p.parse_args()
    rows = []
    for name, (total, active) in MODELS.items():
        s = estimate(total, a.bits, a.gpu_gb, a.gpus_per_node,
                     a.weight_overhead_pct, a.reserve_gb_per_gpu, a.extra_gb)
        row = {'model': name, 'total_B': total, 'active_B': active,
               'raw_weights_GB': round(s.raw_weight_gb, 2),
               'required_GB': round(s.required_gb, 2),
               'min_80GB_equiv_GPUs': s.min_gpus, 'min_nodes': s.min_nodes}
        rows.append(row)
    print(f'Minimum capacity only | {a.bits:g}-bit | {a.gpu_gb:g} GB GPUs | reserve {a.reserve_gb_per_gpu:g} GB/GPU | extra {a.extra_gb:g} GB')
    for r in rows:
        print(f"{r['model']:20} raw {r['raw_weights_GB']:8.1f} GB | required {r['required_GB']:8.1f} GB | GPUs {r['min_80GB_equiv_GPUs']:2} | nodes {r['min_nodes']}")
    if a.csv:
        destination = Path(a.csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'Wrote {destination}')
    print('CAUTION: theoretical capacity, not a validated inference configuration. Check weights format, KV, activation, parallelism and bandwidth.')


if __name__ == '__main__':
    main()
