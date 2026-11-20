"""Educational SSD-backed mixture-of-experts (MoE) inference simulation.

Real experts perform matrix multiplication, selected by a real top-k router.
This does NOT implement Qwen, Metal kernels, or generation.
"""
import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def create_weights(root: Path, n_experts=32, hidden=96, intermediate=192, seed=7):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for eid in range(n_experts):
        # W1: hidden->intermediate; W2: intermediate->hidden
        w1 = (rng.standard_normal((hidden, intermediate)) / np.sqrt(hidden)).astype('float32')
        w2 = (rng.standard_normal((intermediate, hidden)) / np.sqrt(intermediate)).astype('float32')
        np.savez(root / f'expert_{eid:03d}.npz', w1=w1, w2=w2)
    router = (rng.standard_normal((hidden, n_experts)) / np.sqrt(hidden)).astype('float32')
    np.save(root / 'router.npy', router, allow_pickle=False)
    (root / 'config.json').write_text(json.dumps(dict(n_experts=n_experts,hidden=hidden,intermediate=intermediate,seed=seed),indent=2))


class ExpertStore:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root/'config.json').read_text())
        self.bytes_read = 0
        self.loads = 0

    def get(self, eid):
        file = self.root/f'expert_{eid:03d}.npz'
        # On-demand load from a real disk file. Each read materializes one expert.
        with np.load(file, allow_pickle=False) as weights:
            w1 = torch.from_numpy(weights['w1'].copy())
            w2 = torch.from_numpy(weights['w2'].copy())
        self.bytes_read += file.stat().st_size
        self.loads += 1
        return w1, w2


class LRUExpertCache:
    def __init__(self, store, capacity):
        self.store = store
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, eid):
        eid = int(eid)
        if eid in self.entries:
            self.hits += 1
            self.entries.move_to_end(eid)
            return self.entries[eid]
        self.misses += 1
        weights = self.store.get(eid)
        if self.capacity > 0:
            self.entries[eid] = weights
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return weights

    @property
    def resident_bytes(self):
        return sum(sum(w.numel()*w.element_size() for w in weights) for weights in self.entries.values())


def routing(x, router, top_k):
    scores = x @ router
    values, indices = torch.topk(scores, k=top_k, dim=-1)
    gates = torch.softmax(values, dim=-1)
    return indices, gates


def forward_one(x, indices, gates, cache):
    # x shape [hidden], indices shape [top_k]; executes selected experts only.
    result = torch.zeros_like(x)
    for idx, gate in zip(indices.tolist(), gates):
        w1, w2 = cache.get(idx)
        result += gate * (torch.relu(x @ w1) @ w2)
    return result


def generate_inputs(router, count, seed, workload):
    g = torch.Generator().manual_seed(seed)
    h = router.shape[0]
    if workload == 'uniform':
        return torch.randn((count,h),generator=g)
    # Router-aligned vectors create naturally recurring expert choices.
    if workload == 'hotspot':
        center = router[:, 0].clone() * 7
        return center + .25*torch.randn((count,h),generator=g)
    if workload == 'shifting':
        a = router[:, 0].clone() * 7
        b = router[:, router.shape[1]//2].clone() * 7
        chunks = torch.stack([a if i < count//2 else b for i in range(count)])
        return chunks + .25*torch.randn((count,h),generator=g)
    raise ValueError(workload)


def bench(root, capacities, workloads, tokens, top_k, seed):
    cfg = json.loads((root/'config.json').read_text())
    router = torch.from_numpy(np.load(root/'router.npy').copy())
    results = []
    for workload in workloads:
        xs = generate_inputs(router, tokens, seed, workload)
        idxs, gates = routing(xs, router, top_k)
        # precomputation isolates expert cache behavior from router time
        for capacity in capacities:
            store = ExpertStore(root)
            cache = LRUExpertCache(store, capacity)
            checksum = 0.
            t0 = time.perf_counter()
            for x, idx, gate in zip(xs,idxs,gates):
                checksum += forward_one(x,idx,gate,cache).sum().item()
            elapsed = time.perf_counter()-t0
            accesses = cache.hits + cache.misses
            results.append(dict(workload=workload,cache_capacity=capacity,tokens=tokens,
                accesses=accesses,hits=cache.hits,misses=cache.misses,
                hit_rate=round(cache.hits/accesses,4) if accesses else 0,
                bytes_read=store.bytes_read,resident_bytes=cache.resident_bytes,
                elapsed_seconds=round(elapsed,5),tokens_per_second=round(tokens/elapsed,2),
                checksum=round(checksum,5)))
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('expert_store'))
    p.add_argument('--experts',type=int,default=32)
    p.add_argument('--hidden',type=int,default=96)
    p.add_argument('--intermediate',type=int,default=192)
    p.add_argument('--tokens',type=int,default=300)
    p.add_argument('--top-k',type=int,default=2)
    p.add_argument('--capacities',type=int,nargs='+',default=[0,2,4,8,16,32])
    p.add_argument('--workloads',nargs='+',choices=['uniform','hotspot','shifting'],default=['uniform','hotspot','shifting'])
    p.add_argument('--rebuild',action='store_true')
    p.add_argument('--out',type=Path,default=Path('results'))
    a=p.parse_args()
    if a.rebuild or not (a.root/'config.json').exists():
        create_weights(a.root,a.experts,a.hidden,a.intermediate)
    cfg=json.loads((a.root/'config.json').read_text())
    assert 1 <= a.top_k <= cfg['n_experts']
    assert all(0 <= x <= cfg['n_experts'] for x in a.capacities)
    torch.set_num_threads(1)
    results=bench(a.root,a.capacities,a.workloads,a.tokens,a.top_k,seed=42)
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'benchmark.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
    (a.out/'benchmark.json').write_text(json.dumps(results,indent=2))
    print('workload     slots hit-rate disk-MB tok/s checksum')
    for r in results:
        print(f"{r['workload']:<12} {r['cache_capacity']:>3} {r['hit_rate']:>7.1%} {r['bytes_read']/1e6:>7.2f} {r['tokens_per_second']:>7.1f} {r['checksum']:>10.4f}")
    print(f'\nSaved {a.out / "benchmark.csv"}')

if __name__=='__main__': main()
