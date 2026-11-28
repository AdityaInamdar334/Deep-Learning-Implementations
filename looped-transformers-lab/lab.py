# Train and evaluate dense and recurrent transformers on independent pointer-chasing streams.

import argparse, json, random, time, platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import Model
ROOT = Path(__file__).resolve().parent

def batch(g, b, n, hmin, hmax, device):
    edges = torch.randint(n, (b, n), generator=g)
    start = torch.randint(n, (b,), generator=g)
    hops = torch.randint(hmin, hmax + 1, (b,), generator=g)
    y = start.clone()
    for t in range(hmax):
        y = torch.where(hops > t, edges.gather(1, y[:, None]).squeeze(1), y)
    return tuple((x.to(device) for x in (edges, start, hops, y)))

def evaluate(model, g, device, loops, hops, b=256):
    model.eval()
    x, s, h, y = batch(g, b, 8, hops, hops, device)
    with torch.no_grad():
        t = time.perf_counter()
        z, states = model(x, s, h, loops, True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if device.type == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t
        changes = [float((a - b).norm() / a.norm().clamp_min(1e-08)) for a, b in zip(states, states[1:])]
        return {'accuracy': float((z.argmax(-1) == y).float().mean()), 'seconds': elapsed, 'relative_state_changes': changes, 'expert_counts': model.counts.cpu().tolist() if model.counts is not None else None}

def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    configs = [('dense2', 2, False, False, False, 1), ('dense6', 6, False, False, False, 1), ('loop_fixed', 2, True, False, False, 3), ('loop_variable', 2, True, False, False, 0), ('loop_moe', 2, True, True, True, 3)]
    results = []
    for seed in args.seeds:
        for name, depth, shared, moe, gated, fixed in configs:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            g = torch.Generator().manual_seed(seed + 100)
            m = Model(depth=depth, shared=shared, moe=moe, gated=gated).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=0.002)
            start = time.perf_counter()
            applications = 0
            losses = []
            for step in range(args.steps):
                m.train()
                x, s, h, y = batch(g, args.batch, 8, 1, 3, device)
                loops = fixed or rng.choice([1, 2, 3, 4, 5])
                z = m(x, s, h, loops)
                loss = F.cross_entropy(z, y) + 0.01 * m.aux
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                applications += depth * (loops if shared else 1)
                losses.append(float(loss.detach()))
            train_sec = time.perf_counter() - start
            ev = {}
            for loops in [1, 2, 3, 4, 5, 6] if shared else [1]:
                ev[str(loops)] = {str(h): evaluate(m, torch.Generator().manual_seed(seed + 900 + h), device, loops, h) for h in [1, 2, 3, 4, 5, 6]}
            row = {'name': name, 'seed': seed, 'parameters': sum((p.numel() for p in m.parameters())), 'layer_applications_per_batch_sum': applications, 'train_seconds': train_sec, 'loss_first': float(np.mean(losses[:20])), 'loss_last': float(np.mean(losses[-20:])), 'evaluation': ev}
            results.append(row)
            print(name, seed, row['parameters'], round(row['loss_last'], 3), round(np.mean([ev[str(3 if shared else 1)][str(h)]['accuracy'] for h in (1, 2, 3)]), 3), flush=True)
            (ROOT / 'results/results.json').write_text(json.dumps({'config': vars(args), 'device': str(device), 'torch': torch.__version__, 'platform': platform.platform(), 'runs': results}, indent=2))
    plot(results)

def plot(rows):
    names = list(dict.fromkeys((r['name'] for r in rows)))
    groups = {n: [r for r in rows if r['name'] == n] for n in names}
    plt.figure(figsize=(9, 4.8))
    for n, rs in groups.items():
        l = '1' if n.startswith('dense') else '3'
        plt.plot(range(1, 7), [np.mean([r['evaluation'][l][str(h)]['accuracy'] for r in rs]) for h in range(1, 7)], marker='o', label=n)
    plt.axhline(1 / 8, ls=':', label='uniform guess')
    plt.axvline(3.5, ls='--', color='gray')
    plt.xlabel('Pointer hops (training: 1–3)')
    plt.ylabel('Mean test accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / 'figures/hops.png', dpi=180)
    plt.close()
    plt.figure(figsize=(8, 5))
    for n in ['loop_fixed', 'loop_variable', 'loop_moe']:
        rs = groups[n]
        plt.plot(range(1, 7), [np.mean([r['evaluation'][str(l)][str(h)]['accuracy'] for r in rs for h in [1, 2, 3]]) for l in range(1, 7)], marker='o', label=n)
    plt.xlabel('Inference recurrent passes')
    plt.ylabel('Mean accuracy (1–3 hops)')
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(ROOT / 'figures/loop_scaling.png', dpi=180)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.bar(names, [groups[n][0]['parameters'] for n in names])
    plt.ylabel('Unique trainable parameters')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(ROOT / 'figures/parameters.png', dpi=180)
    plt.close()
    plt.figure(figsize=(8, 4))
    for n in ['loop_fixed', 'loop_variable', 'loop_moe']:
        vals = [r['evaluation']['6']['3']['relative_state_changes'] for r in groups[n]]
        plt.plot(range(1, 6), np.mean(vals, axis=0), marker='o', label=n)
    plt.xlabel('Transition between successive passes')
    plt.ylabel('Relative activation change (Frobenius norm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / 'figures/activations.png', dpi=180)
    plt.close()
    counts = np.sum([r['evaluation']['3']['3']['expert_counts'] for r in groups['loop_moe']], axis=0)
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(counts)), counts / counts.sum())
    plt.xlabel('Expert index (pooled over blocks and seeds)')
    plt.ylabel('Fraction of selected slots')
    plt.tight_layout()
    plt.savefig(ROOT / 'figures/routing.png', dpi=180)
    plt.close()
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=360)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=1)
    run(p.parse_args())
