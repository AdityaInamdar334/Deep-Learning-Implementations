"""Controlled learning-rate and gradient-flow probes, CPU only."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(256, 16)
    y = (x[:, 0] + x[:, 1] > 0).long()
    histories = {}
    for lr in [0.0001, 0.1, 10.0]:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        losses, gradients, ratios = [], [], []
        for step in range(100):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            g = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
            w = torch.sqrt(sum(p.detach().square().sum() for p in model.parameters())).item()
            losses.append(loss.item()); gradients.append(g); ratios.append(lr*g/max(w, 1e-12))
            if not np.isfinite(losses[-1]) or not np.isfinite(g):
                break
            opt.step()
        histories[str(lr)] = dict(loss=losses, gradient_norm=gradients, relative_update=ratios)
    fig, ax = plt.subplots(figsize=(8,4.5))
    for lr, h in histories.items(): ax.plot(h['loss'], label=f'LR {lr}')
    ax.set(xlabel='Update', ylabel='Full-batch training cross-entropy', yscale='log'); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/lr_diagnostics.png', dpi=180); plt.close(fig)
    # Exact empirical gradient noise: independent samples with replacement at fixed weights.
    rng = np.random.default_rng(42)
    individual = rng.normal(size=(4096, 8)) + 0.2
    truth = individual.mean(axis=0)
    noise = {}
    for b in [1, 4, 16, 64, 256]:
        sampled = individual[rng.integers(len(individual), size=(1000,b))].mean(axis=1)
        noise[str(b)] = float(np.sqrt(np.mean(np.sum((sampled-truth)**2,axis=1))))
    fig, ax = plt.subplots(figsize=(7,4.5)); bs=np.array(list(map(int,noise)))
    ax.loglog(bs,list(noise.values()),'o-',label='Measured RMS error'); ax.loglog(bs,noise['1']/np.sqrt(bs),'--',label='1/sqrt(B) reference'); ax.set(xlabel='Batch size',ylabel='RMS gradient estimation error'); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/gradient_noise.png',dpi=180); plt.close(fig)
    result = {'learning_rate':histories,'noise':noise}
    (ROOT/'diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for lr,h in histories.items(): print(lr, h['loss'][0],h['loss'][-1], max(h['loss']),max(h['relative_update']))
    print('RMS gradient errors:',noise)
if __name__ == '__main__': main()
