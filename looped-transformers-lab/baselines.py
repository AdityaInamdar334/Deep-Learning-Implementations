"""Non-neural reference policies on exactly the held-out examples used by lab.py."""
import json
from pathlib import Path
import torch
from lab import batch
ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    config = json.loads((ROOT / 'results/results.json').read_text())['config']
    rows = []
    for seed in config['seeds']:
        for hops in range(1, 7):
            e, s, h, y = batch(torch.Generator().manual_seed(seed + 900 + hops), 256, 8, hops, hops, 'cpu')
            one = e.gather(1, s[:, None]).squeeze(1)
            oracle = s.clone()
            for _ in range(hops):
                oracle = e.gather(1, oracle[:, None]).squeeze(1)
            rows.append({'seed': seed, 'hops': hops, 'uniform_expected': 1 / 8, 'start_node': float((s == y).float().mean()), 'one_hop': float((one == y).float().mean()), 'oracle': float((oracle == y).float().mean())})
    (ROOT / 'results/baselines.json').write_text(json.dumps(rows, indent=2))
