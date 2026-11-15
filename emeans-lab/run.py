import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import load_digits, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from cluster import assign, settings, train, peak_step


def experiment(args):
    if args.data == "digits":
        x, y = load_digits(return_X_y=True)
        x /= 16.0
        k = 16
    else:
        x, y = make_moons(n_samples=1800, noise=0.06, random_state=13)
        k = 2
    x, test, _, labels = train_test_split(x, y, test_size=0.25, stratify=y, random_state=17)
    batch = args.batch or settings(len(x), k, args.epochs)[0]
    if not 1 <= batch <= len(x):
        raise ValueError(f"batch must be between 1 and {len(x)}")
    print(f"train={len(x)}, test={len(test)}, k={k}, batch={batch}, "
          f"peak step={peak_step(len(x), k, args.epochs, batch):.4f}")
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        initial = x[rng.choice(len(x), k, replace=False)]
        seeded, _ = kmeans_plusplus(x, k, random_state=seed)
        methods = [('Lloyd', 'lloyd', initial), ('Lloyd++', 'lloyd', seeded),
                   ('Cumulative', 'cumulative', initial), ('e-means', 'ema', initial)]
        for name, method, start in methods:
            for epoch, live, final, splits in train(x, start, method, args.epochs, batch, seed,
                                                   not args.no_revival, not args.constant):
                outputs = [(name, live)]
                if method == 'ema':
                    outputs.append(('e-means final mean', final))
                for variant, centers in outputs:
                    train_ids = assign(x, centers)
                    test_ids = assign(test, centers)
                    rows.append(dict(seed=seed, method=variant, epoch=epoch,
                        train_mse=np.square(x - centers[train_ids]).sum(axis=1).mean(),
                        test_mse=np.square(test - centers[test_ids]).sum(axis=1).mean(),
                        active_centers=len(np.unique(train_ids)), split_count=splits,
                        test_ari=adjusted_rand_score(labels, test_ids)))
    args.out.mkdir(exist_ok=True, parents=True)
    with (args.out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in dict.fromkeys(r['method'] for r in rows):
        curve = []
        for epoch in range(1, args.epochs + 1):
            last = [r for r in rows if r['method'] == name and r['epoch'] == epoch]
            curve.append(np.median([r['test_mse'] for r in last]))
        print(f"{name:20} test MSE {curve[-1]:.4f}   median ARI "
              f"{np.median([r['test_ari'] for r in last]):.3f}")
        ax.plot(range(1, args.epochs + 1), curve, label=name)
    ax.set(xlabel='Training passes', ylabel='Median held-out squared distance',
           title=f'{args.data.capitalize()} • {args.seeds} initializations • k={k}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out/'loss.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['digits', 'moons'], default='digits')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--no-revival', action='store_true')
    parser.add_argument('--constant', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    if min(args.epochs, args.seeds) < 1 or args.batch is not None and args.batch < 1:
        parser.error('epochs, seeds and batch must be positive')
    with threadpool_limits(limits=1):
        experiment(args)
