"""Isolate two-level assignment and its effect on a toy mini-batch trainer."""
import argparse
import numpy as np


def distances(x, centers):
    return np.maximum((x*x).sum(1)[:, None] + (centers*centers).sum(1) - 2 * x @ centers.T, 0)


def cells(centers, count, rng):
    coarse = centers[rng.choice(len(centers), count, replace=False)].copy()
    for _ in range(5):
        labels = distances(centers, coarse).argmin(1)
        for i in range(count):
            group = centers[labels == i]
            if len(group):
                coarse[i] = group.mean(0)
    labels = distances(centers, coarse).argmin(1)
    active = np.unique(labels)
    return coarse[active], [np.flatnonzero(labels == i) for i in active]


def assign(x, centers, index, probes):
    coarse, members = index
    chosen = distances(x, coarse).argsort(axis=1)[:, :probes]
    ids = np.empty(len(x), dtype=int)
    work = len(x) * len(coarse)
    for i, routes in enumerate(chosen):
        candidates = np.sort(np.concatenate([members[j] for j in routes]))
        d = ((centers[candidates] - x[i])**2).sum(1)
        ids[i] = candidates[d.argmin()]
        work += len(candidates)
    return ids, work / len(x)


def train(x, initial, batches, probes):
    centers = initial.copy()
    for step, rows in enumerate(batches):
        batch = x[rows]
        if probes:
            if step % 4 == 0:
                index = cells(centers, 16, np.random.default_rng(100 + step))
            labels, _ = assign(batch, centers, index, probes)
        else:
            labels = distances(batch, centers).argmin(1)
        for i in np.unique(labels):
            centers[i] += 0.35 * (batch[labels == i].mean(0) - centers[i])
    return centers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    modes = rng.normal(size=(64, 12))
    data = modes[rng.integers(64, size=2560)] + rng.normal(scale=.75, size=(2560, 12))
    x, held = data[:2048], data[2048:]
    initial = x[rng.choice(len(x), 256, replace=False)]
    index = cells(initial, 16, rng)
    exact = distances(held, initial).argmin(1)
    oracle_loss = ((held-initial[exact])**2).sum(1).mean()
    batches = [b for _ in range(2) for b in np.split(rng.permutation(len(x)), 8)]
    print("route   comparisons/row   winner agreement   fixed-center excess loss   trained held-out loss")
    for probes in (0, 1, 2, 4, 8, 16):
        ids, work = assign(held, initial, index, probes) if probes else (exact, 256.)
        loss = ((held-initial[ids])**2).sum(1).mean()
        fitted = train(x, initial, batches, probes)
        final_loss = distances(held, fitted).min(1).mean()
        print(f"{probes or 'dense':>5} {work:17.2f} {np.mean(ids==exact):18.2%} "
              f"{loss/oracle_loss-1:24.2%} {final_loss:23.4f}")


if __name__ == "__main__":
    main()
