"""Small exact-assignment CPU solvers for studying e-means."""
import numpy as np


def assign(x, centers):
    # The point norm is constant across centers and cancels in argmin.
    scores = np.sum(centers**2, axis=1) - 2 * (x @ centers.T)
    return scores.argmin(axis=1)


def stats(x, labels, k):
    sums = np.zeros((k, x.shape[1]))
    np.add.at(sums, labels, x)
    return sums, np.bincount(labels, minlength=k)


def means(sums, counts, previous):
    return np.divide(sums, counts[:, None], out=previous.copy(),
                     where=counts[:, None] > 0)


def settings(n, k, epochs):
    coefficient = 0.035 if n <= 2_000_000 else 0.0044
    knee = coefficient * k**0.13 * n**0.88 * epochs**0.81
    batch = min(n, max(1, round(2 ** round(np.log2(knee)))))
    return batch, peak_step(n, k, epochs, batch)


def peak_step(n, k, epochs, batch):
    return min(1.0, 0.824 * k**-0.46 * batch**0.90 * n**-0.45 * epochs**-0.11)


def revive(sums, counts, rng):
    dead = np.flatnonzero(counts < 0.1 * counts.mean())
    live = np.setdiff1d(np.arange(len(counts)), dead)
    pairs = min(len(dead), len(live))
    if not pairs:
        return 0
    donors = rng.choice(live, pairs, replace=False, p=counts[live]/counts[live].sum())
    children = dead[:pairs]
    centers = sums[donors] / counts[donors, None]
    offset = np.where(np.arange(sums.shape[1]) % 2 == 0, 1, -1) / 128
    counts[donors] *= 0.5
    counts[children] = counts[donors]
    sums[children] = centers * (1 + offset) * counts[children, None]
    sums[donors] = centers * (1 - offset) * counts[donors, None]
    return pairs


def train(x, initial, method, epochs, batch, seed, revival=True, schedule=True):
    """Yield epoch, live centers, last-epoch means and cumulative split count."""
    centers = initial.copy()
    k = len(centers)
    counts = np.full(k, 1e-10)
    sums = counts[:, None] * centers
    rng = np.random.default_rng(seed)
    split_rng = np.random.default_rng(seed + 1000)
    total_steps = epochs * ((len(x) + batch - 1) // batch)
    peak = peak_step(len(x), k, epochs, batch)
    step = splits = 0
    for epoch in range(1, epochs + 1):
        if method == 'lloyd':
            s, n = stats(x, assign(x, centers), k)
            centers = means(s, n, centers)
            yield epoch, centers, centers, splits
            continue
        epoch_s, epoch_n = np.zeros_like(sums), np.zeros(k)
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch):
            points = x[order[start:start + batch]]
            s, n = stats(points, assign(points, centers), k)
            epoch_s += s
            epoch_n += n
            if method == 'cumulative':
                sums += s
                counts += n
            elif method == 'ema':
                factor = 0.1 + 0.45 * (1 + np.cos(np.pi * step / total_steps))
                alpha = peak * (factor if schedule else 1)
                sums = (1 - alpha) * sums + alpha * s
                counts = (1 - alpha) * counts + alpha * n
                if revival and (step + 1) % 5 == 0:
                    splits += revive(sums, counts, split_rng)
            else:
                raise ValueError(method)
            centers = means(sums, counts, centers)
            step += 1
        yield epoch, centers, means(epoch_s, epoch_n, centers), splits
