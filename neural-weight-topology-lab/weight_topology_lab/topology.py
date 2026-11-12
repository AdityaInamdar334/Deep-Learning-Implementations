"""Exact small-cloud Vietoris-Rips H0/H1 persistence over F2.

Intentionally restricted to <=16 points; use ripser for serious 64/128-row
neuron clouds. This is a transparent teaching algorithm, not Ripser.py.
"""
import itertools
import numpy as np

MAX_POINTS = 16

def pairwise_euclidean(points):
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError('Expect a finite 2D array with >=1 point')
    differences = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.maximum(np.sum(differences * differences, axis=-1), 0))

def normalize_rows(points, eps=1e-12):
    x = np.asarray(points, dtype=float)
    if x.ndim != 2:
        raise ValueError('Expect 2D matrix')
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)

def rips_h0_h1(points):
    """Return H0/H1 diagrams (n,2) of finite intervals from Rips filtration.

    VR up to triangles suffices for H1, even though we do not construct higher
    simplices. Boundary matrix reduction uses the Z2 field. Nonzero finite
    intervals only; the single essential H0 component is excluded by default.
    """
    dist = pairwise_euclidean(points)
    n = len(dist)
    if n > MAX_POINTS:
        raise ValueError(f'Brute-force demonstration limited to {MAX_POINTS} points; use ripser for larger clouds')
    simplices = []
    for i in range(n):
        simplices.append((0.0, 0, (i,)))
    for verts in itertools.combinations(range(n), 2):
        simplices.append((float(dist[verts[0], verts[1]]), 1, verts))
    for verts in itertools.combinations(range(n), 3):
        a, b, c = verts
        simplices.append((float(max(dist[a, b], dist[a, c], dist[b, c])), 2, verts))
    simplices.sort(key=lambda item: (item[0], item[1], item[2]))
    index = {simplex[2]: i for i, simplex in enumerate(simplices)}
    reduced = {}      # pivot row -> reduced column
    pairs = {0: [], 1: []}
    for column_index, (time, dim, vertices) in enumerate(simplices):
        if dim == 0:
            continue
        boundary = set()
        for face in itertools.combinations(vertices, dim):
            boundary.add(index[face])
        while boundary:
            pivot = max(boundary)
            if pivot not in reduced:
                break
            boundary.symmetric_difference_update(reduced[pivot])
        if boundary:
            pivot = max(boundary)
            reduced[pivot] = boundary
            birth, bdim, _ = simplices[pivot]
            if (time - birth) > 1e-10 and bdim in pairs:
                pairs[bdim].append((birth, time))
    return {k: np.asarray(v, dtype=float).reshape((-1, 2)) for k, v in pairs.items()}

def summarize(diagram):
    d = np.asarray(diagram, dtype=float).reshape((-1, 2))
    if d.size == 0:
        return {'count': 0, 'total': 0.0, 'max': 0.0, 'mean': 0.0, 'entropy': 0.0}
    life = d[:, 1] - d[:, 0]
    life = life[np.isfinite(life) & (life > 1e-12)]
    if not len(life):
        return {'count': 0, 'total': 0.0, 'max': 0.0, 'mean': 0.0, 'entropy': 0.0}
    total = float(life.sum())
    p = life / total
    return {'count': int(len(life)), 'total': total, 'max': float(life.max()),
            'mean': float(life.mean()), 'entropy': float(-(p*np.log(p)).sum())}

def betti(diagram, grid):
    d = np.asarray(diagram, dtype=float).reshape((-1, 2))
    g = np.asarray(grid, dtype=float)
    if not len(d):
        return np.zeros_like(g, dtype=int)
    return ((d[:, 0, None] <= g[None]) & (g[None] < d[:, 1, None])).sum(axis=0)

def weight_cloud(weight_matrix, count=12, normalized=True):
    w = np.asarray(weight_matrix, dtype=float)
    if w.ndim != 2:
        raise ValueError('Expect matrix rows=neurons, columns=incoming weights')
    x = w[:min(count, len(w))].copy()
    return normalize_rows(x) if normalized else x
