# e-means: Outperforming k-means (in One-Tenth the Epochs)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)](test_cluster.py)
[![Paper](https://img.shields.io/badge/arXiv-2611.09501-B31B1B.svg)](https://arxiv.org/html/2611.09501v1)
[![Project](https://img.shields.io/badge/Project-e--means.github.io-4A154B.svg)](https://e-means.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An independent, zero-GPU NumPy teaching laboratory reproducing the core algorithms, dead-cluster revival dynamics, closed-form scaling laws, and geometry tradeoffs of Meta FAIR's **e-means** clustering algorithm.*

Based on the research paper:  
**["e-means: outperforming k-means (in one-tenth the epochs)"](https://arxiv.org/html/2611.09501v1)** (arXiv:2611.09501, November 2026).

</div>

---

## Executive Summary

For over four decades, **Lloyd's algorithm** ($k$-means) and **Mini-Batch $k$-means** (Sculley, 2010) have served as the foundational primitives for vector quantization, unsupervised clustering, and nearest-neighbor indexing (e.g. Faiss IVF indexes). However, both exhibit fundamental scaling bottlenecks when applied to modern billion-token neural embedding sets:

1. **Lloyd's Full-Batch Bottleneck**: Requires complete passes over the entire dataset before updating cluster centroids, demanding dozens of expensive memory-bound synchronization passes.
2. **Mini-Batch Cumulative Stagnation**: Mini-batch $k$-means maintains a cumulative running average of counts $\sum n_t$ and centers. Early iterations overwhelm later updates ($1 / N_t \to 0$), causing the solver to freeze prematurely in suboptimal local minima.
3. **Dead-Center Atrophy**: Centroids initialized in sparse regions quickly lose assignments and become permanently inactive, wasting model capacity.

```
Mini-Batch Cumulative Descent (Sculley 2010):
  Sums:   S_t = S_{t-1} + s_t
  Counts: N_t = N_{t-1} + n_t          <--- As N_t grows, learning rate 1/N_t collapses!
  Center: c_t = S_t / N_t

e-means EMA Descent (Meta FAIR 2026):
  Sums:   S_t = (1 - α_t) S_{t-1} + α_t · s_t   <--- Exponential moving average maintains
  Counts: N_t = (1 - α_t) N_{t-1} + α_t · n_t        adaptive recency memory!
  Center: c_t = S_t / N_t
  Revival: Starved clusters (N_i < 0.1 N_avg) split high-density donor clusters along ±1/128 axes.
```

**e-means** replaces cumulative averaging with **Exponential Moving Average (EMA)** statistics modulated by a cosine-decayed peak step size and periodic dead-cluster donor-splitting. This laboratory isolates each component in pure, readable NumPy without external framework abstractions.

---

## Architecture & Algorithm Design

```
                     ┌──────────────────────────────────────┐
                     │          Training Dataset X          │
                     └──────────────────┬───────────────────┘
                                        │
                         Random / k-means++ Seeding
                                        ▼
                     ┌──────────────────────────────────────┐
                     │      Initial Cluster Centers C_0     │
                     └──────────────────┬───────────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
┌──────────────────────┐                                  ┌──────────────────────┐
│  Lloyd's Algorithm   │                                  │   e-means (EMA)      │
│  (Full Dataset)      │                                  │   (Mini-Batch B)     │
└──────────┬───────────┘                                  └──────────┬───────────┘
           │                                                         │
   Exact Nearest Argmin                                      Exact Nearest Argmin
   Assign all N points                                       Assign mini-batch b_t
           │                                                         │
   Deterministic Mean                                        Compute batch sums s_t, n_t
   C_{t+1} = Σ x / N_c                                               │
           │                                                 EMA Smoothing (α_t Cosine Decay):
           │                                                 S_t = (1-α_t) S_{t-1} + α_t s_t
           │                                                 N_t = (1-α_t) N_{t-1} + α_t n_t
           │                                                         │
           │                                                 Every 5 Steps: Check Dead Clusters
           │                                                 If N_i < 0.1 N_mean:
           │                                                   Donor split + orthogonal jitter
           │                                                         │
           ▼                                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           Held-Out Test Set Evaluation                         │
│   - Test MSE: Held-out squared Euclidean distance to nearest center            │
│   - Active Centers: Count of clusters receiving ≥1 assigned points             │
│   - Adjusted Rand Index (ARI): Alignment against ground-truth labels           │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### 1. Vector Assignment & Distance Cancellation
For points $X \in \mathbb{R}^{B \times D}$ and centers $C \in \mathbb{R}^{K \times D}$, Euclidean distance minimization simplifies by expanding $\|x - c\|^2 = \|x\|^2 - 2 \langle x, c \rangle + \|c\|^2$. Since $\|x\|^2$ is constant across candidate centers, the cluster index is computed via fast matrix multiplication:
$$\text{assign}(X, C) = \arg\min_{j \in \{1,\dots,K\}} \left( \|C_j\|^2 - 2 (X C^T)_j \right)$$

### 2. Exponential Moving Average (EMA) Updates
At step $t$ with mini-batch data $X^{(t)}$:
$$s_t = \sum_{x \in X^{(t)}, \text{assign}(x)=k} x, \quad n_t = \sum_{x \in X^{(t)}} \mathbb{I}(\text{assign}(x) = k)$$
$$S_t = (1 - \alpha_t) S_{t-1} + \alpha_t s_t, \quad N_t = (1 - \alpha_t) N_{t-1} + \alpha_t n_t$$
$$C_t = \frac{S_t}{N_t}$$

### 3. Cosine Step-Size Schedule
The learning rate $\alpha_t$ smoothly transitions from its peak down to a stationary regime:
$$\alpha_t = \alpha_{\text{peak}} \left[ 0.10 + 0.45 \left( 1 + \cos\left( \frac{\pi t}{T_{\text{total}}} \right) \right) \right]$$

### 4. Closed-Form Scaling Laws (No Sweeps)
Rather than tuning $B$ and $\alpha$ empirically, e-means derives both parameters analytically from dataset cardinality $N$, cluster count $K$, and total epochs $E$:
$$B = \text{clamp}\left(1, N, \text{round}\left(2^{\text{round}\left(\log_2(c \cdot K^{0.13} N^{0.88} E^{0.81})\right)}\right)\right)$$
$$\alpha_{\text{peak}} = \min\left(1.0, \, 0.824 \cdot K^{-0.46} B^{0.90} N^{-0.45} E^{-0.11}\right)$$
*(where $c = 0.035$ for $N \le 2\times 10^6$, else $0.0044$)*.

### 5. Dead-Cluster Donor Revival
Clusters with counts $N_i < 0.1 \bar{N}$ are marked as dead. Active donor clusters are sampled with probability proportional to their current volume:
$$p(\text{donor} = d) = \frac{N_d}{\sum_{l \in \text{live}} N_l}$$
The donor's mass is halved and bifurcated using an alternating orthogonal perturbation vector $\delta \in \{-1/128, +1/128\}^D$:
$$C_{\text{child}} = C_{\text{donor}} \odot (1 + \delta), \quad C_{\text{donor}} \leftarrow C_{\text{donor}} \odot (1 - \delta)$$

---

## Empirical Benchmark Results

### Benchmark 1: Handwritten Digits ($N_{\text{train}}=1,347$, $N_{\text{test}}=450$, $K=16$, $B=256$, 12 Epochs, 5 Initializations)

| Solver | Initializer | Median Test MSE $\downarrow$ | Median Test ARI $\uparrow$ | Active Centers | Cumulative Splits |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Lloyd (Standard $k$-means)** | Random | 2.3339 | 0.663 | 16.0 / 16 | 0 |
| **Lloyd++ ($k$-means++)** | Greedy Seed | 2.2935 | 0.651 | 16.0 / 16 | 0 |
| **Cumulative Mini-Batch** | Random | 2.3538 | 0.570 | 16.0 / 16 | 0 |
| **e-means (Online Centers)** | Random | **2.2825** | **0.672** | 16.0 / 16 | 1.0 |
| **e-means (Final Mean Readout)** | Random | **2.2829** | 0.668 | 16.0 / 16 | 1.0 |

> **Key Finding**: `e-means` achieves lower held-out test MSE than both standard Lloyd and greedily seeded Lloyd++ within the exact same 12-pass training budget, while improving cluster alignment (ARI: 0.672 vs 0.663).

---

### Benchmark 2: Non-Convex Moons Manifold ($N=1,800$, $K=2$, 12 Epochs)

| Solver | Median Test MSE $\downarrow$ | Median Test ARI | Geometry Implication |
| :--- | :---: | :---: | :--- |
| **Lloyd** | 0.3997 | 0.278 | Minimizes Euclidean point-to-center distance; cuts across non-convex crescents. |
| **e-means** | **0.3996** | 0.273 | Achieves lower Euclidean distortion, but preserves Euclidean inability to separate intertwined manifolds without spectral/kernel kernels. |

> **Geometry Lesson**: Minimizing Euclidean distortion does not imply recovery of manifold geometry. The moons experiment demonstrates the distinction between metric minimization and topological cluster recovery.

---

## Directory Structure

```text
emeans-lab/
├── README.md              # Documentation, mathematical derivations, benchmarks
├── cluster.py             # Core exact-assignment e-means, Lloyd, and EMA engine
├── run.py                 # Multi-seed CLI benchmark runner with loss curves and CSV export
├── test_cluster.py        # 5 unit tests covering assignment, stats, revival, and Lloyd invariants
├── requirements.txt       # Dependencies (numpy, scikit-learn, matplotlib, threadpoolctl)
└── reference/             # Pre-computed verified benchmark artifacts
    ├── digits/
    │   ├── loss.png       # 12-epoch test MSE convergence curves
    │   └── metrics.csv    # Full trajectory data across seeds and methods
    └── moons/
        ├── loss.png       # 2-cluster manifold comparison curve
        └── metrics.csv    # Moons metrics and ARI evaluations
```

---

## Quickstart & Reproduction

### 1. Installation
```bash
cd emeans-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m unittest test_cluster.py -v
```
*Expected output: 5 passing tests (`test_assign_matches_direct_distance`, `test_counts_weight_updates_and_keep_missing_center`, `test_lloyd_training_error_never_increases`, `test_one_cluster_final_mean_includes_short_batch`, `test_revival_splits_donor`).*

### 3. Reproduce Digits Benchmark
```bash
python run.py --out results/digits
```
Outputs `results/digits/metrics.csv` and `results/digits/loss.png`.

### 4. Ablation Studies
```bash
# Ablation A: Disable dead-cluster revival
python run.py --no-revival --out results/no-revival

# Ablation B: Constant learning rate (disable cosine schedule)
python run.py --constant --out results/constant-step

# Ablation C: Batch size override (B=64 instead of formula-derived 256)
python run.py --batch 64 --out results/batch64

# Ablation D: Non-convex manifold test (Two Moons)
python run.py --data moons --out results/moons
```

---

## References & Attribution

- **Paper**: [e-means: outperforming k-means (in one-tenth the epochs)](https://arxiv.org/html/2611.09501v1) by Meta FAIR & École des Ponts (arXiv:2611.09501, 2026).
- **Official Repository**: [Meta FAIR Faiss contrib/e_means](https://github.com/facebookresearch/faiss)
- **Interactive Authors' Playground**: [e-means.github.io](https://e-means.github.io/)
- **License**: Code in this educational lab is released under the [MIT License](../LICENSE).
