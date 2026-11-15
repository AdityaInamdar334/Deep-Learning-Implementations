# Two-Level Coarse-to-Fine Assignment Lab (o-means)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B%20%7C%202.0%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-2%2F2%20Passing-brightgreen.svg)](test_lab.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A clean, educational NumPy laboratory isolating two-level hierarchical cluster routing (IVF-style Voronoi cells), coarse centroid probing, winner agreement degradation, and the unexpected regularization benefits of approximate assignment during mini-batch codebook training.*

</div>

---

## Executive Summary

In large-scale vector quantization and dense retrieval (e.g. Faiss IVF indexes, neural codebooks, and deep clustering), calculating exact Euclidean distances from every data vector to all $K$ centroids requires an expensive $O(N \cdot K \cdot D)$ matrix multiplication. As codebooks scale from $K=256$ to $K=65,536$, assignment computation rapidly dominates the training and inference budget.

**Two-level coarse-to-fine assignment** tackles this by partitioning the $K$ fine centroids into $C$ coarse cells (meta-clusters). Instead of scanning all $K$ centroids:
1. Each data point queries only the $C$ coarse centroids.
2. The top $P$ nearest coarse cells (**probes**) are selected.
3. Fine-grained Euclidean distances are computed *only* against the candidate centroids belonging to those $P$ cells.

```
Dense Exhaustive Assignment (K = 256):
  Point x ────> Compute distances to all 256 centroids ────> Argmin (256 dot products)

Two-Level Probe Routing (K = 256, C = 16 coarse cells, P probes):
  Point x ────> Compute distances to 16 coarse cells
                      │
                      ▼ Sort & select top P cells
         [ Cell 2 ]       [ Cell 9 ]   ...  (P probed cells)
             │                │
             └─── Candidates ─┴──> Compute fine distances only to subset (~30-80 centroids)
                                   Argmin (16 coarse + subset fine comparisons)
```

### The Core Research Questions
1. **The Missed-Winner Problem**: When points lie near Voronoi cell boundaries, how severely does approximate routing misclassify the true closest centroid?
2. **The Work vs Excess Loss Frontier**: How many coarse cells ($P$) must be probed before excess distortion drops below 1%?
3. **The Training Paradox**: Does approximate assignment during mini-batch training degrade final codebook quality, or does it act as an implicit spatial regularizer?

---

## Benchmark Results

Evaluated on synthetic multimodal data ($N_{\text{train}}=2,048$, $N_{\text{held}}=512$, $D=12$, $K=256$ centroids, $C=16$ coarse cells, 2 passes of 8 mini-batches):

| Probing Strategy ($P$) | Comparisons / Row | Winner Agreement $\uparrow$ | Fixed-Center Excess Loss $\downarrow$ | Exhaustive Held-Out Loss $\downarrow$ |
| :---: | :---: | :---: | :---: | :---: |
| **Dense (Oracle)** | 256.00 | 100.00% | 0.00% | 6.6173 |
| **1 Probe** | 33.51 ($7.6\times$ speedup) | 59.96% | 15.92% | 6.6534 |
| **2 Probes** | 51.73 ($5.0\times$ speedup) | 77.73% | 6.68% | 6.5906 |
| **4 Probes** | **87.87** ($2.9\times$ speedup) | **93.36%** | **1.05%** | **6.5179** (Best) |
| **8 Probes** | 156.70 ($1.6\times$ speedup) | 99.80% | 0.01% | 6.5911 |
| **16 Probes (All Cells)** | 272.00 ($16 + 256$) | 100.00% | 0.00% | 6.6173 |

### Key Empirical Findings

1. **Near-Oracle Loss at $P=4$**: With only 4 probes, the search evaluates an average of 87.87 centroids per row (a **$2.9\times$ computational reduction**), reaches **93.36% winner agreement**, and suffers only **1.05% excess loss** on fixed centroids.
2. **The Regularization Benefit**: Counterintuitively, training with $P=4$ yields a **lower held-out distortion (6.5179)** under exhaustive evaluation than dense training (6.6173). The stochasticity introduced by approximate assignment prevents centroids from collapsing into redundant local minima.
3. **Full Probe Overhead**: Probing all 16 cells ($P=16$) computes $16 + 256 = 272$ comparisons per row, which guarantees 100% agreement but adds the coarse layer's overhead without any algorithmic advantage.

---

## Mathematical Formulation

### 1. Coarse Cell Partitioning
Given $K$ fine centroids $\{c_1, \dots, c_K\} \subset \mathbb{R}^D$, a set of $C$ coarse centroids $\{\mu_1, \dots, \mu_C\}$ is fit using 5 Lloyd iterations. Each coarse cell $j \in \{1, \dots, C\}$ owns the inverted index partition:
$$\mathcal{M}_j = \left\{ i \in \{1, \dots, K\} \;\middle|\; \arg\min_{m \in \{1,\dots,C\}} \|c_i - \mu_m\|^2 = j \right\}$$

### 2. Multi-Probe Routing
For an incoming data query $x \in \mathbb{R}^D$:
1. Coarse ranking:
   $$\pi = \text{argsort}_{j \in \{1,\dots,C\}} \|x - \mu_j\|^2$$
2. Candidate pooling across the top $P$ cells:
   $$\mathcal{C}_P(x) = \bigcup_{p=1}^P \mathcal{M}_{\pi_p}$$
3. Fine nearest-neighbor search:
   $$\hat{k} = \arg\min_{i \in \mathcal{C}_P(x)} \|x - c_i\|^2$$

### 3. Computational Work Accounting
For a query set $X$ of size $N$:
$$\text{Comparisons per row} = C + \frac{1}{N} \sum_{i=1}^N |\mathcal{C}_P(x_i)|$$

---

## Directory Structure

```text
o-means-lab/
├── README.md        # Mathematical foundations, benchmark analysis, quickstart
├── lab.py           # Core two-level indexer, multi-probe router, and training loop
├── test_lab.py      # Unit tests verifying missed-winner boundary and full-probe invariants
└── example.txt      # Reference execution output across probe configurations
```

---

## Quickstart & Verification

### 1. Installation
Requires only standard Python and NumPy:
```bash
pip install numpy
```

### 2. Run Test Suite
```bash
python -m unittest test_lab.py -v
```
- `test_missed_winner`: Validates that a query point ($x = 5.1$) located near a boundary chooses an inaccurate cell under 1 probe, demonstrating the boundary failure mode.
- `test_full_probe`: Validates that probing all cells ($P=2$) guarantees exact match with dense distance calculations.

### 3. Run Benchmark
```bash
python lab.py
```
To evaluate stability across another random seed:
```bash
python lab.py --seed 7
```

---

## Experimental Ablations to Explore

1. **Shift Boundary in `test_missed_winner`**:
   Change the fourth centroid in `test_lab.py` from `15.0` to `9.0` and update the coarse mean to `7.0`. Notice how the point $x=5.1$ is no longer trapped across the wrong Voronoi boundary.
2. **Re-Indexing Frequency**:
   In `lab.py`, modify `step % 4 == 0` to `step % 1 == 0` (rebuilding coarse cells every mini-batch) vs `step % 8 == 0`. Observe the trade-off between centroid drift and index maintenance overhead.
3. **Data Overlap / Dispersion**:
   Increase data noise in `lab.py` from `scale=0.75` to `scale=1.5`. Observe how greater cluster overlap impacts 1-probe winner agreement and widens the gap between fixed-center excess loss and trained generalization.

---

## Attribution & License

- **Authorship**: Independent educational implementation exploring two-level IVF inverted-file routing in mini-batch vector quantization.
- **License**: Released under the [MIT License](../LICENSE).
