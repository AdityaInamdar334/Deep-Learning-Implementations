# Hourglass Transformers: A Controlled PyTorch Laboratory

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Hierarchical%20U--Net%20%7C%20Causal%20Shortening-purple.svg)]()
[![Tests](https://img.shields.io/badge/Tests-25%2F25%20Passing-brightgreen.svg)]()
[![Zero External APIs](https://img.shields.io/badge/Network-100%25%20Offline%20%26%20Hermetic-success.svg)]()
[![Article](https://img.shields.io/badge/Read-Full%20Engineering%20Tutorial-orange.svg)](ARTICLE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A reproducible, offline PyTorch laboratory exploring hierarchical token shortening, causal right-shift downsampling, recursive U-Net Transformer topologies, and sequence length extrapolation without future-token leakage.*

Based on foundational research from NAACL Findings 2022:  
**["Hierarchical Transformers Are More Efficient Language Models"](https://aclanthology.org/2022.findings-naacl.117/)** (Piotr Nawrot, Szymon Tworkowski, Adrian Łańcucki, Edoardo M. Ponti).

</div>

---

## Executive Summary

Autoregressive Transformers process sequences at uniform token resolution across all layers. For sequence length $L$, standard multi-head self-attention scales quadratically as $\mathcal{O}(L^2)$ in time and memory. While low-level representations (such as subword character tokens) require granular resolution, high-level semantic reasoning often operates on coarser conceptual spans.

The **Hourglass Transformer** introduces a hierarchical U-Net topology:
1. An initial causal Transformer block processes full-resolution tokens ($L$).
2. A **causal shortening operator** compresses the token sequence by a downsampling factor $k$, reducing sequence length to $\lceil L/k \rceil$.
3. Intermediate causal Transformer blocks compute self-attention at this coarse resolution with an effective quadratic cost reduction of $(1/k)^2$.
4. An **upsampling operator** expands representations back to length $L$ and recombines them via residual skip connections.
5. A final causal Transformer block computes next-token logits at full resolution.

### The Core Scientific Dilemma: Autoregressive Future-Token Leakage

Shortening sequence representations in autoregressive generation is deceptively dangerous. Naive downsampling (e.g., standard strided average pooling) aggregates tokens $[t, t+1, \dots, t+k-1]$ into coarse index $i$. If coarse token $i$ is upsampled and used to predict token $t$, the representation already contains information from future ground-truth tokens $t+1, \dots, t+k-1$. 

Standard lower-triangular attention masks inside the Transformer layers **do not prevent this leakage**.

To preserve mathematical causality, this laboratory implements and verifies the **causal right-shift operator**:
$$\tilde{x} = \text{ShiftRight}(x, \text{offset} = k - 1)$$
By prepending $k-1$ zero/boundary tokens before downsampling, coarse representation $i$ is guaranteed to depend exclusively on past and current inputs $\{\le t\}$, completely neutralizing future-token leakage ($\Delta_{\text{causal}} = 0.0$).

---

## Rigorous Scope: What This Lab Does, Does Not, and Does Not Claim

To maintain strict scientific and pedagogical integrity:

### What This Lab DOES
- **Implements Clean, Inspectable PyTorch Modules:** Minimal causal self-attention, causal right-shifting, average/linear shortening, repeat/linear upsampling, recursive Hourglass architectures, and standard Transformer baselines.
- **Enforces Mathematical Causality:** Empirically measures future-token leakage $\Delta_{\text{causal}}$ by perturbing future sequence positions and verifying earlier-logit invariance down to machine precision.
- **Includes an Unshifted Negative Control:** Features `hourglass_unshifted_UNSAFE` to prove that unshifted pooling leaks future ground-truth data ($\Delta_{\text{causal}} = 0.7603$).
- **Evaluates 8 Architectural Configurations:** Benchmarks vanilla 4-block Transformers against Hourglass variants ($k=2, 3, 4$), learned linear resampling, unshifted controls, skip-ablation, and factor-dropout across 3 independent seeds.
- **Tests Distribution Shift & Length Extrapolation:** Evaluates copy-region next-token prediction under familiar period ($p=4$), distribution-shifted period ($p=7$), and extended context length ($L=64$).
- **Provides 25 Automated Correctness Tests:** Enforces gradient flow, non-divisible sequence handling, causality constraints, and residual connectivity.

### What This Lab DOES NOT Do
- Does **NOT** replicate large-scale pretraining on enwik8, WikiText-103, or ImageNet-32.
- Does **NOT** implement complex local attention shifts or linear-complexity flash-attention kernels.
- Does **NOT** train multi-billion parameter foundation models or require GPU/cloud infrastructure.
- Does **NOT** rely on external API keys or closed-source LLMs.

### What This Lab Does NOT Claim
- Does **NOT** claim that an isolated $k=3$ downsampler yields an automatic $9\times$ wall-clock speedup in real-world systems. Real latency is bounded by embedding lookup, linear projections, and kernel launch overheads.
- Does **NOT** claim that learned linear downsamplers generalize out-of-distribution without regularization: learned linear resampling achieves **55.84%** on familiar period 4, but drops to **4.39%** on period 7.
- Does **NOT** claim production deployment readiness without careful kernel optimization and causal boundary handling.

---

## Architecture & Causal Mechanics

```
Full Resolution Input: x = [t_0, t_1, t_2, t_3, t_4, t_5, ...]  (Length L)
                                  │
                    ┌─────────────▼─────────────┐
                    │  Causal Transformer Block │
                    └─────────────┬─────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │ (Residual Skip Connection across Hierarchies)    │
         ▼                                                 │
┌───────────────────────────┐                              │
│ Causal Right Shift (k-1)  │ (Prepends k-1 tokens)         │
└────────────┬──────────────┘                              │
             ▼                                             │
┌───────────────────────────┐                              │
│   Downsampling / Pool     │ (Length: L/k)                │
└────────────┬──────────────┘                              │
             ▼                                             │
┌───────────────────────────┐                              │
│ Intermediate Transformers │ (Coarse Attention O(L^2/k^2))│
└────────────┬──────────────┘                              │
             ▼                                             │
┌───────────────────────────┐                              │
│   Upsampling / Resample   │ (Restores Length L)          │
└────────────┬──────────────┘                              │
             ▼                                             │
         ┌───┴─────────────────────────────────────────────┘
         ▼
      [  +  ]  Residual Recombination
         │
┌────────▼──────────────────┐
│  Causal Transformer Block │
└────────┬──────────────────┘
         ▼
Next-Token Logits: y = [p(t_1), p(t_2), ..., p(t_L)]
```

### The Causal Right-Shift Proof

Given downsampling factor $k=3$ and pooling window size $k$:
- **Unshifted Pooling (Unsafe):**
  - Coarse token $C_0 = \frac{1}{3}(x_0 + x_1 + x_2)$
  - Upsampling maps $C_0$ back to fine positions $0, 1, 2$.
  - Position $0$ now contains linear contributions from $x_1$ and $x_2$.
  - When predicting $x_1$ at step $0$, the model can read ground-truth $x_1$ directly through $C_0$. **Future leakage occurs.**
- **Shifted Pooling ($k-1 = 2$ shift, Safe):**
  - Padded sequence: $[0, 0, x_0, x_1, x_2, x_3, \dots]$
  - Coarse token $C_0 = \frac{1}{3}(0 + 0 + x_0)$
  - Coarse token $C_1 = \frac{1}{3}(x_1 + x_2 + x_3)$
  - Fine position $0$ only receives information from $C_0$ (which only knows $x_0$).
  - Fine position $1$ only receives information from $C_0$ or past tokens. **Mathematical causality is strictly preserved.**

---

## Visual Gallery

The laboratory generates five comprehensive publication-grade figures displaying empirical dynamics, causal mechanics, wall-clock savings, and out-of-distribution generalization.

### 1. Hierarchical Hourglass Topology vs. Vanilla Architecture
```
High-resolution token stream ──► Coarse semantic compression ──► Full-resolution restoration
```
![Hourglass Architecture](figures/01_hourglass_architecture.png)
*Figure 1: Full-resolution 4-block causal Transformer versus the U-shaped Hourglass Transformer. The intermediate representations are compressed by factor $k$, enabling quadratic attention savings before upsampling and residual combination.*

---

### 2. Causal Shortening Mechanics & Future Leakage
![Causal Shortening Mechanics](figures/02_causal_shortening_mechanics.png)
*Figure 2: Information flow comparison. The left panel shows how naive unshifted pooling contaminates earlier token representations with future information ($x_1, x_2$). The right panel illustrates how a right shift of $k-1$ positions preserves strict lower-triangular autoregressive causality.*

---

### 3. In-Distribution Accuracy vs. Periodic Distribution Shift
![Familiar vs Shift](figures/03_familiar_vs_shift.png)
*Figure 3: Next-token copy-region accuracy across 8 architectural variants. Learned linear resampling (`hourglass_linear_k3`) achieves state-of-the-art accuracy on familiar period-4 data (55.84%), but suffers sharp degradation under period-7 distribution shift (4.39%), exposing reliance on rigid local alignment.*

---

### 4. Training Wall-Clock Time & Efficiency
![Training Wall Clock](figures/04_training_wall_clock.png)
*Figure 4: Measured training runtime (seconds) across 95 optimization steps on CPU. Standard 4-block Transformer requires 1.20s, whereas Hourglass $k=3$ finishes in 0.78s (~35% end-to-end wall-clock speedup), demonstrating tangible compute savings even on small models.*

---

### 5. Sequence Length Extrapolation ($L=48 \to L=64$)
![Length Extrapolation](figures/05_length_extrapolation.png)
*Figure 5: Evaluation on sequences 33% longer than training context under the original repeat motif. While vanilla and average-pool models maintain modest generalization, the learned linear model sustains 54.63% accuracy, demonstrating robust longitudinal extrapolation.*

---

## Empirical Benchmark Results

Evaluated across **3 independent seeds**, 95 training steps, 48-token synthetic sequences, batch size 12, evaluating 128 validation samples per trial (recorded in `results/benchmark.json`):

| Configuration | Parameters | Causal Shift | Downsample Factor | Training Time (s) | Familiar Acc ($p=4$) | Shifted Acc ($p=7$) | Extrapolate Acc ($L=64$) | Causal Diff ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vanilla_4` | 36,320 | N/A | 1 (None) | $1.20 \pm 0.04$ | $13.29\% \pm 5.58\%$ | $8.89\% \pm 3.17\%$ | $14.28\% \pm 6.30\%$ | **0.0000** | Reference |
| `hourglass_k2` | 36,320 | Shift ($k-1$) | 2 (AvgPool) | $0.82 \pm 0.03$ | $22.11\% \pm 8.24\%$ | $10.04\% \pm 3.86\%$ | $20.92\% \pm 7.95\%$ | **0.0000** | Causal |
| `hourglass_k3` | 36,320 | Shift ($k-1$) | 3 (AvgPool) | **$0.78 \pm 0.02$** | $15.60\% \pm 5.37\%$ | $5.16\% \pm 1.63\%$ | $14.93\% \pm 5.10\%$ | **0.0000** | Causal |
| `hourglass_k4` | 36,320 | Shift ($k-1$) | 4 (AvgPool) | $0.79 \pm 0.02$ | $17.85\% \pm 7.12\%$ | $8.38\% \pm 4.29\%$ | $16.97\% \pm 6.84\%$ | **0.0000** | Causal |
| `hourglass_linear_k3` | 42,592 | Shift ($k-1$) | 3 (Learned) | $0.98 \pm 0.03$ | **$55.84\% \pm 8.92\%$** | $4.39\% \pm 1.48\%$ | **$54.63\% \pm 9.15\%$** | **0.0000** | Causal |
| `hourglass_unshifted_UNSAFE` | 36,320 | **None (0)** | 3 (AvgPool) | $0.78 \pm 0.02$ | $17.38\% \pm 6.89\%$ | $7.28\% \pm 3.10\%$ | $16.51\% \pm 6.72\%$ | **0.7603** | ⚠️ Leaks Future |
| `hourglass_no_skip` | 36,320 | Shift ($k-1$) | 3 (AvgPool) | $0.77 \pm 0.02$ | $23.99\% \pm 9.45\%$ | $4.25\% \pm 1.20\%$ | $23.12\% \pm 9.18\%$ | **0.0000** | Ablation |
| `hourglass_factor_dropout`| 36,320 | Shift ($k-1$) | Dynamic (2/3) | $0.80 \pm 0.02$ | $14.95\% \pm 4.88\%$ | $9.05\% \pm 3.94\%$ | $14.15\% \pm 4.67\%$ | **0.0000** | Causal |

### Key Findings & Insights

1. **Autoregressive Integrity:** The causal shift check modifies future tokens and measures logit variations at earlier time steps. Valid shifted configurations record $\Delta = 0.0000$, whereas `hourglass_unshifted_UNSAFE` exhibits $\Delta = 0.7603$, proving severe information leakage.
2. **Speedup vs. Parameter Count:** Hourglass models with average pooling maintain identical parameter counts ($36,320$) as the 4-block vanilla baseline while reducing training wall-clock time from $1.20$s to $0.78$s (**~35% faster**).
3. **Learned Resampling Trade-off:** The learned linear projection (`hourglass_linear_k3`) provides superior feature mapping for recurring motifs ($55.84\%$ vs. $13.29\%$), but introduces $6,272$ additional parameters and overfits to the trained period cadence ($4.39\%$ on period 7).

---

## Quickstart & Reproducibility

### 1. Installation
The lab requires only standard scientific Python packages (`torch`, `numpy`, `matplotlib`).

```bash
cd hourglass_transformer_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Correctness & Causality Tests
Run all 25 unit tests verifying shape consistency, causal masks, gradient backward passes, and failure modes:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

### 3. Interactive Causality Demonstration
Inspect real-time logit sensitivity when perturbing future inputs:

```bash
python3 -m hourglass_lab.demo
```

### 4. Reproduce the Full 24-Trial Benchmark
Execute the complete multi-seed benchmark across all 8 configurations:

```bash
python3 -m hourglass_lab.benchmark \
  --seeds 3 \
  --steps 95 \
  --batch-size 12 \
  --eval-size 128 \
  --output results/reproduced.json
```

### 5. Generate Figures
Regenerate all 5 publication-ready figures directly from the benchmark data:

```bash
python3 plot_results.py
```

---

## Repository Structure

```
hourglass_transformer_lab/
├── README.md                      # Comprehensive laboratory report & documentation
├── ARTICLE.md                     # Full tutorial article on hierarchical Transformers
├── BENCHMARK_NOTES.md             # Detailed benchmark methodology, variance & audit
├── requirements.txt               # Lightweight runtime dependencies (torch, numpy, matplotlib)
├── plot_results.py                # Standalone figure generation script (dark-mode theme)
├── figures/                       # Generated publication-quality figures (PNG)
│   ├── 01_hourglass_architecture.png
│   ├── 02_causal_shortening_mechanics.png
│   ├── 03_familiar_vs_shift.png
│   ├── 04_training_wall_clock.png
│   └── 05_length_extrapolation.png
├── hourglass_lab/                 # Core modular implementation
│   ├── __init__.py
│   ├── models.py                  # Attention, Resamplers, Hourglass & Vanilla models
│   ├── data.py                    # Synthetic periodic sequence generator & loaders
│   ├── benchmark.py               # 8-config experiment runner & causal verification
│   └── demo.py                    # Demonstration of future-leakage mechanics
├── results/                       # Canonical experimental outputs
│   └── benchmark.json             # 24-trial empirical benchmark data
└── tests/                         # Rigorous test suite
    ├── __init__.py
    └── test_hourglass.py          # 25 test cases covering causality, gradients, shapes
```

---

## Research Citations & Attribution

If you utilize this educational implementation or build upon these experiments, please cite the original foundational research:

```bibtex
@inproceedings{nawrot-etal-2022-hierarchical,
    title = "Hierarchical Transformers Are More Efficient Language Models",
    author = "Nawrot, Piotr and
      Tworkowski, Szymon and
      {\L}a{\'n}cucki, Adrian and
      Ponti, Edoardo Maria",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.117",
    doi = "10.18653/v1/2022.findings-naacl.117",
    pages = "1559--1571"
}
```

Additional reference implementations studied during development:
- **labml.ai**: [Annotated Hourglass Implementation](https://nn.labml.ai/transformers/hour_glass/index.html)
- **Google Trax**: [Official Research Hourglass Model](https://github.com/google/trax/blob/master/trax/models/research/hourglass.py)

---

## License & Copyright

- **Code:** Licensed under the [MIT License](LICENSE).
- **Research Attribution:** Original research concept, mathematical formulation, and architecture belong to Piotr Nawrot et al. and the respective authors.
- **Disclaimer:** This repository is an independent educational laboratory designed for controlled experimentation. It is not affiliated with, sponsored by, or endorsed by Google, NAACL, or labml.ai.
