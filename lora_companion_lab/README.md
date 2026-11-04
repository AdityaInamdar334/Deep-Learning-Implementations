# LoRA from Scratch: A Controlled PyTorch Laboratory

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-PEFT%20%7C%20Low--Rank%20Decomposition-purple.svg)]()
[![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen.svg)]()
[![Parity](https://img.shields.io/badge/Merge%20Parity-%3C%205.6e--6-success.svg)]()
[![Article](https://img.shields.io/badge/Read-Full%20Engineering%20Tutorial-orange.svg)](ARTICLE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A reproducible, transparent PyTorch laboratory dissecting Low-Rank Adaptation (LoRA), gradient backpropagation dynamics, asymmetric initialization, zero-latency inference weight merging, and parameter scaling on GPT-2.*

Based on foundational research from ICLR 2022:  
**["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)** (Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen).

</div>

---

## Executive Summary

When fine-tuning pretrained language models for specialized downstream tasks, the conventional paradigm updates **every parameter** in the network. For even modest models like GPT-2 (124M parameters), full fine-tuning presents steep operational bottlenecks:
1. **Optimizer Memory Bloat**: Modern optimizers (e.g., AdamW) store first and second momentum moments in 32-bit floating point for every trainable parameter, requiring **949.4 MB of RAM** just for optimizer states on a 124M model, and **11.88 GB** on a 1.5B model.
2. **Checkpoint Sprawl**: Storing independent 500 MB to 6 GB weight checkpoints for each downstream domain quickly exhausts storage and complicates versioning.
3. **Serving Latency & Fragmentation**: Serving dozens of fine-tuned variants requires swapping entire model weights into GPU VRAM.

### The Low-Rank Solution

LoRA hypothesizes that the weight updates $\Delta W$ during task adaptation have a low **intrinsic rank** $r \ll \min(d_{\text{in}}, d_{\text{out}})$. Instead of updating the full weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA keeps $W_0$ **completely frozen** and factorizes the update into two small low-rank matrices:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$, with scaling hyperparameter $\alpha$.

By choosing rank $r = 8$:
- Trainable projection parameters for GPT-2 Small drop from **124.4M to 294,912** (**99.76% parameter reduction**).
- AdamW optimizer state footprint collapses from **949.4 MB to 2.25 MB** (**422× memory reduction**).
- In production, $W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B A)$ can be computed **offline before deployment**, resulting in **zero added inference latency and zero additional serving memory**.

---

## Rigorous Scope: What This Lab Does, Does Not, and Does Not Claim

To maintain strict scientific and pedagogical integrity:

### What This Lab DOES
- **Implements Inspectable PyTorch Modules:** Implements `LoRAProjection` and `LoRAEmbedding` from first principles, supporting both standard PyTorch linear layers and GPT-2 style transposed `Conv1D` projections.
- **Audits Initialization & Gradient Dynamics:** Proves mathematically and empirically why asymmetric initialization ($B=0, A \sim \mathcal{U}$) preserves pretrained logits at step 0 while enabling non-zero gradient propagation to $B$.
- **Demonstrates Deadlock Failure Modes:** Empirically demonstrates the pathological deadlock that occurs when both matrices are initialized to zero ($\nabla_A = 0, \nabla_B = 0$).
- **Quantifies Exact Parameter & Memory Scaling:** Analyzes exact parameter reductions, parameter-efficiency ratios, and AdamW memory requirements across GPT-2 Small, Medium, Large, and XL for ranks $r \in \{1, 2, 4, 8, 16, 32, 64\}$.
- **Verifies Numerical Parity:** Formally audits the numerical parity between unmerged forward passes and merged weight matrices across various batch sizes and sequence lengths (max discrepancy $\le 5.6 \times 10^{-6}$).
- **Provides 8 Automated Unit Tests:** Validates frozen gradients, adapter rank validity, merge assertions, and transposed projection support.

### What This Lab DOES NOT Do
- Does **NOT** claim to replicate the full-scale RoBERTa, DeBERTa, or GPT-3 (175B) benchmarks from the original Hu et al. paper.
- Does **NOT** bundle proprietary or massive multi-gigabyte checkpoints; lightweight GPT-2 Small can be fetched on demand.
- Does **NOT** use quantized 4-bit/8-bit base weights (QLoRA); implementations focus strictly on standard full-precision LoRA mechanics.

### What This Lab Does NOT Claim
- Does **NOT** claim that parameter reduction implies an equal reduction in training wall-clock time. Backpropagation still requires computing activations and backward gradients through the frozen base weights to update earlier layers.
- Does **NOT** claim that a 300-step Tiny Shakespeare fine-tuning run achieves competitive literary generation; it serves purely as a proof of end-to-end task adaptation and pipeline plumbing.

---

## Mathematical Formulation & Mechanics

```
Forward Pass (Training):
Input x (B x L x d_in)
    │
    ├─────────────────────────────────────────┐
    ▼                                         ▼
[ Frozen Base W_0 ]                  [ Down-Projection A ] (r x d_in)
    │                                         │
    │                                         ▼ (B x L x r)
    │                                [ Up-Projection B ] (d_out x r)
    │                                         │
    │                                         ▼ (B x L x d_out)
    │                                [ Scale: alpha / r ]
    │                                         │
    ▼                                         ▼
  h_base                                    h_lora
    │                                         │
    └───────────────────► [ + ] ◄─────────────┘
                            │
                            ▼
                    Output h (B x L x d_out)

Inference Deployment (Zero Added Latency):
W_merged = W_0 + (alpha / r) * (B @ A)
h = x @ W_merged^T
```

### The Gradient Asymmetry Proof

Let loss be $\mathcal{L}$ and output correction be $\Delta h = \frac{\alpha}{r} x A^T B^T$.

1. **At Step $t = 0$ ($B = 0, A \sim \text{Kaiming Uniform}$):**
   - Output delta: $\Delta h = 0 \implies h = W_0 x$. Pretrained model behavior is **100% preserved**.
   - Gradient with respect to $B$:
     $$\frac{\partial \mathcal{L}}{\partial B} = \left(\frac{\partial \mathcal{L}}{\partial h}\right)^T (x A^T) \ne 0$$
     Because $x A^T$ produces non-zero activations, $B$ receives a healthy gradient and immediately begins learning.
   - Gradient with respect to $A$:
     $$\frac{\partial \mathcal{L}}{\partial A} = \left(B^T \frac{\partial \mathcal{L}}{\partial h}\right) x^T = 0$$
     Because $B = 0$, $A$ receives zero gradient at step 0. However, as soon as $B$ updates in step 1, $B \ne 0$, and $A$ receives gradient flow for all subsequent iterations.

2. **The Double-Zero Deadlock Failure Mode ($A = 0, B = 0$):**
   - If both matrices are initialized to zero:
     $$\frac{\partial \mathcal{L}}{\partial B} = \left(\frac{\partial \mathcal{L}}{\partial h}\right)^T (x \cdot 0) = 0, \quad \frac{\partial \mathcal{L}}{\partial A} = (0 \cdot \nabla_h \mathcal{L}) x^T = 0$$
   - Both gradient routes are completely blocked. The adapter is trapped in a zero-gradient saddle point and cannot learn.

---

## Visual Gallery

The laboratory generates five comprehensive publication-grade figures displaying architecture, gradient dynamics, scaling, rank recovery, and serving parity.

### 1. Dual-Branch Forward Architecture & Zero-Latency Merge
![LoRA Architecture](figures/01_lora_architecture_and_forward.png)
*Figure 1: Dual-branch training computation versus single-weight inference deployment. Pretrained base weights $W_0$ remain completely frozen while low-rank matrices $A$ and $B$ receive gradient updates. At inference, offline weight merging completely eliminates additional runtime latency.*

---

### 2. Initialization & Step-0 Gradient Dynamics
![Initialization and Gradients](figures/02_initialization_and_gradient_dynamics.png)
*Figure 2: Empirical step-0 gradient norms across layers. Adapter $B$ receives non-zero gradient immediately ($0.276$), while base weights and step-0 matrix $A$ remain at zero. Initializing both $A=0$ and $B=0$ results in a catastrophic zero-gradient deadlock.*

---

### 3. Parameter Scaling & Optimizer Memory Reduction
![Parameter and Memory Efficiency](figures/03_parameter_and_memory_efficiency.png)
*Figure 3: Trainable parameter percentages and AdamW optimizer memory footprints across GPT-2 model variants (Small, Medium, Large, XL). At rank 8, LoRA cuts optimizer memory from 949.4 MB to 2.25 MB on GPT-2 Small, and from 11.88 GB to 37.4 MB on GPT-2 XL (>99.6% reduction).*

---

### 4. Rank Capacity & Target Subspace Recovery
![Rank Convergence](figures/04_rank_capacity_convergence.png)
*Figure 4: Optimization loss trajectories when recovering a ground-truth low-rank ($r^*=4$) target subspace. Ranks $r < 4$ suffer from a capacity bottleneck and plateau at high residual error, while ranks $r \ge 4$ rapidly converge toward machine precision.*

---

### 5. Numerical Parity Audit & Serving Workflow
![Merge Parity](figures/05_zero_latency_merge_parity.png)
*Figure 5: Numerical parity verification between dual-branch unmerged forward passes and merged weights across varied batch sizes and sequence lengths (max discrepancy $\le 5.6 \times 10^{-6}$). The right panel illustrates the multi-tenant architecture sharing a single frozen base model across dozens of domain adapters.*

---

## Empirical Benchmark & Scaling Results

Evaluated across model dimensions and ranks (recorded in `results/benchmark.json`):

### 1. GPT-2 Family Adapter Parameter Breakdown

| Model | Layers | Hidden Dim | Total Base Params | LoRA Rank ($r$) | Trainable Adapter Params | Trainable % | AdamW Memory (Full) | AdamW Memory (LoRA) | Memory Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Small** | 12 | 768 | 124,439,808 | $r=2$ | 73,728 | **0.059%** | 949.4 MB | 0.56 MB | **1,687.8×** |
| **GPT-2 Small** | 12 | 768 | 124,439,808 | $r=4$ | 147,456 | **0.119%** | 949.4 MB | 1.12 MB | **843.9×** |
| **GPT-2 Small** | 12 | 768 | 124,439,808 | $r=8$ | 294,912 | **0.237%** | 949.4 MB | 2.25 MB | **422.0×** |
| **GPT-2 Small** | 12 | 768 | 124,439,808 | $r=16$ | 589,824 | **0.474%** | 949.4 MB | 4.50 MB | **211.0×** |
| **GPT-2 Medium** | 24 | 1,024 | 354,823,168 | $r=8$ | 786,432 | **0.222%** | 2.71 GB | 6.00 MB | **451.2×** |
| **GPT-2 Large** | 36 | 1,280 | 774,030,080 | $r=8$ | 1,474,560 | **0.191%** | 5.91 GB | 11.25 MB | **525.0×** |
| **GPT-2 XL** | 48 | 1,600 | 1,557,611,200 | $r=8$ | 2,457,600 | **0.158%** | 11.88 GB | 18.75 MB | **633.8×** |

### 2. Numerical Parity Across Input Dimensions ($W_{\text{unmerged}}$ vs $W_{\text{merged}}$)

| Batch Size ($B$) | Sequence Length ($L$) | Input Features | Output Features | Max Absolute Discrepancy | Mean Absolute Error | Parity Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 16 | 768 | 2,304 | $3.58 \times 10^{-6}$ | $5.12 \times 10^{-7}$ | ✅ Passed |
| 1 | 64 | 768 | 2,304 | $4.29 \times 10^{-6}$ | $5.34 \times 10^{-7}$ | ✅ Passed |
| 1 | 256 | 768 | 2,304 | $5.11 \times 10^{-6}$ | $5.41 \times 10^{-7}$ | ✅ Passed |
| 4 | 64 | 768 | 2,304 | $4.76 \times 10^{-6}$ | $5.36 \times 10^{-7}$ | ✅ Passed |
| 16 | 64 | 768 | 2,304 | $5.60 \times 10^{-6}$ | $5.38 \times 10^{-7}$ | ✅ Passed |
| 16 | 256 | 768 | 2,304 | $5.48 \times 10^{-6}$ | $5.39 \times 10^{-7}$ | ✅ Passed |

---

## Quickstart & Reproducibility

### 1. Installation
The laboratory requires standard scientific PyTorch and Hugging Face Transformers.

```bash
cd lora_companion_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Correctness & Parity Tests
Run the automated test suite verifying gradient flow, frozen weights, initialization, and weight merge equivalence:

```bash
# Unit test runner
PYTHONPATH=. python3 -m unittest discover -s tests -v

# Standalone check
python3 test_lora.py
```

### 3. Run Benchmark Suite
Run the multi-model parameter scaling and numerical parity benchmark:

```bash
python3 benchmark.py
```

### 4. Generate Publication Figures
Generate all 5 dark-mode figures directly from the benchmark data:

```bash
python3 plot_results.py
```

### 5. Fine-Tune GPT-2 on Tiny Shakespeare (Optional)
Download Tiny Shakespeare and train a rank-8 adapter on GPT-2:

```bash
# Download dataset
python3 download_data.py

# Smoke test (2 steps)
python3 train.py --steps 2 --batch 1 --block 64 --out smoke_adapter

# Full training run (300 steps)
python3 train.py --steps 300 --rank 8 --alpha 16 --out shakespeare_adapter

# Reload adapter and verify zero-shot loss reproduction
python3 train.py --load shakespeare_adapter --steps 0 --out reloaded_adapter
```

---

## Repository Structure

```
lora_companion_lab/
├── README.md                      # Comprehensive laboratory report & documentation
├── ARTICLE.md                     # Complete engineering tutorial on LoRA from scratch
├── requirements.txt               # Dependencies (torch, transformers)
├── benchmark.py                   # Scaling benchmark, gradient audit & parity tests
├── plot_results.py                # Standalone figure generation script (dark-mode theme)
├── lora.py                        # Core LoRAProjection and LoRAEmbedding implementation
├── test_lora.py                   # Minimal standalone correctness assertions
├── download_data.py               # Standard-library Tiny Shakespeare downloader
├── train.py                       # GPT-2 training, evaluation, saving, and merge audit
├── figures/                       # Generated publication-quality figures (PNG)
│   ├── 01_lora_architecture_and_forward.png
│   ├── 02_initialization_and_gradient_dynamics.png
│   ├── 03_parameter_and_memory_efficiency.png
│   ├── 04_rank_capacity_convergence.png
│   └── 05_zero_latency_merge_parity.png
├── results/                       # Experimental benchmarks & metrics
│   └── benchmark.json             # Quantified scaling data and parity audit results
└── tests/                         # Formal unit test suite
    ├── __init__.py
    └── test_lora_suite.py         # 8 test cases covering gradients, ranks, and merges
```

---

## Research Citations & Attribution

If you utilize this educational implementation or build upon these experiments, please cite the foundational research:

```bibtex
@inproceedings{hu2022lora,
    title     = {LoRA: Low-Rank Adaptation of Large Language Models},
    author    = {Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year      = {2022},
    url       = {https://arxiv.org/abs/2106.09685}
}
```

Additional reference implementations and resources:
- **Microsoft LoRA**: [Official Repository](https://github.com/microsoft/LoRA)
- **labml.ai**: [Annotated LoRA Implementation](https://nn.labml.ai/lora/index.html)
- **Hugging Face PEFT**: [Parameter-Efficient Fine-Tuning Library](https://github.com/huggingface/peft)

---

## License & Copyright

- **Code:** Licensed under the [MIT License](LICENSE).
- **Research Attribution:** Original LoRA concept, mathematical formulation, and architecture belong to Edward J. Hu et al. and Microsoft Corporation.
- **Disclaimer:** This repository is an independent educational laboratory designed for controlled experimentation. It is not affiliated with, sponsored by, or endorsed by Microsoft, OpenAI, or Hugging Face.
