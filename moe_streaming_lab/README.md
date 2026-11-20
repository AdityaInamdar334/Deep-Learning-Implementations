# MoE Expert Streaming: On-Demand SSD Offloading & LRU Caching

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Architecture: MoE](https://img.shields.io/badge/Architecture-Sparse%20MoE-purple.svg)](#architecture--mechanism)
[![Strategy: SSD Streaming](https://img.shields.io/badge/Strategy-On--Demand%20Disk%20Streaming-orange.svg)](#how-it-works)
[![Tests Passing](https://img.shields.io/badge/tests-3%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible, real-compute PyTorch benchmark investigating the mechanics of **on-demand SSD expert streaming and bounded LRU caching** for Mixture-of-Experts (MoE) inference.

This lab investigates the core insight behind running massive MoE models (e.g., 180B parameters) on memory-constrained hardware (e.g., a 32GB Mac or single consumer workstation): **while total model parameter footprint is immense, only a tiny active fraction (top-$k$) of experts is required for any individual token**.

---

## Architecture & Mechanism

```
                                  +----------------------------+
                                  |   Token Vector x_t in R^h  |
                                  +----------------------------+
                                                |
                                                v
                                  +----------------------------+
                                  |  Top-k Learned Router W_r  |
                                  +----------------------------+
                                                |
                                 Scores = x_t @ W_r
                                 [values, indices] = topk(Scores, k)
                                 gates = softmax(values)
                                                |
                   +----------------------------+----------------------------+
                   |                                                         |
                   v Selected Expert IDs: [idx_1, idx_2]                     v Gates: [g_1, g_2]
+---------------------------------------------------------------------------------------------------+
| BOUNDED LRU EXPERT CACHE (Capacity = C slots in RAM)                                              |
|                                                                                                   |
|   For each idx in [idx_1, idx_2]:                                                                 |
|                                                                                                   |
|   +-----------------------+              HIT               +----------------------------------+   |
|   | Is idx in LRU Cache?  | -----------------------------> | Return resident tensors (W1, W2) |   |
|   +-----------------------+                                +----------------------------------+   |
|               |                                                              ^                    |
|               | MISS                                                         |                    |
|               v                                                              |                    |
|   +--------------------------------------------------------------------+     |                    |
|   | EXPERT FILE STORE (On-Disk SSD / NVMe Storage)                     |     |                    |
|   | Read 'expert_{idx}.npz' from disk -> materialize (W1, W2)          |     |                    |
|   |                                                                    |     |                    |
|   | Cache eviction policy:                                             |     |                    |
|   | if len(cache) > C: evict Least-Recently Used expert                | ----+                    |
|   +--------------------------------------------------------------------+                          |
+---------------------------------------------------------------------------------------------------+
                                                |
                                                v
               +-----------------------------------------------------------------+
               | PARALLEL EXPERT EXECUTION (Real PyTorch Matrix Multiplications) |
               |                                                                 |
               |  y = SUM_{i=1}^k gate_i * [ ReLU(x_t @ W1_{e_i}) @ W2_{e_i} ]   |
               +-----------------------------------------------------------------+
                                                |
                                                v
                                  +----------------------------+
                                  | Output Token Projection y  |
                                  +----------------------------+
```

---

## Core Principles

### 1. The MoE Memory Bottleneck vs. Sparse Compute
In a dense transformer, every weight tensor must reside in high-speed GPU/unified memory during every token step. In a Sparse Mixture-of-Experts (SMoE) architecture:
- Total parameters scale with the number of experts $E$ (e.g., $E = 32, 64, 256$).
- **Active parameters** per token scale only with top-$k$ (typically $k=2$ or $k=4$, i.e., $3\%\text{--}12\%$ of total weights).

Instead of requiring 180GB+ of high-bandwidth memory to store all experts, the base model (attention + shared embeddings) remains resident in RAM while inactive experts sit on high-speed NVMe storage and are streamed only when summoned by the router.

### 2. Reciprocal Locality & LRU Cache Efficacy
Real-world sequences exhibit high **expert locality**:
- Mathematical queries repeatedly trigger math-specialized experts.
- Programming queries activate syntax, typing, and logic experts.
- Conversational sequences focus attention on distinct semantic subsets.

By maintaining a small in-memory LRU cache holding just $4\text{--}8$ experts, the system satisfies $>85\%\text{--}95\%$ of routing lookups from memory, slashing disk I/O traffic and accelerating decoding throughput by $4\times\text{--}8\times$.

### 3. Bitwise Parity Invariant
Caching is strictly a hardware runtime optimization. The numerical output must remain **strictly identical** regardless of whether an expert was fetched from disk or retrieved from cache:
$$y_{\text{cached}}(x) \equiv y_{\text{uncached}}(x), \quad \forall C \ge 0$$
This lab verifies bitwise output checksum consistency across all cache capacities.

---

## Experimental Workload Types

1. **`hotspot` (High Locality)**: Token vectors cluster along specific semantic directions, simulating a user engaged in a single specialized domain (e.g., coding or math).
2. **`shifting` (Phase Transition)**: Token vectors cluster around topic $A$ for the first half of the sequence, then transition to topic $B$ for the remainder, testing LRU eviction and cache adaptation.
3. **`uniform` (Adversarial Worst-Case)**: Random token representations across the full latent space with zero locality, simulating random context switching.

---

## Empirical Benchmark Results

Measured on 32 experts ($E=32$, $k=2$, hidden dimension 96, intermediate dimension 192, 60 tokens, 120 total expert accesses):

| Workload | Cache Slots ($C$) | Hit Rate | Disk Traffic (MB) | Resident RAM (KB) | Tokens / sec | Output Checksum |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`hotspot`** | **0 (Uncached)** | 0.0% | 17.75 MB | 0 KB | 1,465 tok/s | -356.75679 |
| **`hotspot`** | **2 Slots** | 83.3% | 2.96 MB | 294 KB | 5,545 tok/s | -356.75679 |
| **`hotspot`** | **8 Slots** | **95.8%** | **0.74 MB** | **737 KB** | **12,366 tok/s** | **-356.75679** |
| `shifting` | 0 (Uncached) | 0.0% | 17.75 MB | 0 KB | 1,373 tok/s | -330.35904 |
| `shifting` | 2 Slots | 75.0% | 4.44 MB | 294 KB | 5,069 tok/s | -330.35904 |
| `shifting` | 8 Slots | **90.0%** | **1.78 MB** | **1,179 KB** | **9,275 tok/s** | **-330.35904** |
| `uniform` | 0 (Uncached) | 0.0% | 17.75 MB | 0 KB | 1,322 tok/s | -0.96554 |
| `uniform` | 2 Slots | 6.7% | 16.57 MB | 294 KB | 1,385 tok/s | -0.96554 |
| `uniform` | 8 Slots | 23.3% | 13.61 MB | 1,179 KB | 1,651 tok/s | -0.96554 |

### Key Findings:
- **$8.4\times$ Speedup Under Locality**: On the `hotspot` workload, retaining just 8 of 32 experts in cache reduces disk reads from 17.75 MB to 0.74 MB (a $95.8\%$ reduction in I/O), boosting token throughput from 1,465 tok/s to 12,366 tok/s.
- **Strict Output Invariance**: The mathematical checksum is identical across all cache capacities ($C=0, 2, 8$) for each workload, proving exact numeric parity.
- **Adaptive Cache Warming**: Under the `shifting` workload, cache hit rate reaches $90\%$, demonstrating that LRU eviction gracefully handles domain shifts.

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/moe_streaming_lab

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (NumPy and PyTorch)
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The 3 automated tests verify:
- **`test_cache_matches_no_cache`**: Strict numeric output agreement between uncached ($C=0$) and cached ($C=4$) runs.
- **`test_lru_eviction`**: Correct LRU eviction ordering and hit/miss accounting.
- **`test_routing_probabilities`**: Router logit gating and partition-of-unity probability simplex ($\sum_i g_i = 1$).

---

## Running the Benchmark

Execute the full MoE streaming sweep across workloads and cache configurations:

```bash
python3 moe_lab.py \
  --rebuild \
  --experts 32 \
  --hidden 96 \
  --intermediate 192 \
  --tokens 300 \
  --top-k 2 \
  --capacities 0 2 4 8 16 32 \
  --workloads uniform hotspot shifting \
  --out results
```

### CLI Arguments

| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--root` | `Path` | `expert_store` | Directory storing `.npz` expert weight files |
| `--experts` | `int` | `32` | Total number of expert networks ($E$) |
| `--hidden` | `int` | `96` | Model hidden representation dimension ($h$) |
| `--intermediate` | `int` | `192` | Expert MLP intermediate feed-forward dimension |
| `--tokens` | `int` | `300` | Number of sequential tokens evaluated |
| `--top-k` | `int` | `2` | Active experts selected per token ($k$) |
| `--capacities` | `int...` | `0 2 4 8 16 32` | LRU cache capacities (max resident experts) |
| `--workloads` | `str...` | `uniform hotspot shifting` | Synthetic routing token workloads to evaluate |
| `--rebuild` | `flag` | `False` | Regenerate synthetic expert weights on disk |
| `--out` | `Path` | `results` | Output directory for CSV and JSON benchmark metrics |

---

## What This Demonstrates vs. Does NOT Demonstrate

### What it demonstrates:
- Real top-$k$ routing with softmax gating over on-disk expert stores.
- Accurate LRU cache hit/miss accounting, eviction behavior, and I/O byte tracking.
- The quantitative impact of sequence locality on memory footprint and inference latency.
- Strict numeric equivalence between streamed and resident expert execution.

### What it does NOT demonstrate:
- It is **not** a full language model, tokenizer, or complete attention transformer.
- It does not implement asynchronous double-buffered I/O or speculative expert prefetching.
- Files are read through standard filesystem calls; OS kernel page caching may serve disk requests from unified memory.

---

## File Structure

```
moe_streaming_lab/
├── README.md               # Architecture documentation, math formulations, and benchmark tables
├── requirements.txt        # Python dependencies (numpy, torch)
├── .gitignore              # Ignores __pycache__, bytecode, and results/
├── moe_lab.py              # Core MoE engine, router, LRU cache, and benchmark harness
├── expert_store/           # Pre-generated 32 expert weights (.npz) and router (.npy)
├── sample_results/         # Pre-evaluated empirical benchmark outputs
│   ├── benchmark.csv       # Machine-readable tabular results
│   └── benchmark.json      # Structured benchmark metrics
└── tests/
    └── test_moe_lab.py     # Automated unit test suite (3 tests)
```

---

## References

- [Hugging Face: MoE Expert Streaming Blog](https://huggingface.co/blog/moe-expert-streaming)
- [Hugging Face Transformers PR #50717: MoE Expert Offloading](https://github.com/huggingface/transformers/pull/50717)
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437)
- [Mixtral of Experts (Jiang et al., Mistral AI, 2024)](https://arxiv.org/abs/2401.04088)
- [FastMoE: A Fast Mixture-of-Expert Training System (He et al., 2021)](https://arxiv.org/abs/2103.13262)
