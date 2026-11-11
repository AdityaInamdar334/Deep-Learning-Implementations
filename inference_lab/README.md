# Build an LLM Inference Server from Scratch · Episodes 1–5

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-11%2F11%20Passing-brightgreen.svg)](test_episodes.py)
[![Episodes](https://img.shields.io/badge/Episodes-01%20to%2005%20Complete-success.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-CPU%20%7C%20CUDA%20%7C%20MPS-purple.svg)]()
[![Model](https://img.shields.io/badge/Model-GPT--2%20Toy%20Fixture%20(652KB)-success.svg)](fixture/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An end-to-end, reproducible systems engineering laboratory implementing modern Large Language Model serving engines from first principles in pure PyTorch: from raw un-cached autoregressive recomputation to Static KV Caching, Dynamic Left-Padded Batching, vLLM PagedAttention Virtual Memory, and Orca Continuous Iteration-Level Scheduling.*

Based on foundational inference serving research and systems literature:  
**["FlashAttention: Fast and Memory-Efficient Exact Attention"](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022)  
**["vLLM: Efficient Memory Management for Large Language Model Serving"](https://arxiv.org/abs/2309.06180)** (Kwon et al., SOSP 2023)  
**["Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/conference/osdi22/presentation/yu)** (Yu et al., OSDI 2022)

</div>

---

## Visual Gallery: Architecture & Benchmark Suite

<div align="center">

![Inference Engine Benchmarks](figures/episodes_benchmark.png)

*Figure 1: Comprehensive empirical benchmarks across Episodes 1 through 5: (1) Algorithmic token-position work explosion elimination, (2) Dynamic batching throughput scaling, (3) PagedAttention 34.9x memory reduction & zero external fragmentation, and (4) Continuous vs static batching makespan reduction.*

</div>

---

## Executive Summary & Architecture Roadmap

Serving autoregressive Transformers efficiently is fundamentally different from standard deep learning inference. Because autoregressive generation produces tokens sequentially (one forward pass per token), naive implementations hit catastrophic memory and compute bottlenecks:

1. **The Recomputation Bottleneck**: Re-evaluating the historical prompt at every decode step causes an $\mathcal{O}(G \cdot P + G^2/2)$ computational explosion.
2. **The Memory Bandwidth Trap**: Single-request generation has an arithmetic intensity of $\approx 1 \text{ FLOP/byte}$, leaving high-end tensor cores starved for data.
3. **The Fragmentation Crisis**: Contiguous memory pre-allocation for unpredictable output sequences wastes 60%–80% of accelerator high-bandwidth memory (HBM).
4. **The Head-of-Line Blocking Stall**: Static request-level batching forces fast requests to wait for the slowest completion in the batch, causing severe queue stalls.

This laboratory builds the complete solution stack layer by layer:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      LLM INFERENCE ENGINE STACK                        │
├────────────────────────────────────────────────────────────────────────┤
│  Episode 05: Continuous Batching (Iteration-level Orca Scheduler)      │
│  ├── Dynamic Admission & Retirement per Step                           │
│  └── Eliminates Head-of-Line Blocking & Bubble Idling                  │
├────────────────────────────────────────────────────────────────────────┤
│  Episode 04: PagedAttention & Virtual Memory Management                │
│  ├── Block Table Mapping: Logical Tokens -> Physical Blocks            │
│  └── Zero External Fragmentation, 35x Memory Footprint Reduction       │
├────────────────────────────────────────────────────────────────────────┤
│  Episode 03: Dynamic Request Batching                                  │
│  ├── Left-Padding Collation & Causal Attention Masks                   │
│  └── Independent Per-Request Early-Stopping on EOS                     │
├────────────────────────────────────────────────────────────────────────┤
│  Episode 02: Static Key-Value (KV) Caching                             │
│  ├── Prefill [1, P] -> Decode [1, 1] Incremental Concatenation         │
│  └── Work Complexity: O(G*P + G^2/2) -> O(P + G), Constant TPOT        │
├────────────────────────────────────────────────────────────────────────┤
│  Episode 01: Full-Prefix Recomputation Baseline                        │
│  └── Formal Ground-Truth Oracle, Hardware Sync Barriers, Latency Trace │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Episode Breakdown

### Episode 01: Full-Prefix Recomputation Baseline (`run.py`)

When an autoregressive model generates $G$ tokens from a prompt of length $P$, naive generation re-submits the entire cumulative prefix at every step:
- **Step 1**: Reads $P$ tokens $\to$ predicts token 1.
- **Step 2**: Reads $P+1$ tokens $\to$ predicts token 2.
- $\dots$
- **Step $G$**: Reads $P+G-1$ tokens $\to$ predicts token $G$.

#### Mathematical Work Accounting
$$W_{\text{uncached}} = \sum_{i=0}^{G-1} (P + i) = G \cdot P + \frac{G(G-1)}{2}$$

For a prompt of $P=4,096$ tokens and $G=64$ generated tokens:
$$W = 64 \times 4096 + \frac{64 \times 63}{2} = \mathbf{264,160 \text{ token positions evaluated}}$$
The initial prompt was recomputed 64 distinct times!

#### Hardware Synchronization Barriers
Accelerators (CUDA, MPS) execute kernels asynchronously. Benchmarks without explicit hardware barriers measure the host CPU launch overhead rather than accelerator compute time. `run.py` enforces:
```python
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
```

---

### Episode 02: Static Key-Value (KV) Caching (`kv_cache.py`)

#### The Insight
In causal self-attention, the attention score between query $q_t$ at step $t$ and previous key $k_\tau$ ($\tau < t$) depends only on past activations. Past key and value vectors never change when future tokens are generated!

$$\text{Attention}(Q_t, K_{0:t}, V_{0:t}) = \text{softmax}\left(\frac{Q_t K_{0:t}^T}{\sqrt{d_k}}\right) V_{0:t}$$

#### Mechanism
1. **Prefill Phase (Step 0)**:
   - Feed the full prompt $[1, P]$.
   - Compute and store $K_{\text{past}} \in \mathbb{R}^{H \times P \times d_k}$ and $V_{\text{past}} \in \mathbb{R}^{H \times P \times d_k}$ across all layers.
   - Emit token 1.
2. **Decode Phase (Steps $1 \dots G-1$)**:
   - Feed only the single newly generated token $[1, 1]$.
   - Compute new projections: $K_{\text{new}}, V_{\text{new}} \in \mathbb{R}^{H \times 1 \times d_k}$.
   - Append to cache: $K_{0:t} = [K_{\text{past}}, K_{\text{new}}]$, $V_{0:t} = [V_{\text{past}}, V_{\text{new}}]$.
   - Perform attention across the concatenated history.

#### Complexity & Speedup
$$W_{\text{cached}} = P + (G - 1) = \mathcal{O}(P + G)$$

$$\text{Work Reduction} = 1 - \frac{P + G - 1}{G \cdot P + \frac{G(G-1)}{2}} \approx 1 - \frac{1}{G} \quad (\text{for } P \gg G)$$
For $P=512, G=32$: Work drops from **16,880 positions to 543 positions** ($\mathbf{96.8\% \text{ reduction}}$). TPOT becomes invariant to prompt length!

---

### Episode 03: Dynamic Request Batching (`batching.py`)

Single-request decoding cannot saturate modern matrix cores. Dynamic batching evaluates multiple concurrent requests simultaneously.

#### The Left-Padding Challenge
In causal decoder models, right-padding alters position IDs and invalidates causal attention masks for the active generation position. Therefore, **left-padding** is mandatory:
```
Batch Collation with Left-Padding:
Req 1 (len 3): [PAD, PAD, PAD, PAD,  T0,  T1,  T2] -> Predict Next
Req 2 (len 7): [ T0,  T1,  T2,  T3,  T4,  T5,  T6] -> Predict Next
Req 3 (len 2): [PAD, PAD, PAD, PAD, PAD,  T0,  T1] -> Predict Next
Mask:          [  0,   0,   0,   0,   1,   1,   1]
```

#### Per-Sequence Early Stopping
Different requests terminate at different times due to EOS tokens. Continuing to generate on finished sequences wastes memory bandwidth and corrupts completions. `generate_batched` maintains an active request mask:
```python
for i in range(batch_size):
    if active[i]:
        tok = next_tokens[i].item()
        completions[i].append(tok)
        if tok == eos_token_id:
            active[i] = False
if not any(active):
    break  # Terminate early when all requests finish
```

#### Empirical Throughput Scaling
| Batch Size ($B$) | Total Latency ($p_{50}$) | TTFT ($p_{50}$) | Aggregate Throughput |
| :---: | :---: | :---: | :---: |
| **1** | 0.85 ms | 0.19 ms | **5,672.7 tok/s** |
| **2** | 1.00 ms | 0.23 ms | **5,980.1 tok/s** |
| **4** | 1.04 ms | 0.27 ms | **11,481.6 tok/s** |
| **8** | 1.36 ms | 0.34 ms | **22,744.5 tok/s** |

Throughput scales $\mathbf{4.01\times}$ as batch size expands from 1 to 8 with minimal latency penalty.

---

### Episode 04: PagedAttention & Virtual Block Memory (`paged_attention.py`)

#### The Memory Crisis in LLM Serving
In traditional systems, KV caches are allocated as contiguous tensors. Because generation lengths are unknown upfront, engines pre-allocate memory for the maximum context window ($L_{\max}=1024$ or $2048$).
- **Internal Fragmentation**: Reserved slots in pre-allocated buffers that are never filled.
- **External Fragmentation**: Memory segments scattered in non-contiguous chunks, causing premature Out-Of-Memory (OOM) failures even when 40%–60% of total memory is free.

#### The PagedAttention Architecture
Inspired by operating system virtual memory paging:
- The KV cache is divided into fixed-size **Physical Blocks** (e.g. block size $B_s = 16$ tokens).
- Each sequence maintains a **Block Table** mapping logical blocks to non-contiguous physical blocks:
  $$\text{logical\_block} = \lfloor \tau / B_s \rfloor \implies \text{BlockTable}[\text{logical\_block}] = \text{phys\_block\_id}$$
- Physical blocks are allocated on demand from a centralized pool as tokens are generated.

```
Logical Sequence View:
[Token 0 ... 15] -> Logical Block 0 ───┐
[Token 16 .. 31] -> Logical Block 1 ───┼─┐
[Token 32 .. 36] -> Logical Block 2 ───┼─┼─┐
                                       │ │ │
Physical Memory Pool:                  │ │ │
Physical Block 0:  [Req 1: 0 .. 15]    │ │ │
Physical Block 1:  [Req 0: 0 .. 15] <──┘ │ │
Physical Block 2:  [Req 2: 0 .. 15]      │ │
Physical Block 3:  [Req 0: 16 .. 31]<────┘ │
Physical Block 4:  [Free Block]            │
Physical Block 5:  [Req 0: 32 .. 36]<──────┘ (Unused slots: strictly < 16)
```

#### Mathematical Equivalence & Zero Waste
PagedAttention accumulates attention scores across physical blocks using online softmax without materializing contiguous tensors:
$$A_{\text{online}} = \frac{\sum_{b} \exp(S_b - m_{\text{global}}) V_b}{\sum_b \sum_j \exp(S_{b,j} - m_{\text{global}})}$$
- **Max Absolute Divergence vs Contiguous Reference**: $\mathbf{2.38 \times 10^{-7}}$ (bitwise parity).
- **External Fragmentation**: Strictly $\mathbf{0\%}$.
- **Memory Footprint Reduction**: $\mathbf{34.9\times}$ reduction over static contiguous pre-allocation.

---

### Episode 05: Continuous (Iteration-Level) Batching (`continuous_batching.py`)

#### The Head-of-Line Blocking Problem
In static batching (Episode 3), the entire batch must run until the slowest request finishes. If Request A finishes in 2 tokens while Request B finishes in 100 tokens:
- Request A’s slot sits idle (or wastes compute on pad tokens) for 98 iterations.
- Incoming Requests C, D, and E wait in the queue even though GPU memory and compute capacity are available!

#### Iteration-Level Continuous Scheduling (Orca / vLLM)
Rather than scheduling at the request or batch boundary, continuous batching executes an iteration-level event loop:
1. **Retirement**: At step $k$, inspect all active requests. If any emitted EOS or hit its token limit, retire it immediately and free its KV cache blocks.
2. **Admission**: If the active pool has vacant slots ($< B_{\max}$) and the queue has waiting requests, immediately admit new requests.
3. **Execution**: Execute the forward pass (prefill for newly admitted requests, single-token decode for ongoing requests).
4. **Update**: Append sampled tokens and repeat.

```
Step 0: [Req 0 (prefill)][Req 1 (prefill)][Req 2 (prefill)] -> Req 1 emits EOS! Retired!
Step 1: [Req 0 (decode )][Req 3 (prefill)][Req 2 (decode )] -> Req 3 admitted into Req 1's slot!
Step 2: [Req 0 (decode )][Req 3 (decode )][Req 2 (decode )] -> Req 3 emits EOS! Retired!
Step 3: [Req 0 (decode )][Req 4 (prefill)][Req 2 (decode )] -> Req 4 admitted instantly!
```

#### Comparative Benchmark Results
On a mixed workload of 6 requests with staggered arrivals and heterogeneous lengths:

| Scheduling Strategy | Avg Queue Wait | Avg End-to-End Latency | Total Makespan | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Static Batching** | 2.00 steps | 7.50 steps | 11 steps | Baseline |
| **Continuous Batching** | **0.33 steps** | **3.00 steps** | **8 steps** | **27.3% faster makespan** |

- Queue wait time reduced by **83.5%**.
- Average request turnaround latency reduced by **60.0%**.

---

## Benchmark & Telemetry Summary

| Metric | Episode 01: Baseline | Episode 02: KV Cache | Episode 03: Batching | Episode 04: PagedAttn | Episode 05: Continuous |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Positions ($P=512, G=32$)** | 16,880 | **543** (-96.8%) | 543 | 543 | 543 |
| **TPOT vs Prompt Length** | Degrades $\mathcal{O}(P)$ | **Constant** $\mathcal{O}(1)$ | Constant $\mathcal{O}(1)$ | Constant $\mathcal{O}(1)$ | Constant $\mathcal{O}(1)$ |
| **Throughput ($B=8$)** | ~1,200 tok/s | ~5,600 tok/s | **22,744.5 tok/s** | ~22,000 tok/s | **Highest Under Jitter** |
| **Memory Allocation** | Recomputed | Contiguous | Contiguous | **Paged (34.9x saved)** | Paged / Recycled |
| **External Fragmentation**| N/A | High | High | **0.0%** | **0.0%** |
| **Head-of-Line Blocking** | N/A | N/A | Present | Present | **Eliminated** |

---

## Unit Testing & Formal Verification

The laboratory includes an exhaustive test suite verifying mathematical invariants, token-for-token logit parity, and memory safety:

```bash
cd inference_lab
python3 -m unittest test_run.py test_episodes.py -v
```

```
test_causal_prefix_invariance (test_run.BaselineChecks) ... ok
test_eos_and_single_token_metric (test_run.BaselineChecks) ... ok
test_fixed_length_matches_cached_reference (test_run.BaselineChecks) ... ok
test_nearest_rank_tail (test_run.BaselineChecks) ... ok
test_episode02_cache_inspection_and_termination (test_episodes.InferenceEpisodesTestSuite) ... ok
test_episode02_logit_parity_with_uncached_baseline (test_episodes.InferenceEpisodesTestSuite) ... ok
test_episode03_batch_isolation_parity (test_episodes.InferenceEpisodesTestSuite) ... ok
test_episode03_left_padding_collation_structure (test_episodes.InferenceEpisodesTestSuite) ... ok
test_episode04_block_allocation_and_deallocation (test_episodes.InferenceEpisodesTestSuite) ... ok
test_episode04_paged_attention_numerical_parity (test_episodes.InferenceEpisodesTestSuite) ... ok
test_episode05_early_retirement_and_dynamic_admission (test_episodes.InferenceEpisodesTestSuite) ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.543s

OK
```

---

## Quickstart & CLI Reproduction

### 1. Run Episode 01: Baseline Full Recomputation
```bash
python3 run.py --lengths 16 128 512 --new 8 --runs 10
```

### 2. Run Episode 02: Static KV Caching Sweep
```bash
python3 kv_cache.py --lengths 16 128 512 --new 8 --runs 10
```

### 3. Run Episode 03: Dynamic Batching Scaling
```bash
python3 batching.py --batch-sizes 1 2 4 8 --new 8
```

### 4. Run Episode 04: PagedAttention & Fragmentation Analysis
```bash
python3 paged_attention.py --num-blocks 64 --block-size 16 --num-sequences 6
```

### 5. Run Episode 05: Continuous Batching Simulation
```bash
python3 continuous_batching.py --batch-size 3
```

### 6. Generate Publication Benchmark Plot
```bash
python3 plot_episodes.py
```

---

## Repository Structure

```
inference_lab/
├── ARTICLE.md                  # Comprehensive tutorial and technical deep dive
├── ARTICLE.html                # Offline standalone reading document
├── LICENSE                     # MIT Open Source License
├── README.md                   # Complete architectural guide and benchmark suite
├── requirements.txt            # Minimal dependencies (torch>=2.0.0, transformers, matplotlib)
├── figures/                    # High-resolution benchmark figures
│   ├── cpu_reference.png       # Episode 1 baseline latency distributions
│   └── episodes_benchmark.png  # Episodes 1-5 comparative performance & memory panels
├── fixture/                    # Bundled 2-layer GPT-2 model fixture (652 KB)
│   ├── config.json             # Architecture specification
│   ├── model.safetensors       # Trained model weights
│   └── tokenizer.json          # Specialized sensor vocabulary
├── results/                    # Machine-readable telemetry and benchmark traces
│   ├── batching_run.json       # Episode 3 batch scaling JSON
│   ├── continuous_batching_run.json # Episode 5 scheduler metrics JSON
│   ├── cpu_reference.json      # Episode 1 baseline JSON
│   ├── kv_cache_run.json       # Episode 2 KV cache JSON
│   └── paged_attention_run.json# Episode 4 memory fragmentation JSON
├── run.py                      # Episode 1: Full-prefix recomputation baseline
├── kv_cache.py                 # Episode 2: Static KV caching generator
├── batching.py                 # Episode 3: Dynamic request batching & left-padding
├── paged_attention.py          # Episode 4: PagedAttention virtual memory manager
├── continuous_batching.py      # Episode 5: Continuous iteration-level scheduler
├── plot.py                     # Episode 1 percentile bar chart plotter
├── plot_episodes.py            # Episodes 1-5 multi-panel figure generator
├── test_run.py                 # Episode 1 baseline unit tests (4 tests)
├── test_episodes.py            # Episodes 2-5 invariant unit tests (7 tests)
└── train_fixture.py            # Reproducible training script for the model fixture
```

---

## Roadmap: The Complete Inference Engine

| Episode | Title & Topic | Status | Core Challenge Solved |
| :---: | :--- | :---: | :--- |
| **01** | **Full-Prefix Recomputation Baseline** | **Complete** | Work accounting, sync boundaries, TTFT / TPOT profiles |
| **02** | **Static Key-Value (KV) Caching** | **Complete** | Eliminating repeated prompt compute, $L_{\text{cache}} = L_{\text{output}} - 1$ |
| **03** | **Dynamic Request Batching** | **Complete** | Left-padding masks, padded tensor collation, GPU saturation |
| **04** | **PagedAttention & Virtual Memory** | **Complete** | Eliminating external fragmentation, block-table lookups |
| **05** | **Continuous (Iteration-Level) Batching** | **Complete** | Dynamic prefill/decode scheduling, eliminating head-of-line blocking |
| **06** | **Speculative Decoding** | *Roadmap* | Draft-target verification, acceptance rates, wall-clock speedup |
| **07** | **Quantization & Low-Bit KV Tensors** | *Roadmap* | FP8/INT4 cache compression, memory bandwidth boundaries |
| **08** | **Full Serving Benchmark vs. vLLM** | *Roadmap* | Replaying request traces, throughput-latency Pareto frontiers |

---

## Academic References & Literature

1. **Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)**. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
2. **Gyeong-In Yu et al. (2022)**. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022. [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
3. **Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)**. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
4. **Yaniv Leviathan, Matan Kalman, Yossi Matias (2023)**. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
5. **Noam Shazeer (2019)**. *Fast Transformer Decoding: One Write-Head is All You Need*. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

---

## License

This project is licensed under the [MIT License](LICENSE).
