# Build an Inference Server from Scratch · Episode 1: Full-Prefix Recomputation Baseline

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Work Accounting](https://img.shields.io/badge/Work%20Complexity-%F0%9D%91%82(G%20%C2%B7%20P%20%2B%20G%C2%B2%2F2)-red.svg)]()
[![Tests](https://img.shields.io/badge/Tests-4%2F4%20Passing-brightgreen.svg)](test_run.py)
[![Hardware](https://img.shields.io/badge/Hardware-CPU%20%7C%20CUDA%20%7C%20MPS-purple.svg)]()
[![Model](https://img.shields.io/badge/Model-GPT--2%20Toy%20Fixture%20(652KB)-success.svg)](fixture/)
[![Article](https://img.shields.io/badge/Tutorial-ARTICLE.md%20%26%20HTML-orange.svg)](ARTICLE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A transparent, reproducible PyTorch laboratory dissecting single-request greedy decoding, the 264,160 token-position recomputation explosion, device synchronization boundaries, and exact TTFT / TPOT latency distributions before introducing Key-Value (KV) caching.*

Based on foundational inference serving research and systems engineering:  
**["FlashAttention: Fast and Memory-Efficient Exact Attention"](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022) & **["vLLM: Efficient Memory Management for Large Language Model Serving"](https://arxiv.org/abs/2309.06180)** (Kwon et al., SOSP 2023).

</div>

---

## Executive Summary

When an autoregressive large language model generates a completion, the simplest generation loop re-evaluates the entire accumulated prefix at every single step.

Consider giving a language model a **4,096-token incident report** and requesting a **64-token summary**:
- **Step 1**: The first forward pass reads **4,096** positions to predict token 1.
- **Step 2**: The second forward pass reads **4,097** positions to predict token 2.
- $\dots$
- **Step 64**: The final forward pass reads **4,159** positions to predict token 64.

While a human observer sees a concise 64-token answer, the model has processed:
$$W = \sum_{i=0}^{G-1} (P + i) = G \cdot P + \frac{G(G-1)}{2} = 64 \times 4096 + \frac{64 \times 63}{2} = \mathbf{264,160 \text{ token positions}}$$

The opening sentence was evaluated **64 separate times**.

```
Step 1:  [================ 4096 tokens ================] -> Token 1
Step 2:  [================ 4096 tokens ================][1] -> Token 2
Step 3:  [================ 4096 tokens ================][1][2] -> Token 3
...
Step 64: [================ 4096 tokens ================][1][2]...[63] -> Token 64
```

### Why Build This Baseline First?

Before implementing Key-Value (KV) caching, continuous batching, or PagedAttention, establishing this raw uncached baseline is critical for three engineering reasons:
1. **Mathematical Ground Truth**: Once the full input sequence to every forward pass is explicit, you can predict with mathematical certainty *which exact operations* a cache will eliminate.
2. **Correctness Invariant**: A fast serving engine that produces erroneous tokens is useless. This uncached greedy loop acts as the formal correctness oracle against which future cached implementations are verified down to bitwise logit equivalence.
3. **Hardware Latency Reality**: A work reduction does not automatically map 1:1 to wall-clock latency reduction. On modern memory-bandwidth-bound and compute-bound accelerators, establishing this baseline lets you measure exactly how much theoretical work savings turn into reduced latency on your specific hardware.

---

## Rigorous Scope Disclosures

> [!IMPORTANT]
> **What this laboratory does:**
> - Implements an inspectable, deterministic greedy autoregressive generation loop in pure PyTorch (`run.py`).
> - Bundles an offline, pre-trained 2-layer GPT-2 model fixture (`fixture/model.safetensors`, 652 KB) with a custom sensor vocabulary (`"sensor reports hot"` $\to$ `"fan on"`, `"sensor reports cold"` $\to$ `"fan off"`).
> - Enforces explicit device synchronization barriers across CPU, NVIDIA CUDA, and Apple Silicon MPS to eliminate timing artifacts from asynchronous kernel launches.
> - Captures exact request-level telemetry: Time-To-First-Token (**TTFT**), Inter-Token Latency (**ITL**), Time-Per-Output-Token (**TPOT**), and nearest-rank percentiles ($p_{50}, p_{95}, p_{99}$).
> - Verifies causal prefix invariance, greedy EOS termination, and logit equivalence against cached models across 4 unit tests.
>
> **What this laboratory does NOT do:**
> - It does **not** implement KV caching in this episode (that is the explicit focus of Episode 2).
> - It does **not** implement multi-tenant continuous batching or PagedAttention (Episodes 3 & 4).
> - It does **not** claim laptop CPU timings match datacenter GPU throughput; it isolates the algorithmic work explosion.

---

## Architectural & Mathematical Foundations

### 1. The Core Generation Loop

The minimal uncached autoregressive generator operates as follows:

```python
model.eval()
sequence = input_ids  # shape: [1, prompt_length]

with torch.inference_mode():
    for _ in range(output_length):
        # Full-prefix forward pass without caching
        logits = model(input_ids=sequence, use_cache=False).logits
        
        # Greedy selection at the final sequence position
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # Concatenate and recompute everything next step
        sequence = torch.cat((sequence, token), dim=1)
        del logits

continuation = sequence[:, input_ids.shape[1]:]
```

### 2. Three Critical Implementation Choices

1. **Greedy Decoding (`argmax(dim=-1)`)**: Eliminates sampling stochasticity (temperature, top-$p$, top-$k$), allowing exact deterministic comparisons across different runtime configurations.
2. **Evaluation & Inference Mode (`model.eval()`, `torch.inference_mode()`)**: Disables stochastic layers (dropout, batch-norm updates) and bypasses autograd graph construction, eliminating memory leaks and non-deterministic overhead.
3. **Explicit `use_cache=False`**: Many Hugging Face Transformers models return dynamic Key-Value caches by default. If not explicitly disabled, the runtime incurs the allocation cost of constructing the cache while discarding it at every step.

### 3. Latency Boundaries & Metrics

```
|<-------------------------------- Total Duration -------------------------------->|
|-- Tokenize --|-- Prefill / Forward 1 --|-- Forward 2 --|-- Forward 3 --|-- Detokenize --|
               |<------- TTFT --------->|
                                         |<--- ITL 1 --->|<--- ITL 2 --->|
                                         |<----------- TPOT ------------>|
```

- **Time To First Token (TTFT)**: Starts with prompt token IDs resident on the target device; ends when the first generated token and updated sequence are synchronized on the device.
- **Inter-Token Latency (ITL)**: The incremental duration between token $i$ and token $i+1$.
- **Time Per Output Token (TPOT)**: The arithmetic mean of all inter-token gaps within a request:
  $$\text{TPOT} = \frac{1}{G - 1} \sum_{i=1}^{G-1} \text{ITL}_i$$
  *(Note: A request emitting exactly $G = 1$ token has a valid TTFT but undefined TPOT, recorded cleanly as `null` rather than misleading $0.0$).*
- **Nearest-Rank Quantiles**: Percentiles are calculated without continuous interpolation bias:
  $$p_q = \text{sorted}(X) \left[\left\lceil \frac{q}{100} \cdot N \right\rceil - 1 \right]$$

### 4. The Asynchronous Synchronization Trap

NVIDIA CUDA and Apple MPS run asynchronously with respect to the host CPU. A standard `time.perf_counter()` call around a forward pass will return immediately while the actual matrix multiplication kernel is still queued in the hardware pipeline.

To ensure laboratory-grade timing fidelity, `run.py` forces explicit hardware barriers:
```python
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
```

---

## Visual Gallery: Latency Scaling Under Full Recomputation

The lab generates high-resolution latency distributions comparing prompt prefill latency (TTFT) against generation latency (TPOT):

<div align="center">

![Reference Latency Distribution](figures/cpu_reference.png)

*Figure 1: Request-level latency percentiles ($p_{50}$ bars, $p_{95}$ dashes) across initial prompt lengths of 16, 128, and 512 tokens on CPU (single thread, 100 requests per length group, 8 output tokens).*

</div>

### Key Observations from the Empirical Sweep:
1. **TTFT Scales with Initial Prompt Length**: At 16 tokens, TTFT is $\approx 4.08\text{ ms}$; at 512 tokens, TTFT rises to $\approx 13.64\text{ ms}$ due to the compute cost of processing the longer initial context.
2. **TPOT Degrades Without Caching**: In a cached inference server, TPOT remains nearly constant regardless of prompt length because each decode step processes only 1 new token. Under full-prefix recomputation, however, **TPOT rises from 4.19 ms (16 tokens) to 14.72 ms (512 tokens)** because every single decode step is forced to re-evaluate the entire historical prompt!

---

## Quickstart & Hands-On Exercises

### 1. Environment Setup

```bash
cd inference_lab

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Exercise 1: Run Interactive Generation

Test the bundled toy GPT-2 model on the sensor control vocabulary:

```bash
# Test cold sensor logic -> expected continuation: "fan off"
python3 run.py --prompt "sensor reports cold"

# Test hot sensor logic -> expected continuation: "fan on"
python3 run.py --prompt "sensor reports hot"
```

**Console Output:**
```
Continuation:  fan off
Forward-pass input lengths: [3, 4, 5, 6]
```

### 3. Exercise 2: Trace and Account for Every Position

Run a controlled single-request trace and verify the exact work count:

```bash
python3 run.py --prompt "sensor reports cold" --lengths 3 --runs 1 --warmup 0 --new 4 --out results/trace.json
```

Open `results/trace.json` and verify the mathematical invariant:
- `input_lengths`: `[3, 4, 5, 6]`
- `positions_processed`: $3 + 4 + 5 + 6 = 18$
- `output_tokens`: $4$
- `itl_ms`: Exactly 3 inter-token latency gaps
- Identity: $\text{TTFT} + \sum \text{ITL} = \text{total\_ms}$

### 4. Exercise 3: Run the 100-Request Benchmark Sweep

Benchmark across 16, 128, and 512 token prompt lengths:

```bash
python3 run.py --lengths 16 128 512 --runs 100 --warmup 2 --new 8 --threads 1 --out results/my_cpu.json
```

Generate the latency distribution visualization:

```bash
python3 plot.py results/my_cpu.json --out figures/my_cpu.png
```

### 5. Exercise 4: Run Hardware-Accelerated Inference

On an Apple Silicon Mac:
```bash
python3 run.py --lengths 16 128 512 --device mps --dtype float32 --out results/mps_run.json
python3 plot.py results/mps_run.json --out figures/mps_run.png
```

On an NVIDIA GPU system:
```bash
python3 run.py --lengths 16 128 512 --device cuda --dtype float16 --out results/cuda_run.json
python3 plot.py results/cuda_run.json --out figures/cuda_run.png
```

### 6. Exercise 5: Run the Verification Test Suite

```bash
python3 -m unittest -v
```

**Verification Output:**
```
test_causal_prefix_invariance (test_run.BaselineChecks.test_causal_prefix_invariance) ... ok
test_eos_and_single_token_metric (test_run.BaselineChecks.test_eos_and_single_token_metric) ... ok
test_fixed_length_matches_cached_reference (test_run.BaselineChecks.test_fixed_length_matches_cached_reference) ... ok
test_nearest_rank_tail (test_run.BaselineChecks.test_nearest_rank_tail) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.224s

OK
```

---

## Repository Structure

```
inference_lab/
├── ARTICLE.md                  # Comprehensive technical deep-dive and tutorial
├── ARTICLE.html                # Offline standalone reading document with embedded figures
├── LICENSE                     # MIT Open Source License
├── README.md                   # This detailed engineering and reproduction guide
├── requirements.txt            # Minimal runtime dependencies (torch>=2.0.0, transformers, matplotlib)
├── .gitignore                  # Clean environment and model ignore rules
├── figures/                    # Publication and benchmark plots
│   └── cpu_reference.png       # Latency percentile breakdown (TTFT & TPOT)
├── fixture/                    # Bundled 2-layer GPT-2 model (652 KB)
│   ├── config.json             # Model architecture configuration
│   ├── generation_config.json  # Decoding hyperparameters
│   ├── model.safetensors       # Trained model weights
│   ├── special_tokens_map.json # Special token definitions
│   ├── tokenizer.json          # Vocabulary definition
│   └── tokenizer_config.json   # Tokenizer runtime parameters
├── results/                    # Machine-readable benchmarks and logs
│   ├── benchmark.txt           # Console run log
│   ├── cpu_reference.json      # Raw JSON telemetry from 300 benchmark requests
│   └── tests.txt               # Automated test execution log
├── plot.py                     # High-resolution percentile bar chart generator
├── run.py                      # Core generator, hardware sync, and benchmarking engine
├── test_run.py                 # 4 rigorous behavioral and invariant unit tests
└── train_fixture.py            # Reproducible synthetic training script for the model
```

---

## Roadmap: Building the Complete Inference Server

This laboratory is **Episode 1** of an 8-part series on building a production-grade inference server from first principles:

| Episode | Title & Topic | Core Challenge Solved |
| :---: | :--- | :--- |
| **01** | **Full-Prefix Recomputation Baseline** *(This Lab)* | Work accounting, sync boundaries, TTFT / TPOT profiles |
| **02** | **Static Key-Value (KV) Caching** | Eliminating repeated prompt compute, $L_{\text{cache}} = L_{\text{output}} - 1$ |
| **03** | **Dynamic Request Batching** | Left-padding masks, padded tensor collation, GPU saturation |
| **04** | **PagedAttention & Virtual Memory** | Eliminating external fragmentation, block-table lookups |
| **05** | **Continuous (Iteration-Level) Batching** | Dynamic prefill/decode scheduling, eliminating head-of-line blocking |
| **06** | **Speculative Decoding** | Draft-target verification, acceptance rates, wall-clock speedup |
| **07** | **Quantization & Low-Bit KV Tensors** | FP8/INT4 cache compression, memory bandwidth boundaries |
| **08** | **Full Serving Benchmark vs. vLLM** | Replaying request traces, throughput-latency Pareto frontiers |

---

## Academic References & Literature

1. **Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)**. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
2. **Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)**. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
3. **Gyeong-In Yu et al. (2022)**. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022. [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
4. **Yaniv Leviathan, Matan Kalman, Yossi Matias (2023)**. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
5. **Noam Shazeer (2019)**. *Fast Transformer Decoding: One Write-Head is All You Need*. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
6. **PyTorch Core Team (2024)**. *CUDA Semantics and Asynchronous Execution*. [PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html)
7. **vLLM Team (2024)**. *Serving Benchmark Metrics and Measurement Methodology*. [vLLM Docs](https://docs.vllm.ai/en/latest/cli/bench/serve/)

---

## License

This lab is released under the open-source [MIT License](LICENSE).
