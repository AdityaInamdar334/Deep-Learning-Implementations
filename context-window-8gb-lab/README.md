# The 8GB Context Window Lab: Ollama VRAM Scaling & KV Cache Allocation

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-9%2F9%20Passing-brightgreen.svg)](tests/test_lab.py)
[![Model](https://img.shields.io/badge/Model-Qwen2.5%207B%20%7C%203B-blueviolet.svg)](https://ollama.com/library/qwen2.5)
[![Engine](https://img.shields.io/badge/Engine-Ollama%20Local%20API-black.svg?logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A reproducible, controlled experimental laboratory investigating how **configured context capacity** differs from **actual prompt length**, KV cache preallocation mechanics, prefill and decode throughput degradation, and deterministic needle retrieval on consumer 8GB GPUs.*

</div>

---

## Executive Summary

Modern local models (such as Qwen 2.5 7B) advertise context windows of up to 32K or 128K tokens. However, running these models on consumer hardware (e.g. 8GB VRAM NVIDIA RTX 3070/4060 or Apple Silicon unified memory) introduces a critical physical bottleneck:

1. **Model Weight Floor**: A 7-billion parameter model quantized to 4-bit (`Q4_K_M`) occupies ~4.7 GB of pure model weights before any context is allocated.
2. **KV Cache Expansion**: The Key-Value (KV) cache grows linearly with context length and head configuration. For Qwen2.5 7B ($L=28, H_{\text{KV}}=4, D=128$), an FP16 cache consumes:
   - **4,096 tokens**: $\approx 224 \text{ MiB}$
   - **16,384 tokens**: $\approx 896 \text{ MiB}$
   - **32,768 tokens**: $\approx 1.75 \text{ GiB}$
3. **The Preallocation Cliff**: Depending on engine configuration, the inference runtime may preallocate the entire KV cache buffer for the *configured* context window (`num_ctx`). On an 8GB GPU, configuring `num_ctx: 32768` pushes total memory ($4.7\text{ GB weights} + 1.75\text{ GB KV cache} + \text{CUDA context \& compute buffers}$) over the 8GB ceiling, triggering **silent CPU offloading**, extreme throughput collapse, or out-of-memory (OOM) aborts.

```
8GB Total VRAM Budget
┌───────────────────────────────────────────────┬──────────────────────┬─────────────┐
│  Model Weights (Q4_K_M)                       │  KV Cache (32K F16)  │ Overhead    │
│  ~4.7 GB                                      │  ~1.75 GB            │ ~1.2 GB     │
└───────────────────────────────────────────────┴──────────────────────┴─────────────┘
 ◄───────────────────────── 7.65 GB (Near VRAM Ceiling) ───────────────────────────►
                     ▲
                     │ Any additional buffer allocation triggers
                     │ silent partial layer offloading to system RAM!
```

---

## The Dual-Sweep Experimental Methodology

To separate static runtime buffer preallocation from dynamic sequence processing costs, this lab executes two complementary sweeps across context tiers (2K, 4K, 8K, 16K, 32K):

```
                        ┌────────────────────────────────────────┐
                        │      Target Model: Qwen 2.5 7B/3B      │
                        └───────────────────┬────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
┌────────────────────────────────────────┐            ┌────────────────────────────────────────┐
│  Mode 1: Fixed-Prompt Sweep            │            │  Mode 2: Scaled-Prompt Sweep           │
│  - Byte-identical 120-token prompt     │            │  - Synthetic log stream scaling to 90% │
│  - Sweep num_ctx: 2K ──> 32K           │            │    of configured context size          │
│  - Isolates static KV preallocation    │            │  - Embeds deterministic "needle" at    │
│    and idle runtime memory floor       │            │    1/3 depth for retrieval verification│
└──────────────────┬─────────────────────┘            └──────────────────┬─────────────────────┘
                   │                                                     │
                   └────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│  Granular Hardware & Inference Telemetry                                                     │
│  - GPU Baseline & Peak VRAM (sampled via nvidia-smi every 250ms)                             │
│  - Engine-Reported Allocation (/api/ps size_vram)                                            │
│  - Uncached Prefill Throughput (Prompt Tok/s) & Autoregressive Decode Speed (Tok/s)          │
│  - Exact-Match Needle Retrieval Success vs Prompt Truncation                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical KV-Cache Formulation

For a Transformer with Grouped-Query Attention (GQA):
$$\text{Bytes}_{\text{KV}} = 2 \times N_{\text{layers}} \times N_{\text{KV\_heads}} \times D_{\text{head}} \times N_{\text{tokens}} \times \text{BytesPerElement}$$

For **Qwen2.5 7B** ($N_{\text{layers}}=28, N_{\text{KV\_heads}}=4, D_{\text{head}}=128$, FP16 precision $\text{BytesPerElement}=2.0$):
$$\text{BytesPerToken} = 2 \times 28 \times 4 \times 128 \times 2.0 = 57,344 \text{ bytes/token} \approx 56.0 \text{ KiB/token}$$

```bash
# Calculate theoretical dense KV cache across context targets:
python3 kv_math.py --layers 28 --kv-heads 4 --head-dim 128 --contexts 2048 4096 8192 16384 32768
```
- **2,048 tokens**: $112.0 \text{ MiB}$
- **4,096 tokens**: $224.0 \text{ MiB}$
- **8,192 tokens**: $448.0 \text{ MiB}$
- **16,384 tokens**: $896.0 \text{ MiB}$
- **32,768 tokens**: $1,792.0 \text{ MiB}$ ($1.75 \text{ GiB}$)

---

## Directory Structure

```text
context-window-8gb-lab/
├── README.md              # Methodology, KV cache mathematics, and execution guide
├── RESULTS_TEMPLATE.md    # Standardized markdown report template for hardware runs
├── benchmark.py           # Core benchmark runner (fixed/scaled sweeps, telemetry sampler)
├── kv_math.py             # Pure standard library analytical KV cache capacity calculator
├── plot_results.py        # Zero-dependency SVG chart generator for CSV telemetry data
├── results/               # Destination directory for metadata.json, runs.csv, summary.json
└── tests/
    └── test_lab.py        # 9 unit tests verifying prompt generation, telemetry, and math
```

---

## Quickstart & Reproduction Guide

### 1. Requirements
- Python 3.10+ (standard library only; no pip dependencies).
- [Ollama](https://ollama.com/) running locally.
- NVIDIA GPU with working `nvidia-smi` (optional: CPU execution and non-NVIDIA GPUs capture throughput and latency without VRAM metrics).

### 2. Run Test Suite
```bash
python3 -m unittest discover -s tests -v
```
*Validates prompt generation scaling, rate extraction, needle placement, summary aggregations, SVG creation, and Qwen 7B KV cache math (9/9 pass).*

### 3. Baseline Context Sweep (2K – 8K)
```bash
ollama pull qwen2.5:7b

python3 benchmark.py \
  --model qwen2.5:7b \
  --contexts 2048 4096 8192 \
  --mode both \
  --trials 2 \
  --out results/context-sweep
```

### 4. Extended Boundary Sweep (16K – 32K)
> [!WARNING]
> Run extended context sweeps only after verifying baseline 8K stability. If your GPU has only 8GB VRAM, monitor for system thrashing or layer offloading.

```bash
python3 benchmark.py \
  --model qwen2.5:7b \
  --contexts 2048 4096 8192 16384 32768 \
  --mode both \
  --trials 3 \
  --out results/extended
```

### 5. Generate Zero-Dependency SVG Visualizations
```bash
python3 plot_results.py \
  --csv results/context-sweep/runs.csv \
  --out results/context-sweep/plots
```
Generates clear SVG charts for decode tokens/sec, prefill tokens/sec, peak VRAM, and end-to-end wall latency without external graphics libraries.

---

## Telemetry Metrics Extracted

| Metric | Source | Significance |
| :--- | :--- | :--- |
| `gpu_baseline_mib` | `nvidia-smi` | Total device memory before request initialization. |
| `gpu_peak_mib` | `nvidia-smi` | Maximum sampled device memory during active generation. |
| `gpu_delta_peak_mib` | Calculated | `peak - baseline` memory consumption. |
| `ollama_size_vram_bytes` | Ollama `/api/ps` | Engine's explicit model VRAM footprint. |
| `prefill_tps_uncached` | Ollama timings | Evaluates true prompt ingestion speed excluding cached prefix hits. |
| `decode_tps` | Ollama timings | Autoregressive generation speed (tokens/second). |
| `needle_exact_match` | Evaluator | Binary flag: Did model output retrieve the target UUID? |

---

## Attribution & License

- **Authorship**: Independent educational GPU memory and context scaling laboratory.
- **Official References**:
  - [Ollama API Documentation](https://docs.ollama.com/api/generate)
  - [Qwen2.5 Architecture Configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- **License**: Released under the [MIT License](../LICENSE).
