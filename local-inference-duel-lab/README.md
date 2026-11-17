# Local Inference Duel: Ollama vs llama.cpp

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-4%2F4%20Passing-brightgreen.svg)](test_benchmark.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A controlled, byte-level comparative benchmarking laboratory measuring server-reported prompt prefill throughput, autoregressive token generation throughput, and client-observed end-to-end latency across Ollama and llama.cpp using the exact same local GGUF weights.*

</div>

---

## Executive Summary

Comparing local LLM inference engines is fraught with confounding variables: differing quantization implementations, divergent chat-template tokens, silent prompt caching, dynamic VRAM swapping, and unequal timing boundaries.

This laboratory provides an **unbiased, reproducible test harness** that benchmarks **Ollama** and **llama.cpp `llama-server`** on the **exact same physical GGUF file bytes** on a single machine.

```
                      ┌──────────────────────────────────────────┐
                      │    Local Model Weights (*.gguf)          │
                      │    SHA-256 Hashed for Bitwise Parity     │
                      └─────────────────────┬────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          ┌─────────────────────┐                       ┌─────────────────────┐
          │  Ollama Engine      │                       │  llama.cpp Server   │
          │  Modelfile: FROM    │                       │  llama-server -m    │
          │  Port 11434         │                       │  Port 8080          │
          └──────────┬──────────┘                       └──────────┬──────────┘
                     │                                             │
      /api/generate (raw: true)                               /completion
      temp: 0, seed: 42, ctx: 4096                            temp: 0, seed: 42, ctx: 4096
      stream: false                                           cache_prompt: false
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                     ┌───────────────────────────────────────────┐
                     │          Unified Test Telemetry           │
                     │  - Prefill Throughput (Prompt Tok/s)      │
                     │  - Decode Throughput (Generated Tok/s)    │
                     │  - Client-Observed Wall Latency (s)       │
                     │  - Peak GPU Memory Footprint              │
                     │  - Substring Smoke Verification           │
                     └───────────────────────────────────────────┘
```

### Key Measurement Principles
1. **Identical Weights**: Ollama imports the exact `.gguf` file via `Modelfile` (`FROM /path/to/model.gguf`) with zero re-quantization. Both engines point to identical disk weights verified by SHA-256.
2. **Raw Completion (No Chat Template Bias)**: Standard chat endpoints insert hidden system prompts, role headers (`<|im_start|>`, `[INST]`), and differing whitespace. This harness evaluates raw text completion (`raw: true` in Ollama, `/completion` in llama.cpp) to evaluate pure language modeling throughput on identical prompt byte streams.
3. **Hardware Telemetry Isolation**: Measures server-reported kernel throughput directly from engine response timings, while simultaneously recording client-side end-to-end socket latency.

---

## Prerequisites & Setup

- **Python**: 3.9+ (zero external pip packages required; utilizes standard library `urllib`, `json`, `hashlib`).
- **Ollama**: Installed and running locally.
- **llama.cpp**: `llama-server` compiled with hardware acceleration:
  - **macOS (Apple Silicon)**: Metal (enabled by default).
  - **Linux / Windows (NVIDIA)**: CUDA backend (`-DGGML_CUDA=ON`).
- **Target GGUF File**: Any instruction-tuned or foundation GGUF checkpoint (e.g. 1B–8B Q4_K_M).

---

## Step-by-Step Reproduction Guide

### 1. Build llama.cpp Server

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# macOS / Apple Silicon:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4 --target llama-server llama-bench

# NVIDIA CUDA:
# cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# cmake --build build --config Release -j 4 --target llama-server llama-bench
```

### 2. Import Identical GGUF into Ollama

Assume your target model is at `/absolute/path/to/model.gguf`:

```bash
# Generate Modelfile referencing the exact file
printf 'FROM /absolute/path/to/model.gguf\n' > Modelfile

# Register into Ollama without re-quantization
ollama create duel-gguf -f Modelfile

# Verify registration
ollama show duel-gguf
```

### 3. Start Both Inference Servers

**Terminal A (Ollama):**
```bash
# Limit concurrency to 1 request to prevent internal queue contention
OLLAMA_NUM_PARALLEL=1 ollama serve
```
*(If Ollama is already running as an OS background daemon, ensure `OLLAMA_NUM_PARALLEL=1` is configured).*

**Terminal B (llama.cpp):**
```bash
./build/bin/llama-server \
  -m /absolute/path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 4096 \
  -ngl 99 \
  --parallel 1
```
*(Adjust `-ngl` if unified memory or VRAM cannot offload all layers).*

### 4. Execute the Benchmark Suite

```bash
cd local-inference-duel-lab

python3 benchmark.py \
  --ollama-model duel-gguf \
  --model-file /absolute/path/to/model.gguf \
  --trials 3 \
  --max-tokens 128 \
  --seed 42
```

Outputs written to `results/`:
- `results/metadata.json`: Machine architecture, Python version, SHA-256 of GGUF file, and sampling parameters.
- `results/runs.csv`: Granular per-trial records (prefill tps, decode tps, wall latency, token counts, output strings).
- `results/summary.json`: Aggregated median prefill/decode throughput and smoke test completion counts.

---

## Experimental Controls & Telemetry Details

| Dimension | Ollama (`/api/generate`) | llama.cpp (`/completion`) | Experimental Control |
| :--- | :--- | :--- | :--- |
| **Request Mode** | `raw: true` | Direct prompt | Bypasses template formatting; passes raw bytes. |
| **Sampling** | `temperature: 0` | `temperature: 0` | Deterministic greedy decoding. |
| **RNG Seed** | `seed: 42` | `seed: 42` | Synchronized random seeds. |
| **Context Window** | `num_ctx: 4096` | `-c 4096` | Standardized KV cache buffer dimension. |
| **Max Generation** | `num_predict: 128` | `n_predict: 128` | Uniform upper bound on token generation. |
| **Prompt Caching** | Server keep-alive | `cache_prompt: false` | Cold vs warm cache behavior monitored. |

> [!WARNING]
> **Prompt Caching Asymmetry**: `llama.cpp` explicitly exposes `cache_prompt: false` on its completion endpoint. Ollama's daemon retains model state across requests via `keep_alive`. For strict cold-cache comparisons, restart both daemons between trials or vary prompt prefixes.

---

## Memory Footprint Monitoring

To measure real GPU memory utilization during active inference passes:

**NVIDIA GPUs:**
```bash
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 1 > gpu-memory.csv
```

**Apple Silicon (Unified Memory):**
Use `powermetrics` or Activity Monitor. Unified memory is shared dynamically across CPU and GPU cores; process Resident Set Size (RSS) does not reflect exclusive Metal buffer allocations. Always benchmark engines sequentially rather than concurrently to prevent unified memory pressure.

---

## Test Suite Verification

This lab includes offline unit tests with mock server payloads to verify data extraction, timing parsing, and error-handling logic:

```bash
python3 -m unittest test_benchmark.py -v
```

Tests cover:
- `test_tasks_json_schema`: Validates prompt dataset structure and required test assertions.
- `test_ollama_response_parsing`: Verifies nanosecond conversion to tokens/sec and load duration parsing.
- `test_llama_cpp_response_parsing`: Verifies timing extraction from server payload.
- `test_llama_cpp_stopped_limit`: Validates stop reason normalization when reaching maximum token budget.

---

## Attribution & License

- **Authorship**: Independent educational inference benchmarking laboratory.
- **Reference Documentation**:
  - [Ollama API Documentation](https://docs.ollama.com/api/generate)
  - [llama.cpp Server Documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- **License**: Released under the [MIT License](../LICENSE).
