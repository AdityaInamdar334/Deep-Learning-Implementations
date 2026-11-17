# Apple Silicon Local Inference Lab: MLX vs Ollama vs llama.cpp

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Harness-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-6%2F6%20Passing-brightgreen.svg)](tests/test_benchmark.py)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%28Metal%29-silver.svg?logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5%203B%20Instruct-blueviolet.svg)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Engines](https://img.shields.io/badge/Engines-MLX%20%7C%20Ollama%20%7C%20llama.cpp-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An inspectable, reproducible experimental laboratory evaluating client-observed **Time-to-First-Text (TTFT)**, end-to-end streaming latency, native Metal kernel throughput, unified memory footprints, and thermal behavior across **Apple MLX**, **Ollama**, and **llama.cpp** on Apple Silicon Macs.*

</div>

---

## Executive Summary

On Apple Silicon (M1/M2/M3/M4), local LLM deployment centers around three dominant runtime paradigms:
1. **Apple MLX (`mlx-lm`)**: Apple Silicon native framework utilizing unified memory and Metal Performance Shaders directly via Python/C++ array operations.
2. **llama.cpp (`llama-server`)**: Low-level C++ engine with hand-crafted Metal compute kernels (`ggml-metal`) and GGUF quantized representations.
3. **Ollama**: Developer-friendly daemon wrapping llama.cpp execution with managed model storage, concurrency, and local HTTP endpoints.

Comparing these engines objectively requires accounting for critical Apple Silicon architectural nuances:
- **Unified Memory Architecture**: CPU, GPU, and Neural Engine share a unified memory pool. Standard metrics like process Resident Set Size (`ps -o rss`) reflect operating system paging and mapped files rather than pure GPU buffer allocation.
- **Quantization Disparity**: MLX 4-bit (`mlx_lm.convert`) and llama.cpp `Q4_K_M` use fundamentally different quantization math (block scales vs importance-matrix $k$-quants). Performance differences reflect both runtime efficiency and format design.
- **Client-Observed TTFT**: Measures true user-facing latency to the first delivered text character across streaming HTTP sockets, separating network buffering from server compute.

```
                          ┌──────────────────────────────────────────────┐
                          │   Source Model: Qwen/Qwen2.5-3B-Instruct     │
                          │   Pinned Commit Revision                     │
                          └──────────────────────┬───────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
        ┌─────────────────────────┐                             ┌─────────────────────────┐
        │  GGUF Format (Q4_K_M)   │                             │  MLX Format (4-bit)     │
        │  bartowski/Qwen2.5-3B   │                             │  mlx_lm.convert         │
        └────────────┬────────────┘                             └────────────┬────────────┘
                     │                                                       │
        ┌────────────┴────────────┐                                          │
        ▼                         ▼                                          ▼
┌──────────────┐          ┌──────────────┐                            ┌──────────────┐
│ llama.cpp    │          │ Ollama       │                            │ Apple MLX    │
│ llama-server │          │ Modelfile    │                            │ mlx_lm.server│
│ Port 8080    │          │ Port 11434   │                            │ Port 8081    │
└───────┬──────┘          └───────┬──────┘                            └───────┬──────┘
        │                         │                                          │
        └─────────────────────────┼──────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Standard-Library Streaming Benchmark Harness                     │
│  - Client TTFT: Time to first visible streamed text character                         │
│  - Wall-Clock Streaming Latency (s)                                                    │
│  - Server Prefill Throughput (Prompt Tok/s) & Decode Speed (Tokens/s)                 │
│  - Unified Memory Tracking (Activity Monitor & powermetrics thermal extraction)        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparative Runtime Engine Matrix

| Dimension | Apple MLX (`mlx-lm`) | llama.cpp (`llama-server`) | Ollama (`ollama serve`) |
| :--- | :--- | :--- | :--- |
| **Primary Backend** | Apple MLX / Metal Shaders | `ggml-metal` C++ kernels | Forked `llama.cpp` Metal backend |
| **Target Format** | MLX 4-bit (`*.safetensors`) | GGUF `Q4_K_M` | GGUF `Q4_K_M` via Modelfile |
| **API Protocol** | OpenAI-compatible SSE (`/v1`) | OpenAI-compatible SSE (`/v1`) | Native `/api/generate` (NDJSON) |
| **Default Port** | `8081` | `8080` | `11434` |
| **Hardware Offload** | Unified memory native | `-ngl 99` (Full Metal offload)| Automatic Metal GPU offload |
| **Primary Metric** | Client TTFT & Wall Latency | Client TTFT & `llama-bench` | Server-reported prefill & decode |

---

## Directory Structure

```text
apple-silicon-inference-lab/
├── README.md              # Research framing, setup instructions, and methodology
├── RESULTS_TEMPLATE.md    # Standardized markdown report template for hardware runs
├── benchmark.py           # Standard-library multi-engine streaming benchmark runner
├── thermal_report.py      # powermetrics log parser extracting thermal & power telemetry
├── prompts/
│   └── short.txt          # Standardized 4-sentence diagnostic prompt
├── results/               # Destination for runs.csv, summary.json, and metadata.json
└── tests/
    └── test_benchmark.py  # 6 unit tests verifying request configs and NDJSON/SSE parsers
```

---

## Step-by-Step Reproduction Guide

### 1. Environment & Dependencies

```bash
cd apple-silicon-inference-lab

# 1. Ensure Apple Developer Tools are active
xcode-select --install

# 2. Set up virtual environment for MLX and tools
python3 -m venv .venv
source .venv/bin/activate
pip install -U mlx-lm 'huggingface_hub[cli]'

# 3. Build llama.cpp with native Metal support
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release --target llama-server llama-bench -j 4
```

### 2. Prepare Matching Models from Pinned Checkpoint

```bash
# Download GGUF Q4_K_M
hf download bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q4_K_M.gguf --local-dir models

# Convert original checkpoint to MLX 4-bit
mlx_lm.convert --model Qwen/Qwen2.5-3B-Instruct --mlx-path models/mlx-q4 --q-bits 4 -q

# Register GGUF into Ollama via Modelfile
printf 'FROM %s\n' "$(pwd)/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf" > Modelfile
ollama create apple-duel -f Modelfile
```

### 3. Run Unit Tests

```bash
python3 -m unittest discover -s tests -v
```
*Validates request configurations, NDJSON streaming parser, and OpenAI-compatible SSE events across 6 unit tests (passes in 0.001s).*

### 4. Execute Benchmark Runs

> [!TIP]
> **Sequential Execution Recommended**: On Macs with $\le 16\text{ GB}$ unified memory, execute engines sequentially with other servers stopped to eliminate memory pressure and scheduling jitter.

**Option A: Sequential Evaluation**
```bash
# 1. Test Ollama
python3 benchmark.py --engines ollama --trials 5 --warmups 1 --out results/ollama

# 2. Test llama.cpp (Start llama-server on port 8080 in Terminal 2)
# ./llama.cpp/build/bin/llama-server -m models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --port 8080 -ngl 99
python3 benchmark.py --engines llama --trials 5 --warmups 1 --out results/llama

# 3. Test MLX (Start mlx_lm.server on port 8081 in Terminal 2)
# mlx_lm.server --model models/mlx-q4 --port 8081
python3 benchmark.py --engines mlx --mlx-model models/mlx-q4 --trials 5 --warmups 1 --out results/mlx
```

**Option B: Concurrent Evaluation (for High-RAM 32GB+ Macs)**
```bash
python3 benchmark.py --engines ollama llama mlx --mlx-model models/mlx-q4 --trials 5 --out results/all
```

---

## Native Kernel Throughput (Isolated Hardware Baseline)

To measure raw Metal compute throughput independent of HTTP socket layers:

```bash
# llama.cpp raw Metal kernel throughput
./llama.cpp/build/bin/llama-bench \
  -m models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -ngl 99 -p 512 -n 128 -r 5 -o csv > results/llama_bench.csv

# MLX native generation benchmark
mlx_lm.benchmark --model models/mlx-q4 -p 512 -g 128 -n 5 -b 1 > results/mlx_bench.txt
```

---

## Unified Memory & Thermal Monitoring

Apple Silicon does not feature discrete GPU VRAM; memory bandwidth and capacity are shared dynamically between CPU and GPU clusters:

- **Memory Pressure**: Track `vm_stat` and Activity Monitor memory pressure rather than raw process RSS.
- **Thermal Telemetry (`powermetrics`)**:
  ```bash
  # Log thermal state during active inference pass
  sudo powermetrics --samplers thermal,smc -i 1000 -n 30 > results/thermal_raw.txt
  
  # Parse matching thermal, power, and fan speed lines
  python3 thermal_report.py results/thermal_raw.txt --output results/thermal_summary.json
  ```

---

## Attribution & License

- **Model Checkpoint**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) by Alibaba Cloud.
- **GGUF Weights**: [bartowski/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF).
- **Harness & Profiler**: Released under the [MIT License](../LICENSE).
