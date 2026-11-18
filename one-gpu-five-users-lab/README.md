# One GPU, Five Users: Local LLM Serving Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama API](https://img.shields.io/badge/Ollama-Local%20%2Fapi%2Fgenerate-orange.svg)](https://ollama.com)
[![Workload](https://img.shields.io/badge/workload-closed--loop%20streaming-emerald.svg)](https://docs.ollama.com/api/generate)
[![Concurrency](https://img.shields.io/badge/concurrency-1%20%7C%202%20%7C%204%20%7C%208%20%7C%2016-purple.svg)](#concurrency-sweep-levels)
[![Tests Passing](https://img.shields.io/badge/tests-6%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible, closed-loop concurrency benchmarking suite for local Ollama `/api/generate` servers. It sweeps 1, 2, 4, 8, and 16 concurrent simulated users submitting async streaming requests to quantify throughput saturation, TTFT latency degradation, queue cliffs, and VRAM memory pressure under different server parallelism (`OLLAMA_NUM_PARALLEL`) settings.

---

## Architecture Overview

```
                      +-------------------------------------------------------+
                      |               CLOSED-LOOP CLIENT SUITE                |
                      |  (Virtual Users submitting async streaming requests)  |
                      +-------------------------------------------------------+
                           | User 1       | User 2       | ...       | User N
                           v              v              v           v
                      +-------------------------------------------------------+
                      |         aiohttp ClientSession (Connection Pool)        |
                      +-------------------------------------------------------+
                                                 |
                               HTTP POST /api/generate (stream: true)
                                                 v
+-----------------------------------------------------------------------------------------------+
| LOCAL OLLAMA SERVER ENGINE                                                                    |
|                                                                                               |
|  [ Inbound Request Buffer ]                                                                   |
|          |                                                                                    |
|          v                                                                                    |
|  [ Request Queue (OLLAMA_MAX_QUEUE) ] --------------------------------> 503 Overloaded if full |
|          |                                                                                    |
|          v                                                                                    |
|  [ Server Workers (OLLAMA_NUM_PARALLEL = 1 | 2 | 4) ]                                         |
|          |                                                                                    |
|          +------------ KV Cache Split per Parallel Slot ------------+                         |
|          |                                                          |                         |
|          v                                                          v                         |
|  +-----------------------------------------------------------------------------------------+  |
|  | TARGET HARDWARE (Single GPU / Apple Silicon Unified Memory)                             |  |
|  | - Model Weights (Static VRAM Allocation)                                                |  |
|  | - Parallel KV Cache Slots (Context Length * Slots)                                      |  |
|  | - Compute Engine / Attention Kernels                                                    |  |
|  +-----------------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------------+
                                                 ^
                                                 | Periodic telemetry polling (0.6s)
                      +-------------------------------------------------------+
                      | TELEMETRY MONITOR (nvidia-smi + Ollama /api/ps)       |
                      | - Sampled device VRAM usage & GPU utilization         |
                      | - Runtime-reported model VRAM allocations             |
                      +-------------------------------------------------------+
```

---

## Core Mechanics & Queuing Dynamics

### 1. Closed-Loop vs. Open-Loop Workload
In a **closed-loop** workload, each virtual user awaits completion of their preceding request before dispatching the next request. This prevents unbound queue growth and infinite latency blowups, providing deterministic measurements of system capacity across concurrency levels.

### 2. Time-to-First-Token (TTFT) vs. Decode Speed
- **Time-to-First-Token (TTFT)**: Measured as the wall-clock interval from client dispatch to the first non-empty streamed text chunk:
  $$TTFT = t_{\text{first\_chunk}} - t_{\text{dispatch}}$$
  *Note*: TTFT incorporates client network transit, server queue dwell time, prompt evaluation (prefill), and initial decode token generation.
- **Server Decode Throughput**: Calculated from server-reported metrics:
  $$\text{Decode Speed}_{\text{per-user}} = \frac{\text{eval\_count}}{\text{eval\_duration (s)}}$$
- **Aggregate System Throughput**: Measured across the entire concurrency phase window:
  $$\text{Throughput}_{\text{aggregate}} = \frac{\sum_{i=1}^{M} \text{output\_tokens}_i}{\Delta t_{\text{phase}}}$$

### 3. Server Parallelism (`OLLAMA_NUM_PARALLEL`) Tradeoffs
- `OLLAMA_NUM_PARALLEL=1`: Only one request executes at any instant on the GPU. Concurrent client requests queue in FIFO order. Latency scales linearly with concurrency, but VRAM remains minimal.
- `OLLAMA_NUM_PARALLEL=2` or `4`: Ollama allocates concurrent KV cache execution slots. Multiple requests execute concurrently, boosting aggregate tokens/s up to GPU memory bandwidth limits. However, KV cache memory footprint multiplies, which may exceed VRAM on 8GB-16GB GPUs and trigger CPU offloading.

---

## Quickstart

### 1. Installation

```bash
# Clone and enter workspace
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/one-gpu-five-users-lab

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
python3 -m unittest discover -s tests -v
```

All 6 test cases verify nearest-rank percentiles, summary aggregations, diagnostic warning generation, and live async HTTP streaming handling against an in-process mock server.

---

## Running the Benchmark

### Dry-Run with Mock Server (Zero GPU Required)

To verify the test suite, CSV output pipelines, and chart rendering without a running Ollama daemon:

```bash
# Terminal 1: Start lightweight streaming mock server (runs on port 11435)
python3 mock_server.py

# Terminal 2: Run benchmark sweep against mock server
python3 load_test.py --base-url http://127.0.0.1:11435 --model mock --concurrency 1 2 --requests-per-user 2 --out results/mock

# Terminal 2: Generate SVG plots
python3 plot_results.py --csv results/mock/summary.csv --out results/mock/charts
```

---

### Real GPU Concurrency Sweep against Ollama

#### Step 1: Prepare Ollama Server
Pull the target benchmark model (e.g., `qwen2.5:3b` or `llama3.2:3b`):

```bash
ollama pull qwen2.5:3b
```

Start the Ollama daemon with isolated server parameters. On macOS, ensure the background GUI app is quit first:

```bash
# Parallelism 1 (baseline FIFO queue)
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_QUEUE=64 OLLAMA_CONTEXT_LENGTH=2048 ollama serve
```

#### Step 2: Execute Concurrency Sweep
In another shell session:

```bash
python3 load_test.py \
  --model qwen2.5:3b \
  --concurrency 1 2 4 8 16 \
  --requests-per-user 5 \
  --context 2048 \
  --max-tokens 96 \
  --out results/parallel1
```

#### Step 3: Plot Metric Charts

```bash
python3 plot_results.py --csv results/parallel1/summary.csv --out results/parallel1/charts
```

#### Step 4: Multi-Parallelism Comparison
Rerun the experiment with `OLLAMA_NUM_PARALLEL=2` and `OLLAMA_NUM_PARALLEL=4` (restarting `ollama serve` between tests) to compare capacity cliffs and saturation curves:

```bash
# Test Parallelism 2
OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_QUEUE=64 OLLAMA_CONTEXT_LENGTH=2048 ollama serve
python3 load_test.py --model qwen2.5:3b --concurrency 1 2 4 8 16 --out results/parallel2

# Test Parallelism 4
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_QUEUE=64 OLLAMA_CONTEXT_LENGTH=2048 ollama serve
python3 load_test.py --model qwen2.5:3b --concurrency 1 2 4 8 16 --out results/parallel4
```

---

## Benchmark CLI Options

| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--model` | `str` | `qwen2.5:3b` | Target Ollama model identifier |
| `--base-url` | `str` | `http://127.0.0.1:11434` | Ollama API base endpoint |
| `--concurrency` | `int...` | `1 2 4 8 16` | Concurrency levels (number of simulated active users) |
| `--requests-per-user` | `int` | `5` | Sequential requests submitted per user in each phase |
| `--prompt-words` | `int` | `140` | Length of synthetic input prompt in words |
| `--max-tokens` | `int` | `96` | Max tokens generated (`num_predict`) |
| `--context` | `int` | `2048` | Context window size (`num_ctx`) |
| `--seed` | `int` | `42` | Deterministic generation seed |
| `--keep-alive` | `str` | `20m` | Time to keep model resident in VRAM |
| `--timeout` | `float` | `180.0` | Client-side HTTP streaming request timeout (seconds) |
| `--sample-interval` | `float` | `0.6` | Telemetry sampling interval for `nvidia-smi` / `/api/ps` (seconds) |
| `--gpu-index` | `int` | `0` | GPU device index to query via `nvidia-smi` |
| `--cooldown` | `float` | `2.0` | Pause duration between concurrency phases (seconds) |
| `--think-time` | `float` | `0.0` | Inter-request delay for virtual users (seconds) |
| `--out` | `str` | `results/sweep` | Output directory path |

---

## Output Artifacts

Running a benchmark generates structured machine-readable logs and summary tables:

```
results/parallel1/
├── summary.csv          # Aggregate metrics per concurrency level (throughput, p50/p95 TTFT, VRAM)
├── summary.json         # Summary JSON with automated saturation diagnostics
├── runs.csv             # Granular request-by-request metrics (durations, tokens, HTTP status)
├── runs.json            # Full raw per-request records
├── gpu_samples.json     # Telemetry time-series from nvidia-smi & Ollama /api/ps
├── events.json          # Precise dispatch and completion timestamps
├── metadata.json        # Hardware, server version, and benchmark parameters
└── charts/              # Generated high-resolution SVG metric plots
    ├── throughput_requests.svg
    ├── throughput_tokens.svg
    ├── latency_p50.svg
    ├── latency_p95.svg
    ├── ttft_p95.svg
    ├── failures.svg
    └── gpu_peak.svg
```

---

## Saturation Diagnostics

The benchmark automatically scans results for common local serving bottlenecks:

1. **p95 Latency Cliff**: Flags when $p95 \text{ latency}$ grows by more than $2.5\times$ between adjacent concurrency levels.
2. **Throughput Saturation**: Alerts when aggregate token output flattens ($< 8\%$ gain) despite increasing concurrency.
3. **Queue Rejections (503 / Timeouts)**: Identifies server overload where requests exceed `OLLAMA_MAX_QUEUE`.
4. **VRAM Spillover**: Identifies memory pressure when `OLLAMA_NUM_PARALLEL` forces KV cache spilling to system RAM.

---

## File Structure

```
one-gpu-five-users-lab/
├── README.md               # Architecture documentation, formulas, and CLI guide
├── RESULTS_TEMPLATE.md     # Markdown reporting template for empirical findings
├── requirements.txt        # Python dependencies (aiohttp, matplotlib, pytest)
├── load_test.py            # Async closed-loop concurrency benchmark engine
├── mock_server.py          # Standalone mock streaming HTTP server for offline testing
├── plot_results.py         # SVG chart generator for benchmark metrics
├── results/                # Output directory for benchmarks and plots
│   └── .gitkeep
└── tests/
    └── test_load.py        # Unit test suite (6 tests covering metrics & streaming)
```

---

## References

- [Ollama FAQ & Concurrency Configuration](https://docs.ollama.com/faq)
- [Ollama /api/generate API Reference](https://docs.ollama.com/api/generate)
- [Ollama /api/ps API Reference](https://docs.ollama.com/api/ps)
- [vLLM: Efficient Memory Management with PagedAttention (Kwon et al., SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., 2024)](https://arxiv.org/abs/2403.02310)
