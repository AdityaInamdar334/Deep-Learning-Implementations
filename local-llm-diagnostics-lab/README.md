# Local LLM Diagnostics: Why Your Local Model Is Slow

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen.svg)](tests/test_diagnose.py)
[![Engine](https://img.shields.io/badge/Engine-Ollama%20Local%20API-black.svg?logo=ollama&logoColor=white)](https://ollama.com/)
[![Diagnostics](https://img.shields.io/badge/Diagnostics-8%20Bottleneck%20Flags-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A reproducible, zero-dependency CLI profiling and bottleneck diagnosis laboratory that instruments local LLM inference engines (Ollama) to isolate model loading delays, prompt prefill bottlenecks, autoregressive decode throttling, silent CPU layer offloading, and VRAM memory pressure.*

</div>

---

## Executive Summary

When local language models run slowly, developers often struggle to identify the exact architectural bottleneck:
- Is the latency caused by **model cold-loading** from NVMe into VRAM?
- Is the model spending 90% of its time **ingesting long prompt contexts** (prefill-bound)?
- Is memory bandwidth bottlenecking **token-by-token generation** (decode-bound)?
- Did the runtime silently **offload layers to system RAM** due to VRAM pressure?

This laboratory provides an automated, standard-library diagnostic profiler (`diagnose.py`) that captures streaming tokens, parses server-reported timing breakdowns, samples device GPU telemetry via `nvidia-smi`, and assigns rule-based diagnostic indicators to pinpoint the performance bottleneck.

```
                         ┌──────────────────────────────────────────────┐
                         │      Streaming Client Request (/api/generate)│
                         └──────────────────────┬───────────────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
        ┌─────────────────────────┐                           ┌─────────────────────────┐
        │  Client Wall Telemetry  │                           │  Server Timing Objects  │
        │  - Time-to-First-Text   │                           │  - load_duration        │
        │  - Stream arrival times │                           │  - prompt_eval_duration │
        │  - Total socket elapsed │                           │  - eval_duration        │
        └────────────┬────────────┘                           └────────────┬────────────┘
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Hardware Telemetry & Placement Audit       │
                         │   - Ollama /api/ps size_vram vs total size   │
                         │   - Sampled nvidia-smi GPU compute & memory  │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Rule-Based Bottleneck Classifier                                  │
│                                                                                                │
│   [ LOAD_DOMINATED ]        load_s / wall_s > 0.40                                             │
│   [ PREFILL_DOMINATED ]     prefill_s / compute_s > 0.60 and prompt_tokens > 256               │
│   [ DECODE_DOMINATED ]      decode_s / compute_s > 0.70 and generated_tokens > 32              │
│   [ POSSIBLE_CPU_OFFLOAD ]  size_vram / size < 0.90                                            │
│   [ HIGH_VRAM_PRESSURE ]    peak_vram / total_vram > 0.90                                      │
│   [ LOW_GPU_UTILIZATION ]   mean_util < 15% and wall_s > 2.0s                                  │
│   [ NO_VISIBLE_TOKEN ]      First token never arrived (timeout / error)                        │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Diagnostic Flags & Indicator Semantics

| Diagnostic Flag | Detection Threshold | Root Cause | Engineering Remediation |
| :--- | :--- | :--- | :--- |
| `LOAD_DOMINATED` | $\frac{\text{load\_s}}{\text{wall\_s}} > 0.40$ | Model cold load from disk into memory. | Configure `keep_alive` (e.g. `keep_alive: "30m"`), warm up server before user requests. |
| `PREFILL_DOMINATED` | $\frac{\text{prefill\_s}}{\text{compute\_s}} > 0.60$ and $N_{\text{prompt}} > 256$ | Extensive prompt context processing dominates compute. | Enable prompt caching / prefix reuse, truncate unnecessary context history, use chunked prefill. |
| `DECODE_DOMINATED` | $\frac{\text{decode\_s}}{\text{compute\_s}} > 0.70$ and $N_{\text{gen}} > 32$ | Autoregressive token-by-token loop memory bandwidth bound. | Lower `max_tokens`, implement speculative decoding, quantize KV cache. |
| `POSSIBLE_CPU_OFFLOAD` | $\frac{\text{size\_vram}}{\text{size}} < 0.90$ | Insufficient VRAM; runtime offloaded model layers to system RAM. | Reduce context window (`num_ctx`), select a more aggressively quantized model (`Q4_K_M`), lower `-ngl`. |
| `HIGH_VRAM_PRESSURE` | $\frac{\text{peak\_vram}}{\text{total\_vram}} > 0.90$ | GPU memory near hardware limit ($>90\%$). | Watch out for OOM crashes during long generation or concurrent requests. |
| `LOW_GPU_UTILIZATION` | $\text{mean\_util} < 15\%$ ($t > 2\text{s}$) | Accelerator starved of compute. | Indicates CPU offload, sequential PCIe transfers, or severe CPU thread scheduling bottlenecks. |
| `NO_VISIBLE_TOKEN` | $\text{client\_ttft} = \text{None}$ | No text chunks delivered before termination. | Model hung, server crashed, or generation completed with empty string. |

---

## Directory Structure

```text
local-llm-diagnostics-lab/
├── README.md              # Research context, flag definitions, and execution guide
├── RESULTS_TEMPLATE.md    # Standardized markdown reporting template for hardware runs
├── diagnose.py            # Stream profiler, background GPU monitor, and diagnostic classifier
├── results/               # Output directory for runs.json, runs.csv, summary.json, metadata.json
└── tests/
    └── test_diagnose.py   # 8 unit tests covering streaming, TTFT, and all diagnostic flag rules
```

---

## Quickstart & Execution Guide

### 1. Requirements
- Python 3.10+ (standard library only; zero external dependencies).
- [Ollama](https://ollama.com/) running locally.
- NVIDIA GPU with `nvidia-smi` (optional: Apple Silicon and CPU-only systems record TTFT, prefill, decode, and wall latency with `--no-gpu`).

### 2. Run Offline Unit Tests
```bash
python3 -m unittest discover -s tests -v
```
*Validates that streaming response parsing, throughput calculations, and all 8 diagnostic rules pass 8/8 unit tests.*

### 3. Baseline Diagnostic Run
```bash
ollama pull qwen2.5:3b

python3 diagnose.py \
  --model qwen2.5:3b \
  --runs 3 \
  --warmup 1 \
  --out results/baseline
```
*The CLI prints per-run client TTFT, server prefill tokens/sec, decode tokens/sec, sampled peak GPU memory, and assigned diagnostic flags.*

---

## Controlled A/B Diagnostic Experiments

To validate diagnostic flags through targeted interventions:

### Experiment A: Ingest Bottleneck (Trigger `PREFILL_DOMINATED`)
```bash
# Compare a tiny prompt vs a long document prompt
python3 diagnose.py --model qwen2.5:3b --prompt "What is gradient descent?" --out results/short

# Ingest an 8K-token prompt to observe prefill saturation
python3 diagnose.py --model qwen2.5:3b --prompt-file data/long.txt --context 8192 --out results/long
```

### Experiment B: Generation Bottleneck (Trigger `DECODE_DOMINATED`)
```bash
python3 diagnose.py \
  --model qwen2.5:3b \
  --prompt "Write an exhaustive, comprehensive breakdown of transformer inference." \
  --max-tokens 512 \
  --out results/long-output
```

### Experiment C: Apple Silicon & CPU Profiling
```bash
python3 diagnose.py \
  --model qwen2.5:3b \
  --no-gpu \
  --out results/no-nvidia
```
*(On macOS, use Activity Monitor or Instruments for memory profiling; `nvidia-smi` is disabled).*

---

## Outputs Produced

Every profiling run writes machine-readable artifacts into `--out`:
- `runs.csv` & `runs.json`: Per-trial metrics (TTFT, prefill tps, decode tps, load duration, GPU samples, flags).
- `summary.json`: Statistical medians across successful runs and frequency distribution of diagnostic flags.
- `metadata.json`: Host hardware specs, OS version, Ollama build, and sampling parameters.

---

## Attribution & License

- **Authorship**: Independent educational inference profiling laboratory.
- **Official References**:
  - [Ollama API Documentation](https://docs.ollama.com/api/generate)
  - [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/)
- **License**: Released under the [MIT License](../LICENSE).
