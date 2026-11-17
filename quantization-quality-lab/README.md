# Quantization Quality Lab: Q4_K_M vs Q8_0 vs F16

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-9%2F9%20Passing-brightgreen.svg)](tests/test_lab.py)
[![Model](https://img.shields.io/badge/Model-Qwen2.5%203B%20Instruct-blueviolet.svg)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Formats](https://img.shields.io/badge/Formats-Q4__K__M%20%7C%20Q8__0%20%7C%20F16-orange.svg)]()
[![Engine](https://img.shields.io/badge/Engine-Ollama%20Local%20API-black.svg?logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A controlled, offline-after-download comparative laboratory evaluating the impact of **4-bit (Q4_K_M)**, **8-bit (Q8_0)**, and **16-bit unquantized (F16)** GGUF representations of **Qwen2.5-3B-Instruct** across deterministic reasoning accuracy, strict JSON formatting, prefill/decode throughput, and device-wide GPU VRAM footprint.*

</div>

---

## Executive Summary

When deploying local language models, practitioners universally face the trade-off between **model footprint** and **inference fidelity**:
- Can a 4-bit quantized model (`Q4_K_M`) reliably produce valid structured JSON?
- Does 8-bit quantization (`Q8_0`) recover 99% of unquantized (`F16`) accuracy while halving the required VRAM?
- How much does memory bandwidth compression accelerate prefill and decode throughput on consumer GPUs?

This laboratory isolates these questions by executing a deterministic, multi-trial benchmark across **three precision formats of the exact same model checkpoint (`Qwen2.5-3B-Instruct`)** using Ollama's local raw completion interface and Python standard library tooling.

```
                         ┌─────────────────────────────────────────────────┐
                         │   Source Checkpoint: Qwen2.5-3B-Instruct        │
                         └────────────────────────┬────────────────────────┘
                                                  │
            ┌─────────────────────────────────────┼─────────────────────────────────────┐
            ▼                                     ▼                                     ▼
 ┌──────────────────────┐              ┌──────────────────────┐              ┌──────────────────────┐
 │    Q4_K_M (4-bit)    │              │     Q8_0 (8-bit)     │              │   F16 (16-bit Full)  │
 │ ~1.93 GB File Size   │              │ ~3.29 GB File Size   │              │ ~6.18 GB File Size   │
 │ k-quant imatrix mixed│              │ 8-bit uniform blocks │              │ Half-precision float │
 └──────────┬───────────┘              └──────────┬───────────┘              └──────────┬───────────┘
            │                                     │                                     │
            └─────────────────────────────────────┼─────────────────────────────────────┘
                                                  │
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │      Ollama Local Engine (/api/generate)        │
                         │      raw: true (Bypasses chat template bias)    │
                         │      temperature: 0, top_k: 1, seed: 42         │
                         │      num_ctx: 4096, num_predict: 160            │
                         └────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │           Deterministic Grader Suite            │
                         │  - Arithmetic: Numeric parsing                  │
                         │  - Logic & Reasoning: Exact string token        │
                         │  - Structured Extraction: Strict JSON schema    │
                         │  - Format Adherence: Uppercase / exact match    │
                         │  - Factual Grounding: Bounded entity extraction │
                         └────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │         Automated SVG & CSV Reporting           │
                         │  - Exact-Score Accuracy (%)                     │
                         │  - Prefill & Decode Throughput (tokens/s)       │
                         │  - Incremental GPU VRAM Footprint (MiB)         │
                         │  - End-to-End Wall Latency (s)                  │
                         └─────────────────────────────────────────────────┘
```

---

## Weight Format Comparison

| Metric / Dimension | `Q4_K_M` | `Q8_0` | `F16` |
| :--- | :---: | :---: | :---: |
| **Bits Per Weight (Nominal)** | ~4.5 bits (variable $k$-quants) | 8.5 bits (block scale overhead) | 16.0 bits |
| **GGUF File Size** | 1.93 GB | 3.29 GB | 6.18 GB |
| **VRAM Compression Ratio** | $\approx 3.2\times$ | $\approx 1.88\times$ | $1.0\times$ (Baseline) |
| **Offload Risk on 8GB VRAM** | Negligible | Low | High (near ceiling with 4K context) |
| **Quantization Method** | Importance Matrix ($k$-quant) | Symmetric 32-value blocks | Direct float16 conversion |

---

## Directory Structure

```text
quantization-quality-lab/
├── README.md              # Research framing, execution guide, and methodology
├── RESULTS_TEMPLATE.md    # Standardized publication checklist and results table
├── benchmark.py           # Sequential execution harness, telemetry sampler, strict grading
├── setup_models.py        # Generates Modelfiles and registers GGUF aliases in Ollama
├── plot_results.py        # Dependency-free SVG metric chart generator
├── data/
│   └── tasks.json         # 15 deterministic tasks across 5 capability categories
├── models/                # Local GGUF download directory (untracked *.gguf)
├── results/               # Experiment CSV runs, JSON summaries, and SVG charts
└── tests/
    └── test_lab.py        # 9 unit tests verifying grading parsers and summary calculations
```

---

## The 15-Task Evaluation Battery (`data/tasks.json`)

The test battery isolates five capability dimensions using strict, deterministic graders that eliminate subjective evaluation:

1. **Arithmetic (3 tasks)**: Multi-step expressions `(17 * 23) + 5`, inventory subtraction, multi-day addition. *Grader: `integer` (extracts integer value, tolerating comma formatting).*
2. **Reasoning & Logic (3 tasks)**: Positional ordering (A left of B, C right of B), syllogistic validity, variable inequality comparison. *Grader: `exact` (requires single exact token, e.g. "C", "no", "B").*
3. **Structured Extraction (3 tasks)**: Incident tickets, server logs, weather reports converted to JSON. *Grader: `json` (requires single compact JSON object with exact keys, data types, and zero markdown code fence fences).*
4. **Instruction Formatting (3 tasks)**: Case transformations ("ALPHA"), single-word outputs ("PASS"), ordered selection ("blue"). *Grader: `exact` (rejects extra commentary, quotes, or punctuation).*
5. **Grounding & Evidence (3 tasks)**: Schedule date retrieval, attribution, negative rejection ("unknown" when unstated). *Grader: `exact`.*

---

## Step-by-Step Reproduction Guide

### 1. Download the Three GGUF Files

Download the three checkpoints from [bartowski/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF) into `models/`:

```bash
cd quantization-quality-lab

# Install huggingface-hub CLI if needed
python3 -m pip install -U 'huggingface_hub[cli]'

# Download all three formats
hf download bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q4_K_M.gguf --local-dir models
hf download bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q8_0.gguf --local-dir models
hf download bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-f16.gguf --local-dir models
```

### 2. Register Model Aliases with Ollama

```bash
python3 setup_models.py
ollama list
```
*Creates local aliases `quant-q4`, `quant-q8`, and `quant-f16` without re-quantization.*

### 3. Run Unit Tests

```bash
python3 -m unittest discover -s tests -v
```
*Validates that exact string matchers, integer parsers, strict JSON schemas, and metric aggregators pass 9/9 unit tests.*

### 4. Execute the Benchmark Suite

```bash
# Full 3-model benchmark (3 trials per task, rotated model ordering)
python3 benchmark.py \
  --models quant-q4 quant-q8 quant-f16 \
  --trials 3 \
  --ctx 4096 \
  --max-tokens 160 \
  --out results/main
```

*(On 8GB GPUs, you can benchmark Q4 and Q8 first with `--models quant-q4 quant-q8` to evaluate memory headroom before testing F16).*

### 5. Generate Metric Visualizations

```bash
python3 plot_results.py \
  --csv results/main/runs.csv \
  --out results/main/plots
```
Generates dependency-free SVG charts:
- `accuracy.svg`: Strict exact-score accuracy across representations.
- `decode_tps.svg`: Median token generation speed.
- `prefill_tps.svg`: Median prompt ingestion speed.
- `incremental_gpu_mib.svg`: Incremental VRAM consumption.
- `wall_s.svg`: End-to-end request latency.

---

## Experimental Controls & Methodological Guardrails

1. **Raw Completion vs Chat Templates**: The harness uses Ollama `/api/generate` with `raw: true` and an identical prompt prefix, eliminating template inconsistencies between Modelfiles.
2. **Strict Determinism**: Zero temperature (`temperature: 0`), greedy argmax (`top_k: 1`), fixed seed (`seed: 42`), and uniform context buffers (`num_ctx: 4096`).
3. **Sequential Model Isolation**: Between trials, the harness calls `keep_alive: 0` to unload the previous model and prevent VRAM co-residency or cache sharing.
4. **Hardware Placement Audit**: Inspect `ollama ps` while running. If F16 offloads layers to system RAM due to GPU memory limits, decode speeds will drop drastically; report CPU/GPU placement explicitly in `RESULTS_TEMPLATE.md`.

---

## Attribution & License

- **Model Checkpoint**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) by Alibaba Cloud (licensed under `qwen-research`).
- **GGUF Quantizations**: [bartowski/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF).
- **Harness & Benchmark Code**: Released under the [MIT License](../LICENSE).
