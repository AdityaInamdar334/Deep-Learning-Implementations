# Quantization experiment result sheet

**Do not publish an empirical comparison until the actual benchmark has run.**

- Date/time, timezone:
- GPU name, VRAM, NVIDIA driver, CUDA runtime:
- CPU, system RAM, OS:
- Ollama version, GGUF hashes:
- Quantization provenance / llama.cpp version / imatrix if known:
- All `ollama ps` model placements (100% GPU versus CPU/GPU split):
- Context length, output cap, decoding parameters:
- Background processes and GPU idle memory:
- Benchmark commands and dataset hash:

## Disk sizes (published, NOT measured GPU usage)

| Representation | Approximate published file size |
|---|---:|
| Q4_K_M | 1.93 GB |
| Q8_0 | 3.29 GB |
| F16 | 6.18 GB |

## Real benchmark table (fill from `summary.json` and `runs.csv`)

| Representation | GPU placement | Exact score / n | Median decode tok/s | Median prefill tok/s | Peak sampled VRAM MiB | Median wall s |
|---|---|---:|---:|---:|---:|---:|
| Q4_K_M | TBD | TBD | TBD | TBD | TBD | TBD |
| Q8_0 | TBD | TBD | TBD | TBD | TBD | TBD |
| F16 | TBD | TBD | TBD | TBD | TBD | TBD |

## Per-category comparisons

- Arithmetic: ...
- JSON extraction: ...
- Reasoning: ...
- Grounded factual retrieval: ...
- Instruction following: ...

## Raw-output analysis

Record at least 5 examples where outputs differ. Explain whether the difference is numerical accuracy, instruction compliance, semantic correctness, truncation, or parsing. Avoid treating any output difference as automatically worse.

## Threats to validity

- Small unrepresentative dataset
- Strict exact-match grading
- Prepublished quantizations and importance matrices
- Changes in accelerator placement
- Cold versus warm starts
- Device-wide and sampled memory measurements
- Model sampling and hidden runtime defaults
- Quality of test coverage and prompt effects
