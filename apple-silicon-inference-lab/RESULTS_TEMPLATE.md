# Apple Silicon Benchmark Results (fill only from real runs)

## Environment
- Chip and CPU/GPU cores:
- Unified memory:
- macOS version:
- Power mode (AC/battery):
- Ollama version:
- llama.cpp commit:
- mlx-lm / MLX version:
- GGUF SHA-256:
- MLX model source and revision:
- Model quantization details:
- Ambient temperature, workload order, cooldown:

## Streaming HTTP client measurements

| Engine | Median TTFT to first visible text (s) | Median total wall time (s) | Produced tokens | Notes |
|---|---:|---:|---:|---|
| Ollama | | | | |
| llama.cpp | | | | |
| MLX | | | | |

## Native engine benchmarks (different methodology)

| Engine | Prompt tokens/s (pp512) | Generation tokens/s (tg128) | Tool and version |
|---|---:|---:|---|
| llama-bench | | | |
| mlx_lm.benchmark | | | |
| Ollama server timing, if recorded | | | |

## Memory and thermal observations

| Engine | Process RSS / Mac memory-pressure notes | Peak memory information and source | Thermal pressure / power / temperature, if reported |
|---|---|---|---|
| Ollama | | | |
| llama.cpp | | | |
| MLX | | | |

## Findings and uncertainty
1. Observed bottleneck:
2. Confounders (quantization, tokenizer, cache, residency, warmup, thermal):
3. Repeated measurement variability:
4. Next test:
