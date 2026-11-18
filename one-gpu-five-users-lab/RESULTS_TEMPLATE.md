# Experimental Results Template

## Environment
- Date; model, quantization, and digest:
- GPU model, VRAM, driver, system RAM:
- Ollama version:
- `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_QUEUE`, `OLLAMA_CONTEXT_LENGTH`, cache quantization:
- Client and server on the same host?:
- Prompt length, output cap, requests per user, timeouts:

## Results: closed-loop client sweep

| Concurrent users | Completed/attempted | Requests/s | Aggregate output tok/s | p50 TTFT (s) | p95 TTFT (s) | p50 total latency (s) | p95 total latency (s) | Failures | Peak sampled GPU MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| 4 | | | | | | | | | |
| 8 | | | | | | | | | |
| 16 | | | | | | | | | |

## Analysis

1. At which concurrency did p95 exceed the interactive SLO? Record the SLO first.
2. At what point did request and token throughput stop increasing materially?
3. Did 503 errors or timeouts appear? Was `OLLAMA_MAX_QUEUE` exceeded?
4. Did `ollama ps` show full GPU residency? Did GPU memory pressure increase when `OLLAMA_NUM_PARALLEL` changed?
5. How do results differ for server parallelism 1, 2, and 4?
6. What changed when long-context prompts replaced short prompts?
7. Replicate with open-loop arrivals before extrapolating real user capacity.

## Limitations

These results measure a controlled closed-loop local workload; cannot directly expose internal queue timestamps, production workload mixture, or service capacity under unbounded arrival rates. TTFT is first visible text. The p95 estimate is particularly noisy with small sample counts.
