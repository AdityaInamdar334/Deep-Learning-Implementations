# Results template (fill with your own measured values)

Hardware: [GPU and VRAM], [CPU], [RAM], [OS], [driver version]

Software: [Ollama version], [model tag], [model digest], [quantization], [Flash Attention / KV-cache settings]

## Fixed-prompt sweep

| Context | Actual prompt tokens | Median peak VRAM (MiB) | Median prefill tok/s | Median decode tok/s | Median wall time (s) | Placement |
|---|---:|---:|---:|---:|---:|---|
| 2048 | | | | | | |
| 4096 | | | | | | |
| 8192 | | | | | | |
| 16384 | | | | | | |

## Scaled-prompt sweep

| Context | Actual prompt tokens | Median peak VRAM (MiB) | Median prefill tok/s | Median decode tok/s | Needle correct / trials |
|---|---:|---:|---:|---:|---:|
| 2048 | | | | | |
| 4096 | | | | | |
| 8192 | | | | | |
| 16384 | | | | | |

## Interpretation

- Is memory tied to configured context even when actual prompt stays fixed?
- Does prefill wall time rise with actual prompt tokens?
- Did CPU offloading appear at longer contexts?
- Did needle retrieval fail due to truncation, or a model error? What evidence supports the distinction?
- Were there other processes competing for GPU memory?

Do not treat missing GPU samples as zero usage or a failed needle as proof of context incapacity.
