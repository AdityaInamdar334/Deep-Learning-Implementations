# Trillion-Parameter MoE Serving & Capacity Report

## 1. Hardware Allocation Grid (1 Million Token Context)

| Model | Total Params | Active Params | Precision | Weight VRAM | KV Cache (1M) | Total Needed | H100 GPUs (Nodes) | H200 GPUs (Nodes) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kimi K3.1 Pro** | — | — | 8-bit | 2699.0 GB | 67.0 GB | 3173.9 GB | **40 GPUs (5N)** | 22 GPUs (3N) |
| **Kimi K3.1 Pro** | — | — | 4-bit | 1349.5 GB | 67.0 GB | 1620.4 GB | **21 GPUs (3N)** | 11 GPUs (2N) |
| **Mistral Large 4** | — | — | 8-bit | 1012.1 GB | 81.0 GB | 1249.1 GB | **16 GPUs (2N)** | 9 GPUs (2N) |
| **Mistral Large 4** | — | — | 4-bit | 506.1 GB | 81.0 GB | 671.1 GB | **9 GPUs (2N)** | 5 GPUs (1N) |
| **Tencent Hy4** | — | — | 8-bit | 742.2 GB | 51.2 GB | 913.4 GB | **12 GPUs (2N)** | 7 GPUs (1N) |
| **Tencent Hy4** | — | — | 4-bit | 371.1 GB | 51.2 GB | 482.3 GB | **7 GPUs (1N)** | 4 GPUs (1N) |
| **Inkling 2 Small** | — | — | 8-bit | 572.6 GB | 192.0 GB | 860.6 GB | **11 GPUs (2N)** | 6 GPUs (1N) |
| **Inkling 2 Small** | — | — | 4-bit | 286.3 GB | 192.0 GB | 526.3 GB | **7 GPUs (1N)** | 4 GPUs (1N) |

## 2. Artificial Analysis Intelligence Index v5 vs v4.3 Version Analysis

| Model | v4.3 Score | v5 Score | True Same-Version Gap vs Leader | Public Weights Downloadable |
| :--- | :---: | :---: | :---: | :---: |
| **Claude Opus 5.5 (Closed Baseline)** | 61.0 | **58.0** | Baseline (58.0) | Hosted / Planned |
| **Kimi K3.1 Pro** | 55.0 | **46.0** | -12.0 pts | Yes (Hugging Face) |
| **Tencent Hy4** | N/A | **35.0** | -23.0 pts | Hosted / Planned |
| **Inkling 2 Small** | N/A | **31.0** | -27.0 pts | Hosted / Planned |
| **Mistral Large 4** | 38.0 | **26.0** | -32.0 pts | Yes (Hugging Face) |

## 3. End-to-End Task Economics & Token Verbosity

| Model | Avg Tokens / Task | Task Latency | Hosted Cost / Task | Open-Weight Viability |
| :--- | :---: | :---: | :---: | :---: |
| **Claude Opus 5.5 (Closed Baseline)** | 125,000 | 18.5 min | $4.20 | API / Platform only |
| **Kimi K3.1 Pro** | 140,000 | 36.0 min | $3.68 | Self-hostable |
| **Tencent Hy4** | 115,000 | 24.0 min | $2.10 | API / Platform only |
| **Inkling 2 Small** | 154,000 | 20.8 min | $1.85 | API / Platform only |
| **Mistral Large 4** | 110,000 | 15.2 min | $1.55 | Self-hostable |
