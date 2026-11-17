# Experiment report template

## Hardware and software
GPU, VRAM, CPU, RAM, OS, Ollama version, model tag and digest, quantization, server environment, context, power setting, background activity.

## Experiment matrix
| Condition | Prompt tokens | Generated tokens | Median TTFT (s) | Prefill TPS | Decode TPS | Wall (s) | Peak device VRAM (MiB) | Placement |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | | | | | | | | |
| Longer prompt | | | | | | | | |
| Longer output | | | | | | | | |
| Lower context | | | | | | | | |

## Diagnosis with evidence
Which metric shifted? Which variable changed? Which runtime log or `ollama ps` output corroborates it? Which alternative explanations are possible? Do not infer a causal bottleneck solely from GPU utilization.

## Limitations
Sampling cadence, client first-token buffering, CPU offload uncertainty, cold/warm cache state, model load included/excluded, and trial count.
