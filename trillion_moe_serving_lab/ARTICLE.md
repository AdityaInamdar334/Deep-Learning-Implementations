# The Trillion-Parameter Illusion: Why Open-Weight AI Models Are Harder to Run Than You Think
## Four Major Mixture-of-Experts Models, Up to 2.8 Trillion Parameters, and a Widening Benchmark Gap

> *Imagine you’re building an AI coding agent.*
>
> *You have access to eight NVIDIA H100 GPUs, giving you roughly 640 GB of aggregate GPU memory. That’s serious infrastructure. Enough to run many large language models, process substantial workloads, and serve an application to real users.*
>
> *Then you discover a model with 594 billion parameters.*
>
> *That sounds enormous. But the model only activates 17 billion parameters for each token it processes.*
>
> *Suddenly, it sounds manageable. Seventeen billion active parameters is smaller than many dense language models running on a single modern GPU.*
>
> *So you try to deploy it. And immediately encounter the problem.*
>
> *Your compute budget says 17 billion parameters. Your memory budget still says 594 billion.*

---

## Executive Summary

During the final weeks of October and the beginning of November 2026, four massive sparse Mixture-of-Experts (MoE) models redefined the open-weight frontier:
1. **Kimi K3.1 Pro (Moonshot AI):** 2.8 trillion total parameters, 104 billion active parameters per token, 93 layers, 896 routed experts, 1,048,576 token context window.
2. **Mistral Large 4 ("Le Chonk"):** 1.05 trillion total parameters, 52 billion active parameters per token, 72 layers, 128 routed experts, Multi-head Latent Attention (MLA) + sparse attention indexer.
3. **Tencent Hy4:** 770 billion total parameters, 49 billion active parameters, computer-use and GUI visual interaction (CAD, Blender, terminal).
4. **Inkling 2 Small (Thinking Machines Lab):** 594 billion total parameters, 17 billion active parameters per token, 48 layers, 384 routed experts.

Concurrently, Artificial Analysis updated its Intelligence Index to v5 (introducing FrontierCode 2.0 and Terminal-Bench-Science with 60% private evaluations), causing the reported gap between closed and open models to widen from 3 points to 12 points.

This lab investigates the critical operational reality: **MoE reduces per-token arithmetic in feed-forward layers, but it does not make inactive weights disappear.** Serving trillion-parameter models is primarily a distributed memory and interconnect bandwidth problem, where KV-cache scaling at 1M tokens, expert routing communication, and token verbosity dominate real-world task economics.

---

## Table of Contents

1. [Two Weeks That Changed the Open-Weight Landscape](#1-two-weeks-that-changed-the-open-weight-landscape)
2. [Why a Trillion-Parameter Model Doesn’t Use a Trillion Parameters Per Token](#2-why-a-trillion-parameter-model-doesnt-use-a-trillion-parameters-per-token)
3. [The Four Models and What Makes Each Different](#3-the-four-models-and-what-makes-each-different)
4. [The Memory Calculation Every ML Engineer Should Understand](#4-the-memory-calculation-every-ml-engineer-should-understand)
5. [Why One Million Tokens of Context Is Not Free](#5-why-one-million-tokens-of-context-is-not-free)
6. [The Benchmark Version Trap: How a Three-Point Gap Became Twelve](#6-the-benchmark-version-trap-how-a-three-point-gap-became-twelve)
7. [What the Numbers Actually Say About Model Performance](#7-what-the-numbers-actually-say-about-model-performance)
8. [Open Weights Do Not Always Mean Permission to Do Anything](#8-open-weights-do-not-always-mean-permission-to-do-anything)
9. [How to Choose a Model for Your Own Infrastructure](#9-how-to-choose-a-model-for-your-own-infrastructure)
10. [A Small Python Tool for Estimating Model Memory](#10-a-small-python-tool-for-estimating-model-memory)
11. [What the Next Generation of Open-Weight AI Is Teaching Us](#11-what-the-next-generation-of-open-weight-ai-is-teaching-us)
12. [References and Further Reading](#12-references-and-further-reading)

---

## 1. Two Weeks That Changed the Open-Weight Landscape

Between October 23 and November 5, 2026, four frontier models established sparse Mixture-of-Experts as the undisputed standard for frontier architectures:

* **October 23–24 (Kimi K3.1 Pro):** Moonshot AI released weights on Hugging Face for a 2.8T parameter model activating 104B per token with a 1M token context window [1].
* **October 29 (Mistral Large 4):** Mistral published open weights under the Mistral Ecosystem Open-Weight License for its 1.05T parameter (52B active) flagship, incorporating Multi-head Latent Attention and sparse attention indexing [2][3].
* **November 4 (Tencent Hy4):** Tencent unveiled its 770B parameter (49B active) model targeting multi-modal computer use, GUI control, CAD, and Blender automation [4].
* **November 5 (Inkling 2 Small):** Thinking Machines Lab launched its 594B parameter model with an ultra-sparse 17B active parameters, trained extensively via RL on long-horizon agentic workflows [5][6].

---

## 2. Why a Trillion-Parameter Model Doesn’t Use a Trillion Parameters Per Token

In a standard dense transformer, every token activates 100% of the model's parameters during the forward pass. To double capacity, you must double both compute FLOPs and storage.

Mixture-of-Experts breaks this lockstep coupling:

```
                          [ Input Token x ]
                                  |
                                  v
                        [ Gating Router W_g ]
                                  |
               +------------------+------------------+
               |                  |                  |
               v                  v                  v
         [ Expert 1 ]       [ Expert 2 ]       [ Expert E ]
               |                  |                  |
               \------------------+------------------/
                                  |
                     Top-k Weighted Combination
                                  |
                                  v
                         [ Next Layer ]
```

A learned router assigns routing probabilities $p_i(x) = \text{Softmax}(W_g x)_i$ and dispatches token $x$ exclusively to the top-$k$ experts:

$$y = \sum_{i \in \text{Top-}k} p_i(x) \cdot \text{Expert}_i(x)$$

### The Core Asymmetry
* **Compute (FLOPs):** Scales with **active parameters** $P_{\text{active}}$.
* **Storage & Bandwidth:** Scales with **total parameters** $P_{\text{total}}$.

A model with 2.8 trillion total parameters activating 104 billion parameters executes roughly the arithmetic of a 104B dense model per token, but **requires storing and routing across 2.8 trillion parameters in GPU VRAM**.

---

## 3. The Four Models and What Makes Each Different

| Architectural Feature | Kimi K3.1 Pro | Mistral Large 4 | Tencent Hy4 | Inkling 2 Small |
| :--- | :---: | :---: | :---: | :---: |
| **Total Parameters** | **2,800 B (2.8T)** | **1,050 B (1.05T)** | **770 B** | **594 B** |
| **Active Parameters** | **104 B** | **52 B** | **49 B** | **17 B** |
| **Sparsity Ratio ($P_{\text{active}} / P_{\text{total}}$)** | 3.7% | 5.0% | 6.4% | **2.8%** |
| **Decoder Layers** | 93 | 72 | — | 48 |
| **Routed Experts ($E$)** | 896 | 128 | — | 384 |
| **Top-$k$ Routing** | Top-16 (+2 shared) | Top-4 (+1 shared) | — | Top-8 |
| **Attention Architecture** | 69 Delta + 24 Gated Sparse | MLA + Sparse Indexer | Long-Context Sparse | Local + Global |
| **Context Window** | 1,048,576 (1M) | 1,048,576 (1M) | 1,048,576 (1M) | 1,048,576 (1M) |
| **Weight Availability** | Public (Hugging Face) | Public (HF / vLLM) | Planned Public | Hosted (Planned Apache) |

---

## 4. The Memory Calculation Every ML Engineer Should Understand

Raw theoretical weight footprint is governed by:

$$\text{Memory}_{\text{weights}} = P_{\text{total}} \times \frac{\text{Bits per Parameter}}{8}$$

### Weight Memory Footprint (GB)

| Precision | Bits | Inkling 2 Small (594B) | Tencent Hy4 (770B) | Mistral Large 4 (1.05T) | Kimi K3.1 Pro (2.8T) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP16 / BF16** | 16 | 1,188 GB | 1,540 GB | 2,100 GB | 5,600 GB |
| **FP8** | 8 | 594 GB | 770 GB | 1,050 GB | 2,800 GB |
| **Idealized FP4** | 4 | **297 GB** | **385 GB** | **525 GB** | **1,400 GB (1.4 TB)** |

### Hardware Capacity Thresholds (80 GB H100 GPUs)

| Model (at FP4) | Weight VRAM | Min. 80 GB GPUs | Min. 8-GPU Nodes | Headroom on Node |
| :--- | :---: | :---: | :---: | :---: |
| **Inkling 2 Small** | 297 GB | 4 GPUs (320 GB) | 1 Node (640 GB) | +343 GB (Healthy) |
| **Tencent Hy4** | 385 GB | 5 GPUs (400 GB) | 1 Node (640 GB) | +255 GB (Adequate) |
| **Mistral Large 4** | 525 GB | 7 GPUs (560 GB) | 1 Node (640 GB) | +115 GB (Tight) |
| **Kimi K3.1 Pro** | 1,400 GB | **18 GPUs** | **3 Nodes (1,920 GB)** | +520 GB across 24 GPUs |

> [!WARNING]
> These are raw weight capacity lower bounds. They exclude the KV cache, activation memory, CUDA context buffers, and All-to-All network communication overhead.

---

## 5. Why One Million Tokens of Context Is Not Free

For standard multi-head attention (MHA), the KV cache size grows linearly with sequence length $L$, layers $N$, heads $H_{\text{kv}}$, and dimension $D_h$:

$$\text{KV Bytes} = 2 \times N \times L \times H_{\text{kv}} \times D_h \times \text{BytesPerElement}$$

For a 72-layer model at $L = 1,000,000$ tokens in FP16 with 8 KV heads ($D_h = 128$):
$$\text{KV Bytes} = 2 \times 72 \times 1,000,000 \times 8 \times 128 \times 2 \approx 295 \text{ GB}$$
A single user request consuming 1M context requires nearly **300 GB of pure KV-cache VRAM**!

This is why **Multi-head Latent Attention (MLA)** (compressing KV vectors into a low-dimensional latent space $d_c \ll H \times D_h$) and **Sparse Attention Indexing** are mandatory for serving trillion-parameter models over long sequences.

---

## 6. The Benchmark Version Trap: How a Three-Point Gap Became Twelve

On October 27, 2026, Artificial Analysis launched **Intelligence Index v5**:
* **60% private evaluation tasks or held-out test sets** (mitigating data contamination).
* Introduced **FrontierCode 2.0** (merge-based real-world code criteria) and **Terminal-Bench-Science**.
* Removed SciCode entirely and updated reasoning benchmarks.

Under v4.3, the closed vs. open-weight gap was **~3 points**. Under v5, the gap between Claude Opus 5.5 (58) and Kimi K3.1 Pro (46) was reported as **12 points**.

### The Cross-Version Fallacy
Comparing across methodology versions manufactures artificial gaps:
* Kimi K3.1 Pro scored **55** on v4.3 and **46** on v5.
* Mistral Large 4 scored **38** on v4.3 and **26** on v5.
* Comparing Kimi's old score (55) to Mistral's new score (26) manufactures a **29-point gap**, whereas the true same-version gap on v5 is **20 points**.

```
  v4.3 Evaluation Scale:
  Claude [================================ 61 ]
  Kimi   [============================= 55 ]   <-- 3 pt gap to frontier open

  v5 Evaluation Scale (60% Private Tasks):
  Claude Opus 5.5 [============================ 58 ]
  Kimi K3.1 Pro   [====================== 46 ]  <-- 12 pt gap under new rubric
  Tencent Hy4     [================= 35 ]
  Inkling 2 Small [=============== 31 ]
  Mistral Large 4 [============= 26 ]
```

---

## 7. What the Numbers Actually Say About Model Performance

### Active Parameters $\neq$ Task Cost
* **Inkling 2 Small (17B active):** Scores 31 on v5, but generated **154 million tokens** across evaluation tasks, averaging 20.8 minutes per task due to reasoning verbosity [11].
* **Mistral Large 4 (52B active):** Scores 26 on v5, but completed evaluations in only **110 million tokens** [12].
* **Kimi K3.1 Pro (104B active):** High capability (65.7% Terminal-Bench 4.0), but costs ~$3.68 and ~36 minutes per complex reasoning task [9].

Total task cost is governed by the full pipeline equation:
$$\text{Task Cost} = \text{Tokens Generated} \times \text{Cost per Token} + \text{Tool Calls} + \text{Verification Retries}$$

---

## 8. Open Weights Do Not Always Mean Permission to Do Anything

1. **Mistral Ecosystem Open-Weight License:** Permissive for application development, but explicitly requires a separate commercial license for entities providing Model-as-a-Service (MaaS) hosting [13].
2. **Kimi K3.1 Pro License:** Broad research and commercial rights, but requires custom agreements if licensee/affiliate revenue exceeds $20M over 12 months for MaaS, along with strict attribution thresholds [14].
3. **Inkling 2 Small & Tencent Hy4:** Initial hosted deployment on Tinker / Cloud with commitments to release weights under Apache 2.0 / open licenses following safety audits.

---

## 9. How to Choose a Model for Your Own Infrastructure

1. **Self-Hosting Required?** Only Kimi K3.1 Pro and Mistral Large 4 have downloadable public checkpoints today.
2. **Cluster Topology:** MoE routing introduces heavy **All-to-All** communication across devices. High-speed inter-node fabric (e.g. InfiniBand 3.2 Tbps) is essential for multi-node setups like Kimi (18+ GPUs).
3. **Context Reality:** Do not allocate 1M tokens by default. Use RAG where appropriate to keep KV-cache allocations within single-node budgets.

---

## 10. A Small Python Tool for Estimating Model Memory

```python
from math import ceil

MODELS = {
    "Mistral Large 4": 1050,
    "Kimi K3.1 Pro": 2800,
    "Tencent Hy4": 770,
    "Inkling 2 Small": 594,
}
GPU_MEMORY_GB = 80
GPUS_PER_NODE = 8

def estimate_memory(params_b, bits):
    return params_b * bits / 8

def capacity_plan(memory_gb):
    gpu_count = ceil(memory_gb / GPU_MEMORY_GB)
    node_count = ceil(gpu_count / GPUS_PER_NODE)
    return gpu_count, node_count

for model, params_b in MODELS.items():
    print(f"\n{model} ({params_b}B total)")
    print("-" * 35)
    for precision in [16, 8, 4]:
        mem = estimate_memory(params_b, precision)
        gpus, nodes = capacity_plan(mem)
        print(f"{precision:2d}-bit | {mem:6.1f} GB | {gpus:2d} GPUs | {nodes} nodes")
```

---

## 11. What the Next Generation of Open-Weight AI Is Teaching Us

1. **The Parameter Count Is Not the Deployment Plan:** A model with 2.8T parameters activating 104B per token is neither a dense 2.8T model nor a dense 104B model. It is a distributed systems problem where memory capacity and network topology govern viability.
2. **Active FLOPs Do Not Equal Task Latency:** High reasoning effort and token verbosity can eclipse low per-token compute costs.
3. **The Measurement Layer Shapes the Frontier:** Benchmark methodology revisions shift perceived leaderboards as dramatically as architectural scaling.

---

## 12. References and Further Reading

1. Moonshot AI (2026). *Kimi K3.1 Pro: Official Model Card and Technical Specifications.* [https://huggingface.co/moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)
2. Mistral AI (2026). *Introducing Mistral Large 4.* [https://mistral.ai/news/mistral-large-4/](https://mistral.ai/news/mistral-large-4/)
3. Hugging Face (2026). *Welcome Le Chonk: Mistral Large 4 on Hugging Face.* [https://huggingface.co/blog/mistral-large4](https://huggingface.co/blog/mistral-large4)
4. Tencent (2026). *Tencent Launches Hy4.* [https://www.tencent.com/zh-hk/tencent-launches-hy4-joining-the-ranks-of-the-worlds-leading-open-source-ai-models/](https://www.tencent.com/zh-hk/tencent-launches-hy4-joining-the-ranks-of-the-worlds-leading-open-source-ai-models/)
5. Thinking Machines Lab (2026). *Introducing Inkling 2 Small.* [https://thinkingmachines.ai/news/inkling-2-small/](https://thinkingmachines.ai/news/inkling-2-small/)
6. Thinking Machines Lab (2026). *Inkling 2 Small Model Card.* [https://thinkingmachines.ai/model-card/inkling-2-small/](https://thinkingmachines.ai/model-card/inkling-2-small/)
7. Tencent Cloud (2026). *Hy4 Full Release Technical Overview.* [https://developer.cloud.tencent.com/article/2776227](https://developer.cloud.tencent.com/article/2776227)
8. Artificial Analysis (2026). *Announcing Artificial Analysis Intelligence Index v5.* [https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5)
9. Artificial Analysis (2026). *Kimi K3.1 Pro Analysis.* [https://artificialanalysis.ai/models/kimi-k3-1-pro](https://artificialanalysis.ai/models/kimi-k3-1-pro)
10. Artificial Analysis (2026). *Tencent Hy4 Analysis.* [https://artificialanalysis.ai/models/tencent-hy4](https://artificialanalysis.ai/models/tencent-hy4)
11. Artificial Analysis (2026). *Inkling 2 Small Analysis.* [https://artificialanalysis.ai/models/inkling-2-small](https://artificialanalysis.ai/models/inkling-2-small)
12. Artificial Analysis (2026). *Mistral Large 4 Analysis.* [https://artificialanalysis.ai/models/mistral-large-4](https://artificialanalysis.ai/models/mistral-large-4)
13. Mistral AI (2026). *Mistral Ecosystem Open-Weight License.* [https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-FP8/blob/main/LICENSE](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-FP8/blob/main/LICENSE)
14. Moonshot AI (2026). *Kimi K3.1 Pro License.* [https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE)

