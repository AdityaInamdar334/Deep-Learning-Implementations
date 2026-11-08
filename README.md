# Deep Learning & AI Research Labs

**Reproducible implementations, controlled experiments, and engineering notes on modern machine learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Labs](https://img.shields.io/badge/Research%20Labs-30-238636.svg)](#research-labs)

This repository brings together the companion code for my technical writing on deep learning, AI agents, and applied research. Each lab investigates a specific question through a small implementation, repeatable experiments, and an honest discussion of what the results do and do not show.

The goal is to make research ideas **understandable, executable, and testable**, not just reproduce a diagram from a paper.

Most projects are intentionally smaller than the systems that inspired them. Unless explicitly stated otherwise, the experiments are **independent educational implementations**, not official releases or faithful reproductions of the original authors' large-scale benchmarks.

## Explore the repository

- [Research labs](#research-labs)
- [Getting started](#getting-started)
- [How the experiments are designed](#how-the-experiments-are-designed)
- [Repository structure](#repository-structure)
- [Adding a new lab](#adding-a-new-lab)
- [Research attribution and license](#research-attribution-and-license)

## Research labs

### AI agents, safety, and systems

| Lab | What it explores |
| --- | --- |
| [Elephant-Goldfish Design Review Lab](./elephant_goldfish_review_lab) | Fresh-agent zero-context design reviews, SHA-256 specification binding, and fail-closed gate evaluation. |
| [IDE Trust Boundary Lab](./ide_trust_boundary_lab) | Publisher ID vs content provenance, Win32 canonicalization bypasses, Langflow vulnerability triage, and read-only IOC detection. |
| [Benchmark Auditing & Evaluation Integrity](./benchmark_auditing_lab) | Measurement layer defects, multi-step cascading recovery, Bloom's cognitive complexity collapse, and the rebuilt evaluation stack. |
| [Clocktower Evidence & Belief Updates](./clocktower_belief_lab) | Social deduction benchmarks, Bayesian belief updating under targeted information poisoning, and paired game ratings. |
| [Superagent Context, Authority & Lifecycle](./superagent_lifecycle_lab) | Admission boundaries, immutable mandates against prompt injection, policy cache invalidation, and human stop interlocks. |
| [Trusted Compiler Comparison](./compiler_comparison_lab) | Cyclic round-robin permutation, fresh-process wall time, and bitwise output signature parity across optimization flags. |
| [Distributed Scientific Analysis](./distributed_science_agent_lab) | Specialist workers, parallel execution, holdout verification, and the cost of scientific investigation. |
| [AI Agent Cost Optimization](./agent_cost_optimization_lab) | Token-level accounting, prompt-cache simulations, workflow overhead, and quality-aware optimization. |
| [SpikeGuard Bounded Agent Lab](./spikelab_bounded_agent_lab) | Bounded tool contracts, spike-timing tiling (STTC), surrogate statistics, and independent scientific evaluation. |
| [Roast Room Decision Lab](./roast_room_decision_lab) | Multi-agent councils, correlated errors, conformity dynamics, and non-compensatory ruin caps. |
| [Frontier Model Monitoring](./frontier_monitor_lab) | Layered monitors, independent telemetry, escalation thresholds, and failure under distribution shift. |
| [Spotify Shunt Routing](./spotify_shunt_routing_lab) | Delegating context-heavy tasks to cheaper workers, with token, latency, and information-loss tradeoffs. |

### Model architectures and representation learning

| Lab | What it explores |
| --- | --- |
| [MoE Architecture & Routing Mechanics](./moe_frontier_lab) | NumPy top-k routing implementation, active arithmetic vs stored weight scaling, and CPU dispatch dynamics. |
| [Trillion-Parameter MoE Serving & Capacity](./trillion_moe_serving_lab) | Sparse Mixture-of-Experts memory planning, KV-cache scaling at 1M tokens, MLA compression, and benchmark version traps. |
| [gMLP and Spatial Gating](./gmlp_spatial_gating_lab) | Attention-free token mixing, causal masking, and generalization to unseen token positions. |
| [Grouped-Query Attention (GQA)](./gqa_companion_lab) | Multi-Query vs Grouped-Query attention, KV-cache scaling, and checkpoint mean-pooling. |
| [Vision Transformers](./vit_patch_lab) | Patch embedding, learned positional encodings, classification tokens, and CNN comparisons. |
| [Activation Steering and Interpretability](./llm_steering_interpretability_lab) | Residual-stream steering, neuron interventions, projection edits, and KL-based damage checks. |
| [LoRA from Scratch](./lora_companion_lab) | Low-rank adapter factorizations, asymmetric step-0 gradient dynamics, and zero-latency weight merging. |
| [GPT-NeoX from Scratch](./neox_companion_lab) | Parallel residual blocks, partial RoPE (25%), fused QKV layouts, and causal KV-cache parity. |
| [GPT-NeoX Autoregressive Generation](./neox_generation_lab) | KV-cache prefill vs incremental decoding, request isolation, and sampling dynamics. |
| [Primer EZ Transformer](./primer_ez_transformer_lab) | Squared ReLU, causal Multi-DConv-Head Attention, time-to-quality efficiency, and structural shift. |
| [Hourglass Transformer](./hourglass_transformer_lab) | Hierarchical token shortening, causal right-shift downsampling, and length extrapolation. |
| [Weight-Space Programming](./weight_space_programming_lab) | Direct weight inspection, permutation symmetries, circuit compiling, and gradient-free model repair. |
| [Weight-Space Reading Audit](./weight_space_reading_audit_lab) | Auditing whether LLMs read weight space vs memorize labels; output permutation, shuffle controls, and hidden symmetries. |
| [Qwenputer Educational Lab](./qwenputer_educational_lab) | Delta-rule-inspired addressable memory, binary logic, and a miniature RISC-V execution exercise. |

### Learning, memory, and forecasting

| Lab | What it explores |
| --- | --- |
| [Decision Transformers with Memory](./dt_mem_working_memory_lab) | Content-addressable working memory, task adaptation, and low-rank fine-tuning concepts. |
| [Sleep-Inspired Continual Learning](./sleep_framework_lab) | Replay, frozen-backbone expansion, distillation, and catastrophic forgetting. |
| [TimesFM-3 Forecasting](./timesfm3_forecasting_lab) | Multivariate forecasting, rolling-origin evaluation, strong baselines, and distribution shift. |

### Physics, quantum systems, and scientific computing

| Lab | What it explores |
| --- | --- |
| [Superfluid Helium Physics Lab](./superfluid_helium_lab) | Lambda phase transitions, Bose-Einstein condensate vs superfluid density fractions, Landau critical velocity ($v_c \approx 58.4$ m/s), phonon-roton dispersion, and two-fluid hydrodynamics. |

**Where to begin?** If you're interested in architecture, start with [Vision Transformers](./vit_patch_lab) or [gMLP](./gmlp_spatial_gating_lab). For systems engineering, try [Agent Cost Optimization](./agent_cost_optimization_lab). For evaluation methodology, explore [Frontier Model Monitoring](./frontier_monitor_lab) or [TimesFM-3 Forecasting](./timesfm3_forecasting_lab).

## Getting started

Each directory is designed to be explored independently. Install its dependencies locally rather than treating the monorepo as a single Python package.

**1. Clone the repository and choose a lab.**

```bash
git clone <YOUR_REPOSITORY_URL>
cd Medium-Implementations
cd vit_patch_lab
```

Replace `<YOUR_REPOSITORY_URL>` with this repository's HTTPS or SSH clone URL. If you already have the repository locally, skip the clone step.

**2. Create a virtual environment.**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1`.

**3. Install the selected lab's dependencies.**

```bash
python -m pip install -r requirements.txt
```

**4. Run its tests.**

```bash
python -m unittest discover -s tests -v
```

**5. Run an experiment.** Check that lab's `README.md` for its exact entry point, arguments, expected runtime, and optional dataset or model downloads. For example, the ViT lab provides:

```bash
python -m vit_lab.benchmark --seeds 3 --output results/benchmark.json
```

Some labs use only the Python standard library. Others require PyTorch or additional dependencies. A few provide optional integrations with pretrained models or external datasets; those are separate from their default offline demonstrations.

## How the experiments are designed

These labs follow a few practical principles:

1. **Start with a question.** Every experiment should test a mechanism or hypothesis, not just produce a metric.
2. **Use simple baselines.** A sophisticated architecture is meaningful only when compared with a relevant simpler alternative.
3. **Control randomness.** Where applicable, experiments expose seeds and save machine-readable results.
4. **Probe failure modes.** Ablations, corrupted inputs, distribution shifts, and negative results are part of the research, not exceptions to hide.
5. **Verify the implementation.** Unit tests check tensor shapes, causal behavior, invariants, and other properties where applicable.
6. **Separate evidence from inspiration.** Results from a toy simulation are not presented as replications of claims from the source paper or engineering report.

Depending on the lab, you may find `BENCHMARK_NOTES.md`, `ARTICLE.md`, `results/`, or `figures/` alongside the source code. **Read the lab-specific notes before comparing performance numbers:** some experiments use synthetic datasets, modeled costs, simulated latency, or small CPU-friendly networks. Results are not intended to be compared across unrelated labs.

## Repository structure

```text
Deep-Learning-Implementations/
├── README.md
├── LICENSE
├── .gitignore
├── agent_cost_optimization_lab/
├── benchmark_auditing_lab/
├── clocktower_belief_lab/
├── compiler_comparison_lab/
├── distributed_science_agent_lab/
├── dt_mem_working_memory_lab/
├── elephant_goldfish_review_lab/
├── frontier_monitor_lab/
├── gmlp_spatial_gating_lab/
├── gqa_companion_lab/
├── hourglass_transformer_lab/
├── ide_trust_boundary_lab/
├── llm_steering_interpretability_lab/
├── lora_companion_lab/
├── moe_frontier_lab/
├── neox_companion_lab/
├── neox_generation_lab/
├── primer_ez_transformer_lab/
├── qwenputer_educational_lab/
├── roast_room_decision_lab/
├── sleep_framework_lab/
├── spikelab_bounded_agent_lab/
├── spotify_shunt_routing_lab/
├── superagent_lifecycle_lab/
├── superfluid_helium_lab/
├── timesfm3_forecasting_lab/
├── trillion_moe_serving_lab/
├── vit_patch_lab/
├── weight_space_programming_lab/
└── weight_space_reading_audit_lab/
```

Individual labs generally contain a Python package or scripts, a `tests/` directory, `requirements.txt`, and their own `README.md`. Some also include benchmark artifacts and a companion article. The specific layout is documented inside each folder.

## Adding a new lab

This repository is organized by research topic rather than by publication date. To add an experiment:

1. Create a clearly named folder at the repository root.
2. Include a focused `README.md` covering the research question, setup, commands, evaluation methodology, and limitations.
3. Include executable source code, tests, and a dependency specification. Add saved benchmark outputs where feasible.
4. Cite the original research and clearly label any independent reproductions or deviations.
5. Add the lab to the relevant catalog section above and update the lab count badge.
6. Commit and push the changes.

```bash
git add .
git commit -m "feat: add new research lab"
git push origin main
```

## Research attribution and license

Each project acknowledges the papers, engineering reports, and repositories that motivated it in its own documentation. Original publications and their reported performance figures belong to their respective authors. Unless a lab explicitly says otherwise, the code here is an independent educational implementation and is **not affiliated with or endorsed by** the original research organizations.

The repository's own code is distributed under the [MIT License](LICENSE). Third-party models, datasets, checkpoints, and upstream code may have separate terms; consult their original licenses before redistribution or commercial use.

---

**Read the explanation. Run the experiment. Challenge the result.**
