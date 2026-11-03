# Medium Implementations 🔬🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Total Implementations](https://img.shields.io/badge/Implementations-11%20Labs-brightgreen.svg)]()

A centralized monorepo containing reproducible, self-contained Python laboratories, PyTorch implementations, agentic systems, and research benchmarks accompanying technical Medium articles.

Each subfolder is an independent, reproducible lab equipped with its own codebase, tests, benchmark scripts, and documentation.

---

## 📚 Labs Catalog

| # | Lab / Directory | Domain | Focus & Paper / Article Inspiration |
|---|---|---|---|
| 1 | [`distributed_science_agent_lab`](./distributed_science_agent_lab) | Agentic AI & Science | Multi-agent scientific discovery, independent worker splitting, peer review, and discovery consensus |
| 2 | [`agent_cost_optimization_lab`](./agent_cost_optimization_lab) | LLM Cost Engineering | Offline LLM audit, KV cache reuse, prompt compression, and agent token cost reduction |
| 3 | [`dt_mem_working_memory_lab`](./dt_mem_working_memory_lab) | Reinforcement Learning | Decision Transformers with working memory & episodic recall (inspired by ICML 2024 DT-Mem) |
| 4 | [`frontier_monitor_lab`](./frontier_monitor_lab) | AI Safety & Alignment | Layered safety monitoring, tripwires, and telemetry for frontier AI agents |
| 5 | [`gmlp_spatial_gating_lab`](./gmlp_spatial_gating_lab) | Deep Learning Architectures | gMLP with Spatial Gating Units (SGU), static token mixing, and tiny attention comparison |
| 6 | [`llm_steering_interpretability_lab`](./llm_steering_interpretability_lab) | Mechanistic Interpretability | Activation steering vectors, representation engineering, and neuron interpretability in SwiGLU nets |
| 7 | [`qwenputer_educational_lab`](./qwenputer_educational_lab) | Neural Hardware & Logic | Addressable recurrent memory using DeltaNet delta rules and compiled binary logic weights |
| 8 | [`sleep_framework_lab`](./sleep_framework_lab) | Continual Learning & Neuro-AI | Sleep consolidation, offline memory replay, and mitigating catastrophic forgetting in PyTorch |
| 9 | [`spotify_shunt_routing_lab`](./spotify_shunt_routing_lab) | LLM Context Engineering | Shunt routing architectures for agentic workflows to reduce prompt context tokens |
| 10 | [`timesfm3_forecasting_lab`](./timesfm3_forecasting_lab) | Time-Series Foundation Models | Multivariate forecasting, temporal/variate attention, and rolling-origin evaluation (TimesFM-3) |
| 11 | [`vit_patch_lab`](./vit_patch_lab) | Computer Vision | Vision Transformer (ViT) from scratch: patch embeddings, positional encodings, and inductive bias |

---

## 🛠️ Repository Structure

```text
Medium-Implementations/
├── .gitignore
├── LICENSE
├── README.md
├── distributed_science_agent_lab/
│   ├── science_lab/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
├── agent_cost_optimization_lab/
│   ├── src/
│   ├── tests/
│   └── README.md
├── ... (each lab is fully self-contained)
└── vit_patch_lab/
    ├── vit_lab/
    ├── tests/
    └── README.md
```

---

## 🚀 Running Any Implementation

Every lab is isolated and can be run independently without affecting the others.

1. Navigate to the desired lab folder:
   ```bash
   cd distributed_science_agent_lab
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run tests or experiments:
   ```bash
   pytest
   python plot_results.py # or the primary runner script in that folder
   ```

---

## ➕ Workflow: Adding New Implementations

To add a new lab without cluttering GitHub with individual repositories:

1. **Create a new folder** in the repository root:
   ```bash
   mkdir my_new_lab
   cd my_new_lab
   ```

2. **Add your implementation files**, `requirements.txt`, tests, and a local `README.md`.

3. **Add the new lab to the catalog table** in this root `README.md`.

4. **Commit and push** to the unified monorepo:
   ```bash
   git add .
   git commit -m "feat: add my_new_lab implementation"
   git push origin main
   ```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE). Individual implementations are educational research tools and acknowledge their respective paper and article inspirations in their dedicated READMEs.
