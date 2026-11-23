# Self-Rewriting Weights & Rank-One Projection Editing: Toy Lab

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-013243.svg)](https://numpy.org/)
[![Concept: Weight-Space Editing](https://img.shields.io/badge/Concept-Rank--1%20Projection%20%26%20Paired%20Eval-purple.svg)](#mathematical-mechanics)
[![Tests Passing](https://img.shields.io/badge/tests-5%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A transparent, NumPy-only educational companion to Jiajun Fan et al., *Language Models Can Rewrite Their Own Weights* (arXiv:2611.21885v2, 2026). It isolates three foundational mechanics discussed when evaluating surgical weight-space edits: **rank-one directional projection**, **problem-clustered paired evaluation**, and **output token-budget truncation effects**.

> [!IMPORTANT]
> **Scope & Honest Framing:** This lab uses synthetic matrices and fabricated evaluation draws to demonstrate linear-algebraic and statistical mechanics. It is **not** a replication of the Qwen3.5-27B experiments or the authors' full protocol (public code and model artifacts were pending release as of November 22, 2026). It does not measure actual LLM capability or establish empirical efficacy of self-rewriting.

---

## Mathematical Mechanics

### 1. Rank-One Directional Projection Edit
Given a weight matrix $W \in \mathbb{R}^{d \times m}$, a unit target direction $u \in \mathbb{R}^d$ ($\|u\|_2 = 1$), and an attenuation strength $\alpha \in \mathbb{R}$, the rank-one projection edit is:

$$W' = W - \alpha \, u (u^\top W)$$

This update satisfies three exact algebraic invariants:
- **Rank-One Perturbation**: $\text{rank}(W' - W) = 1$ (provided $u^\top W \neq \mathbf{0}$ and $\alpha \neq 0$).
- **Exact Directional Attenuation**: Projecting along $u$ scales the output by exactly $(1 - \alpha)$:
  $$u^\top W' = u^\top W - \alpha (u^\top u)(u^\top W) = (1 - \alpha) \, u^\top W$$
- **Orthogonal Complement Invariance**: For any vector $v \perp u$ ($v^\top u = 0$), the weight response is completely untouched:
  $$v^\top W' = v^\top W - \alpha (v^\top u)(u^\top W) = v^\top W$$

```
                       RANK-ONE WEIGHT PROJECTION EDIT

    Input Direction v (v _|_ u)  ---->  [ W' = W - alpha * u(u^T W) ]  ---->  v^T W' == v^T W       (Unchanged)
    Target Direction u           ---->  [ W' = W - alpha * u(u^T W) ]  ---->  u^T W' == (1-alpha) u^T W (Attenuated)
```

### 2. Problem-Clustered Paired Evaluation
When benchmarking an edited model against a base model across $P$ problems with $S$ repeated sampling seeds per problem, repeated draws on the same problem are **not independent**. Treating $P \times S$ samples as i.i.d. artificially shrinks standard errors. Instead, we compute per-problem mean differences $\Delta_p$ and cluster standard error over problems:

$$\Delta_p = \frac{1}{S} \sum_{s=1}^S \left( y^{\text{edited}}_{p,s} - y^{\text{base}}_{p,s} \right), \qquad \bar{\Delta} = \frac{1}{P} \sum_{p=1}^P \Delta_p, \qquad \text{SE}_{\text{cluster}} = \frac{\text{std}(\Delta_1, \dots, \Delta_P)}{\sqrt{P}}$$

### 3. Token-Budget Truncation Accounting
Identical reasoning traces can yield dramatically different pass rates purely from output-length cutoffs (e.g., 4,096 vs. 16,384 tokens). Separating truncation effects from genuine reasoning improvements requires reporting both completion rates and accuracy across matched token budgets.

---

## Quickstart

### 1. Setup & Run

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/self_rewriting_weights_lab

# Create virtual environment and install NumPy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all three demonstrations
python3 lab.py

# Or run a specific section: projection | paired | budget
python3 lab.py --section projection
python3 lab.py --section paired
python3 lab.py --section budget
```

### 2. Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The 5 automated unit tests verify:
- Exact rank-1 perturbation and $(1 - \alpha)$ attenuation along the target direction $u$.
- Exact invariance along directions orthogonal to $u$.
- Boundary regimes ($\alpha = 0$ identity and $\alpha = 1$ complete directional nulling).
- Exact problem-clustered paired evaluation means, deltas, and standard errors.
- Clean end-to-end execution of all three CLI demonstrations.

---

## Proposed Independent Controls (Post-Artifact Release)

Once official code and checkpoints are released, an independent replication protocol should:
1. Freeze the 490-problem, 8-seed held-out evaluation before inspecting test metrics.
2. Implement both the session-2 directional projection edit and the session-8 linear-attention time-step-bias edit.
3. Compare against unedited baselines, budget expansion, stop/repetition controls, matched-norm random edits, and manual bias sweeps.
4. Report cluster-level uncertainty over problems alongside both completion and accuracy rates.

---

## File Structure

```
self_rewriting_weights_lab/
├── README.md               # Mathematical derivations, protocol notes, and CLI guide
├── requirements.txt        # numpy>=1.24,<3
├── .gitignore              # Ignores __pycache__ and bytecode
├── lab.py                  # Rank-1 projection, paired evaluation, and budget demos
└── tests/
    └── test_lab.py         # 5 automated unit tests
```

---

## References

- Jiajun Fan et al., *Language Models Can Rewrite Their Own Weights* (2026), [arXiv:2611.21885v2](https://arxiv.org/html/2611.21885v2).
- Project Page & Demos: [jiajunfan.com/projects/self-rewriting](https://www.jiajunfan.com/projects/self-rewriting/).
- Related labs in this repository:
  - [`weight_space_programming_lab/`](../weight_space_programming_lab) — Direct weight inspection, permutation symmetries, and circuit compiling.
  - [`iwsp_companion_lab/`](../iwsp_companion_lab) — Interpretable weight-space programming with handcrafted zero-gradient networks.
  - [`llm_steering_interpretability_lab/`](../llm_steering_interpretability_lab) — Residual-stream steering and projection edits.
