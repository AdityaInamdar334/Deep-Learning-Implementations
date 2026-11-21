# Interpretable Weight-Space Programming (iWSP) Companion Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Paradigm: iWSP](https://img.shields.io/badge/Paradigm-Weight--Space%20Programming-purple.svg)](#what-is-interpretable-weight-space-programming)
[![Optimization: Zero-Gradient](https://img.shields.io/badge/Optimization-Zero--Gradient%20%2F%20Closed--Form-orange.svg)](#mechanics-of-handcrafted-architectures)
[![Tests Passing](https://img.shields.io/badge/tests-4%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational companion lab exploring the foundational concepts behind **Interpretable Weight-Space Programming (iWSP)**, inspired by Mehta et al., *Interpretable weight-space programming builds transparent transformers and brain encoding models* (arXiv:2611.18114, 2026).

This lab demonstrates how algorithmic logic and mathematical state transitions can be **directly compiled into fixed neural network weights** without training, gradient descent (`.backward()`), or optimizer updates.

> [!IMPORTANT]
> **Educational Scope:** This lab is a transparent conceptual companion isolating handcrafted weight construction and deterministic execution. It is **not** a full reproduction of the paper's LLM-driven agentic search, multi-layer transformer compiler, or neuroscience brain-encoding experiments.

---

## What is Interpretable Weight-Space Programming?

Conventional deep learning relies on empirical gradient-based parameter discovery:
$$\theta^* = \arg\min_{\theta} \sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i) \quad \text{via} \quad \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

This optimization yields opaque, entangled weight matrices that require post-hoc mechanistic interpretability to decipher.

In contrast, **Interpretable Weight-Space Programming (iWSP)** treats weight space as a target programming language. Neural networks are constructed or synthesized such that every weight, bias, and activation corresponds to an explicit, provably correct algorithmic primitive:
$$\theta = \text{Compile}(\text{Algorithm Specification})$$

```
  TRADITIONAL GRADIENT OPTIMIZATION (Opaque Representations)
  
       Random Weights \theta_0 ---> [ Gradient Descent / Adam ] ---> Opaque Converged Weights \theta^*
       (Requires millions of training steps; inner circuits are emergent and unverified)

  ---------------------------------------------------------------------------------------------------

  INTERPRETABLE WEIGHT-SPACE PROGRAMMING (Transparent Compilation)
  
       Algorithmic Spec ------------> [ Analytic Circuit Synthesis ] ----> Transparent Fixed Weights \theta
       (e.g., Euclidean GCD, XOR)     (Exact closed-form weight copy)     (Zero training; 100% provable accuracy)
```

---

## Mechanics of Handcrafted Architectures

### 1. `XorNet`: Directional Difference Detectors
The classic XOR function ($x_1 \oplus x_2$) is linearly inseparable for single-layer perceptrons. Instead of fitting random weights through backpropagation, `XorNet` constructs two orthogonal directional detectors:

$$\begin{aligned}
h_1 &= \text{ReLU}(+1 \cdot x_1 - 1 \cdot x_2) = \max(0, x_1 - x_2) \\
h_2 &= \text{ReLU}(-1 \cdot x_1 + 1 \cdot x_2) = \max(0, x_2 - x_1)
\end{aligned}$$

The readout layer sums the two detector outputs:
$$y = 1 \cdot h_1 + 1 \cdot h_2 = |x_1 - x_2|$$

For binary inputs $x \in \{0, 1\}^2$:
- $(0, 0) \implies h_1=0, h_2=0 \implies y = 0$
- $(1, 0) \implies h_1=1, h_2=0 \implies y = 1$
- $(0, 1) \implies h_1=0, h_2=1 \implies y = 1$
- $(1, 1) \implies h_1=0, h_2=0 \implies y = 0$

**Weight Specification:**
$$W_{\text{detect}} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}, \quad W_{\text{readout}} = \begin{bmatrix} 1 & 1 \end{bmatrix}, \quad \text{bias} = \mathbf{0}$$
All parameters have `requires_grad = False`.

---

### 2. `GCDNet`: Iterative Euclidean Subtractive Block
To compute the Greatest Common Divisor $\gcd(a, b)$ of two positive integers, `GCDStep` implements one step of the subtractive Euclidean algorithm as a feed-forward tensor operation:
$$\text{step}(a, b) = \begin{bmatrix} |a - b| \\ \min(a, b) \end{bmatrix}$$

`GCDNet` unrolls this transition block across a fixed budget of $T$ iterations:
$$s_0 = [a, b], \quad s_{t+1} = \text{step}(s_t), \quad \lim_{t \to \infty} s_t = [\gcd(a, b), \gcd(a, b)]$$

Because this is implemented entirely in PyTorch tensor primitives (`torch.abs`, `torch.minimum`, `torch.stack`), it operates seamlessly in batch across arbitrary tensor shapes with zero Python control-flow overhead.

---

## Empirical Verification

Running `handwritten_models.py` evaluates these transparent models against synthetic test suites:

```bash
python3 handwritten_models.py
```

### Empirical Results:
- **`XorNet` Accuracy**: $100.0\%$ exact match across 500 synthetic binary pairs ($250$ identical, $250$ differing).
- **`GCDNet` Accuracy**: $100.0\%$ exact match across all $900$ cartesian pairs in the range $[1, 30] \times [1, 30]$ evaluated against `math.gcd`.
- **Arbitrary Inputs**: Correctly evaluates non-trivial integer inputs (e.g., $\gcd(84, 36) = 12.0$).

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/iwsp_companion_lab

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (pure PyTorch)
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The 4 automated tests verify:
- Complete XOR truth-table fidelity ($4/4$ combinations).
- Invariant check: zero trainable parameters (`requires_grad == False`).
- Non-trivial GCD evaluation correctness across disparate prime and composite pairs.
- Single-step Euclidean state transition fidelity.

---

## Weight-Space Interpretability Exercises

1. **Analytical Weight Perturbation**: Modify $W_{\text{detect}}[0, 1]$ from $-1.0$ to $-0.5$. Which input pairs produce erroneous outputs, and how does the decision boundary shift?
2. **Gradient-Free Mutation Search**: Implement a simple discrete proposal search that mutates integer/half-integer weight matrices to discover the XOR circuit without backpropagation.
3. **Multi-Input Generalization**: Extend `XorNet` to solve $N$-bit parity or majority-voting by compiling wider detection matrices.
4. **Euclidean Convergence Analysis**: For a given maximum input value $M$, what is the worst-case iteration depth $T$ required for `GCDNet` to converge? (Hint: consider consecutive Fibonacci pairs).

---

## File Structure

```
iwsp_companion_lab/
├── README.md               # Theoretical foundation, architecture diagrams, and exercises
├── requirements.txt        # PyTorch dependency
├── .gitignore              # Ignores __pycache__ and Python bytecode
├── handwritten_models.py   # XorNet and GCDNet implementations + benchmark evaluator
└── tests/
    └── test_models.py      # Automated unit tests (4 tests)
```

---

## References

- Karan Mehta, et al., *Interpretable weight-space programming builds transparent transformers and brain encoding models* (arXiv:2611.18114, 2026).
- Official Code Repository: [csinva/weight-space-programming-autoresearch](https://github.com/csinva/weight-space-programming-autoresearch).
- Related labs in this repository:
  - [`weight_space_programming_lab/`](../weight_space_programming_lab) — Direct weight inspection, permutation symmetries, and circuit compiling.
  - [`weight_space_reading_audit_lab/`](../weight_space_reading_audit_lab) — Auditing whether LLMs read weight space vs memorize labels.
