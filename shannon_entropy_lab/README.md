# Shannon Entropy, Optimal Prefix Coding & Language Modeling

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-pure%20standard%20library-success.svg)](#quickstart)
[![Domain: Information Theory](https://img.shields.io/badge/Domain-Information%20Theory-purple.svg)](#theoretical-foundations)
[![Algorithms: Huffman & Cross-Entropy](https://img.shields.io/badge/Algorithms-Huffman%20%7C%20Cross--Entropy-orange.svg)](#mathematical-formulation)
[![Tests Passing](https://img.shields.io/badge/tests-6%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, dependency-free reference implementation of Claude Shannon's foundational **Information Theory** principles. Implemented entirely in the Python standard library, this lab explores:

1. **Shannon Entropy**: The mathematical lower bound on lossless data compression.
2. **Huffman Prefix-Free Coding**: Optimal variable-length code construction achieving the Shannon bound.
3. **Cross-Entropy & KL Divergence**: The geometric cost of using an inaccurate probability model (the foundation of modern LLM loss functions).
4. **Conditional Entropy in Language Modeling**: Demonstrating that conditioning on preceding context ($n$-grams) strictly reduces sequence entropy.

---

## Theoretical Foundations

In his seminal 1948 paper *A Mathematical Theory of Communication*, Claude Shannon proved that the fundamental limit of compression for an information source emitting symbols $x \in \mathcal{X}$ with probabilities $P(x)$ is given by its **entropy**:

$$H(P) = -\sum_{x \in \mathcal{X}} P(x) \log_2 P(x) \quad \text{(bits per symbol)}$$

```
                              SHANNON SOURCE CODING THEOREM
                              
     Source Distribution P(x)                      Mismatched Model Q(x)
     ['UP': 0.50, 'DOWN': 0.25,                    ['UP': 0.25, 'DOWN': 0.25,
      'LEFT': 0.125, 'RIGHT': 0.125]                'LEFT': 0.25, 'RIGHT': 0.25]
                  |                                              |
                  v                                              v
      +------------------------+                     +------------------------+
      |  Optimal Huffman Tree  |                     |  Suboptimal Allocation |
      |  UP    -> 0      (1 b) |                     |  UP    -> 00     (2 b) |
      |  DOWN  -> 10     (2 b) |                     |  DOWN  -> 01     (2 b) |
      |  LEFT  -> 110    (3 b) |                     |  LEFT  -> 10     (2 b) |
      |  RIGHT -> 111    (3 b) |                     |  RIGHT -> 11     (2 b) |
      +------------------------+                     +------------------------+
                  |                                              |
                  v                                              v
         Expected Length L(C)                           Cross-Entropy H(P, Q)
             = 1.75 bits                                    = 2.00 bits
                  |                                              |
                  +----------------------+-----------------------+
                                         |
                                         v
                         Penalty: D_KL(P || Q) = 0.25 bits
```

---

## Mathematical Formulation

### 1. The Shannon Entropy Bound
For a 4-command navigation alphabet with skewed distribution:
$$P(\text{UP}) = \frac{1}{2}, \quad P(\text{DOWN}) = \frac{1}{4}, \quad P(\text{LEFT}) = \frac{1}{8}, \quad P(\text{RIGHT}) = \frac{1}{8}$$

A naïve fixed-length code requires $\lceil \log_2(4) \rceil = 2$ bits per instruction (`00`, `01`, `10`, `11`). However, the theoretical information content is strictly lower:
$$H(P) = -\left(\frac{1}{2}(-1) + \frac{1}{4}(-2) + \frac{1}{8}(-3) + \frac{1}{8}(-3)\right) = \frac{1}{2} + \frac{1}{2} + \frac{3}{8} + \frac{3}{8} = 1.75 \text{ bits/symbol}$$

### 2. Huffman Coding Tree (Exact Shannon Equivalence)
Because all probabilities are negative powers of two ($P(x) = 2^{-\ell_x}$), the optimal codeword lengths $\ell(x) = -\log_2 P(x)$ are exact integers:
- $\text{UP} \implies \ell = 1 \text{ bit} \implies \texttt{"0"}$
- $\text{DOWN} \implies \ell = 2 \text{ bits} \implies \texttt{"10"}$
- $\text{LEFT} \implies \ell = 3 \text{ bits} \implies \texttt{"110"}$
- $\text{RIGHT} \implies \ell = 3 \text{ bits} \implies \texttt{"111"}$

The expected codeword length exactly matches the theoretical minimum:
$$\mathbb{E}[\text{Length}] = \sum_{x \in \mathcal{X}} P(x) \ell(x) = \frac{1}{2}(1) + \frac{1}{4}(2) + \frac{1}{8}(3) + \frac{1}{8}(3) = 1.75000 \text{ bits}$$

### 3. Cross-Entropy & Gibbs' Inequality
When a coder assigns codeword lengths based on an assumed model $Q(x)$ rather than true distribution $P(x)$, the expected bit rate is the **cross-entropy**:
$$H(P, Q) = -\sum_{x} P(x) \log_2 Q(x) = H(P) + D_{\text{KL}}(P \parallel Q)$$
By Gibbs' Inequality, $H(P, Q) \ge H(P)$, with equality if and only if $Q(x) \equiv P(x)$. In machine learning, minimizing cross-entropy is equivalent to minimizing the Kullback-Leibler divergence between the data distribution $P$ and the neural network's predictions $Q_\theta$.

### 4. Conditioning Reduces Entropy
For a sequence of characters $X_1, X_2, \dots, X_T$:
$$H(X_t \mid X_{t-1}) \le H(X_t)$$
Conditioning on preceding context eliminates uncertainty. This lab demonstrates that an autoregressive bigram model compresses repetitive text into **$1.14$ bits/char**, down from **$3.56$ bits/char** under an unconditional unigram model.

---

## Empirical Benchmark Results

Running `entropy_lab.py` executes a full 10,000-symbol Monte Carlo simulation and text prediction trial:

| Metric | Measured Value | Theoretical Value | Notes |
|:---|:---:|:---:|:---|
| **Fixed-Length Code** | `2.00000` bits | $2.00000$ bits | Uncompressed baseline |
| **Shannon Entropy $H(P)$** | `1.75000` bits | $1.75000$ bits | Theoretical compression limit |
| **Expected Huffman Code** | `1.75000` bits | $1.75000$ bits | **$12.5\%$ lossless compression savings** |
| **Sampled Stream Average** | `1.74720` bits | $1.75000$ bits | $N=10,000$ empirical sample |
| **Roundtrip Reversibility** | `True` | `True` | Lossless prefix-free decode |
| **Cross-Entropy $H(P, Q_{\text{uniform}})$** | `2.00000` bits | $2.00000$ bits | $+0.25$ bit penalty for wrong prior |
| **Unigram Character Model** | `3.561` bits/char | — | In-sample vocabulary baseline |
| **Bigram Conditioned Model** | `1.140` bits/char | — | **$68.0\%$ entropy reduction** via context |

---

## Quickstart

### 1. Execution (Zero Dependencies)

```bash
# Clone repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/shannon_entropy_lab

# Run the benchmark (Pure Python 3.9+)
python3 entropy_lab.py
```

### 2. Run Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The 6 automated unit tests verify:
- Analytical Shannon entropy calculation on uniform, skewed, and deterministic distributions.
- Strict prefix-free property of the generated Huffman tree ($\forall i \neq j, c_i \not\sqsubseteq c_j$).
- Complete lossless roundtrip bitstream encoding and decoding.
- Truncated bitstream detection and exception handling.
- Gibbs' Inequality: $H(P, Q) \ge H(P)$ and $H(P, P) \equiv H(P)$.
- Information reduction via conditioning: $H(X_t \mid X_{t-1}) < H(X_t)$.

---

## File Structure

```
shannon_entropy_lab/
├── README.md               # Theoretical derivations, ASCII diagrams, and empirical results
├── requirements.txt        # Pure standard library specification
├── .gitignore              # Ignores __pycache__ and bytecode
├── entropy_lab.py          # Entropy, Huffman tree, bitstream codec, and n-gram LM
└── tests/
    └── test_entropy.py     # 6 automated unit tests
```

---

## References

- Claude E. Shannon, *A Mathematical Theory of Communication*, Bell System Technical Journal, 27(3): 379–423 (1948).
- Claude E. Shannon, *Prediction and Entropy of Printed English*, Bell System Technical Journal, 30(1): 50–64 (1951).
- David A. Huffman, *A Method for the Construction of Minimum-Redundancy Codes*, Proceedings of the IRE, 40(9): 1098–1101 (1952).
- 3Blue1Brown, *Reinventing Entropy* (2026).
- Thomas M. Cover & Joy A. Thomas, *Elements of Information Theory*, 2nd Edition, Wiley-Interscience (2006).
