# Neural Network from Scratch in C: XOR Lab

[![Language: C11](https://img.shields.io/badge/Language-C11-00599C.svg)](https://en.wikipedia.org/wiki/C11_(C_standard_revision))
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Pure%20C%20Stdlib)-success.svg)](#requirements--build)
[![Architecture: MLP](https://img.shields.io/badge/Architecture-2--4--1%20MLP-purple.svg)](#network-architecture)
[![Gradients](https://img.shields.io/badge/Gradients-Analytic%20Backprop%20%2B%20Finite%20Diff-orange.svg)](#mathematical-derivation)
[![Tests Passing](https://img.shields.io/badge/tests-passing%20(make%20test)-brightgreen.svg)](#running-the-test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-contained, dependency-free neural network written in standard C11 from scratch, with zero external libraries. Inspired by Richard van der Oost's `mat.c` pedagogical series, this lab builds a fully functional 2–4–1 Multilayer Perceptron (MLP) that solves the non-linear XOR classification problem using both **analytic backpropagation** and **central finite difference numerical differentiation**.

---

## Network Architecture

```
                      INPUT LAYER               HIDDEN LAYER                 OUTPUT LAYER
                      (2 Inputs + Bias)         (4 Sigmoid Units + Bias)     (1 Sigmoid Unit)

                         [ x_0 ] ------------------ ( h_0 )
                                \                 /        \
                                 \               /          \
                         [ x_1 ] ------------------ ( h_1 )  \
                                 / \           /  \           +---> ( y_hat )
                                /   \         /    \         /
                               /     \       /      \       /
                      [ bias = 1.0 ] ------------- ( h_2 ) /
                                      \     /      /
                                       \   /      /
                                        \ /      /
                                         X ----- ( h_3 )
                                        /
                              [ bias = 1.0 ]

Dimensions:           X in R^(4x2)              Z0 in R^(4x4)                Z1 in R^(4x1)
Augmented:            A0 in R^(4x3)             A1 in R^(4x5)                P  in R^(4x1)
Weights:                       W0 in R^(3x4)                 W1 in R^(5x1)
Parameters:                    12 weights + biases           5 weights + biases  ==> Total 17 Parameters
```

---

## Mathematical Derivation

### 1. Forward Pass
Given input batch $X \in \mathbb{R}^{4 \times 2}$ and binary targets $Y \in \{0, 1\}^{4 \times 1}$:
1. **Input Augmentation**: Append bias column of ones:
   $$A_0 = [X \mid \mathbf{1}] \in \mathbb{R}^{4 \times 3}$$
2. **Hidden Pre-Activation**:
   $$Z_0 = A_0 W_0 \in \mathbb{R}^{4 \times 4}, \quad \text{where } W_0 \in \mathbb{R}^{3 \times 4}$$
3. **Hidden Activation & Bias Augmentation**:
   $$A_1 = [\sigma(Z_0) \mid \mathbf{1}] \in \mathbb{R}^{4 \times 5}$$
   where $\sigma(x)$ is the numerically stabilized sigmoid:
   $$\sigma(x) = \begin{cases} \frac{1}{1 + e^{-x}} & \text{if } x \ge 0 \\ \frac{e^x}{1 + e^x} & \text{if } x < 0 \end{cases}$$
4. **Output Prediction**:
   $$Z_1 = A_1 W_1 \in \mathbb{R}^{4 \times 1}, \quad P = \sigma(Z_1) \in \mathbb{R}^{4 \times 1}$$
5. **Loss (Half Mean Squared Error)**:
   $$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^N (P_i - Y_i)^2$$

---

### 2. Backward Pass (Analytic Backpropagation)
Using the chain rule with sigmoid derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:
1. **Output Error Term**:
   $$\delta_1 = \frac{\partial \mathcal{L}}{\partial Z_1} = \frac{1}{N} (P - Y) \odot P \odot (1 - P) \in \mathbb{R}^{4 \times 1}$$
2. **Output Weight Gradient**:
   $$G_1 = \frac{\partial \mathcal{L}}{\partial W_1} = A_1^T \delta_1 \in \mathbb{R}^{5 \times 1}$$
3. **Hidden Error Term** (backpropagated through active hidden units, excluding the output bias):
   $$\delta_0 = (\delta_1 W_{1, :H}^T) \odot A_{1, :H} \odot (1 - A_{1, :H}) \in \mathbb{R}^{4 \times 4}$$
4. **Hidden Weight Gradient**:
   $$G_0 = \frac{\partial \mathcal{L}}{\partial W_0} = A_0^T \delta_0 \in \mathbb{R}^{3 \times 4}$$

---

### 3. Central Finite Difference Verification
To verify the analytical gradient derivation without trusting the chain-rule code, each parameter $w_k$ is perturbed independently:
$$\frac{\partial \mathcal{L}}{\partial w_k} \approx \frac{\mathcal{L}(w_k + \epsilon) - \mathcal{L}(w_k - \epsilon)}{2\epsilon} + \mathcal{O}(\epsilon^2)$$
With $\epsilon = 10^{-5}$, the maximum absolute discrepancy between analytical backpropagation and finite differences is verified to be within machine precision:
$$\max |G_{\text{analytic}} - G_{\text{numerical}}| < 10^{-11}$$

---

## Requirements & Build

Requires any standard C11 compiler (`cc`, `clang`, or `gcc`) and `make`.

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/c_xor_neural_network_lab
```

### Build Targets

| Command | Action | Description |
|:---|:---|:---|
| `make test` | Runs unit tests | Tests matrix mult, stride-safe slicing, and gradient checking |
| `make run` | Trains the model | Full training using analytic backpropagation ($\approx 0.05$s) |
| `make finite` | Trains with finite diffs | Evaluates $2 \times 17$ forward passes per epoch (demonstrates numerical optimization) |
| `make clean` | Cleans artifacts | Removes compiled `xor` binary |

---

## Running the Test Suite

```bash
make test
```

Output:
```
cc -std=c11 -O2 -Wall -Wextra -Wpedantic main.c -o xor -lm
./xor test
gradient check: max absolute error 2.48e-12
tests passed
```

You can also run the automated Python test runner:
```bash
python3 -m unittest discover -s tests -v
```

---

## Training Demonstration

```bash
make run
```

Output:
```
./xor
gradient check: max absolute error 2.48e-12
epoch     0 loss 0.16182112
epoch  2500 loss 0.00062423
epoch  5000 loss 0.00025601
epoch  7500 loss 0.00015782
epoch 10000 loss 0.00011316
epoch 12500 loss 0.00008783
epoch 15000 loss 0.00007158
epoch 17500 loss 0.00006030
final loss 0.00005202
0 XOR 0 -> 0.01108 (class 0, target 0)
0 XOR 1 -> 0.99066 (class 1, target 1)
1 XOR 0 -> 0.98921 (class 1, target 1)
1 XOR 1 -> 0.00947 (class 0, target 0)
```

The network converges to near-zero loss ($\mathcal{L} \approx 5.2 \times 10^{-5}$) and achieves $100\%$ classification accuracy across all four truth-table cases.

---

## Code Structure

```
c_xor_neural_network_lab/
├── Makefile            # Standard C11 build automation (all, test, run, finite, clean)
├── README.md           # Architecture, math derivations, and verification guide
├── .gitignore          # Ignores compiled binaries and object files
├── main.c              # Complete C11 source: matrix engine, forward pass, autograd, and CLI
└── tests/
    └── test_c_build.py # Automated Python test wrapper verifying make test
```

---

## Key Educational Takeaways

1. **Memory Contiguity & Strides**: 2D matrices are allocated as flat 1D contiguous arrays (`rows * cols * sizeof(double)`). Access is parameterized via the macro `AT(m, r, c)` implementing row-major indexing: `(r * cols) + c`.
2. **Zero-Allocation Bias Injection**: Instead of tracking separate weight vectors and bias vectors, the `copy_top_left` routine copies an $N \times D$ data matrix into the top-left subregion of an $N \times (D+1)$ matrix whose final column is pre-filled with $1.0$. This unifies weight and bias updates into single matrix multiplications.
3. **Numerically Stable Sigmoid**: Exponentiating large positive numbers triggers floating-point overflow. The branching sigmoid evaluates $1 / (1 + e^{-x})$ for $x \ge 0$ and $e^x / (1 + e^x)$ for $x < 0$, guaranteeing strict numerical stability across $[-\infty, \infty]$.
4. **Gradient Checking**: Comparing symbolic analytic derivatives against central numerical approximations is the gold standard for verifying hand-coded backpropagation routines before training.

---

## References

- Richard van der Oost, *Neural Network in C* (`mat.c` pedagogical series).
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning* (MIT Press, 2016), Chapter 6: Deep Feedforward Networks.
- Rumelhart, Hinton, & Williams, *Learning representations by back-propagating errors*, Nature 323, 533–536 (1986).
