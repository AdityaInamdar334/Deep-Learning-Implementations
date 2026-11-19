# World Models vs Pixel Prediction: A Controlled Toy Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Ops-013243.svg)](https://numpy.org/)
[![Concept: JEPA World Models](https://img.shields.io/badge/Concept-JEPA%20%7C%20World%20Models-purple.svg)](#the-world-model-philosophy)
[![Tests Passing](https://img.shields.io/badge/tests-5%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational companion lab exploring the core architectural intuition behind Yann LeCun's **Joint Embedding Predictive Architecture (JEPA)** and latent-space world models versus auto-regressive or generative pixel prediction. 

This lab isolates the fundamental distinction: **predicting in representation space discards unpredictable, high-entropy background details to isolate predictable, action-conditioned transition dynamics**.

> [!IMPORTANT]
> **Scope & Educational Framing:** This lab is a controlled conceptual demonstration, **not a full implementation of I-JEPA, V-JEPA, or V-JEPA 2**. The simulator provides ground-truth compact state coordinates $(x, y)$. The transition operator learns the mapping $[s_t, a_t] \mapsto s_{t+1}$ via linear least-squares. Pixel persistence and structured conditional-expectation baselines illustrate why pixel-level mean-squared error (MSE) is dominated by irreducible visual entropy, even when world dynamics are $100\%$ predictable.

---

## The World Model Philosophy

In classical generative modeling (autoregressive video generation, diffusion models), systems attempt to predict future observations pixel-by-pixel:
$$\min_{\theta} \mathbb{E}\left[ \| x_{t+1} - G_\theta(x_t, a_t) \|^2 \right]$$

When the environment contains stochastic noise, moving leaves, water ripples, or random visual textures, generative models allocate tremendous representational capacity to modeling unpredictable details irrelevant to downstream agency.

In contrast, **Joint Embedding Predictive Architectures (JEPA)** predict in an abstract representation space:
$$\min_{\phi, \psi} \mathbb{E}\left[ D\left( s_{x_{t+1}}, \text{Predictor}_\psi(s_{x_t}, a_t) \right) \right], \quad \text{where } s_x = \text{Encoder}_\phi(x)$$

```
  GENERATIVE PIXEL PREDICTION (High Capacity Wasted on Irreducible Entropy)
  
       Pixel Frame x_t ------------------ Generator G(x_t, a_t) ------------------> Predicted Frame x_{t+1}
       [Dot + Noise]                             ^                                     [Predicted Dot + ??]
                                                 |                                              |
                                              Action a_t                                 Pixel MSE Loss
                                                                                                v
                                                                                   High Error on Background!

  ---------------------------------------------------------------------------------------------------------

  LATENT WORLD MODEL / JEPA PHILOSOPHY (Discards Irreducible Nuisance Details)
  
       Pixel Frame x_t ------------+
             |                     |
     [Feature Encoder]             |
             v                     v
       Latent State s_t ---> [ Transition Model W ] ---> Predicted Latent s_{t+1}
       (e.g., [x, y])              ^                             |
                                   |                       Latent State Loss
                                Action a_t                       v
                                                           Exact Zero Error!
```

---

## Lab Implementation & Mechanics

### 1. The Synthetic Observable World
- **Grid Size**: $24 \times 24$ float32 frames.
- **Controllable Agent**: A single bright point ($1.0$) initialized at position $p_t = (x_t, y_t)$.
- **Action Space**: $5$ discrete cardinal actions:
  $$\mathcal{A} = \{ [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0] \}$$
- **Irreducible Nuisance Variation**: Background pixels are independently resampled at every timestep from a uniform noise distribution:
  $$x_{t}[i, j] \sim \mathcal{U}(0, \sigma_{\text{noise}})$$
  Because the background noise is statistically independent across timesteps, **exact pixel-level prediction of the next frame is mathematically impossible**.

### 2. Latent Transition Estimation
A linear least-squares design matrix combines position coordinates, one-hot encoded actions, and an affine bias:
$$X_{\text{design}} = [s_t, \text{onehot}(a_t), \mathbf{1}] \in \mathbb{R}^{N \times (2 + 5 + 1)}$$
$$W = (X_{\text{design}}^T X_{\text{design}})^{-1} X_{\text{design}}^T s_{t+1}$$

### 3. Baseline Comparisons
1. **Pixel Persistence Baseline**: Assumes no change ($x_{t+1} \approx x_t$).
2. **Structured Conditional Expectation**: Forecasts background pixels at their conditional mean ($\mathbb{E}[noise] = \sigma_{\text{noise}} / 2$) while placing the agent at the predicted latent location:
   $$\hat{x}_{\text{structured}}[y, x] = \begin{cases} 1.0 & \text{if } (y, x) = \text{round}(\hat{s}_{t+1}) \\ \frac{\sigma_{\text{noise}}}{2} & \text{otherwise} \end{cases}$$
3. **Latent Coordinate Accuracy**: Measures mean absolute error (MAE) between predicted and actual coordinates:
   $$\text{MAE}_{\text{latent}} = \frac{1}{N} \sum_{i=1}^N \| \hat{s}_{t+1}^{(i)} - s_{t+1}^{(i)} \|_1$$

### 4. Latent-Space Greedy Planning
Given a start coordinate $s_{\text{start}}$ and a goal coordinate $s_{\text{goal}}$, a greedy planner iterates forward in latent space without ever rendering intermediate pixel frames:
$$a_t^* = \arg\min_{a \in \mathcal{A}} \| \text{Predict}(s_t, a; W) - s_{\text{goal}} \|_2$$

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/world-models-jepa-lab

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (pure NumPy)
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The 5 automated tests verify:
- Data generation reproducibility and determinism across seeds.
- Transition action rule consistency ($s_{t+1} - s_t = \text{ACTION}[a]$).
- Exact linear least-squares transition fitting ($\text{MAE} < 10^{-8}$).
- Pixel prediction error hierarchy ($\text{MSE}_{\text{persistence}} > \text{MSE}_{\text{structured}}$).
- Latent greedy planning convergence to goal within specified horizon.

---

## Running the Experiment

Run the evaluation script to train the transition model, evaluate metrics, and execute a latent trajectory plan:

```bash
python3 toy_world.py --output results --seed 3
```

### Measured Baseline Results

Sample output from `results/metrics.json`:

```json
{
  "seed": 3,
  "train_samples": 1000,
  "test_samples": 200,
  "background_noise_max": 0.25,
  "latent_position_mae": 9.48e-15,
  "rounded_position_accuracy": 1.0,
  "pixel_persistence_mse": 0.01273,
  "structured_pixel_prediction_mse": 0.00521,
  "interpretation": "oracle states from simulator; not learned visual encoder or JEPA"
}
```

### Key Empirical Findings:
1. **Machine-Precision Latent Dynamics**: In the 2D coordinate representation, the action transition model achieves zero test error ($\text{MAE} \approx 10^{-14}$), enabling flawless planning.
2. **Persistent Pixel Uncertainty**: In pixel space, even an oracle that knows the exact foreground destination cannot reduce MSE below $\approx 0.0052$, which is the theoretical variance of the background noise ($\text{Var}(\mathcal{U}(0, 0.25)) = \frac{0.25^2}{12} \approx 0.00521$).

---

## File Structure

```
world-models-jepa-lab/
├── README.md               # Architecture documentation, JEPA philosophy, and formulas
├── requirements.txt        # NumPy dependency
├── .gitignore              # Ignores __pycache__ and bytecode
├── toy_world.py            # Simulation environment, transition fitting, evaluation, and planner
├── results/
│   └── metrics.json        # Output evaluation metrics and planned path trace
└── tests/
    └── test_toy.py         # 5 automated unit tests
```

---

## Conceptual Extensions & Research Prompts

1. **Non-Action Observability**: Remove action conditioning. How does the transition model behave when motion is autonomous or unobserved?
2. **Multi-Object Dynamics & Occlusion**: Introduce two interacting objects. How do collisions or visual occlusions break linear transitions?
3. **Learned Visual Representation**: Replace oracle $(x, y)$ coordinates with an autoencoder or convolutional encoder, and introduce regularization to prevent representation collapse.
4. **Stochastic & Correlated Noise**: Switch background noise from i.i.d. to Ornstein-Uhlenbeck or perlin noise to evaluate how temporal correlation affects pixel prediction baselines.

---

## References

- Yann LeCun, *A Path Towards Autonomous Machine Intelligence* (OpenReview, 2022).
- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas, *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (CVPR 2023), [arXiv:2301.08243](https://arxiv.org/abs/2301.08243).
- Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas, *Revisiting Feature Prediction for Learning Visual Representations from Video* (2024), [arXiv:2404.08471](https://arxiv.org/abs/2404.08471).
- Mahmoud Assran et al., *V-JEPA 2* (2025), [arXiv:2506.09985](https://arxiv.org/abs/2506.09985).
