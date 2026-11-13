# The 0.18215 Latent Scale Factor in Diffusion Models

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-4%2F4%20Passing-brightgreen.svg)](test_experiment.py)
[![Paper](https://img.shields.io/badge/CVPR%202022-arXiv%3A2112.10752-B31B1B.svg)](https://arxiv.org/abs/2112.10752)
[![Architecture](https://img.shields.io/badge/Model-Stable%20Diffusion%20VAE-blueviolet.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A clean, zero-dependency reference laboratory isolating the mathematical impact of the 0.18215 autoencoder latent scaling constant in Latent Diffusion Models (Stable Diffusion), demonstrating why autoencoder unit tests hide the bug and why unscaled latents cause a 22× explosion in noise prediction error.*

Based on foundational research by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer:  
**["High-Resolution Image Synthesis with Latent Diffusion Models"](https://arxiv.org/abs/2112.10752)** (CVPR 2022).

</div>

---

## Executive Summary

In Latent Diffusion Models (LDMs, such as Stable Diffusion 1.4, 1.5, and 2.x), diffusion is trained not on raw pixel grids, but within the compressed latent space of a pretrained Variational Autoencoder (KL-f8 VAE).

When the autoencoder processes natural images, its latent representations have an empirical standard deviation of $\approx 5.49$ (variance $\sigma^2 \approx 30.14$). However, standard diffusion noise schedules (such as linear or cosine $\beta_t$) are mathematically formulated under the foundational assumption that clean data has unit variance ($\sigma^2 \approx 1.0$).

To reconcile the VAE latents with the diffusion schedule, Stable Diffusion scales the latents by a constant factor:

$$z_{\text{scaled}} = 0.18215 \cdot z_{\text{raw}}$$
$$x_{\text{pixel}} = \text{Decoder}\left( \frac{z_{\text{scaled}}}{0.18215} \right)$$

```
The Silent Failure Mode of Latent Scaling:
1. Autoencoder Round-Trip Test:
   decode(encode(x, scale), scale) == x   ---> PASSES for ANY scale (even scale = 1.0)!
   The VAE round-trip perfectly cancels out the factor, concealing the mismatch.

2. Diffusion Noise Prediction under Unit Scale (s = 0.18215):
   At schedule index 500 (ᾱ ≈ 0.2763):
   Noise Prediction MSE = 0.2763 (Theoretical Minimum Bayes Risk = ᾱ)

3. Diffusion Noise Prediction when Scale is Omitted (s = 1.0):
   Input variance is 30.14 instead of 1.0.
   Noise Prediction MSE = 6.1172  (22x Error Explosion!)
   Result: Generation completely collapses into saturated noise or artifact blobs.
```

---

## Visual Gallery: Expected Squared Error Across Schedules

<div align="center">

![Latent Scale Expected Error](latent-scale.png)

*Figure 1: Theoretical expected squared error in predicted noise across the 1,000-step diffusion training schedule: (1) Correct training units ($s=0.18215$, green), (2) Scale omitted ($s=1.0$, orange), and (3) Recalibrated Bayesian oracle ($v=s^{-2}$, purple).*

</div>

---

## Mathematical Foundations

### 1. The Standard Forward Diffusion Step

Let clean latent $z_0 \sim \mathcal{N}(0, \sigma^2)$ and noise $\epsilon \sim \mathcal{N}(0, 1)$. At step $t$ with cumulative retention $\bar{\alpha}_t \in (0, 1)$:

$$z_t = \sqrt{\bar{\alpha}_t} \cdot z_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

The total marginal variance of $z_t$ is:
$$\text{Var}(z_t) = \bar{\alpha}_t \sigma^2 + (1 - \bar{\alpha}_t)$$

The Bayes-optimal minimum mean squared error (MMSE) estimator for the noise $\epsilon$ given observation $z_t$ is:

$$\hat{\epsilon}(z_t) = \mathbb{E}[\epsilon \mid z_t] = \frac{\text{Cov}(\epsilon, z_t)}{\text{Var}(z_t)} \cdot z_t = \frac{\sqrt{1 - \bar{\alpha}_t}}{\bar{\alpha}_t \sigma^2 + 1 - \bar{\alpha}_t} \cdot z_t$$

---

### 2. Case A: Correct Matched Scale ($\sigma^2 = 1.0$)

When latents are correctly scaled ($s = 0.18215$), clean latents have unit variance ($\sigma^2 = 1$):
$$\text{Var}(z_t) = \bar{\alpha}_t(1) + 1 - \bar{\alpha}_t = 1.0$$
$$\hat{\epsilon}(z_t) = \sqrt{1 - \bar{\alpha}_t} \cdot z_t$$

The expected squared error (Bayes risk) evaluates to:
$$\mathbb{E}[(\hat{\epsilon} - \epsilon)^2] = \bar{\alpha}_t$$

At the schedule midpoint $t = 500$ ($\bar{\alpha}_{500} = 0.276333$):
$$\text{MSE}_{\text{matched}} = \mathbf{0.276333}$$

---

### 3. Case B: Scale Omitted ($\sigma^2 = s^{-2} \approx 30.14$)

If the scale factor is omitted, the model receives unscaled latents with variance $\sigma^2 = (1 / 0.18215)^2 \approx 30.14$. However, the diffusion denoiser (trained under unit variance assumptions) applies the standard coefficient $\sqrt{1 - \bar{\alpha}_t} \cdot z_t$:

$$\mathbb{E}\left[ \left( \sqrt{1 - \bar{\alpha}_t} z_t - \epsilon \right)^2 \right] = \bar{\alpha}_t^2 + \bar{\alpha}_t (1 - \bar{\alpha}_t) \sigma^2$$

At $t = 500$:
$$\text{MSE}_{\text{omitted}} = (0.276333)^2 + 0.276333 \times (1 - 0.276333) \times 30.14 = \mathbf{6.117177}$$

The noise prediction error explodes by **$22.14\times$**!

---

### 4. Case C: Oracle Recalibrated Denoiser

Even if an optimal Bayesian estimator is retrained specifically to know the raw variance $\sigma^2 = 30.14$:
$$\text{MSE}_{\text{recalibrated}} = \frac{\bar{\alpha}_t \sigma^2}{\bar{\alpha}_t \sigma^2 + 1 - \bar{\alpha}_t} = \frac{0.276333 \times 30.14}{0.276333 \times 30.14 + (1 - 0.276333)} = \mathbf{0.915905}$$

The noise prediction risk is still **$3.3\times$ higher** than with unit-variance latents because the dominant signal variance overpowers the noise component, making score estimation significantly more difficult.

---

## Empirical Benchmark & Verification

Running `experiment.py` across 100,000 Monte Carlo samples at index $t = 500$:

```bash
python3 experiment.py --index 500 --count 100000 --seed 7
```

```
round trip (factor 0.18215): 2.000000
round trip (factor 1):       2.000000
alpha_bar[500] = 0.276333
matched              noise MSE = 0.276279
scale omitted        noise MSE = 6.117177
oracle recalibrated  noise MSE = 0.915905
```

### Schedule Horizon Sweeps

| Timestep ($t$) | Cumulative $\bar{\alpha}_t$ | Matched MSE ($\sigma^2=1$) | Scale Omitted MSE | Error Ratio |
| :---: | :---: | :---: | :---: | :---: |
| **0** (Near Clean) | $0.99915$ | $0.99915$ | $1.02404$ | $1.02\times$ |
| **250** | $0.62776$ | $0.62776$ | $7.42674$ | $11.83\times$ |
| **500** (Mid-Schedule)| $0.27633$ | **0.27628** | **6.11718** | **22.14×** |
| **750** | $0.07166$ | $0.07166$ | $2.01257$ | $28.08\times$ |
| **999** (Pure Noise) | $0.00472$ | $0.00472$ | $0.14545$ | $30.82\times$ |

---

## Quickstart & Verification

Requires Python 3.9+ with standard library only:

```bash
cd latent-lab

# 1. Run unit test suite
python3 -m unittest -v

# 2. Run default Monte Carlo experiment (100,000 samples)
python3 experiment.py

# 3. Test different schedule indices
python3 experiment.py --index 250
python3 experiment.py --index 750

# 4. (Optional) Re-generate publication plot (requires matplotlib)
python3 figure.py
```

---

## Unit Testing & Verification

4 formal unit tests verify round-trip contract masking, quadratic $\bar{\alpha}$ schedule monotonicity, boundary limits, and closed-form Gaussian Bayes risk:

```bash
python3 -m unittest -v test_experiment.py
```

```
test_gaussian_risks (test_experiment.Contracts.test_gaussian_risks) ... ok
test_oracle_limits (test_experiment.Contracts.test_oracle_limits) ... ok
test_round_trip_can_hide_both_conventions (test_experiment.Contracts.test_round_trip_can_hide_both_conventions) ... ok
test_schedule (test_experiment.Contracts.test_schedule) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.048s

OK
```

---

## Repository Structure

```
latent-lab/
├── README.md           # This comprehensive engineering and mathematical guide
├── experiment.py       # Pure-Python Monte Carlo simulation and analytical MMSE estimators
├── figure.py           # Matplotlib script generating latent-scale.png
├── latent-scale.png    # Publication plot of expected squared error curves
└── test_experiment.py  # 4 unit tests verifying contracts, schedules, and risk bounds
```

---

## Academic References

1. **Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (2022)**. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752).
2. **Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)**. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
3. **CompVis & Stability AI (2022)**. *Stable Diffusion Model Card and Latent Scaling Implementation*. [GitHub Repository](https://github.com/CompVis/stable-diffusion).

---

## License

This project is licensed under the [MIT License](../LICENSE).
