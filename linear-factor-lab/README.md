# Linear Factor Models from Scratch

Companion to the article on reverse-engineering learned representations, based on Chapter 13 of Goodfellow, Bengio and Courville, *Deep Learning* (2016). CPU-only, synthetic data, no downloads or API keys beyond dependencies.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python lab.py
python -m unittest -v
```

On Windows activate with `.venv\Scripts\activate`. Tested with NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, PyTorch 2.10.0+cpu, Matplotlib 3.10.8. The requirements allow compatible versions; exact numeric results can vary.

## Experiments

1. PCA: train on 4,000 examples, test on 2,000. Independent high-variance noise and a binary predictive feature. Compare logistic regression on full inputs and a single training-fitted PC.
2. PPCA and factor analysis: 5,000 training and 2,000 test samples, eight sensors, two Gaussian latent factors, heterogeneous noise. Compare mean held-out log likelihood (higher is better). PPCA is implemented by covariance eigendecomposition; FA uses scikit-learn.
3. ICA: train on 4,000 noiseless mixtures of Laplace and uniform sources, test on 2,000. Match sources by absolute correlation using the Hungarian algorithm; signs and permutation are not identifiable.
4. SFA: generalized eigendecomposition, fit on first 4,000 time steps and test on next 2,000. Compare slow-factor correlation with PCA.
5. ISTA: known normalized 16-by-32 dictionary, three nonzero coefficients per sample, noisy observations. Evaluate five penalties on 400 observations. This is inference with an oracle dictionary, not dictionary discovery.
6. Dictionary learning: fit a separate dictionary on 800 observations; evaluate sparse inference on the 400 held-out observations.
7. Sparse autoencoder: 16 input dimensions, 32 ReLU features, normalized decoder columns, three seeds (11,22,33), three sparsity weights, 60 epochs each, Adam, batch size 128, 3,000 train and 500 test observations per seed. Compare active features and held-out MSE. Not an LLM activation experiment; semantic feature recovery is not established.
8. Identifiability: verify that an invertible change of latent coordinates and inverse decoder transformation preserve reconstruction.

## Files

`lab.py` implements and runs all experiments. `test_lab.py` tests PPCA posterior, SFA, ISTA and source matching. `results/results.json` and `results/run.log` contain actual observations; `results/tests.log` contains test output. `figures/` contains measured plots. `article.md` is the previous article; `ARTICLE-UPDATE.md` provides new measured results and insertion instructions.

## Exercises

Change nuisance variance, replace heterogeneous noise with isotropic noise, replace ICA sources with two Gaussians, vary SFA time frequencies, change dictionary size and ISTA penalty, and explore SAE width and sparsity. Keep the test partition fixed and use validation data for any tuning. These demonstrations are not general benchmark claims; most experiments use one seed and known synthetic generators. No real speech, EEG, or LLM activations are used.

## Reading

- https://www.deeplearningbook.org/contents/linear_factors.html
- https://doi.org/10.1111/1467-9868.00196 (PPCA)
- https://doi.org/10.1016/S0893-6080(00)00026-5 (ICA)
- https://doi.org/10.1162/089976602317318938 (SFA)
- https://doi.org/10.1038/381607a0 (sparse coding)
- https://arxiv.org/abs/2406.04093 (SAEs)
