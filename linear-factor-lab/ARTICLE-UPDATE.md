# Measured companion experiment updates

The previous article uses an analytical counterexample. Keep its population calculation (100/101 = 99.01%) distinct from the new finite-sample measurement (99.02%).

## Suggested headline and opening

# PCA Kept 99% of My Data's Variance. My Classifier Went From 100% Accuracy to a Coin Toss.

*One controlled experiment exposes the difference between preserving data and preserving useful information, from classical factor models to sparse autoencoders.*

I generated two independent features: high-variance Gaussian noise and a binary feature that perfectly determines the target. A logistic regression trained on both features achieved 100% accuracy on 2,000 held-out observations. Then I fitted PCA on the training data and kept only its first principal component. It explained 99.02% of training variance. A new logistic regression trained on that component scored just 50.45% on the same held-out set. The algorithm did exactly what it was designed to do. The objective was simply different from ours.

## Figure placement and captions

- `pca_failure.png`: Section 3, immediately after the variance counterexample (or after the new opening). Training variance retained and held-out accuracy before and after projecting onto one principal component. Separate classifiers trained on each representation.
- `noise.png`: Section 5, after the factor analysis code. True and fitted sensor noise variances for eight synthetic sensors. Held-out mean log likelihood: PPCA -14.2106, FA -11.2741 nats per observation. The generator matches FA assumptions.
- `ica.png`: Section 6, after source recovery code. Mean matched absolute source correlation: PCA 0.81806, FastICA 0.99982 on 2,000 held-out noiseless linear mixtures. Not a speech benchmark.
- `sfa.png`: Section 7, after generalized eigenvalue implementation. Slow factor and rescaled SFA output on held-out time steps. Absolute correlation: PCA 0.08084, SFA 0.99580.
- `sparse_tradeoff.png`: Section 9, after ISTA code. Known dictionary inference; increasing L1 penalty from 0.01 to 1.0 increased fraction of zero coefficients from 49.14% to 96.92%, and increased reconstruction MSE from 0.00008447 to 0.09248. A separately learned dictionary at alpha 0.1 gave MSE 0.002512 and 85.67% zeros.
- `sae.png`: Section 10, after PyTorch implementation. Three-seed mean held-out reconstruction MSE versus mean active features across three sparsity weights. Decoder columns normalized. Synthetic activations, not an LLM interpretability evaluation.

Section 11: The invertible latent-coordinate transformation preserved reconstruction to maximum absolute difference 8.88e-16 in float64 arithmetic.

All experiments executed; four tests passed. See results/results.json for all values, including per-seed SAE results. Most experiments use a single seed and are controlled synthetic demonstrations, not general performance claims.
