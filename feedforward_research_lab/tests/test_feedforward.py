import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from experiment import X, y, forward, loss_grad
from research_experiments import tent

ROOT = Path(__file__).resolve().parents[1]


class TestFeedforwardLab(unittest.TestCase):
    def test_exact_xor_solution(self):
        exact = [np.ones((2, 2)), np.array([0.0, -1.0]), np.array([[1.0], [-2.0]]), np.zeros(1)]
        pred = forward(X, exact)[0].ravel()
        np.testing.assert_allclose(pred, [0.0, 1.0, 1.0, 0.0], atol=1e-12)

    def test_manual_backprop_matches_finite_diff(self):
        rng = np.random.default_rng(7)
        p = [rng.normal(0, 0.5, (2, 8)), np.full(8, 0.1), rng.normal(0, 0.3, (8, 1)), np.zeros(1)]
        _, g = loss_grad(X, y, p)
        err = 0.0
        eps = 1e-6
        for a, ga in zip(p, g):
            for idx in np.ndindex(a.shape):
                old = a[idx]
                a[idx] = old + eps
                plus = loss_grad(X, y, p)[0]
                a[idx] = old - eps
                minus = loss_grad(X, y, p)[0]
                a[idx] = old
                err = max(err, abs((plus - minus) / (2 * eps) - ga[idx]))
        self.assertLess(err, 1e-7)

    def test_bce_vs_mse_saturation_gradient(self):
        z = torch.tensor(-10.0, requires_grad=True)
        bce = nn.functional.binary_cross_entropy_with_logits(z, torch.ones_like(z))
        mse = 0.5 * (z.sigmoid() - 1.0) ** 2
        g_bce = abs(torch.autograd.grad(bce, z)[0].item())
        g_mse = abs(torch.autograd.grad(mse, z)[0].item())
        self.assertGreater(g_bce, 0.99)
        self.assertGreater(g_bce / g_mse, 10000.0)

    def test_tent_map_folding(self):
        pts = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(tent(pts), [0.0, 1.0, 0.0], atol=1e-12)
        grid = np.linspace(0.0, 1.0, 257)
        vals = tent(tent(grid))
        self.assertTrue(np.all((vals >= -1e-12) & (vals <= 1.0 + 1e-12)))

    def test_artifacts_present(self):
        for fname in ('article.md', 'results.txt', 'research_results.txt'):
            self.assertTrue((ROOT / fname).exists(), f'Missing {fname}')
        for fig in ('depth.png', 'gradients.png', 'noisy_xor.png'):
            p = ROOT / 'figures' / fig
            self.assertTrue(p.exists(), f'Missing figure {fig}')
            self.assertGreater(p.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
