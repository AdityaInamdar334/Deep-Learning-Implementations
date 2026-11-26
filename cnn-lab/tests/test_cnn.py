import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lab import mechanics, Net, shift

ROOT = Path(__file__).resolve().parents[1]


class TestCNNLab(unittest.TestCase):
    def test_convolution_mechanics_and_equivariance(self):
        m = mechanics()
        self.assertLess(m['forward_max_error'], 1e-12)
        self.assertTrue(all(e < 1e-12 for e in m['gradient_max_errors']))
        self.assertEqual(m['circular_shift_error'], 0.0)
        self.assertGreater(m['pool_same_grid_mean_change'], 0.1)

    def test_architectures_and_parameter_counts(self):
        x = torch.randn(4, 1, 16, 16)
        counts = {}
        for kind in ('mlp', 'cnn', 'separable'):
            net = Net(kind)
            self.assertEqual(tuple(net(x).shape), (4, 10))
            counts[kind] = sum(p.numel() for p in net.parameters())
        self.assertLess(counts['separable'], counts['cnn'])
        self.assertLess(counts['cnn'], counts['mlp'])

    def test_shift_helper(self):
        x = torch.zeros(1, 1, 8, 8)
        x[0, 0, 2, 3] = 1.0
        y = shift(x, 2, 1)
        self.assertEqual(float(y[0, 0, 4, 4]), 1.0)
        self.assertEqual(float(y.sum()), 1.0)

    def test_artifacts_present(self):
        for fname in ('article.md', 'results.json', 'run.log'):
            self.assertTrue((ROOT / fname).exists(), f'Missing {fname}')
        for fig in ('edge_detection.png', 'shifted_digits.png', 'accuracy.png', 'learning_curves.png', 'learned_filters.png'):
            p = ROOT / 'figures' / fig
            self.assertTrue(p.exists(), f'Missing figure {fig}')
            self.assertGreater(p.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
