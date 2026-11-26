import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from lab import model, evaluate

ROOT = Path(__file__).resolve().parents[1]


class TestRegularizationLab(unittest.TestCase):
    def test_l1_sparsity_vs_l2_shrinkage(self):
        w = np.array([2.0, 2.0, 0.4])
        h = np.array([10.0, 0.1, 1.0])
        alpha = 0.5
        l1 = np.sign(w) * np.maximum(np.abs(w) - alpha / h, 0.0)
        l2 = (h / (h + alpha)) * w
        np.testing.assert_allclose(l1, [1.95, 0.0, 0.0], atol=1e-12)
        self.assertTrue(np.all(np.abs(l2) < np.abs(w)))
        self.assertTrue(np.all(l2 != 0.0))

    def test_dropout_scaling_and_eval_identity(self):
        d = nn.Dropout(0.3)
        x = torch.ones(10000)
        torch.manual_seed(42)
        self.assertLess(abs(d(x).mean().item() - 1.0), 0.05)
        d.eval()
        self.assertTrue(torch.equal(d(x), x))

    def test_model_and_evaluate(self):
        net = model(drop=0.2)
        x = torch.zeros(4, 64)
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        self.assertEqual(tuple(net(x).shape), (4, 10))
        metrics = evaluate(net, x, y)
        self.assertEqual(set(metrics.keys()), {'loss', 'acc', 'ece'})
        self.assertTrue(0.0 <= metrics['acc'] <= 1.0)
        self.assertTrue(0.0 <= metrics['ece'] <= 1.0)

    def test_artifacts_present(self):
        for fname in ('article.md', 'results.json', 'results.txt', 'smoke_tests.py'):
            self.assertTrue((ROOT / fname).exists(), f'Missing {fname}')
        for fig in ('baseline_gap.png', 'ablation.png', 'validation_loss.png', 'validation_acc.png'):
            p = ROOT / 'figures' / fig
            self.assertTrue(p.exists(), f'Missing figure {fig}')
            self.assertGreater(p.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
