import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import quadratic, initialization

ROOT = Path(__file__).resolve().parents[1]


class TestOptimizationLab(unittest.TestCase):
    def test_quadratic_optimizers_ordering(self):
        res = quadratic()
        expected = {'GD', 'Momentum', 'Nesterov', 'AdaGrad', 'RMSProp', 'Adam', 'Newton'}
        self.assertEqual(set(res.keys()), expected)
        self.assertEqual(res['Newton'], 0.0)
        self.assertLess(res['Momentum'], res['GD'])
        self.assertLess(res['Nesterov'], res['Momentum'])
        self.assertLess(res['Adam'], res['GD'])

    def test_initialization_signal_propagation(self):
        res = initialization()
        self.assertEqual(set(res.keys()), {'small', 'xavier', 'he', 'large'})
        for name, stats in res.items():
            self.assertEqual(len(stats['activation_rms']), 20)
            self.assertEqual(len(stats['gradient_rms']), 20)
        self.assertLess(res['small']['activation_rms'][-1], 1e-10)
        self.assertGreater(res['he']['activation_rms'][-1], 0.1)
        self.assertLess(res['he']['activation_rms'][-1], 10.0)

    def test_artifacts_present(self):
        for fname in ('article.md', 'results.json', 'run.log'):
            self.assertTrue((ROOT / fname).exists(), f'Missing {fname}')
        for fig in ('paths.png', 'quadratic_loss.png', 'activation_rms.png', 'gradient_rms.png', 'validation.png'):
            p = ROOT / 'figures' / fig
            self.assertTrue(p.exists(), f'Missing figure {fig}')
            self.assertGreater(p.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
