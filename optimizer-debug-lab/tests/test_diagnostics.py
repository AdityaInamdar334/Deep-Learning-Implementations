import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestOptimizerDebugLab(unittest.TestCase):
    def test_lr_regimes_in_diagnostics(self):
        data = json.loads((ROOT / 'diagnostics.json').read_text())
        lrs = data['learning_rate']
        self.assertEqual(set(lrs.keys()), {'0.0001', '0.1', '10.0'})
        # Tiny LR barely moves
        self.assertLess(max(lrs['0.0001']['relative_update']), 1e-4)
        # Healthy LR decreases loss smoothly
        self.assertLess(lrs['0.1']['loss'][-1], 0.3)
        self.assertLess(max(lrs['0.1']['loss']), 1.0)
        # Huge LR exhibits a dramatic early spike and >100% weight update ratio
        self.assertGreater(max(lrs['10.0']['loss']), 10.0)
        self.assertGreater(max(lrs['10.0']['relative_update']), 1.0)

    def test_gradient_noise_inverse_sqrt_scaling(self):
        data = json.loads((ROOT / 'diagnostics.json').read_text())
        noise = data['noise']
        base = noise['1']
        for b in (4, 16, 64, 256):
            expected = base / math.sqrt(b)
            self.assertAlmostEqual(noise[str(b)] / expected, 1.0, delta=0.05)

    def test_artifacts_present(self):
        for fname in ('article.md', 'lab.py', 'diagnostics.py', 'results.json', 'diagnostics.json', 'run.log', 'diagnostics.log'):
            self.assertTrue((ROOT / fname).exists(), f'Missing {fname}')
        for fig in (
            'lr_diagnostics.png',
            'gradient_noise.png',
            'paths.png',
            'quadratic_loss.png',
            'activation_rms.png',
            'gradient_rms.png',
            'validation.png',
        ):
            p = ROOT / 'figures' / fig
            self.assertTrue(p.exists(), f'Missing figure {fig}')
            self.assertGreater(p.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
