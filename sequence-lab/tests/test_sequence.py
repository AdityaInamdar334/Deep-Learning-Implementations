import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lab import batch, MemoryModel, check_bptt

ROOT = Path(__file__).resolve().parents[1]


class TestSequenceLab(unittest.TestCase):
    def test_manual_bptt_matches_autograd(self):
        errors = check_bptt()
        self.assertEqual(len(errors), 2)
        self.assertLess(max(errors), 1e-12)

    def test_batch_contract(self):
        rng = torch.Generator().manual_seed(0)
        x, y = batch(16, 12, rng)
        self.assertEqual(tuple(x.shape), (16, 12, 3))
        self.assertEqual(tuple(y.shape), (16,))
        self.assertTrue(torch.all(x[:, 0, 2] == 1.0))
        self.assertTrue(torch.all(x[:, 1:, 2] == 0.0))

    def test_memory_models_and_shapes(self):
        rng = torch.Generator().manual_seed(1)
        x, _ = batch(8, 16, rng)
        counts = {}
        for kind in ('RNN', 'GRU', 'LSTM', 'attention'):
            m = MemoryModel(kind)
            self.assertEqual(tuple(m(x).shape), (8, 2))
            counts[kind] = sum(p.numel() for p in m.parameters())
        self.assertLess(counts['attention'], counts['RNN'])

    def test_artifacts_present(self):
        for fname in ('article.md', 'results.json', 'run.log'):
            self.assertTrue((ROOT / fname).exists(), f'Missing {fname}')
        for fig in ('gradient.png', 'training.png', 'memory.png'):
            p = ROOT / 'figures' / fig
            self.assertTrue(p.exists(), f'Missing figure {fig}')
            self.assertGreater(p.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
