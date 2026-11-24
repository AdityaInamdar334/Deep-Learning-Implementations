import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from swiglu_lab import PlainFFN, SwiGLUFFN, ToySwiGLU, target


class TestSwiGLULab(unittest.TestCase):
    def test_parameter_matching(self):
        a = PlainFFN(96, 384)
        b = SwiGLUFFN(96, 256)
        self.assertEqual(
            sum(x.numel() for x in a.parameters()),
            sum(x.numel() for x in b.parameters()),
        )

    def test_shapes(self):
        x = torch.randn(2, 7, 96)
        self.assertEqual(PlainFFN(96, 384)(x).shape, x.shape)
        self.assertEqual(SwiGLUFFN(96, 256)(x).shape, x.shape)

    def test_exact_gate_representation(self):
        m = ToySwiGLU()
        with torch.no_grad():
            m.gate.weight.copy_(torch.tensor([[2.5, 0.0]]))
            m.value.weight.copy_(torch.tensor([[0.0, 1.0]]))
        x = torch.randn(128, 2)
        self.assertTrue(torch.allclose(m(x), target(x), atol=1e-6))

    def test_silu_not_bounded(self):
        vals = F.silu(torch.tensor([-1.0, 0.0, 3.0]))
        self.assertTrue(vals[0] < 0 and vals[1] == 0 and vals[2] > 1)


if __name__ == '__main__':
    unittest.main()
