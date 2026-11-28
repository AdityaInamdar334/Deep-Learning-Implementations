# Fast CPU unit tests for graph labels, weight sharing, sparse dispatch and memory arithmetic.

import sys, unittest, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Model, SparseMoE
from lab import batch
from memory import kv_bytes, weight_bytes

class Tests(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_pointer_labels(self):
        x, s, h, y = batch(torch.Generator().manual_seed(42), 32, 8, 1, 6, 'cpu')
        for e, a, k, b in zip(x, s, h, y):
            for _ in range(int(k)):
                a = e[a]
            self.assertEqual(a.item(), b.item())

    def test_shared_parameter_count(self):
        m = Model(shared=True)
        count = sum((p.numel() for p in m.parameters()))
        x, s, h, y = batch(torch.Generator().manual_seed(1), 4, 8, 1, 3, 'cpu')
        z, states = m(x, s, h, 3, True)
        self.assertEqual(len(states), 3)
        self.assertEqual(z.shape, (4, 8))
        self.assertEqual(count, sum((p.numel() for p in m.parameters())))
        z.sum().backward()
        self.assertTrue(all((p.grad is not None for p in m.parameters())))

    def test_moe_dispatch(self):
        m = SparseMoE(8)
        x = torch.randn(3, 5, 8, requires_grad=True)
        y = m(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(m.counts.sum().item(), 30)
        (y.square().mean() + m.aux).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_moe_reference(self):
        m = SparseMoE(8)
        x = torch.randn(7, 8)
        scores, ids = m.router(x).topk(2, dim=-1)
        w = scores.softmax(-1)
        all_out = torch.stack([e(x) for e in m.experts], 1)
        ref = (all_out.gather(1, ids[..., None].expand(-1, -1, 8)) * w[..., None]).sum(1)
        torch.testing.assert_close(m(x), ref)

    def test_memory(self):
        self.assertEqual(kv_bytes(10, 2, 3, 4, 1, 8, 2), 1024)
        self.assertEqual(weight_bytes(100, 4), 50)
if __name__ == '__main__':
    unittest.main()
