import tempfile
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from model import Config, TinyGQA, grouped_attention, cache_bytes
from checkpoints import save_model, load_model, pool_mha
from train import matched_initialization


class GQATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.ids = torch.randint(0, 256, (2, 12))

    def model(self, kv=2, backend='repeat'):
        return TinyGQA(Config(hidden_size=64, num_heads=8, num_kv_heads=kv,
            num_layers=2, intermediate_size=128, max_seq_len=64, backend=backend)).eval()

    def test_group_mapping_and_wrong_repeat(self):
        q = torch.zeros(1, 8, 1, 2)
        k = torch.zeros(1, 2, 1, 2)
        v = torch.tensor([[[[10., 11.]], [[20., 21.]]]])
        expected = v.repeat_interleave(4, dim=1)
        allowed = torch.ones(1, 1, dtype=torch.bool)
        actual = grouped_attention(q, k, v, allowed)
        torch.testing.assert_close(actual, expected)
        wrong = F.scaled_dot_product_attention(q, k.repeat(1, 4, 1, 1), v.repeat(1, 4, 1, 1))
        self.assertFalse(torch.equal(actual, wrong))

    def test_grouped_reference_and_repeat_gradients(self):
        for kv in (8, 2, 1):
            with self.subTest(kv=kv):
                q = torch.randn(2, 8, 3, 8, requires_grad=True)
                k = torch.randn(2, kv, 7, 8, requires_grad=True)
                v = torch.randn(2, kv, 7, 8, requires_grad=True)
                allowed = torch.arange(7)[None, :] <= torch.arange(4, 7)[:, None]
                a = grouped_attention(q, k, v, allowed)
                ga = torch.autograd.grad(a.square().sum(), (q, k, v), retain_graph=True)
                b = F.scaled_dot_product_attention(q, k.repeat_interleave(8 // kv, 1),
                    v.repeat_interleave(8 // kv, 1), attn_mask=allowed, dropout_p=0.0)
                gb = torch.autograd.grad(b.square().sum(), (q, k, v))
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
                for x, y in zip(ga, gb):
                    torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-5)

    def test_native_backend_parity(self):
        for kv in (8, 2, 1):
            a = self.model(kv)
            b = self.model(kv, 'native')
            b.load_state_dict(a.state_dict())
            try:
                actual = b(self.ids)[0]
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                self.skipTest('Native GQA unavailable: ' + str(exc))
            torch.testing.assert_close(a(self.ids)[0], actual, rtol=1e-4, atol=1e-5)

    def test_cache_chunk_parity_all_variants(self):
        for kv in (8, 2, 1):
            m = self.model(kv)
            full = m(self.ids)[0]
            for chunks in ([1] * 12, [5, 3, 4]):
                cache = None
                offset = 0
                pieces = []
                for n in chunks:
                    logits, cache = m(self.ids[:, offset:offset+n], cache, use_cache=True)
                    pieces.append(logits)
                    offset += n
                torch.testing.assert_close(full, torch.cat(pieces, 1), rtol=1e-4, atol=1e-5)

    def test_causality_all_variants(self):
        changed = self.ids.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % 256
        for kv in (8, 2, 1):
            m = self.model(kv)
            torch.testing.assert_close(m(self.ids)[0][:, :6], m(changed)[0][:, :6])

    def test_compact_cache_bytes(self):
        for kv in (8, 2, 1):
            m = self.model(kv)
            _, cache = m(self.ids, use_cache=True)
            self.assertEqual(cache[0][0].shape, (2, kv, 12, 8))
            self.assertEqual(cache_bytes(cache), 2 * 2 * 2 * 12 * kv * 8 * 4)

    def test_pooling_weights_bias_and_noop(self):
        m = self.model(8)
        noop = pool_mha(m, 8)
        torch.testing.assert_close(m(self.ids)[0], noop(self.ids)[0])
        with torch.no_grad():
            for layer in m.layers:
                for proj in (layer.attn.k_proj, layer.attn.v_proj):
                    proj.weight.copy_(torch.arange(8.)[:, None, None].expand(8, 8, 64).reshape(64, 64))
                    proj.bias.copy_(torch.arange(8.)[:, None].expand(8, 8).reshape(64))
        g = pool_mha(m, 2)
        self.assertTrue(torch.equal(g.layers[0].attn.k_proj.bias,
                                   torch.tensor([1.5] * 8 + [5.5] * 8)))
        torch.testing.assert_close(m.layers[0].attn.q_proj.weight, g.layers[0].attn.q_proj.weight)
        self.assertEqual(g.layers[0].attn.k_proj.weight[0, 0].item(), 1.5)

    def test_conversion_is_not_function_preserving(self):
        m = self.model(8)
        g = pool_mha(m, 2)
        self.assertGreater((m(self.ids)[0] - g(self.ids)[0]).abs().max().item(), 1e-5)

    def test_checkpoint_roundtrip_and_generation(self):
        for kv in (8, 2, 1):
            m = self.model(kv)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'model.pt'
                save_model(m, path)
                r = load_model(path).eval()
                torch.testing.assert_close(m(self.ids)[0], r(self.ids)[0])
                self.assertTrue(torch.equal(r.generate(self.ids[:, :3], 4),
                                           r.generate(self.ids[:, :3], 4, cached=False)))

    def test_initialization_matches_shared_weights(self):
        a = self.model(8)
        b = self.model(2)
        matched_initialization(a, 42)
        matched_initialization(b, 42)
        for name, value in a.state_dict().items():
            if value.shape == b.state_dict()[name].shape:
                self.assertTrue(torch.equal(value, b.state_dict()[name]), name)

    def test_trainability_all_variants(self):
        for kv in (8, 2, 1):
            m = self.model(kv).train()
            optimizer = torch.optim.AdamW(m.parameters(), lr=0.01)
            def loss():
                return F.cross_entropy(m(self.ids[:, :-1])[0].reshape(-1, 256),
                                       self.ids[:, 1:].reshape(-1))
            initial = loss().item()
            for _ in range(25):
                optimizer.zero_grad(set_to_none=True)
                val = loss()
                val.backward()
                optimizer.step()
            self.assertLess(loss().item(), initial * 0.7)

    def test_invalid_group_count(self):
        with self.assertRaises(ValueError):
            Config(num_heads=8, num_kv_heads=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
