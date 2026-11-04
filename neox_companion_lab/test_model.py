import io
import unittest
import torch
from model import Config, TinyNeoX

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.c = Config(hidden_size=32, num_heads=4, num_layers=2,
                        intermediate_size=64, max_seq_len=64)
        self.model = TinyNeoX(self.c).eval()
        self.ids = torch.randint(0, 256, (2, 12))

    def test_shape_and_untied_embeddings(self):
        logits, cache = self.model(self.ids)
        self.assertEqual(logits.shape, (2, 12, 256))
        self.assertIsNone(cache)
        self.assertNotEqual(self.model.embedding.weight.data_ptr(),
                            self.model.head.weight.data_ptr())

    def test_future_cannot_change_past(self):
        changed = self.ids.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % 256
        a, _ = self.model(self.ids)
        b, _ = self.model(changed)
        torch.testing.assert_close(a[:, :6], b[:, :6], rtol=1e-5, atol=1e-6)

    def test_cache_single_token_and_multi_token_chunks(self):
        full, _ = self.model(self.ids)
        for chunks in ([1] * 12, [5, 3, 4]):
            cache, offset, pieces = None, 0, []
            for length in chunks:
                output, cache = self.model(self.ids[:, offset:offset+length],
                                           cache, use_cache=True)
                pieces.append(output)
                offset += length
            torch.testing.assert_close(full, torch.cat(pieces, dim=1),
                                       rtol=1e-4, atol=1e-5)

    def test_generation_cache_parity(self):
        a = self.model.generate(self.ids[:, :5], 8, cached=True)
        b = self.model.generate(self.ids[:, :5], 8, cached=False)
        self.assertTrue(torch.equal(a, b))

    def test_rotary_preserves_norm_and_pass_through(self):
        rope = self.model.layers[0].attn.rope
        x = torch.randn(2, 4, 7, 8)
        y = rope(x, offset=11)
        torch.testing.assert_close(x[..., :rope.dim].norm(dim=-1),
                                   y[..., :rope.dim].norm(dim=-1))
        self.assertTrue(torch.equal(x[..., rope.dim:], y[..., rope.dim:]))

    def test_checkpoint_roundtrip(self):
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = TinyNeoX(self.c).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(self.model(self.ids)[0], restored(self.ids)[0])

    def test_tiny_batch_can_be_learned(self):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        def loss():
            logits, _ = self.model(self.ids[:, :-1])
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256), self.ids[:, 1:].reshape(-1))
        initial = loss().item()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            value = loss()
            value.backward()
            optimizer.step()
        self.assertLess(loss().item(), initial * 0.5)

    def test_invalid_rotary_dimension(self):
        with self.assertRaises(ValueError):
            Config(hidden_size=24, num_heads=4, rotary_pct=0.25)

if __name__ == '__main__':
    unittest.main(verbosity=2)
