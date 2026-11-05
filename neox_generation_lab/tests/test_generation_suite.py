import tempfile
import unittest
from pathlib import Path
import torch
from model import *
from generation import *


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.m = TinyNeoX().eval()

    @torch.inference_mode()
    def test_cached_logits_match_full_prefix(self):
        ids = torch.randint(0, 257, (1, 20))
        _, cache = self.m(ids[:, :8], use_cache=True)
        for j in range(8, 20):
            got, cache = self.m(ids[:, j:j+1], cache, True)
            want, _ = self.m(ids[:, :j+1])
            torch.testing.assert_close(got[:, -1], want[:, -1], atol=2e-5, rtol=2e-5)

    @torch.inference_mode()
    def test_chunked_prefill(self):
        ids = torch.randint(0, 257, (1, 13))
        full, _ = self.m(ids)
        _, cache = self.m(ids[:, :5], use_cache=True)
        got, _ = self.m(ids[:, 5:], cache, True)
        torch.testing.assert_close(got, full[:, 5:], atol=2e-5, rtol=2e-5)

    @torch.inference_mode()
    def test_causality(self):
        a = torch.randint(0, 257, (1, 12))
        b = a.clone()
        b[:, 7:] = (b[:, 7:] + 1) % 257
        x, _ = self.m(a)
        y, _ = self.m(b)
        torch.testing.assert_close(x[:, :7], y[:, :7])

    def test_greedy_equivalence(self):
        for prompt in ([1], [1, 2, 3], encode('The cache ')):
            a, _ = generate(self.m, prompt, 24, True, eos_id=None)
            b, _ = generate(self.m, prompt, 24, False, eos_id=None)
            self.assertEqual(a, b)

    def test_pending_final_token(self):
        output, trace = generate(self.m, [1, 2, 3], 7, eos_id=None)
        self.assertEqual([r['input_tokens'] for r in trace], [3] + [1] * 6)
        self.assertEqual(trace[-1]['cache_after'], len(output) - 1)
        self.assertEqual(trace[-1]['cache_bytes'], 2 * 2 * 1 * 4 * 9 * 24 * 4)

    def test_zero_budget_and_bounds(self):
        self.assertEqual(generate(self.m, [1], 0), ([1], []))
        with self.assertRaises(ValueError):
            generate(self.m, [], 2)
        with self.assertRaises(ValueError):
            generate(self.m, [1] * 256, 1)
        with self.assertRaises(ValueError):
            generate(self.m, [1], 1, temperature=-1)

    def test_eos(self):
        first, _ = generate(self.m, [1, 2], 1, eos_id=None)
        out, trace = generate(self.m, [1, 2], 20, eos_id=first[-1])
        self.assertEqual(len(trace), 1)

    def test_sampling_reproducibility_and_topk(self):
        a, _ = generate(self.m, [1, 2], 8, temperature=.8, top_k=8, seed=4, eos_id=None)
        b, _ = generate(self.m, [1, 2], 8, temperature=.8, top_k=8, seed=4, eos_id=None)
        self.assertEqual(a, b)
        g, _ = generate(self.m, [1, 2], 8, eos_id=None)
        k, _ = generate(self.m, [1, 2], 8, temperature=.8, top_k=1, eos_id=None)
        self.assertEqual(g, k)

    def test_fresh_request(self):
        a, _ = generate(self.m, [1, 2], 5)
        generate(self.m, [99, 33, 21], 6)
        b, _ = generate(self.m, [1, 2], 5)
        self.assertEqual(a, b)

    @torch.inference_mode()
    def test_interleaved_requests(self):
        a = torch.tensor([[1, 2, 3]])
        b = torch.tensor([[4, 5]])
        _, ca = self.m(a, use_cache=True)
        _, cb = self.m(b, use_cache=True)
        ga, _ = self.m(torch.tensor([[6]]), ca, True)
        gb, _ = self.m(torch.tensor([[7]]), cb, True)
        wa, _ = self.m(torch.tensor([[1, 2, 3, 6]]))
        wb, _ = self.m(torch.tensor([[4, 5, 7]]))
        torch.testing.assert_close(ga[:, -1], wa[:, -1], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(gb[:, -1], wb[:, -1], atol=2e-5, rtol=2e-5)

    def test_utf8_stream(self):
        s = ByteStream()
        text = 'नमस्ते café 🧠'
        got = ''.join(s.push(x) for x in encode(text)) + s.finish()
        self.assertEqual(got, text)

    def test_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'model.pt'
            save(self.m, p)
            other = load(p)
            self.assertEqual(generate(self.m, [1, 2], 8)[0], generate(other, [1, 2], 8)[0])

    @torch.inference_mode()
    def test_last_only_projection(self):
        ids = torch.tensor([[1, 2, 3]])
        full, _ = self.m(ids)
        last, _ = self.m(ids, last_only=True)
        torch.testing.assert_close(full[:, -1:], last, atol=2e-5, rtol=2e-5)

    def test_requires_eval(self):
        with self.assertRaises(ValueError):
            generate(self.m.train(), [1], 1)

    @torch.inference_mode()
    def test_wrong_history_changes_output(self):
        _, ca = self.m(torch.tensor([[1, 2, 3]]), use_cache=True)
        _, cb = self.m(torch.tensor([[8, 9, 10]]), use_cache=True)
        a, _ = self.m(torch.tensor([[4]]), ca, True)
        b, _ = self.m(torch.tensor([[4]]), cb, True)
        self.assertGreater((a - b).abs().max().item(), 1e-3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
