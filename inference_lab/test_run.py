import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run import ROOT, generate, quantiles

class BaselineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = AutoModelForCausalLM.from_pretrained(ROOT/"fixture").eval()
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT/"fixture")

    def test_fixed_length_matches_cached_reference(self):
        ids = torch.tensor([self.tokenizer.encode("sensor reports cold")])
        tokens, trace = generate(self.model, ids, 6)
        with torch.inference_mode():
            current = ids
            cache = None
            reference = []
            for _ in range(6):
                output = self.model(current, past_key_values=cache, use_cache=True)
                current = output.logits[:, -1, :].argmax(-1, keepdim=True)
                cache = output.past_key_values
                reference.append(current)
        self.assertTrue(torch.equal(tokens, torch.cat(reference, dim=1)))
        self.assertEqual(trace["input_lengths"], [3,4,5,6,7,8])
        self.assertEqual(trace["positions_processed"],33)
        self.assertAlmostEqual(trace["total_ms"],trace["ttft_ms"]+sum(trace["itl_ms"]))

    def test_eos_and_single_token_metric(self):
        ids = torch.tensor([self.tokenizer.encode("sensor reports hot . fan on .")])
        tokens,trace = generate(self.model,ids,5,self.tokenizer.eos_token_id)
        self.assertEqual(tokens.tolist(),[[self.tokenizer.eos_token_id]])
        self.assertIsNone(trace["tpot_ms"])

    def test_causal_prefix_invariance(self):
        a = torch.tensor([self.tokenizer.encode("sensor reports hot .")])
        b = torch.tensor([self.tokenizer.encode("sensor reports cold .")])
        with torch.inference_mode():
            x = self.model(a,use_cache=False)
            y = self.model(b,use_cache=False)
        torch.testing.assert_close(x.logits[:,:2],y.logits[:,:2])
        self.assertIsNone(x.past_key_values)

    def test_nearest_rank_tail(self):
        self.assertEqual(quantiles(range(1,101)),{"p50":50,"p95":95,"p99":99})
        self.assertIsNone(quantiles([])["p99"])

if __name__ == "__main__":
    unittest.main()
