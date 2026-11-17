import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark
import plot_results

class LabTests(unittest.TestCase):
    def test_fixed_prompt_identical(self):
        self.assertEqual(benchmark.make_prompt(2048, "fixed"), benchmark.make_prompt(16384, "fixed"))

    def test_scaled_prompt_grows(self):
        self.assertLess(len(benchmark.make_prompt(2048, "scaled")[0]), len(benchmark.make_prompt(8192, "scaled")[0]))

    def test_needle_unique(self):
        prompt, answer = benchmark.make_prompt(4096, "scaled")
        self.assertEqual(prompt.count(answer), 1)

    def test_rate_excludes_cached(self):
        uncached, prefill, decode = benchmark.rates({"prompt_eval_count": 100, "prompt_eval_cached_count": 80, "prompt_eval_duration": 1000000000, "eval_count": 20, "eval_duration": 2000000000})
        self.assertEqual((uncached, prefill, decode), (20, 20, 10))

    def test_rate_handles_missing(self):
        self.assertEqual(benchmark.rates({}), (None, None, None))

    def test_summary_median_and_error(self):
        rows = [{"mode": "fixed", "configured_context": 2048, "error": "", "decode_tps": 20, "needle_exact_match": 1},
                {"mode": "fixed", "configured_context": 2048, "error": "", "decode_tps": 40, "needle_exact_match": 0},
                {"mode": "fixed", "configured_context": 2048, "error": "OOM", "decode_tps": None}]
        result = benchmark.summarise(rows)[0]
        self.assertEqual(result["median_decode_tps"], 30)
        self.assertEqual(result["failed_runs"], 1)
        self.assertEqual(result["needle_exact_matches"], 1)

    def test_gpu_unavailable(self):
        with patch("benchmark.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(benchmark.gpu_snapshot(0))

    def test_svg_created(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "chart.svg"
            plot_results.chart([{"mode": "fixed", "configured_context": "2048", "error": "", "decode_tps": "12.5"}], "decode_tps", "Test", target)
            self.assertIn("<svg", target.read_text())

if __name__ == "__main__":
    unittest.main()

# Extra sanity test for the illustrative Qwen2.5 7B cache math.
import kv_math
class KVMathTests(unittest.TestCase):
    def test_qwen_7b_4096(self):
        self.assertEqual(kv_math.kv_bytes(4096, 28, 4, 128, 2) / (1024**2), 224)
