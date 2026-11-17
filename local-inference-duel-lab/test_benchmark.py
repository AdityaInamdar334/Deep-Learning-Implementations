import json
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmark import post, run


class LocalInferenceDuelTests(unittest.TestCase):
    def test_tasks_json_schema(self):
        tasks_file = Path(__file__).parent / "prompts" / "tasks.json"
        self.assertTrue(tasks_file.is_file())
        tasks = json.loads(tasks_file.read_text())
        self.assertGreater(len(tasks), 0)
        for task in tasks:
            self.assertIn("id", task)
            self.assertIn("prompt", task)
            self.assertIn("expected_substring", task)
            self.assertIsInstance(task["id"], str)
            self.assertIsInstance(task["prompt"], str)

    @patch("benchmark.post")
    def test_ollama_response_parsing(self, mock_post):
        mock_post.return_value = {
            "response": "12345",
            "prompt_eval_count": 20,
            "prompt_eval_duration": 200_000_000,  # 0.2s -> 100 tok/s
            "eval_count": 10,
            "eval_duration": 200_000_000,        # 0.2s -> 50 tok/s
            "load_duration": 50_000_000,         # 50 ms
            "done_reason": "stop"
        }
        res = run("ollama", "http://127.0.0.1:11434", "test-model", "test prompt", 42, 128, 30)
        self.assertEqual(res["output"], "12345")
        self.assertEqual(res["prompt_tokens"], 20)
        self.assertEqual(res["generated_tokens"], 10)
        self.assertAlmostEqual(res["prefill_tps"], 100.0)
        self.assertAlmostEqual(res["decode_tps"], 50.0)
        self.assertAlmostEqual(res["load_ms"], 50.0)
        self.assertEqual(res["stop_reason"], "stop")

    @patch("benchmark.post")
    def test_llama_cpp_response_parsing(self, mock_post):
        mock_post.return_value = {
            "content": "391",
            "stop_type": "eos",
            "stopped_limit": False,
            "timings": {
                "prompt_n": 18,
                "prompt_per_second": 142.5,
                "predicted_n": 6,
                "predicted_per_second": 62.8
            }
        }
        res = run("llama.cpp", "http://127.0.0.1:8080", "unused", "test prompt", 42, 128, 30)
        self.assertEqual(res["output"], "391")
        self.assertEqual(res["prompt_tokens"], 18)
        self.assertEqual(res["generated_tokens"], 6)
        self.assertAlmostEqual(res["prefill_tps"], 142.5)
        self.assertAlmostEqual(res["decode_tps"], 62.8)
        self.assertIsNone(res["load_ms"])
        self.assertEqual(res["stop_reason"], "eos")

    @patch("benchmark.post")
    def test_llama_cpp_stopped_limit(self, mock_post):
        mock_post.return_value = {
            "content": "truncated text...",
            "stop_type": None,
            "stopped_limit": True,
            "timings": {
                "prompt_n": 10,
                "prompt_per_second": 100.0,
                "predicted_n": 128,
                "predicted_per_second": 40.0
            }
        }
        res = run("llama.cpp", "http://127.0.0.1:8080", "unused", "test prompt", 42, 128, 30)
        self.assertEqual(res["stop_reason"], "limit")


if __name__ == "__main__":
    unittest.main()
