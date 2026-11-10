"""Automated unit test suite for the CPU-ARC Research Laboratory.

Verifies:
1. Multi-template linguistic normalization and parameter binding.
2. Parameter scaling (input count and capacity scaling).
3. Semantic boundary enforcement and rejection of out-of-scope qualifiers.
4. Input validation (zero capacity, negative counts, missing policy).
5. Exhaustive small-input slicing oracle (8,320 cases verified by physical list chunking).
6. Break-even economics across all 8 NVIDIA paper benchmarks (GSM8K, AIME, MATH, etc.).
7. Cost ratio parity against Table 2 reported values.
"""

from decimal import Decimal as D
import math
import unittest

from cpu_arc_lab.cache import Request, execute, answer
from cpu_arc_lab.costs import ROWS, break_even


class TestCpuArc(unittest.TestCase):
    """Test suite covering procedure caching, semantic boundary checks, and break-even math."""

    def test_01_template_phrasing_equivalence(self):
        """Test 1: Verify that differing surface templates produce identical bound execution."""
        q1 = "Batch 103 examples at 32 per batch, keep the last partial batch"
        q2 = "With batches of 32, group 103 examples and keep the remainder."
        res1 = answer(q1)
        res2 = answer(q2)
        self.assertIsNotNone(res1)
        self.assertEqual(res1, res2)
        self.assertEqual(res1["batches"], 4)
        self.assertEqual(res1["full"], 3)
        self.assertEqual(res1["remainder"], 7)
        self.assertEqual(res1["partial_batches_used"], 1)

    def test_02_new_parameters_scaling(self):
        """Test 2: Verify procedure execution with new parameters (129 examples at 32 capacity)."""
        q = "Batch 129 examples at 32 per batch, keep the last partial batch"
        res = answer(q)
        self.assertIsNotNone(res)
        self.assertEqual(res["full"], 4)
        self.assertEqual(res["remainder"], 1)
        self.assertEqual(res["batches"], 5)

    def test_03_combined_parameter_changes(self):
        """Test 3: Verify execution when both capacity and total count change."""
        q = "With batches of 64, group 129 examples and keep the remainder"
        res = answer(q)
        self.assertIsNotNone(res)
        self.assertEqual(res["full"], 2)
        self.assertEqual(res["remainder"], 1)
        self.assertEqual(res["batches"], 3)

    def test_04_semantic_boundary_and_qualifier_rejection(self):
        """Test 4: Verify policy clauses and ensure unsupported qualifiers trigger a cache miss."""
        # Policy: drop the remainder
        drop_res = answer("With batches of 32, group 103 examples and drop the remainder")
        self.assertIsNotNone(drop_res)
        self.assertEqual(drop_res["batches"], 3)
        self.assertEqual(drop_res["partial_batches_used"], 0)

        # Unsupported qualifier: "keep the remainder per worker" MUST NOT match
        unsupported = answer("With batches of 32, group 103 examples and keep the remainder per worker")
        self.assertIsNone(unsupported)

    def test_05_invalid_input_rejection(self):
        """Test 5: Verify zero capacity and malformed queries return None (cache miss)."""
        self.assertIsNone(answer("With batches of 0, group 103 examples and keep the remainder"))
        self.assertIsNone(answer("Batch 103 examples at 32 per batch"))
        self.assertIsNone(answer("Just compute batches for 103 items"))

    def test_06_exhaustive_slicing_oracle(self):
        """Test 6: Verify mathematical execution against an exhaustive physical list-slicing oracle."""
        # 130 example sizes x 32 capacities x 2 policies = 8,320 test combinations
        for n in range(130):
            items = list(range(n))
            for b in range(1, 33):
                chunks = [items[i:i + b] for i in range(0, n, b)]
                for keep in (True, False):
                    expected = sum(keep or len(chunk) == b for chunk in chunks)
                    req = Request(examples=n, capacity=b, keep_partial=keep)
                    self.assertEqual(execute(req)["batches"], expected)

    def test_07_paper_breakeven_thresholds(self):
        """Test 7: Verify exact break-even hit thresholds from NVIDIA Table 2 rounded inputs."""
        expected_break_evens = {
            "MATH-500": 31,
            "AIME 2024": 13,
            "AIME 2025": 13,
            "AIME 2026": 13,
            "GPQA Diamond": 26,
            "GSM8K": 90,
            "MMLU-Pro": 43,
            "SciBench": 30,
        }
        for name, s, b, c, _ in ROWS:
            computed_hits = break_even(D(s), D(b), D(c))
            self.assertEqual(computed_hits, expected_break_evens[name], f"Mismatch for {name}")

    def test_08_cost_savings_ratios(self):
        """Test 8: Verify reported invocation speedup ratios range between 16.3x and 291.1x."""
        ratios = [row[4] for row in ROWS]
        self.assertAlmostEqual(min(ratios), 16.3, places=1)
        self.assertAlmostEqual(max(ratios), 291.1, places=1)


if __name__ == "__main__":
    unittest.main()

