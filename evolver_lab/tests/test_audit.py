"""Automated unit test suite for the Evolver Research Laboratory.

Verifies:
1. Exact reconstruction of the 320 paired benchmark outcomes from Zhang et al. (2026).
2. Net accuracy percentage point gain (+4.375 pp).
3. Token expenditure reduction percentages (All pairs: 24.24%, Both pass: 17.57%).
4. Break-even economics across 20%, 50%, and 80% useful retrieval hit rates.
5. Algorithmic fidelity of the pure Python verify_gene_recall implementation across:
   - Injected-only (unknown)
   - Different reply (unused)
   - Promised action (used)
   - Negated action (used) -> demonstrating lexical grounding limitations.
"""

from decimal import Decimal
import math
import unittest

from evolver_lab.recall_verifier import verify_gene_recall, extract_keywords


class TestEvolverAudit(unittest.TestCase):
    """Test suite verifying mathematical reconstructions and recall verification mechanics."""

    def test_01_paired_outcomes_reconstruction(self):
        """Test 1: Verify exact integer reconstruction of the 320 paired execution outcomes."""
        total_pairs = 320
        # Per-harness reported rates from paper section 4.2:
        # Harness 1 (160 tasks): 85.0% pass
        # Harness 2 (160 tasks): 81.25% pass
        baseline_passes = int(160 * Decimal("0.85") + 160 * Decimal("0.8125"))
        # Augmented pass rate across all 320 tasks: 87.5%
        guided_passes = int(320 * Decimal("0.875"))
        # Reported intersection (tasks that passed on both attempts)
        both_pass = 265

        repaired_gains = guided_passes - both_pass
        regression_losses = baseline_passes - both_pass
        both_fail = total_pairs - both_pass - repaired_gains - regression_losses

        self.assertEqual(baseline_passes, 266)
        self.assertEqual(guided_passes, 280)
        self.assertEqual(both_pass, 265)
        self.assertEqual(repaired_gains, 15)
        self.assertEqual(regression_losses, 1)
        self.assertEqual(both_fail, 39)
        self.assertEqual(both_pass + repaired_gains + regression_losses + both_fail, 320)

    def test_02_net_accuracy_gain(self):
        """Test 2: Verify net accuracy improvement is +4.375 percentage points."""
        pairs = 320
        gains = 15
        losses = 1
        net_pp = 100.0 * (gains - losses) / pairs
        self.assertAlmostEqual(net_pp, 4.375, places=3)

    def test_03_token_reduction_arithmetic(self):
        """Test 3: Verify token reductions from published rounded means."""
        # Section 4.4 reported means (in thousands of tokens)
        # All pairs: 1,641.1k -> 1,243.3k
        red_all = 100.0 * (1.0 - 1243.3 / 1641.1)
        self.assertAlmostEqual(red_all, 24.24, places=2)

        # Both pass: 1,278.2k -> 1,053.6k
        red_both = 100.0 * (1.0 - 1053.6 / 1278.2)
        self.assertAlmostEqual(red_both, 17.57, places=2)

    def test_04_breakeven_economics(self):
        """Test 4: Verify expected token economics and break-even attempt thresholds."""
        capture_cost = 60000
        saving_per_hit = 12000
        cost_per_miss = 1000

        # Case 20% hit rate: net saving = 0.2*12000 - 0.8*1000 = 1600 tokens/attempt
        saving_20 = 0.2 * saving_per_hit - 0.8 * cost_per_miss
        self.assertEqual(saving_20, 1600)
        self.assertEqual(math.ceil(capture_cost / saving_20), 38)

        # Case 50% hit rate: net saving = 0.5*12000 - 0.5*1000 = 5500 tokens/attempt
        saving_50 = 0.5 * saving_per_hit - 0.5 * cost_per_miss
        self.assertEqual(saving_50, 5500)
        self.assertEqual(math.ceil(capture_cost / saving_50), 11)

        # Case 80% hit rate: net saving = 0.8*12000 - 0.2*1000 = 9400 tokens/attempt
        saving_80 = 0.8 * saving_per_hit - 0.2 * cost_per_miss
        self.assertEqual(saving_80, 9400)
        self.assertEqual(math.ceil(capture_cost / saving_80), 7)

    def test_05_recall_verifier_injected_only(self):
        """Test 5: Verify 'unknown' status when prompt is injected but assistant has not spoken."""
        strategy = "Inspect partition metadata before counting filtered tokens."
        gene = {"geneId": "test-gene", "strategy": [strategy]}
        turns = [{"role": "user", "text": strategy}]
        res = verify_gene_recall(gene, turns)
        self.assertEqual(res["recalled"], "unknown")
        self.assertEqual(res["score"], 0.0)
        self.assertEqual(res["matched"], [])

    def test_06_recall_verifier_different_reply(self):
        """Test 6: Verify 'unused' status when assistant discusses unrelated subject."""
        strategy = "Inspect partition metadata before counting filtered tokens."
        gene = {"geneId": "test-gene", "strategy": [strategy]}
        turns = [
            {"role": "user", "text": strategy},
            {"role": "assistant", "text": "I will examine the configuration."},
        ]
        res = verify_gene_recall(gene, turns)
        self.assertEqual(res["recalled"], "unused")
        self.assertEqual(res["score"], 0.0)

    def test_07_recall_verifier_promised_action(self):
        """Test 7: Verify 'used' status with score 1.00 when assistant promises action."""
        strategy = "Inspect partition metadata before counting filtered tokens."
        gene = {"geneId": "test-gene", "strategy": [strategy]}
        turns = [
            {"role": "assistant", "text": "I plan to inspect partition metadata before counting filtered tokens."}
        ]
        res = verify_gene_recall(gene, turns)
        self.assertEqual(res["recalled"], "used")
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(len(res["matched"]), 6)

    def test_08_recall_verifier_negated_action(self):
        """Test 8: Verify the lexical vulnerability where negation is marked as 'used'."""
        strategy = "Inspect partition metadata before counting filtered tokens."
        gene = {"geneId": "test-gene", "strategy": [strategy]}
        turns = [
            {"role": "assistant", "text": "I did not inspect partition metadata before counting filtered tokens."}
        ]
        res = verify_gene_recall(gene, turns)
        # Lexical matching cannot distinguish negation from execution
        self.assertEqual(res["recalled"], "used")
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(len(res["matched"]), 6)

    def test_09_recall_verifier_partial_threshold(self):
        """Test 9: Verify threshold sensitivity for partial vocabulary overlap."""
        strategy = "Inspect partition metadata before counting filtered tokens."
        gene = {"geneId": "test-gene", "strategy": [strategy]}
        # Keywords are 6 terms: counting, filtered, inspect, metadata, partition, tokens
        # Matching 2 terms: 2/6 = 0.3333 >= 0.30 -> used
        turns_above = [{"role": "assistant", "text": "Let us inspect the tokens."}]
        res_above = verify_gene_recall(gene, turns_above, threshold=0.30)
        self.assertEqual(res_above["recalled"], "used")
        self.assertAlmostEqual(res_above["score"], 0.3333, places=3)

        # Matching 1 term: 1/6 = 0.1667 < 0.30 -> unused
        turns_below = [{"role": "assistant", "text": "Let us inspect the file."}]
        res_below = verify_gene_recall(gene, turns_below, threshold=0.30)
        self.assertEqual(res_below["recalled"], "unused")
        self.assertAlmostEqual(res_below["score"], 0.1667, places=3)


if __name__ == "__main__":
    unittest.main()

