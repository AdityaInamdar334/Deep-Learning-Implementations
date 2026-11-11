"""Automated unit test suite for the Majority Vote Reliability Laboratory.

Verifies:
1. Binomial vote probability symmetry and vote amplification/collapse bifurcation.
2. PC-Dr.DPO within-prompt objective and analytical gradient correctness via finite difference.
3. Behavior at reference policy agreement (zero margins).
4. Frequency mode selection vs trajectory log-likelihood score selection.
5. Unanimous group caching and conflict replay triggers.
6. Empirical audit of Table 1 across all 30 model-benchmark settings.
"""

from collections import Counter
import json
from math import isclose, log
from pathlib import Path
import unittest

from vote_lab.votes import majority_correct, pair_objective, select, replay_decision
from vote_lab.audit import compare


class TestVoteReliability(unittest.TestCase):
    """Test suite covering majority voting dynamics, DPO objectives, and paper audit."""

    def test_01_binary_symmetry(self):
        """Test 1: Verify exact binomial vote probability symmetry."""
        for n in (1, 9, 33, 65):
            self.assertAlmostEqual(majority_correct(0.6, n) + majority_correct(0.4, n), 1.0, places=10)

    def test_02_majority_vote_bifurcation(self):
        """Test 2: Verify that scaling rollouts amplifies competence but severely deepens misconceptions."""
        # When model has slight edge (p=0.60): vote amplifies correctness
        self.assertAlmostEqual(majority_correct(0.60, 1), 0.6000, places=4)
        self.assertAlmostEqual(majority_correct(0.60, 9), 0.7334, places=4)
        self.assertAlmostEqual(majority_correct(0.60, 33), 0.8789, places=4)
        self.assertAlmostEqual(majority_correct(0.60, 65), 0.9490, places=4)

        # When model has slight misconception (p=0.40): vote drives accuracy to zero
        self.assertAlmostEqual(majority_correct(0.40, 1), 0.4000, places=4)
        self.assertAlmostEqual(majority_correct(0.40, 9), 0.2666, places=4)
        self.assertAlmostEqual(majority_correct(0.40, 33), 0.1211, places=4)
        self.assertAlmostEqual(majority_correct(0.40, 65), 0.0510, places=4)

    def test_03_pc_dr_dpo_gradient_finite_difference(self):
        """Test 3: Verify analytical gradient of pair_objective against numerical central difference."""
        for kappa in (18 / 32, 28 / 32):
            margins = [-3.0, 1.0, 7.0]
            loss, weights, grad = pair_objective(margins, kappa)
            self.assertAlmostEqual(sum(weights), 1.0, places=10)

            eps = 1e-5
            for i in range(len(margins)):
                plus = margins.copy()
                minus = margins.copy()
                plus[i] += eps
                minus[i] -= eps
                num_grad = (pair_objective(plus, kappa)[0] - pair_objective(minus, kappa)[0]) / (2.0 * eps)
                self.assertAlmostEqual(num_grad, grad[i], places=7)

    def test_04_zero_margin_reference_behavior(self):
        """Test 4: Verify that at reference policy agreement, loss is log(2) and weights are uniform."""
        res_a = pair_objective([0.0, 0.0, 0.0], 0.55)
        res_b = pair_objective([0.0, 0.0, 0.0], 0.90)

        self.assertTrue(isclose(res_a[0], log(2), rel_tol=1e-7))
        # Tilting weights should be uniform [1/3, 1/3, 1/3] regardless of kappa
        for w in res_a[1]:
            self.assertAlmostEqual(w, 1.0 / 3.0, places=7)
        self.assertEqual(res_a[1:], res_b[1:])

    def test_05_frequency_vs_trajectory_score_selection(self):
        """Test 5: Verify frequency selection vs trajectory-score selection."""
        candidates = ["A", "A", "A", "B", "B"]
        # Frequency mode picks 'A' (3 votes vs 2 votes)
        self.assertEqual(select(candidates), "A")

        # When trajectory scores favor 'B' (0.1*3 = 0.3 vs 0.6*2 = 1.2)
        scores = [0.1, 0.1, 0.1, 0.6, 0.6]
        self.assertEqual(select(candidates, scores), "B")

    def test_06_unanimous_caching_and_replay_trigger(self):
        """Test 6: Verify caching of unanimous groups and replay triggering upon majority conflict."""
        cache = {}
        # Initial unanimous rollout -> caches and skips fresh pairs
        status_1 = replay_decision(cache, "task_1", ["A", "A", "A", "A"])
        self.assertEqual(status_1, "unanimous: skip fresh pairs")
        self.assertIn("task_1", cache)

        # Later round on same task with conflicting majority 'B' -> triggers replay
        status_2 = replay_decision(cache, "task_1", ["B", "B", "B", "A"])
        self.assertEqual(status_2, "replay cached completions")

        # Uncached task with non-unanimous answers -> proceeds with fresh pairs
        status_3 = replay_decision(cache, "task_2", ["B", "B", "A", "A"])
        self.assertEqual(status_3, "use fresh pairs")

    def test_07_table1_audit_outcomes(self):
        """Test 7: Verify exact outcome breakdown of Table 1 (30 settings: 18 wins, 4 ties, 6 losses, 2 collapses)."""
        table1_path = Path(__file__).resolve().parent.parent / "table1.json"
        data = json.loads(table1_path.read_text())
        rows = list(compare(data))
        self.assertEqual(len(rows), 30)

        counts = Counter(r[-1] for r in rows)
        self.assertEqual(counts["higher"], 18)
        self.assertEqual(counts["numeric tie"], 4)
        self.assertEqual(counts["lower"], 6)
        self.assertEqual(counts["all collapsed"], 2)
        # Matches or beats baseline in 24 of 30 settings (18 + 4 + 2)
        self.assertEqual(counts["higher"] + counts["numeric tie"] + counts["all collapsed"], 24)


if __name__ == "__main__":
    unittest.main()

