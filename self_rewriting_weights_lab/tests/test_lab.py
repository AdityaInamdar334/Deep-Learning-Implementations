import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lab import (
    projection_edit,
    paired_evaluation,
    demonstrate_projection,
    demonstrate_paired_scores,
    demonstrate_token_budget,
)


class TestSelfRewritingWeightsLab(unittest.TestCase):
    def test_projection_edit_rank_and_attenuation(self):
        rng = np.random.default_rng(7)
        W = rng.normal(size=(8, 5))
        direction = rng.normal(size=8)
        u = direction / np.linalg.norm(direction)
        alpha = 0.15

        W_new = projection_edit(W, direction, alpha)
        delta = W_new - W

        self.assertEqual(np.linalg.matrix_rank(delta, tol=1e-9), 1)
        np.testing.assert_allclose(u @ W_new, (1.0 - alpha) * (u @ W), atol=1e-12)

    def test_projection_edit_orthogonal_invariance(self):
        rng = np.random.default_rng(11)
        W = rng.normal(size=(6, 4))
        u = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # orthogonal to u

        W_new = projection_edit(W, u, alpha=0.4)
        np.testing.assert_allclose(v @ W_new, v @ W, atol=1e-12)

    def test_projection_edit_extremes(self):
        rng = np.random.default_rng(19)
        W = rng.normal(size=(5, 3))
        d = rng.normal(size=5)
        u = d / np.linalg.norm(d)

        # alpha = 0 -> identity
        np.testing.assert_allclose(projection_edit(W, d, 0.0), W, atol=1e-12)
        # alpha = 1 -> complete nulling along u
        np.testing.assert_allclose(u @ projection_edit(W, d, 1.0), np.zeros(3), atol=1e-12)

    def test_paired_evaluation_metrics(self):
        base = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=float)
        edited = np.array([[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=float)
        b, e, delta, se = paired_evaluation(base, edited)

        self.assertAlmostEqual(b, 0.5, places=6)
        self.assertAlmostEqual(e, 0.75, places=6)
        self.assertAlmostEqual(delta, 0.25, places=6)
        # Problem-level diffs: [0.25, 0.0, 0.5] -> std(ddof=1) = 0.25, SE = 0.25 / sqrt(3)
        self.assertAlmostEqual(se, 0.25 / np.sqrt(3.0), places=6)

        # Identical arrays -> zero delta and zero SE
        _, _, zero_delta, zero_se = paired_evaluation(base, base)
        self.assertEqual(zero_delta, 0.0)
        self.assertEqual(zero_se, 0.0)

    def test_cli_demonstrations_run_cleanly(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            demonstrate_projection()
            demonstrate_paired_scores()
            demonstrate_token_budget()
        out = buf.getvalue()
        self.assertIn('TOY PROJECTION EDIT', out)
        self.assertIn('TOY PAIRED EVALUATION', out)
        self.assertIn('TOY TOKEN-BUDGET EXAMPLE', out)


if __name__ == '__main__':
    unittest.main()
