"""Verification suite for applied-deep-learning-lab."""

import itertools
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))


class TestAppliedDeepLearningLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (LAB_DIR / "results.json").open() as f:
            cls.results = json.load(f)

    def test_distillation_and_cascade_summary(self) -> None:
        self.assertEqual(len(self.results["runs"]), 3)
        summary = self.results["summary"]
        self.assertGreater(summary["teacher"], 0.97)
        self.assertGreater(summary["student"], 0.95)
        self.assertGreater(summary["distilled"], 0.95)
        # Confidence cascade lifts student accuracy while routing only a small fraction to teacher.
        self.assertGreater(summary["cascade_accuracy"], summary["student"])
        self.assertLess(summary["teacher_fraction"], 0.10)

    def test_quantization_cliff_and_payload_scaling(self) -> None:
        q8 = [r["quantization"]["8"]["accuracy"] for r in self.results["runs"]]
        q4 = [r["quantization"]["4"]["accuracy"] for r in self.results["runs"]]
        q2 = [r["quantization"]["2"]["accuracy"] for r in self.results["runs"]]
        self.assertGreater(np.mean(q8), 0.97)
        self.assertGreater(np.mean(q4), 0.96)
        self.assertLess(np.mean(q2), 0.70)
        # Ideal payload decreases monotonically with bit width.
        q = self.results["runs"][0]["quantization"]
        self.assertGreater(q["8"]["ideal_payload_bytes"], q["4"]["ideal_payload_bytes"])
        self.assertGreater(q["4"]["ideal_payload_bytes"], q["2"]["ideal_payload_bytes"])

    def test_ctc_enumeration_matches_pytorch(self) -> None:
        ctc = self.results["ctc"]
        self.assertEqual(ctc["paths"], 5)
        self.assertAlmostEqual(ctc["probability"], 0.494, places=12)

        # Live parity check against torch.nn.functional.ctc_loss.
        probs = torch.tensor([[0.2, 0.7, 0.1], [0.4, 0.5, 0.1], [0.2, 0.2, 0.6]], dtype=torch.float64)
        collapse = lambda path: [a for i, a in enumerate(path) if a != 0 and (i == 0 or a != path[i - 1])]
        paths = [p for p in itertools.product(range(3), repeat=3) if collapse(p) == [1, 2]]
        total = sum(float(np.prod([probs[t, a].item() for t, a in enumerate(p)])) for p in paths)
        loss = F.ctc_loss(probs.log().unsqueeze(1), torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([2]), reduction="sum")
        self.assertEqual(len(paths), 5)
        self.assertLess(abs(float((-loss).exp()) - total), 1e-12)

    def test_ngram_and_matrix_factorization(self) -> None:
        probs = self.results["ngram"]["smoothed_probabilities"]
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(sum(probs), 1.0, places=12)
        self.assertAlmostEqual(probs[1], 0.5, places=12)

        rec = self.results["recommendation"]
        self.assertGreater(rec["mean_baseline_rmse"], 1.5)
        self.assertLess(rec["factor_rmse"], 1e-4)

    def test_companion_artifacts_exist(self) -> None:
        expected = [
            "lab.py",
            "article.md",
            "results.json",
            "run.log",
            "figures/distillation.png",
            "figures/quantization.png",
            "figures/cascade.png",
        ]
        for rel in expected:
            path = LAB_DIR / rel
            self.assertTrue(path.is_file(), f"Missing artifact: {rel}")
            self.assertGreater(path.stat().st_size, 0, f"Empty artifact: {rel}")


if __name__ == "__main__":
    unittest.main()
