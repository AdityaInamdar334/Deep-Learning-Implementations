"""Verification suite for ml-methodology-lab."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))


class TestMLMethodologyLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (LAB_DIR / "results.json").open() as f:
            cls.results = json.load(f)

    def test_group_leakage_vs_unseen_groups(self) -> None:
        self.assertGreater(self.results["row"]["overlap"], 0)
        self.assertEqual(self.results["row"]["accuracy"], 1.0)
        self.assertEqual(self.results["group"]["overlap"], 0)
        self.assertLess(self.results["group"]["accuracy"], 0.60)

    def test_asymmetric_cost_thresholding(self) -> None:
        th = self.results["threshold"]
        self.assertLess(th["selected"]["cost"], th["default"]["cost"])
        self.assertLess(th["default"]["cost"], th["all_negative"]["cost"])
        # Verify cost formula fp + 20 * fn holds across all three policies.
        for key in ("default", "selected", "all_negative"):
            row = th[key]
            self.assertEqual(row["cost"], row["fp"] + 20 * row["fn"])

    def test_selective_prediction_tradeoff(self) -> None:
        sel = self.results["selective"]
        all_cov, all_acc = sel["all"]
        chosen_cov, chosen_acc = sel["chosen"]
        val_cov, val_acc = sel["validation"]
        self.assertEqual(all_cov, 1.0)
        self.assertLess(chosen_cov, 1.0)
        self.assertGreater(chosen_acc, all_acc)
        self.assertGreaterEqual(val_acc, 0.98)
        self.assertGreater(val_cov, 0.90)

    def test_learning_curves_bias_vs_variance(self) -> None:
        rows = self.results["learning"]
        self.assertEqual(len(rows), 36)  # 6 sizes x 3 seeds x 2 depths
        d2_1000 = [r["val"] for r in rows if r["n"] == 1000 and r["depth"] == "2"]
        dn_40 = [r["val"] for r in rows if r["n"] == 40 and r["depth"] == "None"]
        dn_1000 = [r["val"] for r in rows if r["n"] == 1000 and r["depth"] == "None"]
        # High-bias depth=2 tree plateaus low; unconstrained tree improves strongly with data.
        self.assertLess(np.mean(d2_1000), 0.40)
        self.assertGreater(np.mean(dn_1000), np.mean(dn_40) + 0.30)

    def test_random_search_beats_grid(self) -> None:
        search = self.results["search"]
        self.assertLess(search["random_median"], search["grid"])
        self.assertGreater(search["random_win_fraction"], 0.80)

    def test_debugging_sanity_and_live_finite_diff(self) -> None:
        dbg = self.results["debug"]
        self.assertEqual(dbg["tiny_accuracy"], 1.0)
        self.assertLess(dbg["tiny_loss"], 1e-2)
        self.assertLess(dbg["gradient_max_error"], 1e-8)

        # Live central finite-difference verification.
        w = np.array([0.2, -0.4, 1.0])
        f = lambda x: float(np.sum(np.sin(x) + x * x))
        eps = 1e-5
        num = np.array([(f(w + eps * e) - f(w - eps * e)) / (2 * eps) for e in np.eye(3)])
        exact = np.cos(w) + 2.0 * w
        self.assertLess(float(np.max(np.abs(num - exact))), 1e-8)

    def test_companion_artifacts_exist(self) -> None:
        expected = [
            "lab.py",
            "article.md",
            "results.json",
            "run.log",
            "figures/leakage.png",
            "figures/precision_recall.png",
            "figures/coverage.png",
            "figures/learning_curves.png",
            "figures/search.png",
        ]
        for rel in expected:
            path = LAB_DIR / rel
            self.assertTrue(path.is_file(), f"Missing artifact: {rel}")
            self.assertGreater(path.stat().st_size, 0, f"Empty artifact: {rel}")


if __name__ == "__main__":
    unittest.main()
