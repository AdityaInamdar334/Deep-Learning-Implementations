"""Verification suite for neural-memory-lab."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))


def hillis_steele_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inclusive Hillis-Steele associative prefix scan for h_t = a_t * h_{t-1} + b_t."""
    A = a.copy()
    B = b.copy()
    offset = 1
    while offset < len(A):
        A0 = A.copy()
        B0 = B.copy()
        A[offset:] = A0[offset:] * A0[:-offset]
        B[offset:] = A0[offset:] * B0[:-offset] + B0[offset:]
        offset *= 2
    return B


def serial_recurrence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Serial reference recurrence h_t = a_t * h_{t-1} + b_t."""
    h = np.zeros(a.shape[1], dtype=a.dtype)
    out = []
    for ai, bi in zip(a, b):
        h = ai * h + bi
        out.append(h.copy())
    return np.array(out)


class TestNeuralMemoryLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results_path = LAB_DIR / "memory_results.json"
        with cls.results_path.open() as f:
            cls.results = json.load(f)

    def test_selective_recall_vs_decay(self) -> None:
        recall = self.results["recall"]
        for length in ("8", "32", "128", "512"):
            self.assertIn(length, recall)
            self.assertEqual(recall[length]["selective"], 1.0)
        # Fixed exponential decay collapses toward chance at T=512.
        self.assertGreater(recall["8"]["decay"], 0.99)
        self.assertLess(recall["512"]["decay"], 0.55)

    def test_delta_overwrite_vs_additive(self) -> None:
        overwrite = self.results["overwrite"]
        for writes in ("1", "2", "4", "8", "16", "32"):
            self.assertIn(writes, overwrite)
            self.assertEqual(overwrite[writes]["delta"], 1.0)
        # Additive accumulation degrades to ~50% latest-value accuracy once n >= 2.
        self.assertEqual(overwrite["1"]["additive"], 1.0)
        for writes in ("2", "4", "8", "16", "32"):
            self.assertLess(overwrite[writes]["additive"], 0.60)

    def test_saved_associative_scan_error(self) -> None:
        self.assertLess(self.results["scan_max_error"], 1e-12)

    def test_live_hillis_steele_scan_parity(self) -> None:
        rng = np.random.default_rng(7)
        for seq_len in (1, 16, 65, 129):
            a = rng.uniform(0.1, 0.99, (seq_len, 4))
            b = rng.normal(size=(seq_len, 4))
            parallel = hillis_steele_scan(a, b)
            serial = serial_recurrence(a, b)
            max_err = float(np.max(np.abs(parallel - serial)))
            self.assertLess(max_err, 1e-12)

    def test_companion_artifacts_exist(self) -> None:
        expected_files = [
            "memory.py",
            "lab.py",
            "article.md",
            "memory_results.json",
            "results.json",
            "figures/selective_recall.png",
            "figures/overwrite.png",
            "figures/gradient.png",
            "figures/training.png",
            "figures/memory.png",
        ]
        for rel in expected_files:
            path = LAB_DIR / rel
            self.assertTrue(path.is_file(), f"Missing expected artifact: {rel}")
            self.assertGreater(path.stat().st_size, 0, f"Empty artifact: {rel}")


if __name__ == "__main__":
    unittest.main()
