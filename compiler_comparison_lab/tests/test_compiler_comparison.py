"""Unit tests for the trusted compiler comparison harness."""

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure compiler_comparison_lab root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare import run_comparison


class TestCompilerComparison(unittest.TestCase):
    def setUp(self):
        self.py = sys.executable

    def test_clean_parity_run(self):
        """Reference, candidate, and fast producing identical output must pass stability and parity."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workload = tmp_path / "work.py"
            workload.write_text("print('HELLO COMPILER WORLD')")

            report_path, data = run_comparison(
                cwd=str(tmp_path),
                runs=3,
                timeout=10.0,
                reference=f"{self.py} work.py",
                candidate=f"{self.py} -u work.py",
                fast=f"{self.py} -B work.py",
                output=str(tmp_path / "out")
            )

            self.assertTrue(report_path.exists())
            self.assertTrue(data["summary"]["reference"]["stable_output"])
            self.assertTrue(data["summary"]["candidate"]["stable_output"])
            self.assertTrue(data["summary"]["fast"]["stable_output"])
            self.assertTrue(data["summary"]["candidate"]["matches_reference"])
            self.assertTrue(data["summary"]["fast"]["matches_reference"])

    def test_divergent_output_detected(self):
        """Candidate producing different output must be marked as not matching reference."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ref_script = tmp_path / "ref.py"
            cand_script = tmp_path / "cand.py"
            ref_script.write_text("print('OUTPUT 1')")
            cand_script.write_text("print('OUTPUT 2')")

            report_path, data = run_comparison(
                cwd=str(tmp_path),
                runs=3,
                timeout=10.0,
                reference=f"{self.py} ref.py",
                candidate=f"{self.py} cand.py",
                fast=f"{self.py} ref.py",
                output=str(tmp_path / "out")
            )

            self.assertTrue(data["summary"]["candidate"]["stable_output"])
            self.assertFalse(data["summary"]["candidate"]["matches_reference"])
            self.assertTrue(data["summary"]["fast"]["matches_reference"])

    def test_unstable_nondeterministic_output_detected(self):
        """A script that prints random non-deterministic output must be flagged as unstable."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            unstable_script = tmp_path / "unstable.py"
            unstable_script.write_text("import time; print('TIME', time.time())")

            report_path, data = run_comparison(
                cwd=str(tmp_path),
                runs=3,
                timeout=10.0,
                reference=f"{self.py} unstable.py",
                candidate=f"{self.py} unstable.py",
                fast=f"{self.py} unstable.py",
                output=str(tmp_path / "out")
            )

            self.assertFalse(data["summary"]["reference"]["stable_output"])
            self.assertFalse(data["summary"]["reference"]["matches_reference"])

    def test_timeout_detection(self):
        """A command that exceeds timeout must be recorded with timeout=True and flagged as invalid."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sleep_script = tmp_path / "hang.py"
            sleep_script.write_text("import time; time.sleep(5)")

            report_path, data = run_comparison(
                cwd=str(tmp_path),
                runs=2,
                timeout=0.2,
                reference=f"{self.py} -c pass",
                candidate=f"{self.py} hang.py",
                fast=f"{self.py} -c pass",
                output=str(tmp_path / "out")
            )

            self.assertFalse(data["summary"]["candidate"]["valid"])
            self.assertIsNone(data["summary"]["candidate"]["median_seconds"])

    def test_round_robin_execution_order(self):
        """The execution order must shift cyclicly across rounds to eliminate thermal and cache bias."""
        runs = 5
        commands = {'reference': 'a', 'candidate': 'b', 'fast': 'c'}
        expected_orders = [
            ['reference', 'candidate', 'fast'],
            ['candidate', 'fast', 'reference'],
            ['fast', 'reference', 'candidate'],
            ['reference', 'candidate', 'fast'],
            ['candidate', 'fast', 'reference']
        ]

        actual_orders = []
        for i in range(runs):
            names = list(commands)
            names = names[i % 3:] + names[:i % 3]
            actual_orders.append(names)

        self.assertEqual(actual_orders, expected_orders)

    def test_invalid_arguments_raise_error(self):
        """Non-positive runs or timeouts must raise ValueError."""
        with self.assertRaises(ValueError):
            run_comparison(runs=0, reference="a", candidate="b", fast="c")
        with self.assertRaises(ValueError):
            run_comparison(timeout=-1.0, reference="a", candidate="b", fast="c")


if __name__ == "__main__":
    unittest.main()
