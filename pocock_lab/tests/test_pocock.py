"""Automated unit test suite for the Pocock Skills and Contract Audit Laboratory.

Tests:
1. State-machine lifecycle contracts for task cancellation.
2. Mutation detection catching race conditions where workers claim cancelled runs.
3. Historical Git tag audit consistency (v1.0.0 through v1.3.2).
4. Dependency-aware interview question grouping (v1.1.0 serial vs v1.3.2 batched).
5. Idempotent cancellation and retry safety.
"""

from dataclasses import dataclass
import unittest

from pocock_lab.audit import HISTORICAL_AUDIT
from pocock_lab.queue_demo import Run, check_contract


class TestPocockSkillsAndContracts(unittest.TestCase):
    """Test suite covering lifecycle contracts, mutation verification, and release audits."""

    def test_01_run_lifecycle_contract(self):
        """Test 1: Verify correct state-machine transitions and queued-only cancellation."""
        run = Run()
        self.assertEqual(run.state, "queued")

        # Cancelling a queued run succeeds
        self.assertEqual(run.cancel(), "cancelled")
        # Idempotent re-cancellation
        self.assertEqual(run.cancel(), "cancelled")
        # Worker must never claim a cancelled run
        self.assertFalse(run.claim())
        self.assertEqual(run.state, "cancelled")

        # Fresh run claimed by worker
        active_run = Run()
        self.assertTrue(active_run.claim())
        self.assertEqual(active_run.state, "running")
        # Queued-only contract: running run cannot be cancelled via this transition
        self.assertEqual(active_run.cancel(), "running")

    def test_02_mutation_detection(self):
        """Test 2: Verify that mutating the state check allows worker claim and fails contract."""
        @dataclass
        class MutantRun:
            state: str = "queued"

            def cancel(self):
                if self.state == "queued":
                    self.state = "cancelled"
                return self.state

            def claim(self):
                # Buggy mutation: only rejects if already running, allowing cancelled runs to be claimed!
                if self.state == "running":
                    return False
                self.state = "running"
                return True

        with self.assertRaises(AssertionError) as ctx:
            check_contract(MutantRun)
        self.assertEqual(str(ctx.exception), "worker claimed a cancelled run")

    def test_03_historical_audit_counts(self):
        """Test 3: Verify the exact historical Git tag counts from the release audit."""
        self.assertEqual(HISTORICAL_AUDIT["v1.0.0"]["grilling_lines"], 10)
        self.assertEqual(HISTORICAL_AUDIT["v1.0.0"]["manifest_entries"], 17)
        self.assertEqual(HISTORICAL_AUDIT["v1.0.0"]["skill_files"], 34)

        self.assertEqual(HISTORICAL_AUDIT["v1.1.0"]["grilling_lines"], 12)
        self.assertEqual(HISTORICAL_AUDIT["v1.1.0"]["manifest_entries"], 21)
        self.assertEqual(HISTORICAL_AUDIT["v1.1.0"]["skill_files"], 38)

        self.assertEqual(HISTORICAL_AUDIT["v1.2.0"]["grilling_lines"], 22)
        self.assertEqual(HISTORICAL_AUDIT["v1.2.0"]["manifest_entries"], 25)
        self.assertEqual(HISTORICAL_AUDIT["v1.2.0"]["skill_files"], 35)

        self.assertEqual(HISTORICAL_AUDIT["v1.3.0"]["grilling_lines"], 28)
        self.assertEqual(HISTORICAL_AUDIT["v1.3.0"]["manifest_entries"], 27)
        self.assertEqual(HISTORICAL_AUDIT["v1.3.0"]["skill_files"], 37)

        self.assertEqual(HISTORICAL_AUDIT["v1.3.2"]["grilling_lines"], 30)
        self.assertEqual(HISTORICAL_AUDIT["v1.3.2"]["manifest_entries"], 27)
        self.assertEqual(HISTORICAL_AUDIT["v1.3.2"]["skill_files"], 38)

    def test_04_interview_dependency_grouping(self):
        """Test 4: Simulate dependency-ordered question scheduling (v1.1.0 serial vs v1.3.2 batched)."""
        questions = {
            "Q1_scope": {"text": "Which runs can be cancelled?", "depends_on": None},
            "Q2_auth": {"text": "Who may request cancellation?", "depends_on": None},
            "Q3_signal": {"text": "How to signal process termination?", "depends_on": "Q1_scope"},
        }

        # Round 1 in v1.3.2 batches independent questions
        round_1 = [q_id for q_id, q in questions.items() if q["depends_on"] is None]
        self.assertEqual(round_1, ["Q1_scope", "Q2_auth"])

        # Dependent questions wait for round 2
        round_2 = [q_id for q_id, q in questions.items() if q["depends_on"] == "Q1_scope"]
        self.assertEqual(round_2, ["Q3_signal"])

    def test_05_idempotent_cancellation_semantics(self):
        """Test 5: Verify cancellation idempotency under simulated network retry clicks."""
        run = Run()
        # First user cancellation click
        status_1 = run.cancel()
        self.assertEqual(status_1, "cancelled")

        # Simulated browser timeout retry click
        status_2 = run.cancel()
        self.assertEqual(status_2, "cancelled")
        self.assertEqual(run.state, "cancelled")


if __name__ == "__main__":
    unittest.main()
