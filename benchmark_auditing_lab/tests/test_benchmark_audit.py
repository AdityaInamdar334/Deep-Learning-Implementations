"""Unit tests for the Benchmark Auditing and Integrity Suite."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_engine import BenchmarkAuditor, SubProblem, TaskProblem, DefectType
from cognitive_complexity import CognitiveComplexityEngine, ModelProfile, CognitiveLevel, create_standard_cognitive_suite
from epoch_rubric import EpochReviewEngine, BenchmarkVerdict


class TestBenchmarkAuditing(unittest.TestCase):

    def test_undocumented_constraint_rejection(self):
        """Grader demanding undocumented helper name must reject clean implementation under original rules."""
        sub = SubProblem(
            id="sub_01",
            name="Sort and deduplicate",
            description="Return sorted unique integers",
            analytical_solution=42.0,
            defect_type=DefectType.UNDOCUMENTED_CONSTRAINT,
            requires_helper_name="deduplicate_and_sort",
            default_tolerance=1e-4,
        )
        task = TaskProblem(id="task_01", title="Dedup Task", domain="General", subproblems=[sub])
        auditor = BenchmarkAuditor([task])

        # Candidate output is correct, but globals lack the hidden helper
        res = auditor.evaluate_subproblem(sub, 42.0, candidate_globals={}, is_memorizer=False)
        self.assertFalse(res.passed_original, "Must fail original flawed grader due to missing undocumented helper")
        self.assertTrue(res.passed_verified, "Must pass verified grader on mathematical correctness")

    def test_unreasonable_numerical_tolerance(self):
        """Excessively strict tolerance on numerical simulation must fail under original, pass under verified."""
        sub = SubProblem(
            id="sub_02",
            name="Numerical ODE",
            description="Euler integration",
            analytical_solution=0.665132,
            flawed_tolerance=1e-8,
            default_tolerance=1e-3,
            defect_type=DefectType.UNREASONABLE_TOLERANCE,
        )
        auditor = BenchmarkAuditor([])
        # Actual numerical output slightly perturbed by step-size discretization
        candidate_val = 0.665031

        res = auditor.evaluate_subproblem(sub, candidate_val, candidate_globals={}, is_memorizer=False)
        self.assertFalse(res.passed_original, "Original strict tolerance (1e-8) must reject valid numerical solution")
        self.assertTrue(res.passed_verified, "Verified tolerance (1e-3) must accept defensible numerical solution")

    def test_cascading_whole_problem_failure(self):
        """A single subproblem failure must cause whole-problem score to collapse to zero."""
        subs = [
            SubProblem(id="s1", name="Step 1", description="", analytical_solution=1.0),
            SubProblem(id="s2", name="Step 2", description="", analytical_solution=2.0),
            SubProblem(
                id="s3",
                name="Step 3",
                description="",
                analytical_solution=3.0,
                defect_type=DefectType.UNDOCUMENTED_CONSTRAINT,
                requires_helper_name="secret_fn",
            ),
            SubProblem(id="s4", name="Step 4", description="", analytical_solution=4.0),
        ]
        task = TaskProblem(id="task_cascade", title="Multi-step Cascade", domain="Physics", subproblems=subs)
        auditor = BenchmarkAuditor([task])

        outputs = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0}
        w_orig, w_verif, res_map = auditor.evaluate_whole_problem(task, outputs, candidate_globals={}, is_memorizer=False)

        self.assertFalse(w_orig, "Whole problem must fail in original suite because Step 3 failed")
        self.assertTrue(w_verif, "Whole problem must pass in verified suite because all steps are mathematically correct")

    def test_memorizer_vs_reasoning_model_divergence(self):
        """Memorizer passing flawed answer key must fail verified suite, while reasoning model passes."""
        sub = SubProblem(
            id="sub_err_ref",
            name="Normalization Constant",
            description="Derived constant",
            analytical_solution=21.3535,
            flawed_reference_solution=21.35,  # Erroneous reference answer
            defect_type=DefectType.DEFECTIVE_REFERENCE,
            default_tolerance=1e-3,
        )
        auditor = BenchmarkAuditor([])

        # 1. Reasoning Model: outputs true analytical value (21.3535)
        res_reasoning = auditor.evaluate_subproblem(sub, 21.3535, candidate_globals={}, is_memorizer=False)
        self.assertFalse(res_reasoning.passed_original, "Reasoning model must fail flawed grader that expects 21.35")
        self.assertTrue(res_reasoning.passed_verified, "Reasoning model must pass verified grader")

        # 2. Memorizer Model: outputs flawed reference value (21.35)
        res_memorizer = auditor.evaluate_subproblem(sub, 21.35, candidate_globals={}, is_memorizer=True)
        self.assertTrue(res_memorizer.passed_original, "Memorizer must pass flawed grader by matching erroneous key")
        self.assertFalse(res_memorizer.passed_verified, "Memorizer must fail verified grader on true scientific value")

    def test_cognitive_complexity_degradation(self):
        """Model accuracy must degrade as cognitive depth progresses from Recall to Synthesis."""
        suite = create_standard_cognitive_suite()
        engine = CognitiveComplexityEngine(suite)
        model = ModelProfile(name="TestModel", base_knowledge_capacity=0.95, reasoning_depth_factor=0.60)

        scores = engine.evaluate_model(model, seed=42)
        recall = scores[CognitiveLevel.RECALL]
        application = scores[CognitiveLevel.APPLICATION]
        analysis = scores[CognitiveLevel.ANALYSIS]
        synthesis = scores[CognitiveLevel.SYNTHESIS]

        self.assertGreaterEqual(recall, application, "Recall score must be >= application score")
        self.assertGreater(recall, analysis, "Recall score must exceed analysis score")
        self.assertGreater(analysis, synthesis, "Analysis score must exceed synthesis score")
        # Ensure synthesis shows significant drop (~50%+ relative decline)
        self.assertLess(synthesis, recall * 0.60, "Synthesis score must show major drop from recall")

    def test_epoch_rubric_classification(self):
        """Epoch rubric must correctly classify Verified, Flawed, and Not Enough Info."""
        # 1. Verified benchmark (<20% defect rate)
        res_v = EpochReviewEngine.evaluate_benchmark(
            "Benchmark-Clean", "2026-01-01", True, True, 100, 5
        )
        self.assertEqual(res_v.verdict, BenchmarkVerdict.VERIFIED)

        # 2. Flawed benchmark (>=20% defect rate)
        res_f = EpochReviewEngine.evaluate_benchmark(
            "Benchmark-Buggy", "2026-01-01", True, True, 100, 25
        )
        self.assertEqual(res_f.verdict, BenchmarkVerdict.FLAWED)

        # 3. Not Enough Info (non-inspectable tasks)
        res_n = EpochReviewEngine.evaluate_benchmark(
            "Benchmark-Secret", "2026-01-01", False, True, 100, 0
        )
        self.assertEqual(res_n.verdict, BenchmarkVerdict.NOT_ENOUGH_INFO)


if __name__ == "__main__":
    unittest.main(verbosity=2)
