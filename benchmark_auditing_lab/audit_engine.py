"""Benchmark Auditing Engine.

Simulates automated code evaluation pipelines, multi-step dependency cascades,
benchmark defect taxonomies (undocumented constraints, numerical tolerances,
defective ground truths), and model evaluation under original vs verified benchmarks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class DefectType(Enum):
    NONE = "none"
    UNDOCUMENTED_CONSTRAINT = "undocumented_constraint"
    UNREASONABLE_TOLERANCE = "unreasonable_tolerance"
    DEFECTIVE_REFERENCE = "defective_reference"
    INCONSISTENT_CONVENTION = "inconsistent_convention"


@dataclass
class SubProblem:
    id: str
    name: str
    description: str
    analytical_solution: float
    flawed_reference_solution: Optional[float] = None
    default_tolerance: float = 1e-5
    flawed_tolerance: float = 1e-8
    defect_type: DefectType = DefectType.NONE
    defect_description: str = ""
    requires_helper_name: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)


@dataclass
class TaskProblem:
    id: str
    title: str
    domain: str  # Physics, Mathematics, Chemistry, etc.
    subproblems: List[SubProblem]


@dataclass
class EvaluationResult:
    passed_original: bool
    passed_verified: bool
    defect_encountered: DefectType
    actual_output: float
    expected_original: float
    expected_verified: float
    details: str


class BenchmarkAuditor:
    """Evaluates candidate model implementations against original and verified benchmark suites."""

    def __init__(self, problems: List[TaskProblem]):
        self.problems = problems

    def evaluate_subproblem(
        self,
        sub: SubProblem,
        candidate_val: float,
        candidate_globals: Dict[str, Any],
        is_memorizer: bool = False,
    ) -> EvaluationResult:
        """Evaluate a single subproblem under both original and verified standards."""
        # 1. Original (flawed) grader logic
        passed_orig = True
        expected_orig = (
            sub.flawed_reference_solution
            if sub.flawed_reference_solution is not None
            else sub.analytical_solution
        )
        expected_verif = sub.analytical_solution

        # Defect A: Undocumented helper constraint
        if sub.defect_type == DefectType.UNDOCUMENTED_CONSTRAINT and sub.requires_helper_name:
            if sub.requires_helper_name not in candidate_globals:
                passed_orig = False

        # Defect B: Unreasonable tolerance
        tol_orig = (
            sub.flawed_tolerance
            if sub.defect_type == DefectType.UNREASONABLE_TOLERANCE
            else sub.default_tolerance
        )
        if abs(candidate_val - expected_orig) > tol_orig:
            passed_orig = False

        # If model is a memorizer that learned the flawed reference answer:
        if is_memorizer and sub.flawed_reference_solution is not None:
            # Memorizer reproduces the benchmark's mistake, so it passes the flawed grader
            passed_orig = True
            # But fails the verified grader
            passed_verif = abs(candidate_val - expected_verif) <= sub.default_tolerance
        else:
            # Reasoning model produces the correct analytical result
            passed_verif = abs(candidate_val - expected_verif) <= sub.default_tolerance

        return EvaluationResult(
            passed_original=passed_orig,
            passed_verified=passed_verif,
            defect_encountered=sub.defect_type,
            actual_output=candidate_val,
            expected_original=expected_orig,
            expected_verified=expected_verif,
            details=sub.defect_description if sub.defect_type != DefectType.NONE else "Clean subproblem",
        )

    def evaluate_whole_problem(
        self,
        task: TaskProblem,
        candidate_sub_outputs: Dict[str, float],
        candidate_globals: Dict[str, Any],
        is_memorizer: bool = False,
    ) -> Tuple[bool, bool, Dict[str, EvaluationResult]]:
        """Evaluate a multi-step task where whole problem credit requires ALL subproblems to pass."""
        sub_results: Dict[str, EvaluationResult] = {}
        whole_orig = True
        whole_verif = True

        for sub in task.subproblems:
            val = candidate_sub_outputs.get(sub.id, 0.0)
            res = self.evaluate_subproblem(sub, val, candidate_globals, is_memorizer)
            sub_results[sub.id] = res

            if not res.passed_original:
                whole_orig = False
            if not res.passed_verified:
                whole_verif = False

        return whole_orig, whole_verif, sub_results


def create_synthetic_scicode_suite(num_problems: int = 65, seed: int = 42) -> List[TaskProblem]:
    """Generates a representative suite of multi-step scientific coding problems

    modeling the 263 defects documented in the SciCode audit.
    """
    rng = np.random.default_rng(seed)
    domains = ["Quantum Physics", "Astrophysics", "Fluid Dynamics", "Molecular Chemistry", "Statistical Mechanics"]
    problems = []

    defect_pool = [
        (DefectType.UNDOCUMENTED_CONSTRAINT, "Grader expects undocumented helper function in globals()"),
        (DefectType.UNREASONABLE_TOLERANCE, "Grader imposes 1e-8 tolerance on ill-conditioned numerical ODE"),
        (DefectType.DEFECTIVE_REFERENCE, "Reference implementation uses incorrect normalization reciprocal (1/21.35 vs 1/21.3535)"),
        (DefectType.INCONSISTENT_CONVENTION, "Grader expects unstated 1-based indexing for boundary grid nodes"),
    ]

    for p_idx in range(1, num_problems + 1):
        num_subs = rng.integers(3, 7)  # 3 to 6 subproblems per main problem
        domain = rng.choice(domains)
        subs = []

        for s_idx in range(1, num_subs + 1):
            sub_id = f"P{p_idx:02d}_S{s_idx}"
            true_val = float(rng.uniform(0.1, 100.0))

            # ~45% of subproblems have a defect in the flawed suite (matching SciCode's 192/400+ subproblem defects)
            has_defect = rng.random() < 0.45
            if has_defect:
                d_type, d_desc = defect_pool[rng.integers(0, len(defect_pool))]
                flawed_val = true_val * 1.05 if d_type == DefectType.DEFECTIVE_REFERENCE else None
                helper_req = "deduplicate_and_sort" if d_type == DefectType.UNDOCUMENTED_CONSTRAINT else None
                subs.append(
                    SubProblem(
                        id=sub_id,
                        name=f"Substep {s_idx}",
                        description=f"Compute numerical step {s_idx} for {domain}",
                        analytical_solution=true_val,
                        flawed_reference_solution=flawed_val,
                        default_tolerance=1e-4,
                        flawed_tolerance=1e-9 if d_type == DefectType.UNREASONABLE_TOLERANCE else 1e-4,
                        defect_type=d_type,
                        defect_description=d_desc,
                        requires_helper_name=helper_req,
                    )
                )
            else:
                subs.append(
                    SubProblem(
                        id=sub_id,
                        name=f"Substep {s_idx}",
                        description=f"Compute numerical step {s_idx} for {domain}",
                        analytical_solution=true_val,
                        default_tolerance=1e-4,
                        defect_type=DefectType.NONE,
                    )
                )

        problems.append(TaskProblem(id=f"P{p_idx:02d}", title=f"Scientific Problem {p_idx}", domain=domain, subproblems=subs))

    return problems
