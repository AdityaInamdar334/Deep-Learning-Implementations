"""Epoch AI Benchmark Review Rubric & Registry Simulator.

Formalizes the three-verdict benchmark audit methodology:
- VERIFIED: Review found no score-affecting defects exceeding the 20% threshold.
- FLAWED: Serious scoring, task formulation, or harness defect affects >= 20% of the sample.
- NOT ENOUGH INFO: Tasks, rules, or environment lack sufficient inspection transparency.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class BenchmarkVerdict(Enum):
    VERIFIED = "Verified"
    FLAWED = "Flawed"
    NOT_ENOUGH_INFO = "Not enough info"


@dataclass
class ReviewAssessment:
    benchmark_name: str
    release_date: str
    inspectable_tasks: bool
    inspectable_scoring_rules: bool
    inspected_sample_size: int
    score_affecting_defects: int
    false_negative_rate: float
    false_positive_rate: float
    reward_hacking_detected: bool
    data_contamination_risk: str  # "Low", "Medium", "High"
    verdict: BenchmarkVerdict
    rationale: str


class EpochReviewEngine:
    """Implements the formal evaluation rubric published by Epoch AI."""

    DEFECT_THRESHOLD_PERCENT: float = 20.0

    @classmethod
    def evaluate_benchmark(
        cls,
        benchmark_name: str,
        release_date: str,
        inspectable_tasks: bool,
        inspectable_scoring_rules: bool,
        inspected_sample_size: int,
        score_affecting_defects: int,
        false_negative_rate: float = 0.0,
        false_positive_rate: float = 0.0,
        reward_hacking_detected: bool = False,
        data_contamination_risk: str = "Low",
    ) -> ReviewAssessment:
        # Check condition 1: Transparency / Inspectability
        if not inspectable_tasks or not inspectable_scoring_rules:
            return ReviewAssessment(
                benchmark_name=benchmark_name,
                release_date=release_date,
                inspectable_tasks=inspectable_tasks,
                inspectable_scoring_rules=inspectable_scoring_rules,
                inspected_sample_size=inspected_sample_size,
                score_affecting_defects=score_affecting_defects,
                false_negative_rate=false_negative_rate,
                false_positive_rate=false_positive_rate,
                reward_hacking_detected=reward_hacking_detected,
                data_contamination_risk=data_contamination_risk,
                verdict=BenchmarkVerdict.NOT_ENOUGH_INFO,
                rationale="Benchmark tasks or scoring harness cannot be independently inspected.",
            )

        # Check condition 2: Sample size validity
        if inspected_sample_size <= 0:
            defect_pct = 0.0
        else:
            defect_pct = (score_affecting_defects / inspected_sample_size) * 100.0

        # Check condition 3: Defect Threshold & Severe Exploits
        is_flawed = defect_pct >= cls.DEFECT_THRESHOLD_PERCENT or reward_hacking_detected

        if is_flawed:
            verdict = BenchmarkVerdict.FLAWED
            reason = (
                f"Score-affecting defects affected {defect_pct:.1f}% of inspected sample "
                f"(threshold: {cls.DEFECT_THRESHOLD_PERCENT}%)."
            )
            if reward_hacking_detected:
                reason += " Evaluator infrastructure enables reward hacking/signal spoofing."
        else:
            verdict = BenchmarkVerdict.VERIFIED
            reason = f"Defect rate ({defect_pct:.1f}%) is below the {cls.DEFECT_THRESHOLD_PERCENT}% threshold."

        return ReviewAssessment(
            benchmark_name=benchmark_name,
            release_date=release_date,
            inspectable_tasks=inspectable_tasks,
            inspectable_scoring_rules=inspectable_scoring_rules,
            inspected_sample_size=inspected_sample_size,
            score_affecting_defects=score_affecting_defects,
            false_negative_rate=false_negative_rate,
            false_positive_rate=false_positive_rate,
            reward_hacking_detected=reward_hacking_detected,
            data_contamination_risk=data_contamination_risk,
            verdict=verdict,
            rationale=reason,
        )


def get_historical_epoch_registry() -> List[ReviewAssessment]:
    """Returns the historical 2026 Epoch AI benchmark review registry snapshot (26 benchmarks)."""
    return [
        EpochReviewEngine.evaluate_benchmark(
            "SciCode (Original)", "2026-08-15", True, True, 65, 59, 0.45, 0.02, False, "Medium"
        ),
        EpochReviewEngine.evaluate_benchmark(
            "Humanity's Last Exam (HLE)", "2026-08-11", True, True, 48, 22, 0.35, 0.11, False, "High"
        ),
        EpochReviewEngine.evaluate_benchmark(
            "SWE-bench Verified (Audited Failures)", "2026-02-23", True, True, 138, 82, 0.59, 0.05, False, "High"
        ),
        EpochReviewEngine.evaluate_benchmark(
            "SWE-bench Pro (Public Split)", "2026-07-08", True, True, 731, 219, 0.30, 0.04, False, "Medium"
        ),
        EpochReviewEngine.evaluate_benchmark(
            "Terminal-Bench 4.0.0", "2026-09-04", True, True, 66, 30, 0.25, 0.15, True, "Medium"
        ),
        EpochReviewEngine.evaluate_benchmark(
            "GPQA Diamond Clean", "2026-05-10", True, True, 189, 9, 0.04, 0.01, False, "Medium"
        ),
        EpochReviewEngine.evaluate_benchmark(
            "FrontierCode 2.0 (Private)", "2026-10-27", False, True, 132, 0, 0.0, 0.0, False, "Low"
        ),
        EpochReviewEngine.evaluate_benchmark(
            "HLE-Diamond (1000-Task)", "2026-10-27", True, True, 100, 8, 0.06, 0.02, False, "Low"
        ),
    ]
