"""DeepQuestion Cognitive Complexity Analyzer.

Implements Bloom's Taxonomy evaluation tiers (Recall -> Application -> Analysis -> Synthesis)
and measures capability degradation curves as tasks progress to higher cognitive depth.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple
import numpy as np


class CognitiveLevel(IntEnum):
    RECALL = 1
    APPLICATION = 2
    ANALYSIS = 3
    SYNTHESIS = 4


@dataclass
class CognitiveQuestion:
    id: str
    topic: str
    level: CognitiveLevel
    prompt: str
    requires_cross_variable_reasoning: bool
    requires_open_synthesis: bool


@dataclass
class ModelProfile:
    name: str
    base_knowledge_capacity: float  # 0.0 - 1.0 (memorization & recall)
    reasoning_depth_factor: float   # 0.0 - 1.0 (depth retention under complexity)


class CognitiveComplexityEngine:
    """Evaluates how language models perform as tasks move up Bloom's taxonomy."""

    def __init__(self, questions: List[CognitiveQuestion]):
        self.questions = questions

    def evaluate_model(self, model: ModelProfile, seed: int = 42) -> Dict[CognitiveLevel, float]:
        """Calculates expected pass rates across cognitive levels based on model capability."""
        rng = np.random.default_rng(seed)
        results_by_level: Dict[CognitiveLevel, List[bool]] = {lvl: [] for lvl in CognitiveLevel}

        for q in self.questions:
            # Cognitive decay model:
            # Level 1 (Recall): pass probability ~ base_knowledge_capacity
            # Level 2 (Application): base * (0.85 + 0.15 * reasoning_depth)
            # Level 3 (Analysis): base * (reasoning_depth ** 0.8)
            # Level 4 (Synthesis): base * (reasoning_depth ** 1.8)
            if q.level == CognitiveLevel.RECALL:
                p_success = model.base_knowledge_capacity * 0.98
            elif q.level == CognitiveLevel.APPLICATION:
                p_success = model.base_knowledge_capacity * (0.75 + 0.20 * model.reasoning_depth_factor)
            elif q.level == CognitiveLevel.ANALYSIS:
                p_success = model.base_knowledge_capacity * (model.reasoning_depth_factor ** 0.85) * 0.88
            elif q.level == CognitiveLevel.SYNTHESIS:
                # Up to 70% drop from recall, matching DeepQuestion empirical findings
                p_success = model.base_knowledge_capacity * (model.reasoning_depth_factor ** 1.9) * 0.65

            passed = rng.random() < p_success
            results_by_level[q.level].append(passed)

        return {lvl: float(np.mean(vals)) if vals else 0.0 for lvl, vals in results_by_level.items()}


def create_standard_cognitive_suite() -> List[CognitiveQuestion]:
    """Generates a standardized 100-question suite spanning Bloom's taxonomy."""
    suite = []
    topics = [
        "Regularization & Overfitting",
        "Attention Mechanisms",
        "Numerical Stability & Floating Point",
        "Distributed Parallelism & All-Reduce",
        "Reinforcement Learning from Human Feedback",
    ]

    q_idx = 1
    for topic in topics:
        for lvl in CognitiveLevel:
            for rep in range(5):  # 5 questions per topic per level = 100 questions total
                suite.append(
                    CognitiveQuestion(
                        id=f"CQ_{q_idx:03d}",
                        topic=topic,
                        level=lvl,
                        prompt=f"Task {q_idx} on {topic} at Bloom Level {lvl.name}",
                        requires_cross_variable_reasoning=(lvl >= CognitiveLevel.ANALYSIS),
                        requires_open_synthesis=(lvl == CognitiveLevel.SYNTHESIS),
                    )
                )
                q_idx += 1

    return suite
