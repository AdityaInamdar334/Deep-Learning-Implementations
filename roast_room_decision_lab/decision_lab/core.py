"""Synthetic decisions, specialist observations, and transparent verdict rules.

This is NOT a Claude Code, Roast Room, or any real-LLM implementation.
We deliberately expose oracle ground truth for offline evaluation only.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Mapping

SEATS = ('strategy', 'product', 'pricing', 'marketing', 'people', 'operations', 'capital')
OPTIONS = ('A', 'B', 'C')


@dataclass(frozen=True)
class Case:
    """Unknown to the policy, available only to the evaluator."""
    values: tuple[tuple[float, ...], ...]  # [option][dimension]
    weights: tuple[int, ...]               # relevance weights 1..3
    ruin: tuple[bool, ...]                 # option-specific irreversible downside
    shared_framing: tuple[float, ...]      # shared bias the agents might inherit

    def true_scores(self) -> tuple[float, ...]:
        return tuple(sum(v * w for v, w in zip(row, self.weights)) / sum(self.weights)
                     for row in self.values)

    def best(self) -> str:
        scores = self.true_scores()
        return OPTIONS[max(range(len(OPTIONS)), key=lambda i: scores[i])]


@dataclass(frozen=True)
class SeatCard:
    seat: str
    scores: tuple[float, ...]
    weight: int
    ruin_flags: tuple[bool, ...]


@dataclass(frozen=True)
class Verdict:
    recommended: str
    score: float
    label: str
    spread: float
    capped: bool
    cards: tuple[SeatCard, ...]
    calls: int
    proxy_tokens: int


def make_case(rng: random.Random) -> Case:
    # Each dimension has partially shared attractiveness, plus specialist factors.
    raw = []
    base = [rng.uniform(3.2, 7.5) for _ in OPTIONS]
    for i in range(3):
        raw.append(tuple(max(0, min(10, base[i] + rng.gauss(0, 2.15))) for _ in SEATS))
    w = [1] * len(SEATS)
    leaders = rng.sample(range(len(SEATS)), 2)
    for j in leaders: w[j] = 3
    for j in rng.sample([k for k in range(len(SEATS)) if k not in leaders], 2): w[j] = 2
    # 12% of options have a structural high-impact downside. This is separate from
    # financial attractiveness, which tests a council's ability to flag tail risk.
    ruin = tuple(rng.random() < 0.13 for _ in OPTIONS)
    common = tuple(rng.gauss(0, 1.85) for _ in OPTIONS)
    return Case(tuple(raw), tuple(w), ruin, common)


def clamp(x: float) -> float:
    return max(0.0, min(10.0, x))


def band(score: float) -> str:
    if score >= 7.5: return 'GO'
    if score >= 6.0: return 'GO WITH CONDITIONS'
    if score >= 4.5: return 'RESHAPE'
    return "DON'T DO IT"


def aggregate(cards: tuple[SeatCard, ...], *, ruin_cap: bool, calls: int) -> Verdict:
    if not cards:
        raise ValueError('at least one seat is required')
    denom = sum(c.weight for c in cards)
    if denom <= 0:
        raise ValueError('seat weights must be positive')
    scores = [sum(c.weight * c.scores[j] for c in cards) / denom for j in range(3)]
    idx = max(range(3), key=lambda i: scores[i])
    score = scores[idx]
    spread = max(c.scores[idx] for c in cards) - min(c.scores[idx] for c in cards)
    cap = ruin_cap and any(c.ruin_flags[idx] for c in cards)
    label = band(score)
    if cap and label in ('GO', 'GO WITH CONDITIONS'):
        label = 'RESHAPE'
    # Warning: the cap does not change which option wins, consistent with the
    # Roast Room documented rule. It restricts the verdict label.
    return Verdict(OPTIONS[idx], round(score, 4), label, round(spread, 4), bool(cap),
                   cards, calls, proxy_tokens=220 + 650 * calls + 45 * len(cards))


def selected_seats(weights: tuple[int, ...], quick: bool) -> tuple[int, ...]:
    if not quick:
        return tuple(range(7))
    top = sorted(range(7), key=lambda i: (-weights[i], i))[:2]
    # A transparent analogue of the source's optional downside/checker seat.
    for i in (6, 0, 3, 1, 2, 4, 5):
        if i not in top:
            top.append(i)
            break
    for i in (3, 2, 1, 4, 5, 0, 6):
        if i not in top:
            top.append(i)
            break
    return tuple(top)


def make_cards(case: Case, rng: random.Random, *, quick: bool = False,
               correlated: bool = False, evidence: bool = False,
               debate: bool = False) -> tuple[SeatCard, ...]:
    indices = selected_seats(case.weights, quick)
    cards = []
    # Independently randomized specialist observations. An auditor can reveal
    # noisy features much more accurately, without ever exposing option utility.
    for j in indices:
        estimates = []
        flags = []
        for i in range(3):
            specialized_sigma = 1.45 / (2.5 if evidence else 1)
            noise = rng.gauss(0, specialized_sigma)
            framing = case.shared_framing[i] * (0.90 if correlated else 0.16)
            estimates.append(clamp(case.values[i][j] + noise + framing))
            # Independent downside audit with sensitivity and specificity.
            sensitivity = 0.87 if evidence else 0.64
            false_alert = 0.018 if evidence else 0.075
            flags.append(rng.random() < (sensitivity if case.ruin[i] else false_alert))
        cards.append(SeatCard(SEATS[j], tuple(estimates), case.weights[j], tuple(flags)))
    if debate:
        # Conversation-only averaging reduces disagreement without adding factual
        # information. This is a control for the fallacy 'consensus = accuracy'.
        mean = [sum(c.scores[i] for c in cards) / len(cards) for i in range(3)]
        cards = [SeatCard(c.seat, tuple(clamp(0.48 * s + 0.52 * mean[i])
                     for i, s in enumerate(c.scores)), c.weight, c.ruin_flags)
                 for c in cards]
    return tuple(cards)


def simulate(case: Case, rng: random.Random, policy: str) -> Verdict:
    if policy == 'solo_generalist':
        avg = case.true_scores()
        # Generalist sees the same task brief but no specialized observations.
        # It receives one noisy summary of whole-option utility.
        cards = (SeatCard('generalist', tuple(clamp(x + rng.gauss(0, 2.4) +
                    0.22 * case.shared_framing[i]) for i, x in enumerate(avg)),
                    1, tuple(rng.random() < (0.45 if risk else 0.04) for risk in case.ruin)),)
        return aggregate(cards, ruin_cap=False, calls=1)
    if policy == 'quick_four':
        return aggregate(make_cards(case, rng, quick=True), ruin_cap=False, calls=4)
    if policy == 'seven_independent':
        return aggregate(make_cards(case, rng), ruin_cap=False, calls=7)
    if policy == 'seven_correlated':
        return aggregate(make_cards(case, rng, correlated=True), ruin_cap=False, calls=7)
    if policy == 'debate_no_evidence':
        return aggregate(make_cards(case, rng, correlated=True, debate=True), ruin_cap=False, calls=21)
    if policy == 'audited_specialists':
        return aggregate(make_cards(case, rng, evidence=True), ruin_cap=False, calls=14)
    if policy == 'audited_with_ruin_cap':
        return aggregate(make_cards(case, rng, evidence=True), ruin_cap=True, calls=14)
    raise ValueError(f'unknown policy: {policy}')

POLICIES = ('solo_generalist', 'quick_four', 'seven_independent',
            'seven_correlated', 'debate_no_evidence',
            'audited_specialists', 'audited_with_ruin_cap')
