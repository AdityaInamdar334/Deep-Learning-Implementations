"""Evaluate synthetic councils against an oracle with known option values."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics
from .core import OPTIONS, POLICIES, make_case, simulate


def single_seed(seed: int, n: int = 600) -> dict:
    case_rng = random.Random(seed)
    cases = [make_case(case_rng) for _ in range(n)]
    result = {}
    for pidx, policy in enumerate(POLICIES):
        # Independent, repeatable RNG for each policy and seed.
        # Paired comparisons reuse the same random evidence where appropriate.
        matched = {'debate_no_evidence': 'seven_correlated',
                   'audited_with_ruin_cap': 'audited_specialists'}
        source_idx = POLICIES.index(matched.get(policy, policy))
        rng = random.Random(seed * 30031 + source_idx * 7001 + 50)
        correct = regret = risky_go = risky_options = passed_alert = 0.0
        spread = cost = calls = 0.0
        for case in cases:
            v = simulate(case, rng, policy)
            best = case.best()
            correct += v.recommended == best
            true_scores = case.true_scores()
            regret += max(true_scores) - true_scores[OPTIONS.index(v.recommended)]
            is_risky = case.ruin[OPTIONS.index(v.recommended)]
            risky_options += is_risky
            risky_go += is_risky and v.label == 'GO'
            passed_alert += is_risky and v.label not in ('GO', 'GO WITH CONDITIONS')
            spread += v.spread
            cost += v.proxy_tokens
            calls += v.calls
        result[policy] = {
            'oracle_choice_accuracy': correct / n,
            'mean_regret': regret / n,
            'unsafe_go_rate_per_case': risky_go / n,
            'risky_recommendation_rate': risky_options / n,
            'risk_held_rate_per_case': passed_alert / n,
            'mean_spread': spread / n,
            'mean_calls': calls / n,
            'mean_proxy_tokens': cost / n,
        }
    return result


def run(seeds: int = 5, cases: int = 600) -> dict:
    if seeds < 1 or cases < 1:
        raise ValueError('seeds and cases must be positive')
    trials = [single_seed(seed, cases) for seed in range(seeds)]
    means = {}
    for policy in POLICIES:
        means[policy] = {}
        for key in trials[0][policy]:
            numbers = [trial[policy][key] for trial in trials]
            means[policy][key] = {'mean': statistics.mean(numbers),
                                 'sample_std': statistics.stdev(numbers) if len(numbers) > 1 else 0.0}
    return {'metadata': {'seeds': seeds, 'cases_per_seed': cases,
                         'dataset': 'synthetic case generator with known oracle',
                         'agents': 'scripted statistical simulations, no LLM calls',
                         'cost': 'fictional proxy tokens; NOT actual API usage',
                         'source_relationship': 'inspired by Roast Room; not an implementation or reproduction'},
            'means': means, 'trials': trials}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--cases', type=int, default=600)
    ap.add_argument('--output', default='results/benchmark.json')
    args = ap.parse_args()
    result = run(args.seeds, args.cases)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {target} | {args.seeds} seeds x {args.cases} decisions')
    print('policy                 accuracy  regret  unsafe GO  spread  calls  proxy tokens')
    for name, r in result['means'].items():
        m = {k: x['mean'] for k, x in r.items()}
        print(f'{name:23s} {m["oracle_choice_accuracy"]:7.3%} {m["mean_regret"]:7.3f}'
              f' {m["unsafe_go_rate_per_case"]:9.3%} {m["mean_spread"]:7.3f}'
              f' {m["mean_calls"]:5.1f} {m["mean_proxy_tokens"]:12.1f}')

if __name__ == '__main__':
    main()
