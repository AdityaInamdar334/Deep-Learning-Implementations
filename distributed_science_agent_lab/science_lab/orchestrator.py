"""Deterministic agent-role simulation: scope, analyze, audit, synthesize."""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import numpy as np
from .stats import inspect_feature, contrast, benjamini_hochberg


def prioritize_features(study):
    """Discovery-set-only ranking; holdout may not influence task selection."""
    features = []
    for i in range(len(study.names)):
        p, d = contrast(study.x_train, study.y_train, i)
        features.append((i, abs(d)))
    return [i for i, _ in sorted(features, key=lambda v: (-v[1], v[0]))]


def investigate_one(study, idx, latency_s=0.0):
    # Represents a remote literature / tool call, not measured LLM inference.
    if latency_s:
        time.sleep(latency_s)
    return inspect_feature(study, idx)


def execute(study, mode='distributed', workers=1, max_features=None,
            latency_s=0.0, correction=True, verification=True,
            discovery_alpha=0.05, holdout_alpha=0.05):
    if workers < 1 or latency_s < 0:
        raise ValueError('workers >= 1 and latency_s >= 0 required')
    if mode not in ('generalist', 'distributed'):
        raise ValueError('unknown mode')
    rank = prioritize_features(study)
    if max_features is None:
        max_features = 8 if mode == 'generalist' else len(rank)
    if not 1 <= max_features <= len(rank):
        raise ValueError('max_features invalid')
    chosen = rank[:max_features]
    t0 = time.perf_counter()
    if mode == 'distributed' and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            findings = list(pool.map(lambda ix: investigate_one(study, ix, latency_s), chosen))
    else:
        findings = [investigate_one(study, i, latency_s) for i in chosen]
    elapsed = time.perf_counter() - t0
    pvals = np.array([f.discovery_p for f in findings])
    qvals = benjamini_hochberg(pvals) if correction else pvals.copy()
    claims = []
    for f, q in zip(findings, qvals):
        candidate = q < discovery_alpha
        holdout_ok = (f.holdout_p < holdout_alpha and f.effect_d * f.holdout_effect_d > 0)
        accepted = bool(candidate and (holdout_ok if verification else True))
        claims.append({**asdict(f), 'adjusted_discovery_p': float(q),
                       'candidate': bool(candidate), 'replicated': bool(holdout_ok),
                       'accepted': accepted, 'synthetic_null': bool(f.feature_index >= study.observed_count)})
    accepted = [r for r in claims if r['accepted']]
    # A verified claim means independent same-direction holdout evidence. That is
    # statistical replication, not causal proof or a novel biological finding.
    true_heldout = sum(c['replicated'] for c in accepted)
    false_null = sum(c['synthetic_null'] for c in accepted)
    # Cost ledger is a declared proxy, not API-reported token usage.
    overhead = 380
    per_task_input = 230
    per_task_output = 175
    synth_tokens = 250
    calls = 1 if mode == 'generalist' else len(chosen) + 1
    token_proxy = (overhead + max_features * per_task_input + max_features * per_task_output
                   + synth_tokens + (len(chosen) * 80 if mode == 'distributed' else 0))
    # Hypothetical USD / 1M tokens, not vendor pricing.
    estimated_dollars = token_proxy / 1e6 * 3.0
    return {'mode': mode, 'workers': workers, 'seed': study.seed,
            'features_examined': len(chosen), 'features_available': len(rank),
            'accepted_claims': len(accepted), 'independently_replicated_claims': true_heldout,
            'known_null_claims': false_null, 'claims_without_replication': len(accepted)-true_heldout,
            'covered_observed_features': sum(c['accepted'] and not c['synthetic_null'] for c in claims),
            'testable_precision': true_heldout/max(1,len(accepted)),
            'runtime_seconds': elapsed, 'simulated_io_seconds_per_task': latency_s,
            'proxy_tokens': token_proxy, 'proxy_dollars': estimated_dollars,
            'requests_proxy': calls, 'correction': correction, 'verification': verification,
            'claims': claims}
