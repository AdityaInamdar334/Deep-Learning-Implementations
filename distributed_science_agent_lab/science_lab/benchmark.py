"""CLI benchmark for serial budgeted, full serial, and parallel specialists."""
import argparse, json, statistics, time
from pathlib import Path
from .data import make_study
from .orchestrator import execute


def benchmark(seeds=5, null_features=24, latency_ms=20.0, workers=6):
    modes = [
        ('generalist_8', dict(mode='generalist', workers=1, max_features=8)),
        ('specialists_8_verified', dict(mode='distributed', workers=1, max_features=8)),
        ('specialists_all_serial', dict(mode='distributed', workers=1)),
        ('specialists_all_parallel', dict(mode='distributed', workers=workers)),
        ('no_holdout_verification', dict(mode='distributed', workers=workers, verification=False)),
        ('no_multiple_testing_correction', dict(mode='distributed', workers=workers, correction=False)),
        ('neither_safeguard', dict(mode='distributed', workers=workers, correction=False, verification=False)),
    ]
    experiments = []
    for s in range(seeds):
        study = make_study(seed=s, null_count=null_features)
        for label, cfg in modes:
            row = execute(study, latency_s=latency_ms/1000.0, **cfg)
            row['configuration'] = label
            experiments.append(row)
    summary = {}
    metrics = ('features_examined','accepted_claims','independently_replicated_claims',
               'known_null_claims','claims_without_replication','testable_precision',
               'runtime_seconds','proxy_tokens','proxy_dollars')
    for label, _ in modes:
        rows = [r for r in experiments if r['configuration'] == label]
        summary[label] = {metric: {'mean': statistics.mean(r[metric] for r in rows),
                                    'std': statistics.stdev(r[metric] for r in rows) if len(rows)>1 else 0.0}
                          for metric in metrics}
    return {'metadata': {'dataset': 'sklearn Wine, classes 0 and 1',
                         'null_features': null_features, 'seeds': seeds,
                         'simulated_io_delay_ms': latency_ms, 'workers': workers,
                         'purpose':'Architecture demo, no LLM calls, no biological discovery'},
            'summary': summary, 'runs': experiments}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--null-features', type=int, default=24)
    parser.add_argument('--latency-ms', type=float, default=20)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output', default='results/benchmark.json')
    args = parser.parse_args()
    result = benchmark(args.seeds,args.null_features,args.latency_ms,args.workers)
    path = Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    for name, group in result['summary'].items():
        print(f"{name:36} examined={group['features_examined']['mean']:.1f} "
              f"accepted={group['accepted_claims']['mean']:.2f} "
              f"replicated={group['independently_replicated_claims']['mean']:.2f} "
              f"null={group['known_null_claims']['mean']:.2f} "
              f"elapsed={group['runtime_seconds']['mean']:.3f}s")
    print('Saved',path)

if __name__ == '__main__':
    main()
