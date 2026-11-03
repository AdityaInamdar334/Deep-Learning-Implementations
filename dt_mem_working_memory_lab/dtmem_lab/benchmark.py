"""Five-seed task adaptation experiment and controlled memory ablations."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .memory import SoftWorkingMemory
from .toy_tasks import baseline_predict, draw_support, sample_task, train_context_baseline

CONFIGS = (
    ('baseline_no_memory', 0, 8, 0.0, False),
    ('memory_8_slots_support_16', 16, 8, 0.0, False),
    ('memory_8_slots_support_4', 4, 8, 0.0, False),
    ('memory_8_slots_support_32', 32, 8, 0.0, False),
    ('memory_4_slots_support_16', 16, 4, 0.0, False),
    ('memory_2_slots_support_16', 16, 2, 0.0, False),
    ('memory_8_slots_corrupt_25pct', 16, 8, 0.25, False),
    ('memory_8_slots_shuffled_labels', 16, 8, 0.0, True),
)


def evaluate(seed, n_tasks=100, query_per_task=64, baseline_steps=180):
    g = torch.Generator().manual_seed(1000 + seed)
    model = train_context_baseline(seed, steps=baseline_steps)
    settings = {config[0]: {'correct': 0, 'total': 0, 'seen_correct': 0, 'seen_total': 0,
                             'unseen_correct': 0, 'unseen_total': 0} for config in CONFIGS}
    for task_i in range(n_tasks):
        mapping = sample_task(g)
        query = torch.randint(0, 8, (query_per_task,), generator=g)
        target = mapping[query]
        baseline = baseline_predict(model, query)
        for name, support_n, slots, corruption, shuffle in CONFIGS:
            if support_n == 0:
                predictions = baseline
                seen = torch.zeros_like(query, dtype=torch.bool)
            else:
                # Re-seed each setting per task. Makes comparisons repeatable.
                local_g = torch.Generator().manual_seed(seed * 100000 + task_i * 17 + 300)
                support_ctx, support_labels = draw_support(mapping, support_n, local_g, corruption=corruption)
                if shuffle:
                    support_labels = support_labels[torch.randperm(support_n, generator=local_g)]
                memory = SoftWorkingMemory(slots=slots)
                memory.write(support_ctx, support_labels)
                predictions = memory.predict(query, baseline)
                seen = (query[:, None] == support_ctx[None, :]).any(-1)
            s = settings[name]
            success = predictions.eq(target)
            s['correct'] += int(success.sum())
            s['total'] += query_per_task
            s['seen_correct'] += int(success[seen].sum())
            s['seen_total'] += int(seen.sum())
            s['unseen_correct'] += int(success[~seen].sum())
            s['unseen_total'] += int((~seen).sum())
    for s in settings.values():
        s['accuracy'] = s['correct'] / s['total']
        s['seen_accuracy'] = s['seen_correct'] / s['seen_total'] if s['seen_total'] else None
        s['unseen_accuracy'] = s['unseen_correct'] / s['unseen_total'] if s['unseen_total'] else None
    return settings


def run(seeds=5, n_tasks=100, query_per_task=64, baseline_steps=180):
    torch.set_num_threads(1)
    records = []
    start = time.perf_counter()
    for seed in range(seeds):
        records.append(evaluate(seed, n_tasks=n_tasks, query_per_task=query_per_task,
                                baseline_steps=baseline_steps))
    summaries = {}
    for name, *_ in CONFIGS:
        vals = [r[name]['accuracy'] for r in records]
        summaries[name] = {
            'mean_accuracy': statistics.mean(vals),
            'sample_sd': statistics.stdev(vals) if seeds > 1 else None,
            'per_seed_accuracy': vals,
            'mean_seen_accuracy': statistics.mean([r[name]['seen_accuracy'] for r in records
                          if r[name]['seen_accuracy'] is not None]) if any(r[name]['seen_accuracy'] is not None for r in records) else None,
            'mean_unseen_accuracy': statistics.mean([r[name]['unseen_accuracy'] for r in records
                          if r[name]['unseen_accuracy'] is not None]) if any(r[name]['unseen_accuracy'] is not None for r in records) else None,
        }
    return {'metadata': {'synthetic': True, 'paper_reproduction': False, 'seeds': seeds,
                         'heldout_tasks_per_seed': n_tasks, 'queries_per_task': query_per_task,
                         'baseline_training_steps': baseline_steps,
                         'torch_version': torch.__version__,
                         'elapsed_seconds': round(time.perf_counter()-start, 3)},
            'configurations': summaries}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--tasks', type=int, default=100)
    p.add_argument('--queries', type=int, default=64)
    p.add_argument('--steps', type=int, default=180)
    p.add_argument('--output', default='results/benchmark.json')
    a = p.parse_args()
    if min(a.seeds, a.tasks, a.queries, a.steps) < 1:
        p.error('all counts must be positive')
    result = run(a.seeds, a.tasks, a.queries, a.steps)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
