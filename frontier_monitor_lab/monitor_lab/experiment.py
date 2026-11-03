"""Train probes and run paired 3-seed synthetic ablation evaluation."""
import argparse
import json
from pathlib import Path
import torch
from .data import generate_dataset
from .models import (LinearProbe, TextReviewer, build_vocab, features_text, features_probe,
                     train_model, probabilities, recall_threshold)
from .pipeline import Config, evaluate
from .metrics import summarize, mean_sd


def run_seed(seed, train_n=700, val_n=300, test_n=500, epochs=65):
    torch.manual_seed(seed)
    train = generate_dataset(train_n, seed, 'train')
    val = generate_dataset(val_n, seed, 'val')
    test = generate_dataset(test_n, seed, 'test')
    shift = generate_dataset(test_n, seed, 'shift', shift=True)
    joint_shift = generate_dataset(test_n, seed, 'joint_shift', shift=True, activation_shift=True)
    vocab = build_vocab(train)
    x_train = features_probe(train)
    probe = LinearProbe(x_train.shape[1])
    probe = train_model(probe, x_train, [c.label for c in train], seed, epochs=epochs)
    val_ps = probabilities(probe, features_probe(val))
    pt = recall_threshold(val_ps, [c.label for c in val], target_recall=.90)
    model = TextReviewer(len(vocab))
    model = train_model(model, features_text(train, vocab), [c.label for c in train], seed+80, epochs=epochs)
    val_s = probabilities(model, features_text(val, vocab))
    lt = recall_threshold(val_s, [c.label for c in val], target_recall=.90)
    # Separate text-only model, trained from scratch on the same examples.
    text_model = TextReviewer(len(vocab))
    text_model = train_model(text_model, features_text(train, vocab, text_only=True),
                             [c.label for c in train], seed+90, epochs=epochs)
    text_t = recall_threshold(probabilities(text_model, features_text(val, vocab, text_only=True)),
                              [c.label for c in val], target_recall=.90)
    data = {}
    scenarios = {}
    for split, cases in [('in_distribution', test), ('wording_shift', shift), ('joint_shift', joint_shift)]:
        p0 = probabilities(probe, features_probe(cases))
        p1 = probabilities(model, features_text(cases, vocab))
        txt_scores = probabilities(text_model, features_text(cases, vocab, text_only=True))
        configs = {
            'layered_5pct_audit': (Config(probe_threshold=pt, l2_threshold=lt), p1),
            'layered_no_audit': (Config(probe_threshold=pt, l2_threshold=lt, audit_rate=0), p1),
            'all_to_reviewer': (Config(use_probe=False, l2_threshold=lt), p1),
            'full_investigation': (Config(use_probe=False, l2_threshold=0.), p1),
            'no_l3_telemetry': (Config(probe_threshold=pt, l2_threshold=lt, use_telemetry=False), p1),
            'skip_compaction': (Config(probe_threshold=pt, l2_threshold=lt, skip_compaction=True), p1),
            'text_only_reviewer': (Config(probe_threshold=pt, l2_threshold=text_t), txt_scores),
        }
        for name, (cfg, scores) in configs.items():
            records = evaluate(cases, p0, scores, cfg, seed)
            data[f'{split}/{name}'] = summarize(records)
        scenarios[split] = {}
        records = evaluate(cases, p0, p1, configs['layered_5pct_audit'][0], seed)
        for scenario in sorted({c.scenario for c in cases}):
            selected = [r for r in records if r['scenario'] == scenario]
            scenarios[split][scenario] = {'n': len(selected), 'l1_fraction':sum(r['screened'] for r in selected)/len(selected),
                                          'l2_flag_fraction':sum(r['security_queue'] for r in selected)/len(selected)}
    return dict(seed=seed, thresholds=dict(probe=pt, reviewer=lt, text_only=text_t),
                results=data, scenarios=scenarios)


def run_experiment(seeds=(0, 1, 2), train_n=700, val_n=300, test_n=500, epochs=65):
    trials = [run_seed(s, train_n, val_n, test_n, epochs) for s in seeds]
    metrics = ['l1_recall', 'l2_recall', 'p0_recall', 'l2_precision', 'l2_fpr',
               'p0_fpr', 'mean_cost_units', 'l1_fraction', 'coverage']
    aggregate = {}
    for name in trials[0]['results']:
        aggregate[name] = {}
        for metric in metrics:
            avg, sd = mean_sd([trial['results'][name][metric] for trial in trials])
            aggregate[name][metric] = {'mean': avg, 'sd': sd}
    return dict(metadata=dict(description='Synthetic educational proxy monitor, not OpenAI monitoring',
                              seeds=list(seeds), train_n=train_n, val_n=val_n, test_n=test_n, epochs=epochs,
                              cost_units='fictional illustrative units, not dollars or measured GPU time'),
                aggregate=aggregate, trials=trials)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--train', type=int, default=700)
    parser.add_argument('--val', type=int, default=300)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--output', default='results/benchmark.json')
    args = parser.parse_args()
    result = run_experiment(tuple(range(args.seeds)), args.train, args.val, args.test, args.epochs)
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Synthetic benchmark only. No frontier LLM or proprietary monitor is executed.')
    for name, d in result['aggregate'].items():
        print(f'{name:42s} L1 recall {d["l1_recall"]["mean"]:.3f}, L2 recall {d["l2_recall"]["mean"]:.3f}, '
              f'P0 recall {d["p0_recall"]["mean"]:.3f}, cost {d["mean_cost_units"]["mean"]:.1f}')
    print('Saved', path)


if __name__ == '__main__':
    main()
