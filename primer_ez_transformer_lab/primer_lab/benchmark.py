"""Synthetic ablations with explicit timing, seeds, and independent validation sets."""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
import torch
from torch import nn
from .data import periodic_sequences, teacher_forcing
from .model import make_model, num_parameters, ABLATIONS


def evaluate(model, batches, period: int, noise: float, vocab_size: int, context: int, seed: int) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses, correct, n = [], 0, 0
    with torch.inference_mode():
        for _ in range(batches):
            x, y = teacher_forcing(periodic_sequences(64, context, period, vocab_size, gen, noise))
            logits = model(x)
            per_token = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction='none')
            # Startup tokens are inherently unpredictable. Score only positions where
            # the required antecedent has appeared in the context.
            active = torch.arange(context) >= (period - 1)
            losses.append(float(per_token.reshape(64, context)[:, active].mean()))
            predict = logits.argmax(-1)
            correct += int((predict[:, active] == y[:, active]).sum())
            n += int(y[:, active].numel())
    return {'nll': sum(losses) / len(losses),
            'perplexity': math.exp(min(30, sum(losses) / len(losses))),
            'accuracy': correct / n, 'evaluated_tokens': n}


def run_trial(name: str, seed: int, steps=120, batch_size=32, context=24,
              d_model=48, layers=2, d_ff=144, vocab_size=16, period=4,
              noise=.025, eval_batches=4) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    model = make_model(name, vocab_size=vocab_size, context=context, d_model=d_model,
                       d_ff=d_ff, layers=layers)
    params = num_parameters(model)
    optim = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    gen = torch.Generator().manual_seed(10_000 + seed)
    training_curve = []
    model.train()
    start = time.perf_counter()
    for step in range(1, steps + 1):
        samples = periodic_sequences(batch_size, context, period, vocab_size, gen, noise)
        x, y = teacher_forcing(samples)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step in {1, max(1, steps // 4), max(1, steps // 2), steps}:
            training_curve.append({'step': step, 'loss': float(loss.detach())})
    elapsed = time.perf_counter() - start
    # Independent generator seeds, shared across configurations. OOD varies motif period only.
    in_dist = evaluate(model, eval_batches, period, noise, vocab_size, context, 90_000 + seed)
    shifted = evaluate(model, eval_batches, min(period + 2, context - 1), noise,
                       vocab_size, context, 90_000 + seed)
    return {'name': name, 'seed': seed, 'parameters': params, 'training_seconds': elapsed,
            'training_tokens': batch_size * context * steps,
            'train_tokens_per_second': batch_size * context * steps / elapsed,
            'in_distribution': in_dist, 'shifted_period': shifted,
            'training_curve': training_curve}


def summarize(trials):
    output = {}
    for name in sorted(set(x['name'] for x in trials)):
        rows = [x for x in trials if x['name'] == name]
        keys = {'id_nll': [r['in_distribution']['nll'] for r in rows],
                'id_accuracy': [r['in_distribution']['accuracy'] for r in rows],
                'ood_nll': [r['shifted_period']['nll'] for r in rows],
                'ood_accuracy': [r['shifted_period']['accuracy'] for r in rows],
                'time_seconds': [r['training_seconds'] for r in rows],
                'tokens_per_second': [r['train_tokens_per_second'] for r in rows]}
        output[name] = {'parameters': rows[0]['parameters'],
                        **{k: {'mean': statistics.mean(v),
                               'sample_std': statistics.stdev(v) if len(v) > 1 else 0.0}
                           for k, v in keys.items()}}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--start-seed', type=int, default=0,
                   help='Seed offset to support independently reproducible chunks')
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--context', type=int, default=24)
    p.add_argument('--eval-batches', type=int, default=4)
    p.add_argument('--configs', nargs='+', default=['baseline_relu', 'squared_relu_only',
                                                   'dconv_only', 'primer_ez',
                                                   'gelu_baseline', 'primer_per_head',
                                                   'q_only', 'k_only', 'v_only'],
                   choices=sorted(ABLATIONS))
    p.add_argument('--output', default='results/benchmark.json')
    args = p.parse_args()
    if args.seeds < 1 or args.steps < 1:
        p.error('--seeds and --steps must be positive')
    torch.set_num_threads(min(2, torch.get_num_threads()))
    trials = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        for name in args.configs:
            row = run_trial(name, seed, steps=args.steps, batch_size=args.batch_size,
                            context=args.context, eval_batches=args.eval_batches)
            trials.append(row)
            print(f'{name:21s} seed={seed} id_nll={row["in_distribution"]["nll"]:.3f} '
                  f'ood_nll={row["shifted_period"]["nll"]:.3f} '
                  f'time={row["training_seconds"]:.2f}s', flush=True)
    report = {'protocol': {'seeds': args.seeds, 'start_seed': args.start_seed, 'steps': args.steps, 'batch_size': args.batch_size,
                           'context': args.context, 'period_train': 4, 'period_shift': 6,
                           'vocab_size': 16, 'noise': .025, 'dataset': 'synthetic periodic motif',
                           'device': 'CPU', 'torch': torch.__version__,
                           'eval_batches': args.eval_batches,
                           'notes': 'Timing is environment-specific, not hardware-controlled. '
                                    'These are not the official Primer/Google benchmarks.'},
              'trials': trials, 'summary': summarize(trials)}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print('Saved', target, flush=True)


if __name__ == '__main__':
    main()
