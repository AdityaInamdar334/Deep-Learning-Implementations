"""Compare trusted compiler commands. Python 3.9+, standard library only."""
import argparse
import json
import shlex
import statistics
import subprocess
import time
from pathlib import Path


def run_comparison(cwd='.', runs=7, timeout=120.0, reference=None, candidate=None, fast=None, output='compiler-comparison', overwrite=False):
    if runs < 1 or timeout <= 0:
        raise ValueError('runs and timeout must be positive')
    out = Path(output).resolve()
    if out.exists() and overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=False)

    commands = {
        'reference': shlex.split(reference) if isinstance(reference, str) else list(reference),
        'candidate': shlex.split(candidate) if isinstance(candidate, str) else list(candidate),
        'fast': shlex.split(fast) if isinstance(fast, str) else list(fast)
    }

    rows = []
    for i in range(runs):
        names = list(commands)
        # Cyclic round-robin permutation to neutralize cold-cache and thermal throttling bias
        names = names[i % 3:] + names[:i % 3]
        for name in names:
            start = time.perf_counter()
            status, timed_out = None, False
            try:
                r = subprocess.run(commands[name], cwd=cwd, capture_output=True, timeout=timeout)
                stdout, stderr, status = r.stdout, r.stderr, r.returncode
            except subprocess.TimeoutExpired as e:
                stdout, stderr, timed_out = e.stdout or b'', e.stderr or b'', True
            except OSError as e:
                stdout, stderr = b'', str(e).encode()
            elapsed = time.perf_counter() - start
            stem = f'{name}-{i + 1}'
            (out / f'{stem}.stdout').write_bytes(stdout)
            (out / f'{stem}.stderr').write_bytes(stderr)
            rows.append(dict(name=name, round=i + 1, seconds=elapsed, exit=status, timeout=timed_out, log=stem))

    def signature(row):
        return (
            row['exit'],
            row['timeout'],
            (out / (row['log'] + '.stdout')).read_bytes(),
            (out / (row['log'] + '.stderr')).read_bytes()
        )

    reference_rows = [r for r in rows if r['name'] == 'reference']
    baseline = signature(reference_rows[0])
    report = [
        '# Compiler comparison',
        '',
        'Fresh-process wall time includes launcher overhead. No memory measurement.',
        '',
        '| Mode | Median seconds | Stable output | Matches reference |',
        '| --- | ---: | --- | --- |'
    ]

    summary = {}
    for name in commands:
        group = [r for r in rows if r['name'] == name]
        valid = all(r['exit'] in (0, 1, 2) and not r['timeout'] for r in group)
        stable = all(signature(r) == signature(group[0]) for r in group)
        matches = valid and all(signature(r) == baseline for r in group) and all(signature(r) == baseline for r in reference_rows)
        median_sec = statistics.median(r["seconds"] for r in group) if valid else None
        median_str = f'{median_sec:.4f}' if valid else 'invalid run'
        report.append(f'| {name} | {median_str} | {stable} | {matches} |')
        summary[name] = {
            'median_seconds': median_sec,
            'stable_output': stable,
            'matches_reference': matches,
            'valid': valid,
            'run_times': [r['seconds'] for r in group]
        }

    results_data = dict(
        cwd=str(Path(cwd).resolve()),
        commands=commands,
        runs=rows,
        summary=summary
    )

    (out / 'results.json').write_text(json.dumps(results_data, indent=2))
    report_text = '\n'.join(report) + '\n'
    (out / 'REPORT.md').write_text(report_text)

    return out / 'REPORT.md', results_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cwd', default='.')
    p.add_argument('--runs', type=int, default=7)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--fast', required=True)
    p.add_argument('--output', default='compiler-comparison')
    p.add_argument('--overwrite', action='store_true', help='Overwrite output directory if exists')
    a = p.parse_args()

    report_path, _ = run_comparison(
        cwd=a.cwd,
        runs=a.runs,
        timeout=a.timeout,
        reference=a.reference,
        candidate=a.candidate,
        fast=a.fast,
        output=a.output,
        overwrite=a.overwrite
    )
    print(report_path)


if __name__ == '__main__':
    main()

