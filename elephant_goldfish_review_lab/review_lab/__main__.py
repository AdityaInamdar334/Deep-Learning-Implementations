from __future__ import annotations
import argparse
import json
from pathlib import Path
from .core import prepare, lint_spec, gate


def main():
    ap = argparse.ArgumentParser(description='Educational fresh-context spec review packet generator and conservative gate')
    subs = ap.add_subparsers(dest='cmd', required=True)
    a = subs.add_parser('lint', help='Run limited deterministic section and placeholder checks')
    a.add_argument('spec', type=Path)
    a = subs.add_parser('prepare', help='Generate self-contained packets for manual fresh sessions')
    a.add_argument('spec', type=Path)
    a.add_argument('--out', type=Path, default=Path('out/review1'))
    a = subs.add_parser('gate', help='Validate manually saved JSON reviews; fail closed')
    a.add_argument('spec', type=Path)
    a.add_argument('--responses', type=Path, required=True)
    a.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.cmd == 'lint':
        results = lint_spec(args.spec.read_text(encoding='utf-8'))
        print(json.dumps({'heuristic_findings': results, 'not_a_security_or_readiness_verdict': True}, indent=2))
    elif args.cmd == 'prepare':
        print(json.dumps(prepare(args.spec, args.out), indent=2))
    elif args.cmd == 'gate':
        results = gate(args.spec, args.responses)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(results, indent=2))
        if results['verdict'].startswith('NO-GO'):
            raise SystemExit(2)

if __name__ == '__main__':
    main()
