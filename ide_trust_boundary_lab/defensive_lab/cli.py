import argparse
import json
from pathlib import Path
from .provenance import verify_artifact
from .paths import naive_protected_check, canonical_protected_check
from .versions import langflow_triage
from .ioc import scan_root


def main():
    parser = argparse.ArgumentParser(description='Offline, read-only trust-boundary lab')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('verify', help='Compare local bytes to independently trusted SHA-256')
    p.add_argument('file')
    p.add_argument('trusted_sha256')
    p = sub.add_parser('scan', help='Read-only scan of explicit root')
    p.add_argument('root')
    p = sub.add_parser('langflow', help='Offline version triage from published bulletin')
    p.add_argument('version')
    sub.add_parser('paths', help='Demonstrate raw versus normalized filename checks')
    args = parser.parse_args()
    if args.command == 'verify':
        result = verify_artifact(args.file, args.trusted_sha256)
    elif args.command == 'scan':
        result = scan_root(args.root)
    elif args.command == 'langflow':
        result = langflow_triage(args.version)
    else:
        protected = {'protected.conf'}
        result = {
            item: {'naive_blocked': naive_protected_check(item, protected),
                   'canonical_blocked': canonical_protected_check(item, protected)}
            for item in ('protected.conf', 'protected.conf.', 'protected.conf ')
        }
        result['caveat'] = 'Pure string toy; never substitute for OS path authorization.'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
