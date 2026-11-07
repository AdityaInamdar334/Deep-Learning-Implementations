"""Read-only offline scan for selected published Cursor incident indicators.

Checks only a user-specified directory tree. Not an exhaustive forensic scanner.
"""
from pathlib import Path
from .provenance import sha256_file

KNOWN_BAD_EXTENSION_SHA256 = 'ecb58cb29078d9646ad1dcdbe51526f2d9cfea06be5fb2ab183e8ecc3ab6bf15'
MARKER = b'svc-telemetry'
C2_MARKER = b'134.209.82.225'


def scan_root(root: str | Path) -> dict:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f'not a directory: {base}')
    findings = []
    checks = 0
    patterns = [
        '.ssh/authorized_keys',
        'ProgramData/ssh/administrators_authorized_keys',
        '.config/.smt', '.config/.smtk', '.config/smt.cmd',
        'Library/LaunchAgents/com.apple.sysmond.plist',
        'Startup/sysmon.cmd',
    ]
    for relative in patterns:
        path = base / relative
        checks += 1
        if not path.is_file() or path.is_symlink():
            continue
        if relative.endswith('authorized_keys'):
            # Restrict reading size; no print-out of secrets or key contents.
            with path.open('rb') as f:
                found = MARKER in f.read(5_000_000)
            if found:
                findings.append({'kind': 'ssh_key_marker', 'path': str(path)})
        else:
            findings.append({'kind': 'persistence_path_present', 'path': str(path)})

    # Look for extension bundles under the supplied root. Read only likely JS paths.
    for path in base.rglob('extension.js'):
        if not path.is_file() or path.is_symlink():
            continue
        if 'anthropic.claude-code' not in str(path).lower():
            continue
        checks += 1
        if sha256_file(path) == KNOWN_BAD_EXTENSION_SHA256:
            findings.append({'kind': 'known_malicious_extension_hash', 'path': str(path)})
        else:
            with path.open('rb') as f:
                content = f.read(8_000_000)
            if MARKER in content or C2_MARKER in content:
                findings.append({'kind': 'extension_indicator_string', 'path': str(path)})
    return {
        'scanned_root': str(base), 'checks_performed': checks, 'findings': findings,
        'assessment': 'Review findings manually. No findings does NOT prove a host is clean.',
    }
