"""Compare an artifact to an independently obtained, trusted SHA-256 manifest."""
import hashlib
import hmac
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(path: str | Path, trusted_sha256: str) -> dict:
    """Checksum is only meaningful if its expected value comes from a trusted channel."""
    expected = trusted_sha256.strip().lower()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('expected SHA-256 must be 64 hex characters')
    observed = sha256_file(path)
    return {
        'artifact': str(path), 'expected_sha256': expected,
        'observed_sha256': observed,
        'matches_trusted_manifest': hmac.compare_digest(observed, expected),
    }
