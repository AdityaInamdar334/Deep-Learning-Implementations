"""Safe *string-only* Windows filename normalization demonstration.

Not a general Win32 path parser or a replacement for OS-level path policy.
It never accesses or writes a filesystem path.
"""


def simulate_win32_basename(name: str) -> str:
    """Models common Win32 trimming of trailing ASCII spaces and dots."""
    if '/' in name or '\\' in name or not name:
        raise ValueError('demo accepts a single basename only')
    return name.rstrip(' .').casefold()


def naive_protected_check(raw_name: str, protected: set[str]) -> bool:
    """Unsafe example: compare raw spelling, not the target name."""
    return raw_name.casefold() in {n.casefold() for n in protected}


def canonical_protected_check(raw_name: str, protected: set[str]) -> bool:
    """Safe *for the toy model*: check the same form used by the simulated writer."""
    return simulate_win32_basename(raw_name) in {
        simulate_win32_basename(name) for name in protected
    }
