"""Version triage reflects only the public IBM bulletin dated 2026-10-29.

Later branch patch statuses need authoritative confirmation; do not infer from ordering.
"""
import re


def langflow_triage(version: str) -> dict:
    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', version.strip())
    if not match:
        raise ValueError('provide a release version such as 1.12.8; pre-releases require review')
    parsed = tuple(map(int, match.groups()))
    if (1, 0, 0) <= parsed <= (1, 12, 8):
        status = 'affected_in_IBM_bulletin'
        action = 'Isolate and upgrade using vendor guidance; 1.13.0 is named as a fixed release.'
    elif parsed in {(1, 12, 9), (1, 12, 10)}:
        status = 'later_backport_not_verified_by_cited_bulletin'
        action = 'Do not assume the CVEs were fixed. Request vendor confirmation or upgrade to supported 1.13.x.'
    elif parsed >= (1, 13, 0):
        status = 'outside_original_affected_range'
        action = 'Confirm installed build and later advisories; this is not a full vulnerability clearance.'
    else:
        status = 'outside_bulletin_scope_or_unrecognized'
        action = 'Check current vendor advisories before interpreting this version.'
    return {'version': version, 'status': status, 'action': action}
