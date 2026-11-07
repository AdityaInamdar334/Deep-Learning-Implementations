# IDE Trust Boundary Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Standard Library Only](https://img.shields.io/badge/dependencies-stdlib_only-green.svg)](https://docs.python.org/3/library/)
[![Tests](https://img.shields.io/badge/tests-10%2F10%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Defensive / Read-Only](https://img.shields.io/badge/safety-read--only%20defensive-lightgrey.svg)](defensive_lab/)

A **fully offline, standard-library-only, read-only** companion to the article about the October 2026 Cursor, Langflow, and Visual Studio Copilot security incidents.

## Safety and scope

This lab contains **no malware**, no reverse SSH tunnel, no exploit, no network client, and no secret-collection code. It **does not** install or remove anything, contact a service, launch an extension, or alter a machine. All potentially dangerous behaviors are replaced with harmless text fixtures and offline classification. The path comparison is a conceptual simulation of Win32 basename normalization, not an actual filesystem operation.

**Not a forensic clearance tool.** Negative findings do not establish that a machine is uncompromised. Refer to the vendor's official remediation scripts and qualified incident response team for real incidents. Only scan directories you own or are authorized to inspect.

## Run

Requires Python 3.10+; no dependencies.

```bash
python -m unittest discover -s tests -v
python -m defensive_lab.cli paths
python -m defensive_lab.cli langflow 1.12.8
python -m defensive_lab.cli langflow 1.12.10
python -m defensive_lab.cli langflow 1.13.1
python -m defensive_lab.cli scan fixtures
```

### Exercise 1: Publisher ID versus content provenance

Both fixtures carry the exact same `publisher`, `name`, and `version` strings, but contain different bytes. Run:

```bash
python -m defensive_lab.cli verify fixtures/publisher-clean.txt \
  "$(python -c 'from defensive_lab.provenance import sha256_file; print(sha256_file("fixtures/publisher-clean.txt"))')"
python -m defensive_lab.cli verify fixtures/mirror-different-bytes.txt \
  "$(python -c 'from defensive_lab.provenance import sha256_file; print(sha256_file("fixtures/publisher-clean.txt"))')"
```

The first matches and the second fails. In a real system, the expected hash must come from a genuinely independent, authenticated channel. A checksum delivered by the same compromised mirror proves nothing about authenticity.

### Exercise 2: Compare the checked name to the written name

```bash
python -m defensive_lab.cli paths
```

A toy policy checks `protected.conf` against a raw spelling. `protected.conf.` evades that raw comparison while mapping to the protected name under simulated Win32 normalization. Correct authorization requires OS-aware canonicalization and validation of the **same final target** the writer will use. This lab has no actual file-writing operations.

### Exercise 3: Local version inventory

```bash
python -m defensive_lab.cli langflow 1.12.8
python -m defensive_lab.cli langflow 1.12.10
python -m defensive_lab.cli langflow 1.13.1
```

Based solely on IBM's October 29 bulletin, 1.0.0–1.12.8 are affected and 1.13.0 was named as the fixed version. A 1.12.9 or 1.12.10 maintenance release is *not verified as safe by that bulletin*. Confirm later fixes explicitly with the vendor instead of assuming version ordering is proof.

### Exercise 4: Read-only local IOC triage

```bash
python -m defensive_lab.cli scan /path/to/a/directory-you-control
```

Only searches files inside the given root for selected published markers and persistence file names. Does not enumerate system processes, scheduled tasks, network connections, or external logs. Start with a synthetic fixture directory, not your production machine. A finding is a prompt for human triage, not a conviction. An empty report is **not proof of safety**.

## Reproducibility

No randomness, downloads, private data, external services, or platform-specific dependencies. Ten `unittest` cases cover all four exercises.

## File organization

```
defensive_lab/     checksum validation, pure filename simulation, version triage, local IOC search
fixtures/          safe text-only artifacts
results/           example outputs
 tests/            ten unit tests
SOURCES.md         vendor references, scope and limitations
```
