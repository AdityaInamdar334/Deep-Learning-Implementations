# Superagent: Context, Authority, and Lifecycle Lab

[![Node.js 24](https://img.shields.io/badge/node.js-24%2B-green.svg)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-types--transform-blue.svg)](https://www.typescriptlang.org/)
[![Tests](https://img.shields.io/badge/tests-11%2F11%20passed-brightgreen.svg)](tests.mjs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Network / Zero External API](https://img.shields.io/badge/environment-hermetic-lightgrey.svg)](harness.mjs)

> A rigorous study and tutorial implementation of autonomous agent context management, admission authority boundaries, evaluation cache invalidation, mandate immutability against prompt injection, and human-in-the-loop lifecycle coordination. Upstream core vendored from DeepSeek Superagent prototype (`vendor/core/`).

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Explicit Scope & Disclosures](#explicit-scope--disclosures)
   - [What Runs](#what-runs)
   - [What This Lab Does Not Claim](#what-this-lab-does-not-claim)
   - [Copyright & Vendored Licensing](#copyright--vendored-licensing)
3. [Visual Architecture Gallery](#visual-architecture-gallery)
4. [Lifecycle & Authority Invariants](#lifecycle--authority-invariants)
5. [Quickstart & Execution Guide](#quickstart--execution-guide)
6. [Formal Unit Test Verification Matrix](#formal-unit-test-verification-matrix)
7. [Repository File Map](#repository-file-map)

---

## Executive Summary

Autonomous agent architectures in enterprise and developer workflows often suffer from structural failure modes:
1. **Context Pollution & Mandate Drift:** External comments, chat notifications, or third-party webhooks overwrite the initial system prompt or user mandate.
2. **Replay & Duplicate Execution:** Webhook retries or network blips re-trigger irreversible actions.
3. **Stale Admission Policies:** Policy or security rules change, yet cached worker decisions continue running against obsolete evaluations.
4. **Autonomous Runaway:** Completed tasks automatically reawaken when bots or unauthorized actors leave comments.

The **Superagent Context, Authority, and Lifecycle Lab** implements an end-to-end hermetic test harness validating the core governance primitives of the DeepSeek Superagent prototype:
- **Immutable Mandate Snapshotting:** Once work is accepted, outsider messages are quarantined to trigger event streams and cannot mutate the underlying mandate.
- **Idempotent Intake:** Stable SHA-256 event IDs prevent duplicate work intake.
- **Versioned Policy Evaluations:** Cache invalidation ensures that changing policy versions immediately invalidates previously authorized evaluations.
- **Human-in-the-Loop Interlocks:** Human stop signals halt workflows permanently against bot re-awakenings; only authorized human interventions can resume completed work.

---

## Explicit Scope & Disclosures

### What Runs
- `vendor/core` is verbatim upstream code at commit `8c32a9dc5d72c033816c6a8110b37f40cf94fba4` from [deepseek-ai/dsh-superagent-prototype](https://github.com/deepseek-ai/dsh-superagent-prototype). Upstream license is MIT (`vendor/LICENSE`).
- `harness.mjs` provides tutorial adapters: an Alice-only admission policy, captured assessments, simulated workers, in-memory message buses, and upstream `MemoryRepositories`.
- Requires Node.js 24 (`node --experimental-transform-types`). Zero npm dependencies, zero network requests, zero model API costs, and zero GPU requirements.

### What This Lab Does Not Claim
- **No LLM Semantic Guarantee:** These tests verify the state machine, admission logic, and repository invariants. They do not claim that an unconstrained LLM will always follow semantics if OS boundaries are absent.
- **No SQLite Durability Claim:** The demo uses in-memory repositories; production systems require transactional persistence and write-ahead logging (WAL).
- **Not a Production Runtime:** This harness demonstrates the architectural boundary patterns. It does not provide real sandboxed container execution or remote RPC delivery.

### Copyright & Vendored Licensing
- Vendored upstream code (`vendor/core/`) retains DeepSeek AI copyright under the MIT License.
- Companion tutorial harness, demo, tests, visualizations, and documentation are authored by Aditya Inamdar and released under the MIT License ([LICENSE](LICENSE)). Detailed background and walk-throughs are available in [ARTICLE.md](ARTICLE.md).

---

## Visual Architecture Gallery

### Figure 1: Context, Authority & Lifecycle Architecture
![Superagent Architecture](figures/01_superagent_lifecycle_architecture.png)
*Figure 1: End-to-end component layout: Intake Gatekeeper, Policy Assessor, Mandate Vault, Invalidation Engine, and Lifecycle Coordinator.*

### Figure 2: Mandate Integrity & Outsider Injection Isolation
![Mandate Integrity](figures/02_mandate_integrity_and_isolation.png)
*Figure 2: Contrast between vulnerable agent designs (where third-party comments mutate prompt context) and the Superagent mandate vault (where the original objective remains bitwise identical).*

### Figure 3: Finite State Machine of Superagent Lifecycle
![State Machine](figures/03_state_transition_lifecycle.png)
*Figure 3: Full state machine: Intake -> Assessment -> Assigned -> Completed / Human Stopped, showing permission guards on transitions.*

### Figure 4: Formal Test Suite Validation Matrix
![Validation Matrix](figures/04_test_suite_validation_matrix.png)
*Figure 4: Summary matrix of all 11 lifecycle invariants verified under the native Node.js 24 test runner.*

---

## Lifecycle & Authority Invariants

```
 [Event Received]
        |
        v
 [Deduplication]  ---> Duplicate ID? ---> [Drop / Idempotent Ack]
        |
        v
 [Policy Assessor] ---> Actor != Alice? -> [Reject / Unauthorized]
        |
        v
 [Snapshot Mandate] ---> Stored in Immutable Record
        |
        +<--- [Outsider Comment] (Quarantined to Trigger Stream)
        |
        v
 [Worker Execution] ---> Preparation Error? ---> [Transactional Rollback]
        |
        v
 [Ready State] --------> No active threads? ----> [Completed]
        |
        +<--- [Human Stop Signal] -------------> [Halted (Permanent)]
```

---

## Quickstart & Execution Guide

### Prerequisites
Node.js 24+ (e.g. `v24.2.0` or higher). No `npm install` needed.

### Run the Interactive Demo
```bash
node --experimental-transform-types demo.mjs
```
Expected output:
```
Accepted mandate: Correct one spelling error in README.
Trigger inputs received: 1
Mandate after outsider comment: Correct one spelling error in README.
Final outcome: completed
No model was called and no external message was sent.
```

### Run the Invariant Unit Test Suite
```bash
node --experimental-transform-types --test tests.mjs
```

### Generate High-Resolution Figures
```bash
python3 plot_results.py
```

---

## Formal Unit Test Verification Matrix

All 11 tests pass deterministically in under 150ms:

| # | Invariant Under Test | Security / Architectural Property | Result |
| :-: | :--- | :--- | :---: |
| 1 | Stable event IDs prevent duplicate intake | Deduplication & Idempotency | **PASS** (2.12ms) |
| 2 | Unauthorized new work never reaches the assessor | Admission Boundary Enforcement | **PASS** (0.20ms) |
| 3 | Accepted work snapshots a mandate & starts assignment | Mandate Immutability | **PASS** (0.49ms) |
| 4 | Subscribed outsider context cannot rewrite stored mandate | Prompt Injection Quarantine | **PASS** (0.93ms) |
| 5 | Policy version change makes old assessment unavailable | Evaluation Cache Invalidation | **PASS** (0.20ms) |
| 6 | Explicit invalidation replaces an evaluation | Dynamic Security Policy Evolution | **PASS** (0.40ms) |
| 7 | Preparation failure rolls back ticket acceptance | Atomic Transaction Rollback | **PASS** (0.24ms) |
| 8 | Ready work without followed completion threads ends | Clean Lifecycle Termination | **PASS** (0.46ms) |
| 9 | Bot followup cannot resume completed work | Privilege Escalation Defense | **PASS** (0.30ms) |
| 10 | Authorized human followup can resume completed work | Safe State Resumption | **PASS** (0.43ms) |
| 11 | Human stop prevents automatic ticket resumption | Human-in-the-Loop Override | **PASS** (0.27ms) |

---

## Repository File Map

```
superagent_lifecycle_lab/
├── ARTICLE.md               # In-depth architectural treatise & analysis
├── README.md                # Technical guide, scope disclosures, and benchmark results
├── LICENSE                  # MIT License
├── PROVENANCE.json          # Upstream commit and SHA-256 integrity verification
├── demo.mjs                 # Interactive walk-through script
├── harness.mjs              # Clean tutorial test harness & memory adapter
├── tests.mjs                # 11-test lifecycle invariant verification suite
├── plot_results.py          # Dark-slate publication figure generator
├── figures/                 # 300 DPI publication-grade figures
│   ├── 01_superagent_lifecycle_architecture.png
│   ├── 02_mandate_integrity_and_isolation.png
│   ├── 03_state_transition_lifecycle.png
│   └── 04_test_suite_validation_matrix.png
├── demo-output.txt          # Recorded terminal output of demo run
├── lab-validation.txt       # Recorded test suite validation log
├── upstream-validation.txt  # Recorded upstream check logs (1,197 tests passed)
└── vendor/                  # Vendored DeepSeek core prototype
    ├── LICENSE              # Upstream MIT License
    └── core/                # Upstream TypeScript core modules
```
