# Skills for Real Engineers: Contract Auditing, Mutation Testing & Release Evolution

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Git Audited](https://img.shields.io/badge/Git%20Audited-v1.0.0%20%E2%86%92%20v1.3.2-success.svg?logo=git&logoColor=white)](audit.py)
[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)](tests/test_pocock.py)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-purple.svg)]()
[![Upstream](https://img.shields.io/badge/Upstream-mattpocock%2Fskills-informational.svg)](https://github.com/mattpocock/skills)
[![Article](https://img.shields.io/badge/Tutorial-ARTICLE.md-orange.svg)](ARTICLE.md)
[![Fact Check](https://img.shields.io/badge/Fact%20Check-Editorial%20Verification-teal.svg)](Editorial_Fact_Check.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A reproducible software engineering and contract audit laboratory examining Matt Pocock’s AI skills collection, the evolution of the 12-line to 30-line `grilling` interview primitive, the 32.4M install telemetry metrics, and mutation testing against state-machine race conditions.*

Based on real-world engineering agent skills, release tags, and contract design:  
**["Skills for Real Engineers"](https://github.com/mattpocock/skills)** (Matt Pocock, Total TypeScript, 2026).

</div>

---

## Executive Summary

Give an autonomous coding agent this ticket: **"Let users cancel a training run."**

A typical agent will quickly generate a Cancel button in the frontend, wire up an API endpoint, turn the status badge gray upon click, and write a passing unit test verifying that clicking the button changes the UI state. Meanwhile, deep in the backend pipeline, a background worker can still pick up the cancelled task from the queue and proceed to spend six hours of expensive GPU compute training a model nobody wants.

Before anyone writes a single line of UI code, someone has to ask a fundamental architectural question:  
**"Does cancellation mean the user no longer wants the result, or have we guaranteed that a queued run will never be claimed by a worker?"**

This is the exact class of ambiguity that Matt Pocock's `grilling` interview primitive is designed to surface. An early release (`v1.1.0`) fit in 12 lines of Markdown; by `v1.3.2`, it expanded to 30 lines. 

```
v1.1.0 (12 lines):
[Ask Q1] -> [Wait for User] -> [Ask Q2] -> [Wait for User] -> [Ask Q3] (Serial interview)

v1.3.2 (30 lines):
[Tool Exploration: Check Codebase] -> [Batch Independent Q1 + Q2 with Defaults] -> [Wait] -> [Ask Dependent Q3]
```

This laboratory investigates:
1. **The Evolution of the Interview**: How `grilling` shifted from a strictly serial question loop to dependency-ordered batching with recommended defaults, while offloading factual environmental questions to automated codebase exploration.
2. **Fact-Checking the 32.4M Install Claims**: Auditing the real release history across Git tags (`v1.0.0` through `v1.3.2`) to separate collection-wide download telemetry from individual skill line counts.
3. **State-Machine Lifecycle Contracts & Mutation Testing**: Demonstrating why naive unit tests written by agents miss state-machine race conditions, and how contract-based mutation testing (`queue_demo.py`) catches silent bugs before production.

---

## The Release Evolution & Fact-Check

A common viral headline claims: *"20 million engineers downloaded a 12-line markdown file."*

An editorial audit of `skills.sh` and Git release blobs reveals the true provenance of these metrics:
- **32.4 Million Total Installs**: Recorded on `skills.sh` as the collection-wide download count across all 27+ skills combined (not unique users or single-file downloads).
- **1.1 Million for `grilling`**: The specific download count for the `grilling` interview primitive.
- **1.5 Million for `grill-me`**: The CLI slash-command wrapper that delegates to `grilling`.
- **12 Lines to 30 Lines**: The 12-line version existed in `v1.1.0`; by `v1.3.2`, the core skill expanded to 30 lines (plus a separate 7-line wrapper).

### Authoritative Git Release Audit Table

Using `audit.py` to inspect the historical Git tree without modifying checkouts:

| Release Tag | Tag Date | `grilling/SKILL.md` Lines | Manifest Entries (`plugin.json`) | `skills/**/SKILL.md` Files | Key Architectural Changes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`v1.0.0`** | Initial Release | **10** | 17 | 34 | First public skills release |
| **`v1.1.0`** | Early Iteration | **12** | 21 | 38 | Serial question interview; recommended defaults |
| **`v1.2.0`** | Mid Iteration | **22** | 25 | 35 | Introduction of dependent question ordering |
| **`v1.3.0`** | Oct 4, 2026 | **28** | 27 | 37 | Glossary renaming; whole-spec implementation promotion |
| **`v1.3.2`** | Oct 29, 2026 | **30** | 27 | 38 | Mutation verification; batched independent questions; ask-matt |

---

## State-Machine Lifecycle Contracts & Mutation Testing

In `queue_demo.py`, we model the training-run cancellation lifecycle:

```
[queued] ---- cancel() ----> [cancelled] (worker claim MUST fail)
   |
 claim()
   v
[running] --- cancel() ----> [running]   (queued-only contract: in-flight jobs continue)
```

### The Subtle Agent Mutation

When asked to implement worker claiming, an agent may write:

```python
# Baseline Contract (Correct)
def claim(self):
    if self.state != "queued":
        return False
    self.state = "running"
    return True
```

An agent might inadvertently write an inverted check:

```python
# Agent Mutation (Buggy)
def claim(self):
    if self.state == "running":
        return False
    self.state = "running"
    return True
```

### Why Standard Tests Miss It:
- A test that creates a run, claims it, and verifies it is `"running"` will pass.
- A test that claims an already running job and expects `False` will pass.
- A UI test that clicks Cancel on a queued run and verifies the status is `"cancelled"` will pass.

### How Contract Mutation Testing Catches It:
The contract test explicitly tests the transition sequence:
$$\text{queued} \xrightarrow{\text{cancel()}} \text{cancelled} \xrightarrow{\text{claim()}} \mathbf{False}$$

Under the mutant, `claim()` checks `if state == "running"` (False, since state is `"cancelled"`), transitions to `"running"`, and returns `True`! The worker claims a cancelled job. The contract test immediately raises:
`AssertionError: worker claimed a cancelled run`.

---

## The Interview Architecture: Serial vs. Batched Questions

The diff between `v1.1.0` and `v1.3.2` reveals an intentional UX evolution:

1. **Separating Environment Facts from Human Decisions**:
   - *Bad prompt*: Asking the human *"Which database table stores training jobs?"*
   - *Pocock prompt*: Instructs the agent to inspect the codebase directly via grep/tools, reserving human questions strictly for product choices.
2. **Dependency-Ordered Batching**:
   - Independent questions (e.g. *Q1: Which runs can be cancelled?*, *Q2: Who is authorized to cancel?*) are grouped into a single round with recommended defaults.
   - Dependent questions (e.g. *Q3: How should worker processes be signalled?*) wait until Q1 confirms whether running tasks are in scope.
3. **The Anchoring Effect of Recommended Defaults**:
   - Framing answers with defaults (e.g. *"I recommend queued-only runs because terminating running processes requires SIGTERM handling"*) allows the developer to reply with a simple `"yes"`.
   - The trade-off: A busy reviewer might approve an assumption without realizing they just committed to allowing in-flight runs to consume budget.

---

## Rigorous Scope Disclosures

> [!IMPORTANT]
> **What this laboratory does:**
> - Audits Matt Pocock's published open-source Git tags (`v1.0.0` through `v1.3.2`) to verify file sizes, line counts, and manifest entries.
> - Demonstrates mutation testing on state-machine contracts (`queue_demo.py`) using pure Python standard library.
> - Provides an offline reference audit (`audit.py`) with optional live verification against upstream checkouts.
> - Includes 5 automated unit tests verifying lifecycle state transitions, mutation catching, and dependency scheduling.
>
> **What this laboratory does NOT do:**
> - It does **not** install skills into your local agent environment.
> - It does **not** host LLM API endpoints or benchmark model generation performance.
> - It does **not** claim to measure developer productivity gains; it inspects the structural evolution of the prompt templates.

---

## Quickstart & Hands-On Exercises

### 1. Run the Mutation Contract Demonstration

```bash
cd pocock_lab
python3 queue_demo.py
```

**Console Output:**
```diff
--- baseline
+++ mutant
@@ -8,7 +8,7 @@
         return self.state
 
     def claim(self):
-        if self.state != "queued":
+        if self.state == "running":
             return False
         self.state = "running"
         return True
Baseline passed. Mutant caught: worker claimed a cancelled run
```

### 2. Run the Git Release Tag Audit (Offline)

```bash
python3 audit.py
```

**Console Output:**
```
Notice: Local checkout 'skills' not found. Displaying verified historical release audit.
To run a live audit, clone upstream: git clone https://github.com/mattpocock/skills.git

v1.0.0: 10 grilling lines; 17 manifest entries; 34 skills/**/SKILL.md files
v1.1.0: 12 grilling lines; 21 manifest entries; 38 skills/**/SKILL.md files
v1.2.0: 22 grilling lines; 25 manifest entries; 35 skills/**/SKILL.md files
v1.3.0: 28 grilling lines; 27 manifest entries; 37 skills/**/SKILL.md files
v1.3.2: 30 grilling lines; 27 manifest entries; 38 skills/**/SKILL.md files
```

### 3. Run the Automated Unit Test Suite

```bash
PYTHONPATH=. python3 -m unittest discover -s pocock_lab/tests -v
```

**Test Execution:**
```
test_01_run_lifecycle_contract ......................................... ok
test_02_mutation_detection ............................................. ok
test_03_historical_audit_counts ........................................ ok
test_04_interview_dependency_grouping .................................. ok
test_05_idempotent_cancellation_semantics .............................. ok

Ran 5 tests in 0.000s (OK)
```

### 4. Optional: Live Upstream Verification

```bash
git clone https://github.com/mattpocock/skills.git
python3 audit.py skills
git -C skills diff v1.1.0 v1.3.2 -- skills/productivity/grilling/SKILL.md
```

---

## Repository Structure

```
pocock_lab/
├── ARTICLE.md                  # Comprehensive essay analyzing Matt Pocock's skills repository
├── Editorial_Fact_Check.md     # Point-by-point fact check of install counters and metrics
├── LICENSE                     # MIT Open Source License
├── README.md                   # This detailed engineering and audit guide
├── requirements.txt            # Zero third-party dependencies (Python standard library)
├── .gitignore                  # Clean ignore rules for checkouts and environments
├── __init__.py                 # Python package definitions
├── audit.py                    # Tag auditor (supports offline reference and live Git queries)
├── audit-output.txt            # Verified ground-truth audit console output
├── queue_demo.py               # State machine cancellation contract and mutation test runner
├── demo-output.txt             # Verified output from queue_demo.py
└── tests/                      # Automated unit test suite
    ├── __init__.py
    └── test_pocock.py          # 5 unit tests verifying contracts, mutations, and tag history
```

---

## Academic & Practitioner References

1. **Matt Pocock (2026)**. *Skills for Real Engineers*. GitHub repository: [mattpocock/skills](https://github.com/mattpocock/skills).
2. **Gergely Orosz (2026)**. *AI Skills with Matt Pocock*. The Pragmatic Engineer. [Link](https://newsletter.pragmaticengineer.com/p/ai-skills-with-matt-pocock).
3. **Thoughtworks Technology Radar (November 2026)**. *Matt Pocock's Skills (Trial)*. [Thoughtworks Radar](https://www.thoughtworks.com/radar/tools/mattpocock-skills).
4. **Alibaba Cloud Community (2026)**. *Analysis of AI Agent Skills and Developer Workflows*.
5. **skills.sh Registry (2026)**. *Ecosystem Install Telemetry and Distribution Counts*. [skills.sh](https://skills.sh/mattpocock/skills).

---

## License

This lab is released under the open-source [MIT License](LICENSE).
