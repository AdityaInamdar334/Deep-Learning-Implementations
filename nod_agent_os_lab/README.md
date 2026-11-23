# Agent With Its Own Computer: Mini Agent OS Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-pure%20standard%20library-success.svg)](#quickstart)
[![Storage: SQLite](https://img.shields.io/badge/Persistence-SQLite%20%2B%20Audit%20Log-003B57.svg)](#architecture--permission-flow)
[![Safety: Read/Write Gates](https://img.shields.io/badge/Safety-Allowlist%20%2B%20Approval%20Gate-red.svg)](#capability-matrix)
[![Tests Passing](https://img.shields.io/badge/tests-5%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-contained, zero-dependency Python simulation exploring core architectural primitives behind persistent computer-using agents like **n8n Nod**: strict tool allowlists, read-versus-write permission boundaries, explicit human approval gates for persistent mutations, SQLite state storage, and an append-only UTC audit trail.

> [!IMPORTANT]
> **Educational Scope:** This lab is **not** a live Nod integration and does not run an LLM, headless browser, Docker container, or n8n server. The mock page is hard-coded with zero network calls or secrets, isolating the permission and audit architecture from runtime sandboxing.

---

## Architecture & Permission Flow

```
                        +----------------------------------------------------------+
                        |                 AGENT ACTION REQUEST                     |
                        |      request(task, action, args, approved=False)         |
                        +----------------------------------------------------------+
                                                     |
                                                     v
                        +----------------------------------------------------------+
                        | [1] ALLOWLIST CHECK                                      |
                        |     allowed = {read_mock_page, list_notes, save_note}    |
                        +----------------------------------------------------------+
                                   | No                                | Yes
                                   v                                   v
                         [Log: 'denied']                 +----------------------------+
                         return error                    | [2] ARGUMENT VALIDATION    |
                                                         |     Exact keys, str, 1-500 |
                                                         +----------------------------+
                                                              | Invalid       | Valid
                                                              v               v
                                                      [Log: 'denied']   +-------------+
                                                      return error      | Read or     |
                                                                        | Write?      |
                                                                        +-------------+
                                                                        /             \
                                                                  Read /               \ Write (save_note)
                                                                      v                 v
                                                           [Execute Read]      [approved == True?]
                                                                 |               /             \
                                                                 |          No  /               \ Yes
                                                                 |             v                 v
                                                                 |  [Log: 'approval_required']  [INSERT INTO notes]
                                                                 |  return preview              [Log: 'completed']
                                                                 |                              return saved
                                                                 v
                                                        [Log: 'completed']
                                                        return result
                                                                 |
                                                                 v
                        +----------------------------------------------------------+
                        |        SQLITE PERSISTENCE (notes + events tables)        |
                        |  - Durable notes across runs                             |
                        |  - Immutable UTC-timestamped event audit trail           |
                        +----------------------------------------------------------+
```

---

## Capability Matrix

| Tool | Access Class | Argument Schema | Approval Required | Side Effect |
|:---|:---:|:---|:---:|:---|
| `read_mock_page` | Read-only | `{}` (empty) | No | Returns deterministic status payload |
| `list_notes` | Read-only | `{}` (empty) | No | Queries ordered `notes` table |
| `save_note` | Persistent Write | `{"text": str}` (1–500 chars) | **Yes** (`approved=True`) | Inserts row into SQLite `notes` table |
| *Any other tool (e.g., `run_shell`)* | Denied | — | Blocked | Logged as `denied`, returns error |

---

## Quickstart

### 1. Run Unit Tests & Demo (Zero Dependencies)

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/nod_agent_os_lab

# Run automated unit tests (in-memory SQLite)
python3 -m unittest discover -s tests -v

# Run persistent CLI demonstration
python3 agent_os.py --db demo.db
```

The first run creates `demo.db`, reads the mock status page, proposes a persistent note (which pauses for approval), commits it once approved, rejects `run_shell`, and prints the full audit log. Re-running persists previously approved notes.

---

## Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The 5 automated unit tests verify:
- **`test_read`**: Read-only `read_mock_page` executes without approval.
- **`test_approval`**: Unapproved `save_note` returns `requires_approval: True` without mutating SQLite; approved call commits cleanly.
- **`test_denied`**: Unallowlisted capabilities (`run_shell`) fail closed.
- **`test_bad_args`**: Non-string or malformed arguments are rejected even when `approved=True`.
- **`test_audit`**: Every action writes an ordered status record to the `events` audit table.

---

## File Structure

```
nod_agent_os_lab/
├── README.md               # Architecture flow, capability matrix, and exercises
├── requirements.txt        # Pure Python standard library note
├── .gitignore              # Ignores __pycache__, bytecode, and generated *.db files
├── agent_os.py             # MiniAgentOS runtime, SQLite schema, and CLI demo
└── tests/
    └── test_agent_os.py    # 5 automated unit tests
```

---

## Suggested Exercises

1. Add a read-only `find_note` tool with exact argument validation.
2. Implement an idempotency/duplicate-action guard to prevent re-executing completed writes.
3. Add a `scheduled_tasks` SQLite table processed via an explicit manual `run-due` CLI subcommand (no background daemon).
4. Write a prompt-injection test proving that text returned by `read_mock_page` cannot invoke tools or set `approved=True`.
5. For real browser-using agents, enforce OS/container process isolation and egress network policies—this minimal simulation is not a production security boundary.

---

## References

- [n8n Nod Repository](https://github.com/n8n-io/nod)
- [Nod Documentation: Install from Source](https://nod.n8n.io/docs/get-started/install-from-source/)
- Related labs in this repository:
  - [`botpsy_offline_agent_lab/`](../botpsy_offline_agent_lab) — Sub-billion parameter offline tool calling with schema gates.
  - [`superagent_lifecycle_lab/`](../superagent_lifecycle_lab) — Immutable mandates, policy cache invalidation, and human stop interlocks.
  - [`n8n-agent-lab/`](../n8n-agent-lab) — Split-tool vs atomic by-ID architectures in n8n workflows.
