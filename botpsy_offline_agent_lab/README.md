# BotPsy Offline Agent: Edge Tool-Calling & Safety Interlocks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-pure%20standard%20library-success.svg)](#quickstart)
[![Inference: llama.cpp](https://img.shields.io/badge/inference-llama.cpp%20(local)-orange.svg)](https://github.com/ggerganov/llama.cpp)
[![Target Model: BotPsy-0.8B](https://img.shields.io/badge/Model-BotPsy--0.8B%20GGUF-purple.svg)](https://huggingface.co/qvac/BotPsy-0.8B)
[![Safety: Fail-Closed](https://img.shields.io/badge/Safety-Allowlist%20%2B%20Approval%20Gate-red.svg)](#security-threat-model--safety-invariants)
[![Tests Passing](https://img.shields.io/badge/tests-5%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, zero-dependency offline personal assistant agent designed to pair with sub-billion parameter local models (specifically the **BotPsy-0.8B** family by Tether/QVAC) hosted via `llama.cpp`.

This lab explores the architectural boundaries required to make ultra-compact edge language models reliably handle structured tools: **strict JSON-schema enforcement, a bounded execution governor, atomic local persistence, and human-in-the-loop approval interlocks**.

---

## Architecture & Safety Interlocks

```
                        +-------------------------------------------------------------+
                        |                      USER INSTRUCTION                       |
                        |      "Remember to review the experiment results tonight"    |
                        +-------------------------------------------------------------+
                                                       |
                                                       v
                        +-------------------------------------------------------------+
                        |                 LOCAL RUNTIME DISPATCHER                    |
                        |  - System Prompt Framing (Tool Declarations & Contracts)    |
                        |  - Bounded Step Governor (MAX_STEPS = 4, max 2 calls/turn)  |
                        +-------------------------------------------------------------+
                                                       |
                                                       | HTTP POST /v1/chat/completions
                                                       v
+---------------------------------------------------------------------------------------------------------------+
| LOCAL LLAMA.CPP SERVER (Offline Localhost Endpoint)                                                           |
|                                                                                                               |
|   Target Model: BotPsy-0.8B (Q4_K_M or Q8_0 GGUF)                                                             |
|   - Native Jinja tool template parsing                                                                        |
|   - Emits OpenAI-compatible tool_calls: {"name": "add_note", "arguments": {"text": "..."}}                   |
+---------------------------------------------------------------------------------------------------------------+
                                                       |
                                                       v Raw tool call response
                        +-------------------------------------------------------------+
                        |               SCHEMA & SAFETY INTERCEPTOR                   |
                        |                                                             |
                        |  [1] Tool Name in Allowlist?                                |
                        |      Allowed: {add_note, list_notes, search_notes,          |
                        |                add_task, list_tasks, complete_task}         |
                        |                                                             |
                        |  [2] Exact Parameter Matching?                              |
                        |      Strict equality: keys(args) == schema(tool)            |
                        |      (Rejects unrecognized or injected parameter keys)      |
                        |                                                             |
                        |  [3] Input Length & Type Sanitizer?                         |
                        |      Requires non-empty string, length <= 400 chars         |
                        |                                                             |
                        |  [4] Human-in-the-Loop Approval Gate?                       |
                        |      Destructive actions (e.g. complete_task) pause for     |
                        |      literal 'YES' user confirmation before execution       |
                        +-------------------------------------------------------------+
                                                       |
                                                       v Passed All Interlocks
                        +-------------------------------------------------------------+
                        |                   ATOMIC LOCAL STORAGE                      |
                        |  - Atomic write via .tmp + os.replace (crash-safe)          |
                        |  - Zero external network calls or cloud telemetry           |
                        +-------------------------------------------------------------+
```

---

## Security Threat Model & Safety Invariants

Small sub-billion parameter models (0.8B) possess limited reasoning depth and can be susceptible to prompt injection, hallucinated arguments, or runaway tool recursion. The host client enforces mathematical invariants:

### 1. Fail-Closed Dispatch Invariant
A tool call is executed if and only if it strictly satisfies four gating predicates:
$$\text{Dispatch}(f, \mathbf{a}) = \begin{cases} \text{Execute}(f, \mathbf{a}) & \text{if } f \in \mathcal{F}_{\text{allowed}} \land \text{keys}(\mathbf{a}) \equiv \text{schema}(f) \land \forall v \in \mathbf{a}: \text{valid}(v) \land \text{Approve}(f, \mathbf{a}) \\ \text{Error}(\text{"Rejected"}) & \text{otherwise} \end{cases}$$

### 2. Zero Unchecked Text Execution
The client explicitly ignores arbitrary model text and will **never** execute regex-parsed shell commands or script snippets. Only structured OpenAI-compatible `tool_calls` payloads produced by the model template are parsed.

### 3. Human Confirmation for Destructive State Changes
Read operations (`list_notes`, `search_notes`, `list_tasks`) and additive operations (`add_note`, `add_task`) execute automatically. State transitions that modify or close records (`complete_task`) block for explicit user confirmation (`YES`), preventing false-positive completions from model hallucination.

### 4. Bounded Turn Governor
To eliminate infinite tool recursion loops, execution terminates unconditionally after at most $4$ dialogue turns (`MAX_STEPS = 4`), and enforces a maximum of $2$ tool calls per turn.

---

## Quickstart

### 1. Deterministic Smoke Test (Zero Model Required)

The lab includes a built-in mock inference engine to verify tool execution, schema gating, and atomic file storage without downloading weights:

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/botpsy_offline_agent_lab

# Run deterministic demo
python3 agent.py --demo
```

Output:
```json
{
  "answer": "The local tool returned a result. Check the trace for confirmation.",
  "trace": [
    {
      "tool": "add_note",
      "args": {
        "text": "Read the BotPsy paper"
      },
      "result": {
        "saved": {
          "id": "34c8cfe5",
          "text": "Read the BotPsy paper",
          "created": "2026-11-22T05:52:47.442371+00:00"
        }
      }
    }
  ]
}
```

### 2. Run Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The 5 automated unit tests verify:
- `test_notes`: Substring search and note appending.
- `test_reject_unknown_tool`: Rejection of unauthorized tools (e.g. `run_shell`) with zero filesystem side effects.
- `test_reject_bad_args`: Rejection of injected extra keys or mismatched schemas.
- `test_approval`: Verification that task completion fails when unapproved, and succeeds only when approved.
- `test_mock_agent`: End-to-end multi-step agent dialogue loop with mock responses.

---

## Real Offline Inference with `llama.cpp` & `BotPsy-0.8B`

### Step 1: Launch Local Server with Tool Template
Download the model and start `llama-server` with Jinja tool template support enabled:

```bash
# Using llama-server from llama.cpp
llama-server \
  -hf qvac/BotPsy-0.8B-GGUFs:Q4_K_M \
  --jinja \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8080
```

*For completely offline environments after initial download, point directly to the cached `.gguf` file:*
```bash
llama-server -m /path/to/BotPsy-0.8B-Q4_K_M.gguf --jinja -c 8192 --host 127.0.0.1 --port 8080
```

### Step 2: Interact with the Agent

```bash
# Create a note
python3 agent.py 'Please save a note: review the experimental results tonight'

# Search stored notes
python3 agent.py 'What notes do I have about experiments?'

# Add and manage tasks
python3 agent.py 'Add a task to submit the benchmark paper'
python3 agent.py 'What tasks are currently open?'
python3 agent.py 'Complete task 34c8cfe5'
```

---

## File Structure

```
botpsy_offline_agent_lab/
├── README.md               # Architecture documentation, security invariants, and CLI guide
├── requirements.txt        # Zero external dependencies (pure Python standard library)
├── .gitignore              # Ignores __pycache__, bytecode, and local runtime databases
├── agent.py                # Full agent runtime: tools, validation, llama-server client, and CLI
└── tests/
    └── test_agent.py       # 5 automated unit tests covering tool safety and gatekeeping
```

---

## References

- [QVAC / Tether: BotPsy-0.8B Model Card](https://huggingface.co/qvac/BotPsy-0.8B)
- [BotPsy-0.8B-GGUFs Repository](https://huggingface.co/qvac/BotPsy-0.8B-GGUFs)
- [llama.cpp: OpenAI-Compatible Server & Jinja Tool Calling](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)
- Related labs in this repository:
  - [`local-five-tools-lab/`](../local-five-tools-lab) — Small-model tool calling fidelity and AST sandboxing.
  - [`superagent_lifecycle_lab/`](../superagent_lifecycle_lab) — Policy cache invalidation and human stop interlocks.
  - [`spikelab_bounded_agent_lab/`](../spikelab_bounded_agent_lab) — Bounded tool contracts and surrogate statistics.
