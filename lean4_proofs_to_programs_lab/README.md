# From Proofs to Programs in Lean 4

[![Language: Lean 4](https://img.shields.io/badge/Language-Lean%204-2E3440.svg)](https://lean-lang.org/)
[![Toolchain: elan + Lake](https://img.shields.io/badge/Build-Lake%20(TOML)-blue.svg)](https://lean-lang.org/install/manual/)
[![Proofs: Zero sorry](https://img.shields.io/badge/Proofs-Zero%20sorry%20%7C%20Zero%20custom%20axioms-purple.svg)](#proven-theorems--programs)
[![Tests Passing](https://img.shields.io/badge/tests-3%20passing%20(structure)-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal, editor-agnostic **Lean 4** lab demonstrating how formal theorem proving and executable programming live in the same language—with no VS Code, `mathlib`, or C FFI dependencies required. Built around the Curry–Howard correspondence (propositions as types, proofs as terms), it pairs a zero-`sorry` library (`src/Proofs.lean`) with an executable Lake target (`src/Main.lean`).

> [!NOTE]
> **Verification Status:** The Python structure test suite verifies source presence, manifest integrity, and the absence of `sorry`. Compilation via `lake build` should be verified on a machine with the Lean 4 toolchain (`elan`) installed.

---

## Proofs-to-Programs Pipeline

```
  src/Proofs.lean (Library)                  src/Main.lean (Executable)
  ├── theorem identity : p -> p              ├── import Proofs
  ├── theorem add_zero : n + 0 = n           └── def main : IO Unit
  ├── theorem add_comm : a + b = b + a             calls Proofs.double 21
  └── def double (n : Nat) := n + n
        ├── double_zero : double 0 = 0
        └── double_one  : double 1 = 2
                     |                                     |
                     +------------------+------------------+
                                        |
                                        v
                          [ Lean 4 Kernel + Lake Build ]
                          1. Type-checks proof terms (rejects unsound claims)
                          2. Compiles executable binary: lake exe lean4_lab
```

---

## Proven Theorems & Programs

| Declaration | Type / Statement | Proof Strategy |
|:---|:---|:---|
| `Proofs.identity` | `(p : Prop) : p → p` | Direct implication (`intro hp; exact hp`) |
| `Proofs.add_zero` | `(n : Nat) : n + 0 = n` | Definitional reflexivity (`rfl`) |
| `Proofs.add_comm` | `(a b : Nat) : a + b = b + a` | Standard library reuse (`Nat.add_comm`) |
| `Proofs.double` | `(n : Nat) : Nat := n + n` | Executable function |
| `Proofs.double_zero` | `double 0 = 0` | Definitional reduction (`rfl`) |
| `Proofs.double_one` | `double 1 = 2` | Definitional reduction (`rfl`) |

No `sorry` or custom `axiom` appears anywhere in the codebase.

---

## Quickstart

### 1. Install Lean 4 via `elan`

Follow the [official Lean 4 installation guide](https://lean-lang.org/install/manual/). On macOS/Linux (inspect scripts before piping into a shell):

```sh
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
lean --version
lake --version
```

### 2. Build & Run with Lake (Recommended)

```sh
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/lean4_proofs_to_programs_lab

lake build
lake exe lean4_lab
```

Direct single-file checking without Lake:
```sh
lean src/Proofs.lean
```

### 3. Learn by Breaking a Proof

1. Change `add_zero` from `n + 0 = n` to `n + 1 = n` with `by rfl` and observe Lean's kernel reject the invalid equality.
2. Change `double_one` from `= 2` to `= 3` and verify that type-checking fails before any binary is emitted.

---

## Test Suite

```sh
python3 -m unittest discover -s tests -v
```

The 3 automated structural tests verify:
- **`test_sources_present`**: Both `src/Proofs.lean` and `src/Main.lean` exist.
- **`test_no_unfinished_proof`**: Zero `sorry` placeholders in `src/Proofs.lean`.
- **`test_manifest_present`**: `lakefile.toml` declares the `lean4_lab` executable target.

---

## File Structure

```
lean4_proofs_to_programs_lab/
├── README.md                 # Curry–Howard overview, build guide, and exercises
├── .gitignore                # Ignores .lake/, build/, and Python bytecode
├── lakefile.toml             # Lake package manifest (Proofs lib + lean4_lab exe)
├── lean-toolchain            # Pins leanprover/lean4:stable
├── src/
│   ├── Proofs.lean           # Zero-sorry proofs and verified double function
│   └── Main.lean             # Executable entry point printing Proofs.double 21
└── tests/
    └── test_structure.py     # 3 automated structural checks
```

---

## FFI & Native Code Note

This lab intentionally omits untested C FFI bindings (e.g., Raylib), because crossing into C introduces ABI, ownership, and memory-safety obligations that Lean's kernel does not verify automatically. For official FFI patterns, see:
- [Lean 4 Foreign Function Interface Reference](https://lean-lang.org/doc/reference/latest/Run-Time-Code/Foreign-Function-Interface/)
- [Lake Official FFI Examples](https://github.com/leanprover/lean4/tree/master/src/lake/examples/ffi)
- [Raylib.lean](https://github.com/KislyjKisel/Raylib.lean)
