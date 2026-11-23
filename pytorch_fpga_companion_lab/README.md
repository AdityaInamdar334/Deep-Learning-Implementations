# From Tensors to Circuits: Mini FPGA Compiler Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-pure%20standard%20library-success.svg)](#quickstart)
[![RTL Target: SystemVerilog](https://img.shields.io/badge/RTL-SystemVerilog%20(IEEE%201800)-orange.svg)](#optional-rtl-simulation)
[![Architecture: Spatial Lane Scheduler](https://img.shields.io/badge/Hardware-Int8%20Dot--Product%20Lanes-purple.svg)](#lane-parallelism-vs-cycle-tradeoffs)
[![Tests Passing](https://img.shields.io/badge/tests-3%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, dependency-free educational compiler lab demonstrating the core intermediate lowering stages from a mathematical tensor operation to synthesizable **SystemVerilog RTL** and simulation testbenches. Given a fixed 8-element signed `int8` dot product, the compiler parameterizes spatial parallelism (`lanes` $\in \{1, 2, 4, 8\}$), builds a deterministic cycle-by-cycle accumulation schedule, and emits both the behavioral RTL module (`dot8_lanes_K.sv`) and a self-checking testbench (`tb_K.sv`).

> [!IMPORTANT]
> **Educational Scope:** This lab models a single fixed `int8` dot-product kernel to illustrate scheduling and RTL codegen—it is **not** a general PyTorch-to-FPGA compiler. It does not import PyTorch graphs, generate FPGA bitstreams, or claim vendor-specific DSP/LUT utilization, clock frequency ($F_{\max}$), or place-and-route timing.

---

## Compiler Lowering Pipeline

```
  [1] Mathematical Spec          y = SUM_{i=0}^{7} a[i] * b[i],   a[i], b[i] in [-128, 127] (signed int8)
            |
            v
  [2] Spatial Lane Scheduler     Choose parallelism factor L in {1, 2, 4, 8}  -->  G = 8 / L accumulation cycles
            |                    Cycle g evaluates L parallel products: SUM_{i=0}^{L-1} a[g*L+i] * b[g*L+i]
            v
  [3] Behavioral SystemVerilog   Emit FSM controller (idle -> active -> done) + combinational lane adder tree
            |                    (generated/dot8_lanes_L.sv)
            v
  [4] Self-Checking Testbench    Emit stimulus driver + reference check (result === 120) + watchdog timeout
            |                    (generated/tb_L.sv)
            v
  [5] Optional RTL Simulation    iverilog / Verilator functional verification
```

---

## Lane Parallelism vs. Cycle Tradeoffs

For an 8-element signed `int8` dot product accumulated into a signed 32-bit register (`result`), the worst-case magnitude is $8 \times (-128)^2 = 131{,}072 \ll 2^{31} - 1$, guaranteeing **zero arithmetic overflow**.

| Parallel Lanes ($L$) | Accumulation Cycles ($8/L$) | Combinational Multipliers / Cycle | Generated Module |
|:---:|:---:|:---:|:---|
| **1** | 8 cycles | 1 signed `8×8` MAC | `dot8_lanes_1` |
| **2** | 4 cycles | 2 parallel multipliers + adder | `dot8_lanes_2` |
| **4** | 2 cycles | 4 parallel multipliers + adder tree | `dot8_lanes_4` |
| **8** | 1 cycle | 8 parallel multipliers + full reduction | `dot8_lanes_8` |

*Note: Abstract accumulation cycles exclude the 1-cycle `start` handshake and do not predict post-synthesis $F_{\max}$ or DSP slice count.*

---

## Quickstart

### 1. Run Unit Tests & Generate RTL (Zero Dependencies)

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/pytorch_fpga_companion_lab

# Run Python unit tests
python3 -m unittest discover -s tests -v

# Generate schedules, SystemVerilog RTL, and testbenches
python3 mini_compiler.py --lanes 1 --out generated
python3 mini_compiler.py --lanes 4 --out generated
```

### 2. Optional RTL Simulation

Requires **Icarus Verilog** (`iverilog` with `-g2012` SystemVerilog support) or **Verilator**:

```bash
# Using Icarus Verilog
iverilog -g2012 -s tb -o generated/sim generated/dot8_lanes_4.sv generated/tb_4.sv
vvp generated/sim

# Or using Verilator
verilator --binary --timing -Wno-fatal --top-module tb generated/dot8_lanes_4.sv generated/tb_4.sv
./obj_dir/Vtb
```

---

## Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The 3 automated unit tests verify:
- **`test_all_parallelism_factors`**: Exact numeric parity between `scheduled_dot` and `dot_ref` across all lane counts (`1, 2, 4, 8`) with signed positive/negative inputs, plus exact cycle count `8 // lanes`.
- **`test_generates_modules`**: Valid SystemVerilog module header emission for all supported lane counts.
- **`test_invalid`**: Rejection of non-divisor lane counts (e.g., `lanes=3`) via `ValueError`.

---

## File Structure

```
pytorch_fpga_companion_lab/
├── README.md                       # Compiler pipeline, lane tradeoffs, and simulation guide
├── requirements.txt                # Pure Python standard library note
├── .gitignore                      # Ignores __pycache__, obj_dir/, and compiled sim binaries
├── mini_compiler.py                # Reference dot product, lane scheduler, and SV codegen
├── generated/
│   ├── dot8_lanes_4.sv             # Pre-generated 4-lane SystemVerilog RTL module
│   └── tb_4.sv                     # Pre-generated self-checking SystemVerilog testbench
└── tests/
    └── test_mini_compiler.py       # 3 automated unit tests
```

---

## Suggested Extensions

1. **Memory-Port Stalls**: Introduce a 2-read-port SRAM bandwidth constraint and model pipeline stalls when `lanes > 2`.
2. **Graph IR Pass**: Add a minimal tensor IR with double-transpose elimination before scheduling.
3. **Multi-Layer Quantized MLP**: Chain two scheduled `int8` matrix-vector products with ReLU and bit-shift requantization against a Python reference.
4. **Vendor Synthesis**: Synthesize `dot8_lanes_1` through `dot8_lanes_8` in Vivado or Quartus and plot actual DSP48/LUT usage vs. $F_{\max}$.
