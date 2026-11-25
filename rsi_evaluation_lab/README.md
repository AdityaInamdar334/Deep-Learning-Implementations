# RSI Evaluation Blind Spots: Coverage, Bottlenecks & Judge Bias Lab

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-pure%20standard%20library-success.svg)](#quickstart)
[![Domain: AI Evaluation & Safety](https://img.shields.io/badge/Domain-Benchmark%20Coverage%20%26%20Bias-purple.svg)](#core-mechanics)
[![Tests Passing](https://img.shields.io/badge/tests-6%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A zero-dependency Python thought-experiment lab demonstrating why **narrow benchmark coverage**, **missing bottleneck dimensions**, and **positively biased automated judges** can make an autonomous AI R&D trajectory appear far closer to full recursive self-improvement (RSI) than the underlying end-to-end loop actually is.

> [!IMPORTANT]
> **Synthetic Thought Experiment:** This lab uses arbitrarily chosen synthetic trajectories to illustrate measurement mechanics. It contains **no** real model scores, **no** Vals AI or METR data, and makes **no** empirical forecast of when recursive self-improvement will occur.

---

## Core Mechanics

```
  FIVE SYNTHETIC R&D COMPETENCE DIMENSIONS

  [ Narrowly Benchmarked (Fast Growth) ]          [ Under-Measured Bottlenecks (Slow Growth) ]
  ├── pretraining   (init 0.43, +0.030/mo)        ├── judgment  (init 0.19, +0.009/mo)
  ├── posttraining  (init 0.35, +0.026/mo)        └── systems   (init 0.22, +0.011/mo)
  └── harness       (init 0.40, +0.035/mo)
                   |                                                   |
                   v                                                   v
      All 3 >= 0.80 at Month 18                          All 5 >= 0.80 at Month 68
      (Apparent narrow score: 0.929)                     (50-month blind-spot gap!)
```

### 1. Narrow vs. Full Coverage (`apparent_score` & `first_crossing`)
Averaging only the three easily benchmarked skills (`pretraining`, `posttraining`, `harness`) masks slow progress in `judgment` and `systems`. Requiring **all** required dimensions to cross a capability threshold ($0.80$) exposes the slowest bottleneck rather than hiding it inside a mean.

### 2. Automated Judge Bias (`judge_example`)
When an LLM-as-a-judge exhibits positive self-preference or rubric inflation ($b = +0.07$), a modest true gain ($\Delta = +0.04$) is reported as $\hat{\Delta} = +0.11$—nearly $3\times$ the actual improvement.

### 3. Dev vs. Independent Holdout Separation (`evaluation_signals`)
Tracking an agent's self-reported development gain (`+0.04`) separately from an independent holdout evaluation (`-0.01`) exposes overfitting to the harness or evaluation rubric.

---

## Synthetic Baseline Output (`sample_output.json`)

| Month | Narrow 3-Domain Avg | All 5-Domain Avg |
|:---:|---:|---:|
| 0 | `0.393` | `0.318` |
| 6 | `0.575` | `0.451` |
| 12 | `0.757` | `0.584` |
| 18 | `0.929` | `0.712` |

- **First month all 3 narrow domains $\ge 0.80$**: Month `18`
- **First month all 5 domains $\ge 0.80$**: Month `68`
- **Judge bias example**: `true = +0.04`, `bias = +0.07`, `estimated = +0.11`

---

## Quickstart

### 1. Run Simulation & Unit Tests (Zero Dependencies)

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/rsi_evaluation_lab

# Run CLI simulation
python3 rsi_lab.py
python3 rsi_lab.py --output sample_output.json

# Run automated test suite
python3 -m unittest discover -s tests -v
```

### 2. Test Suite

The 6 automated unit tests verify:
- **`test_all_dimensions`**: Every trajectory step tracks all five R&D domains.
- **`test_covered_score`**: Narrow 3-domain average strictly overstates performance when bottleneck domains lag.
- **`test_monotonic`**: Monotonic non-decreasing progression across months.
- **`test_no_extrapolation_from_short_horizon`**: Short-horizon trajectories return `None` rather than falsely claiming a crossing.
- **`test_judge_bias`**: Additive judge bias inflates estimated improvements cleanly.
- **`test_provenance`**: Output payload explicitly carries the `SYNTHETIC` provenance disclaimer.

---

## File Structure

```
rsi_evaluation_lab/
├── README.md                 # Blind-spot mechanics, baseline table, and caveats
├── requirements.txt          # Pure Python standard library note
├── .gitignore                # Ignores __pycache__ and bytecode
├── rsi_lab.py                # 5-domain trajectory, crossing, and judge-bias simulator
├── sample_output.json        # Saved synthetic JSON output
└── tests/
    └── test_rsi_lab.py       # 6 automated unit tests
```
