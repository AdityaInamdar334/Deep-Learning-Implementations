# Capability & Oversight: A Synthetic Pacing Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-pure%20standard%20library-success.svg)](#quickstart)
[![Domain: AI Governance & Safety](https://img.shields.io/badge/Domain-Capability%20vs.%20Oversight%20Pacing-purple.svg)](#scenario-comparison)
[![Tests Passing](https://img.shields.io/badge/tests-6%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A zero-dependency Python companion lab contrasting three synthetic **capability-versus-oversight** trajectories over a 24-month horizon: unpaced capability growth, slowed capability growth alone, and paced capability growth paired with accelerated investment in oversight.

> [!IMPORTANT]
> **Synthetic Toy Model:** Every parameter in this lab is invented for educational clarity. The linear growth curves and `uncovered_gap = max(0, capability - oversight)` proxy are **not** calibrated risk probabilities, forecasts of AGI/RSI, or evaluations of any real AI lab or model.

---

## Model & Scenarios

At each month $t \in \{0, \dots, T\}$:
$$C(t) = \min\big(1.0,\; C_0 + r_C \cdot t\big), \qquad O(t) = \min\big(1.0,\; O_0 + r_O \cdot t\big), \qquad G(t) = \max\big(0.0,\; C(t) - O(t)\big)$$

```
  MONTH-24 ENDPOINTS (C_0 = 0.20, O_0 = 0.35)

  fast_progress         (r_C=0.032, r_O=0.009)  C=0.968 | O=0.566 | Uncovered Gap = 0.402
  paced_progress        (r_C=0.023, r_O=0.009)  C=0.752 | O=0.566 | Uncovered Gap = 0.186
  paced_plus_oversight  (r_C=0.023, r_O=0.025)  C=0.752 | O=0.950 | Uncovered Gap = 0.000
```

### Scenario Comparison (`sample_results.csv`, Month 24)

| Scenario | Cap. Growth ($r_C$) | Oversight Growth ($r_O$) | First Gap Month | Month-24 Capability | Month-24 Oversight | Month-24 Uncovered Gap |
|:---|---:|---:|:---:|---:|---:|---:|
| `fast_progress` | `0.032/mo` | `0.009/mo` | Month 7 | `0.968` | `0.566` | `0.402` |
| `paced_progress` | `0.023/mo` | `0.009/mo` | Month 11 | `0.752` | `0.566` | `0.186` |
| `paced_plus_oversight` | `0.023/mo` | `0.025/mo` | *None* | `0.752` | `0.950` | `0.000` |

---

## Quickstart

### 1. Run Simulation & Unit Tests (Zero Dependencies)

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/ai_pacing_oversight_lab

# Generate CSV trajectory table
python3 pacing_lab.py --months 24 --csv results.csv

# Run automated test suite
python3 -m unittest discover -s tests -v
```

### 2. Test Suite

The 6 automated unit tests verify:
- **`test_zero_month`**: Horizon `0` returns a single initial row.
- **`test_all_scores_bounded`**: Capability, oversight, and uncovered gap stay strictly inside $[0, 1]$.
- **`test_more_oversight_reduces_gap`**: Accelerating oversight strictly reduces the final uncovered gap.
- **`test_pacing_reduces_gap`**: Slowing capability growth reduces the final uncovered gap.
- **`test_invalid_months`** & **`test_invalid_growth`**: Negative horizons or negative growth rates raise `ValueError`.

---

## File Structure

```
ai_pacing_oversight_lab/
├── README.md                 # Equations, scenario comparison, and caveats
├── requirements.txt          # Pure Python standard library note
├── .gitignore                # Ignores __pycache__ and generated results.csv
├── pacing_lab.py             # Scenario dataclass, simulator, and CSV CLI
├── sample_results.csv        # Saved 24-month trajectory table across all 3 scenarios
└── tests/
    └── test_pacing_lab.py    # 6 automated unit tests
```
