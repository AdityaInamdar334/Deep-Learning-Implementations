# Intelligence as Finding Another Way: Replanning & Optionality Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: Zero](https://img.shields.io/badge/dependencies-pure%20standard%20library-success.svg)](#quickstart)
[![Concept: Replanning & Optionality](https://img.shields.io/badge/Concept-Adaptive%20Detours%20%26%20Reachability-purple.svg)](#core-mechanics)
[![Tests Passing](https://img.shields.io/badge/tests-5%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A zero-dependency Python companion lab contrasting **fixed-route execution** against **adaptive online replanning** when an unexpected obstacle closes a previously open path, alongside a finite-horizon **reachable-state count** serving as a transparent toy proxy for future optionality.

> [!IMPORTANT]
> **Interpretation Caveat:** Inspired conceptually by William James (1890), Michael Levin (2022), and Wissner-Gross & Freer (2013), this lab models *behavioral response flexibility* via deterministic BFS graph search. It does **not** implement biological morphogenesis, causal entropic forces, consciousness, or a general measure of intelligence.

---

## Core Mechanics

```
  7 x 5 GRID WORLD AFTER UNEXPECTED OBSTACLE AT (3, 2)

       0 1 2 3 4 5 6
    0  . . . # . . .      S = Start (0, 2)
    1  . . * * * * .      G = Goal  (6, 2)
    2  S * * # . * G      # = Walls: initial {(3,0), (3,4)} + surprise {(3,2)}
    3  . . . . . . .      * = Adaptive agent's replanned detour
    4  . . . # . . .

  Fixed-Plan Agent:  Computes shortest path through (3,2) -> halts at (2,2) -> FAILS
  Adaptive Agent:    Detects blockage at (3,2) -> adds wall to belief map -> replans -> SUCCEEDS
```

### 1. Fixed vs. Adaptive Policies
- **`fixed_plan_agent`**: Computes an optimal BFS path under `walls_before`, then executes blindly. When `NEW_WALL = (3, 2)` appears, it terminates at `(2, 2)`.
- **`adaptive_agent`**: Tracks a live `known_walls` belief set. Upon encountering a blocked next cell, it updates `known_walls`, triggers online BFS replanning from its current cursor, and routes around the obstruction.

### 2. Reachable-State Optionality Proxy
`reachable_state_count(start, walls, max_steps=3)` computes the cardinality of the $k$-step reachable set:
$$\Omega_k(s) = \left| \left\{ s' \in \mathcal{S} \;\middle|\; d_{\text{grid}}(s, s') \le k \right\} \right|$$
Open interior cells preserve more reachable future states than constrained corners or bottleneck corridors.

---

## Quickstart

### 1. Run Simulation & Unit Tests (Zero Dependencies)

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/intelligence_detours_lab

# Run CLI demonstration
python3 agent_detours.py

# Run automated test suite
python3 -m unittest discover -s tests -v
```

### 2. Test Suite

The 5 automated unit tests verify:
- **`test_original_route`**: Initial shortest path starts at `START`, ends at `GOAL`, and passes through `NEW_WALL`.
- **`test_fixed_fails_when_road_closes`**: Fixed-plan agent fails when `(3, 2)` is blocked.
- **`test_adaptive_replans`**: Adaptive agent replans at least once, avoids `NEW_WALL`, and reaches `GOAL`.
- **`test_impossible_goal`**: Returns `None` when the goal is completely walled off.
- **`test_option_count`**: Reachable state count stays strictly within $(0, |\mathcal{S}|]$.

---

## File Structure

```
intelligence_detours_lab/
├── README.md                    # Concepts, ASCII map, and caveats
├── requirements.txt             # Pure Python standard library note
├── .gitignore                   # Ignores __pycache__ and bytecode
├── agent_detours.py             # Grid world, BFS planner, agents, and reachability counter
└── tests/
    └── test_agent_detours.py    # 5 automated unit tests
```

---

## References

- Michael Levin (2022), *Technological Approach to Mind Everywhere*, Frontiers in Systems Neuroscience. [Link](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2022.768201/full)
- William James (1890), *The Principles of Psychology*, Chapter 1. [Link](https://psychclassics.yorku.ca/James/Principles/prin1.htm)
- A. D. Wissner-Gross & C. E. Freer (2013), *Causal Entropic Forces*, Physical Review Letters 110, 168702. [doi:10.1103/PhysRevLett.110.168702](https://doi.org/10.1103/PhysRevLett.110.168702)
