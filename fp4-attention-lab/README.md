# BitDecoding-FP4: Native FP4 Attention & KV Cache Quantization

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)](test_lab.py)
[![Paper](https://img.shields.io/badge/arXiv-2611.09825-B31B1B.svg)](https://arxiv.org/html/2611.09825v1)
[![Code](https://img.shields.io/badge/GitHub-OpenBitSys%2FBitDecoding--FP4-181717.svg?logo=github)](https://github.com/OpenBitSys/BitDecoding-FP4)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A clean, zero-dependency educational laboratory isolating the mathematics of 4-bit floating-point (FP4) KV cache attention, the catastrophic collective-weight underflow in long-context decoding, the small-block scale repair rule, and dual-residual query quantization.*

Based on the research paper:  
**["BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm-System Co-Design"](https://arxiv.org/html/2611.09825v1)** (arXiv:2611.09825, 2026).

</div>

---

## Executive Summary

Serving Large Language Models across long context windows (e.g. 8K to 128K tokens) is heavily bounded by the memory capacity and memory bandwidth of Key-Value (KV) caches. 

Modern hardware like NVIDIA Blackwell introduces native **FP4 Tensor Cores** (supporting microscaling formats such as NVFP4 and MXFP4) to quadruple throughput and cut memory requirements by $3.5\times$ to $4\times$. However, applying 4-bit microscaling to attention decoding introduces a catastrophic algorithmic failure mode: **Collective Weight Underflow**.

```
Standard Attention Sum:
Tokens:   [ Token 1 ] [ Token 2 ] ... [ Token 2048 ]    [ Recent Token ]
Weights:  [ 1 / 2048] [ 1 / 2048] ... [  1 / 2048  ]    [     1.0      ]
          └───────────────────┬────────────────────┘            │
                     Sum = 1.0 (50% of Attention)           Weight = 1.0 (50%)
                              │
                    Block Scale Rounding
                              ▼
           Max value in block = 1/2048 ≈ 0.000488
           Nearest E4M3FN Scale (< 2^-10) rounds to 0.0!
                              ▼
           All 2,048 tokens quantized to EXACTLY 0.0!
           Attention Output: 0.500000 ──> 0.000000 (Catastrophic Loss)
```

This laboratory reproduces:
1. **The Scale Eradication Trap**: Why rounding block scales to the finite 8-bit float grid (`E4M3FN`) zeroes out long-context history even though the collective sum of those weights constitutes half the attention mass.
2. **The Small-Block Repair Rule**: How shifting the quantization target from 6 to 2 raises the scale factor above the underflow threshold $2^{-10}$ into $2^{-9}$, completely restoring the expected 0.500000 attention output.
3. **The Physical Scale Horizon**: Why at 8,192 tokens even the repaired scale underflows, demonstrating the mathematical limits of microscaling grids without token compaction.
4. **Dual-Residual Query Quantization**: How splitting query projections into two additive FP4 components cut query-induced logit Root Mean Squared Error (RMSE) by more than half.

---

## Mathematical Foundations

### 1. The Microscaling Number Systems

#### FP4 E2M1 Grid
The standard 4-bit floating-point format (E2M1) specifies 8 unsigned discrete magnitude levels:
$$\text{GRID} = \{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$$
Any value $x$ in a block is quantized as:
$$\hat{x} = \text{sign}(x) \cdot \text{Round}\left(\frac{|x|}{S}, \text{GRID}\right) \cdot S$$
where $S$ is the block scale.

#### E4M3FN Scale Grid
In NVFP4 microscaling, each block of 16 elements shares an 8-bit floating-point scale factor chosen from the 127 nonnegative finite `E4M3FN` values:
- **Subnormal values**: $S_i = i \cdot 2^{-9}$ for $i \in \{0, 1, \dots, 7\}$
- **Normal values**: $S = (1 + m / 8) \cdot 2^e$ for $e \in \{-6, \dots, 8\}, m \in \{0, \dots, 7\}$
- **Minimum positive scale**: $S_{\min} = 2^{-9} = \frac{1}{512} \approx 0.001953125$
- **Underflow threshold**: Any requested scale $S_{\text{req}} < 2^{-10} = \frac{1}{1024}$ rounds down to **0.0**!

### 2. The Collective Weight Underflow Phenomenon

Consider $N = 2,048$ historical tokens, each with value $V_{\text{old}} = 1.0$, and one recent token with $V_{\text{recent}} = 0.0$ and attention logit weight $W_{\text{recent}} = 1.0$.

Each historical token has attention probability:
$$P_i = \frac{\text{lift}}{N} = \frac{8}{2048} = \frac{1}{256} = 0.00390625$$
The true expected attention output is:
$$\text{Output}_{\text{ref}} = \frac{\sum_{i=1}^{N} P_i \cdot V_i}{W_{\text{recent}} + \sum_{i=1}^N P_i} = \frac{8 \cdot 1.0}{8 + 8} = \mathbf{0.500000}$$

#### What happens during block quantization?
Each block of 16 tokens has maximum value $M = 8 / 2048 = 0.00390625$.
With the standard quantization target $T = 6$ (the maximum FP4 value):
$$S_{\text{requested}} = \frac{M}{T} = \frac{1/256}{6} = \frac{1}{1536} \approx 0.000651$$
Because $\frac{1}{1536} < 2^{-10} \approx 0.0009765$, the nearest `E4M3FN` scale rounds to **0.0**!
Every single token in all $2048 / 16 = 128$ blocks is multiplied by scale 0:
$$\text{Output}_{\text{quantized}} = \mathbf{0.000000}$$
Half of the entire model's context is completely erased.

### 3. The Small-Block Repair Rule

To prevent the scale from collapsing into zero, the **small-block rule** dynamically adjusts the scaling target from $T = 6$ down to $T = 2$:
$$S_{\text{repaired}} = \frac{M}{2} = \frac{1/256}{2} = \frac{1}{512} = 2^{-9}$$
Because $2^{-9}$ is exactly the first positive subnormal value on the `E4M3FN` scale grid:
$$S = 2^{-9} \quad (\text{Valid Non-Zero Scale})$$
Quantizing each normalized weight:
$$\frac{|x|}{S} = \frac{1/256}{1/512} = 2.0 \in \text{GRID}$$
The restored output recovers to exactly:
$$\text{Output}_{\text{repaired}} = \mathbf{0.500000}$$

---

### 4. Dual-Residual Query Quantization

Quantizing query vectors $Q \in \mathbb{R}^{d}$ to 4-bit causes significant score variance when multiplied against unquantized or quantized keys.

BitDecoding-FP4 introduces a two-stage residual quantization scheme:
1. **Coarse Step**: Compute coarse 4-bit approximation $Q_{\text{coarse}} = \text{Quantize}(Q, S_1)$.
2. **Residual Step**: Compute remainder $R = Q - Q_{\text{coarse}}$.
3. **Refinement Step**: Quantize the remainder with its own independent scale: $Q_{\text{residual}} = \text{Quantize}(R, S_2)$.
4. **Reconstructed Query**:
   $$\hat{Q} = Q_{\text{coarse}} + Q_{\text{residual}}$$

When evaluated against 256 random Gaussian keys at dimension $d=16$, dual-residual quantization reduces query-only logit RMSE from $\approx 0.26$ down to $\approx 0.09$ (a $> 65\%$ error reduction).

---

## Empirical Walkthrough & Output

Running the standalone educational script:

```bash
python3 lab.py --tokens 2048 --seed 7
```

```
Reference attention output = 0.500000
ordinary                  scale=0.00000000  output=0.000000
lift only                 scale=0.00000000  output=0.000000
lift + small-block rule   scale=0.00195312  output=0.500000
Query-only logit RMSE: 0.259972 -> 0.089849
```

### Probing the Scale Horizon (8,192 tokens)

```bash
python3 lab.py --tokens 8192
```

```
Reference attention output = 0.500000
ordinary                  scale=0.00000000  output=0.000000
lift only                 scale=0.00000000  output=0.000000
lift + small-block rule   scale=0.00000000  output=0.000000
Query-only logit RMSE: 0.259972 -> 0.089849
```

At $N=8,192$, the required scale with target 2 is:
$$S = \frac{8 / 8192}{2} = \frac{1}{2048} = 2^{-11} < 2^{-10}$$
Because $2^{-11}$ falls below the midpoint between 0 and $2^{-9}$, even the small-block rule underflows to 0. This demonstrates why extreme long-context microscaling requires page-level or token-compacted mixed precision.

---

## Unit Testing & Verification

The lab provides 5 formal unit tests checking grid bounds, tie-breaking behavior, collective-weight recovery, and query residual improvements:

```bash
python3 -m unittest -v test_lab.py
```

```
test_collective_weight (test_lab.AttentionChecks.test_collective_weight) ... ok
test_query_residual (test_lab.AttentionChecks.test_query_residual) ... ok
test_repair_has_a_limit (test_lab.AttentionChecks.test_repair_has_a_limit) ... ok
test_scale_grid (test_lab.AttentionChecks.test_scale_grid) ... ok
test_signed_round_trip (test_lab.AttentionChecks.test_signed_round_trip) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.012s

OK
```

---

## File Structure

```
fp4-attention-lab/
├── README.md       # This comprehensive technical guide and theoretical walkthrough
├── lab.py          # Pure Python 3.9+ implementation of E4M3FN scales, FP4 block quantization,
│                   # collective weight attention simulation, and dual-residual queries
└── test_lab.py     # 5 unit tests verifying mathematical correctness and invariant thresholds
```

---

## Academic References

1. **BitDecoding-FP4 Team (2026)**. *BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm-System Co-Design*. arXiv preprint [arXiv:2611.09825](https://arxiv.org/html/2611.09825v1).
2. **NVIDIA Corporation (2024)**. *NVIDIA Blackwell Architecture Technical Whitepaper: 4-Bit Microscaling and Tensor Cores*.
3. **OpenBitSys Community (2026)**. *BitDecoding-FP4 Reference Implementation*. [GitHub Repository](https://github.com/OpenBitSys/BitDecoding-FP4).

---

## License

This project is licensed under the [MIT License](../LICENSE).
