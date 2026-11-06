"""Deterministic computational workload for compiler and interpreter benchmarking.

Exercises:
1. Integer matrix multiplication & modular reductions.
2. Floating point Taylor series expansions (sin/cos/exp).
3. Cryptographic hashing of state for bit-for-bit output verification.
4. Clean stdout reporting.
"""

import hashlib
import math
import sys


def compute_matrix_hash(n=120):
    # Deterministic integer matrix multiplication
    a = [[(i * 3 + j * 7 + 1) % 100 for j in range(n)] for i in range(n)]
    b = [[(i * 5 + j * 2 + 3) % 100 for j in range(n)] for i in range(n)]
    c = [[0] * n for _ in range(n)]

    for i in range(n):
        for k in range(n):
            aik = a[i][k]
            for j in range(n):
                c[i][j] = (c[i][j] + aik * b[k][j]) % 1000003

    checksum = sum(c[i][i] for i in range(n))
    return checksum


def compute_taylor_series(terms=10000):
    # Floating-point numerical series
    total = 0.0
    for k in range(1, terms + 1):
        x = k * 0.001
        total += math.sin(x) * math.cos(x) + math.exp(-0.0001 * k)
    return total


def main():
    mat_sum = compute_matrix_hash(n=100)
    float_sum = compute_taylor_series(terms=8000)

    # Cryptographic digest of complete computational state
    state_str = f"MAT:{mat_sum}|FLOAT:{float_sum:.8f}|PY:{sys.version_info.major}.{sys.version_info.minor}"
    digest = hashlib.sha256(state_str.encode("utf-8")).hexdigest()

    print(f"STATUS=OK CHECKSUM={digest[:16]} MAT_DIAG={mat_sum} FLOAT_VAL={float_sum:.6f}")


if __name__ == "__main__":
    main()

