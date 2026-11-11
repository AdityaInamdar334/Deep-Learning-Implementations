"""Two CPU experiments in block-scaled attention. Python 3.9+, standard library."""
import argparse
import math
import random

GRID = (0, .5, 1, 1.5, 2, 3, 4, 6)
# Nonnegative finite E4M3FN values in encoding order.
SCALES = [i * 2**-9 for i in range(8)]
SCALES += [(1 + m / 8) * 2**e for e in range(-6, 9)
           for m in range(8) if e < 8 or m < 7]


def nearest(value, grid):
    # Break a midpoint tie in favor of the even encoding.
    index = min(range(len(grid)), key=lambda i: (abs(grid[i] - value), i % 2))
    return grid[index]


def block(values, target=6):
    scale = nearest(max(map(abs, values)) / target, SCALES)
    if scale == 0:
        return [0.0] * len(values), scale
    restored = [math.copysign(nearest(abs(x) / scale, GRID) * scale, x) for x in values]
    return restored, scale


def old_token_output(n, lift, target):
    # Value=1 for the n old tokens. The recent token has value=0 and weight=1.
    # All old blocks are identical, so one block represents n/16 blocks.
    old, scale = block([lift / n] * 16, target)
    numerator = sum(old) * (n // 16)
    denominator = lift * (1 + n * (1 / n))
    return scale, numerator / denominator


def query_experiment(seed):
    rng = random.Random(seed)
    q = [rng.gauss(0, 1) for _ in range(16)]
    coarse, _ = block(q)
    remainder, _ = block([x - y for x, y in zip(q, coarse)])
    repaired = [a + b for a, b in zip(coarse, remainder)]
    keys = [[rng.gauss(0, 1) for _ in q] for _ in range(256)]

    def score_rmse(candidate):
        errors = [sum((a - b) * k for a, b, k in zip(candidate, q, key)) / 4
                  for key in keys]
        return math.sqrt(sum(e * e for e in errors) / len(errors))

    return score_rmse(coarse), score_rmse(repaired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.tokens < 16 or args.tokens % 16:
        parser.error("--tokens must be a positive multiple of 16")
    print("Reference attention output = 0.500000")
    for label, lift, target in [("ordinary", 1, 6), ("lift only", 8, 6),
                                ("lift + small-block rule", 8, 2)]:
        scale, output = old_token_output(args.tokens, lift, target)
        print(f"{label:25s} scale={scale:.8f}  output={output:.6f}")
    coarse, repaired = query_experiment(args.seed)
    print(f"Query-only logit RMSE: {coarse:.6f} -> {repaired:.6f}")
