"""Inspect matched masks, early stops and the paper's paired counts."""
import argparse
import math
import random


def schedules(seed, steps=100, heads=20):
    """One packet per time bin. IDs represent vulnerable layer/head slots."""
    if steps % 5 or steps // 5 < 7 or heads < 5:
        raise ValueError("Use five equal bins of at least seven steps and at least five heads.")
    rng = random.Random(seed)
    width = steps // 5
    masks = {name: [set() for _ in range(steps)] for name in ("burst", "spaced", "diffuse")}
    for start in range(0, steps, width):
        first = start + rng.randrange(width - 6)
        selected = rng.sample(range(heads), round(.4 * heads))
        for name, offsets in [("burst", (0, 1, 2)), ("spaced", (0, 3, 6))]:
            for offset in offsets:
                masks[name][first + offset].update(selected)
        # Match even the identity counts: each selected head is masked three times.
        for head in selected:
            for time in rng.sample(range(start, start + width), 3):
                masks["diffuse"][time].add(head)
    return masks


def toy_decode(mask, threshold):
    """Deterministic binary copy task with explicit head contributions.

    Start with bit 1. Each live head reads the last bit. The projected sum
    competes with a fixed bias favoring zero. Ties go to zero.
    """
    bit = 1
    first_error = None
    for time, missing in enumerate(mask, start=1):
        one_score = bit * (20 - len(missing))
        bit = int(one_score > threshold)
        if bit == 0 and first_error is None:
            first_error = time
    return bit, first_error


def mcnemar(losses, gains):
    n = losses + gains
    tail = sum(math.comb(n, k) for k in range(min(losses, gains) + 1))
    return min(1.0, 2 * tail / 2**n)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stop", type=int, default=47)
    p.add_argument("--bias", type=float, default=13)
    args = p.parse_args()
    if not 1 <= args.stop <= 100:
        p.error("--stop must be 1..100")
    for name, mask in schedules(args.seed).items():
        exposed = mask[:args.stop]
        counts = [len(row) for row in mask]
        bit, error = toy_decode(exposed, args.bias)
        print(f"{name:7s} planned={sum(counts)} exposed={sum(map(len, exposed))}"
              f" peak={max(counts)} final_bit={bit} first_error={error}")
    for model, losses, gains in [("Qwen", 39, 8), ("Llama", 36, 18)]:
        print(f"{model}: difference={100*(gains-losses)/250:.1f} pp,"
              f" exact paired p={mcnemar(losses, gains):.8g}")
