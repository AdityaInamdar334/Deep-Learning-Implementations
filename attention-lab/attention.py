"""Inspect Stable Diffusion v1 attention geometry and one exact attention row."""
import argparse
import math
import random


def scores(height, width, keys, heads=8, batch=1):
    return batch * heads * height * width * keys * 2 / 2**20


def dense(logits, values):
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)


def tiled(logits, values, block):
    if block < 1:
        raise ValueError("block must be positive")
    peak, mass, weighted = -math.inf, 0., 0.
    for start in range(0, len(logits), block):
        chunk = logits[start:start + block]
        new_peak = max(peak, max(chunk))
        correction = math.exp(peak - new_peak)
        mass *= correction
        weighted *= correction
        for offset, score in enumerate(chunk):
            weight = math.exp(score - new_peak)
            mass += weight
            weighted += weight * values[start + offset]
        peak = new_peak
    return weighted / mass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.resolution < 64 or args.resolution % 64 or args.batch < 1:
        parser.error("use a positive batch and a resolution divisible by 64")
    for level, width in enumerate([320, 640, 1280, 1280]):
        side = args.resolution // (8 * 2**level)
        spatial = scores(side, side, side**2, batch=args.batch)
        text = scores(side, side, 77, batch=args.batch)
        print(f"{side:3} x {side:<3}  head width {width//8:3}  "
              f"self {spatial:9.3f} MiB  text {text:7.3f} MiB")
    rng = random.Random(19)
    logits = [rng.uniform(-25, 25) for _ in range(77)]
    values = [rng.uniform(-1, 1) for _ in logits]
    print(f"Dense/tiled absolute error: {abs(dense(logits, values)-tiled(logits, values, 8)):.3g}")
