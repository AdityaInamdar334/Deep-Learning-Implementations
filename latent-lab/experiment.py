"""Run with Python 3.9+. All latents in this experiment are scalar."""
import argparse
import math
import random
from itertools import accumulate
from operator import mul

SCALE = 0.18215


def schedule():
    start = math.sqrt(0.00085)
    stop = math.sqrt(0.012)
    alphas = (1 - (start + (stop - start) * i / 999) ** 2 for i in range(1000))
    return list(accumulate(alphas, mul))


def encode(raw, factor):
    return factor * raw


def decode(latent, factor):
    return latent / factor


def mix(clean, noise, alpha_bar):
    return math.sqrt(alpha_bar) * clean + math.sqrt(1 - alpha_bar) * noise


def predict_noise(noisy, alpha_bar, clean_variance=1.0):
    variance = alpha_bar * clean_variance + 1 - alpha_bar
    return math.sqrt(1 - alpha_bar) * noisy / variance


def experiment(index=500, count=100000, seed=7):
    rng = random.Random(seed)
    a = schedule()[index]
    squared = [0.0, 0.0, 0.0]
    for _ in range(count):
        raw = rng.gauss(0, 1) / SCALE
        noise = rng.gauss(0, 1)
        expected = mix(encode(raw, SCALE), noise, a)
        wrong = mix(encode(raw, 1), noise, a)
        predictions = (predict_noise(expected, a), predict_noise(wrong, a),
                       predict_noise(wrong, a, SCALE ** -2))
        for i, predicted in enumerate(predictions):
            squared[i] += (predicted - noise) ** 2
    return [s / count for s in squared]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=range(1000), metavar="0..999", default=500)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    for scale in (SCALE, 1):
        print(f"round trip (factor {scale}): {decode(encode(2.0, scale), scale):.6f}")
    print(f"alpha_bar[{args.index}] = {schedule()[args.index]:.6f}")
    for label, mse in zip(("matched", "scale omitted", "oracle recalibrated"),
                          experiment(args.index, args.count, args.seed)):
        print(f"{label:20} noise MSE = {mse:.6f}")
