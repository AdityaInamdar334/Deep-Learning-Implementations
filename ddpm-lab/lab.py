"""One-dimensional DDPM. NumPy only; no downloads."""
import argparse
import numpy as np

CENTER, WIDTH = 2., .30


def schedule(steps):
    if steps < 2:
        raise ValueError("use at least two steps")
    beta = np.linspace(.0001, .02, steps)
    alpha = 1 - beta
    return beta, alpha, np.cumprod(alpha)


def data(rng, size):
    return CENTER * rng.choice([-1., 1.], size) + WIDTH * rng.normal(size=size)


def forward(x0, noise, a):
    return np.sqrt(a) * x0 + np.sqrt(1-a) * noise


def oracle(x, a):
    # At t, each Gaussian component has mean +/-m and variance v.
    m = np.sqrt(a) * CENTER
    v = a * WIDTH**2 + 1-a
    return np.sqrt(1-a) / v * (x - m * np.tanh(m*x/v))


def features(x):
    # Odd functions encode only the known left/right symmetry.
    return np.column_stack((x, np.tanh(.5*x), np.tanh(x), np.tanh(2*x),
                            np.tanh(4*x), np.tanh(8*x), np.tanh(16*x)))


def train(rng, bars, examples=2048):
    weights = []
    for a in bars:
        noise = rng.normal(size=examples)
        x = forward(data(rng, examples), noise, a)
        phi = features(x)
        gram = phi.T @ phi / examples
        weights.append(np.linalg.solve(gram + 1e-5*np.eye(phi.shape[1]),
                                       phi.T @ noise / examples))
    return np.array(weights)


def reverse_step(x, predicted, beta, alpha, bar, z):
    mean = (x - beta / np.sqrt(1-bar) * predicted) / np.sqrt(alpha)
    return mean + np.sqrt(beta) * z


def sample(rng, sched, predict, count):
    beta, alpha, bars = sched
    x = rng.normal(size=count)
    for i in reversed(range(len(beta))):
        z = rng.normal(size=count) if i else 0.
        x = reverse_step(x, predict(x, i), beta[i], alpha[i], bars[i], z)
    return x


def run(steps, seed, count):
    sched = schedule(steps)
    bars = sched[2]
    weights = train(np.random.default_rng(seed), bars)
    predictors = {"learned": lambda x,i: features(x) @ weights[i],
                  "oracle": lambda x,i: oracle(x, bars[i]),
                  "always zero": lambda x,i: np.zeros_like(x)}
    print(f"steps={steps} terminal signal coefficient={np.sqrt(bars[-1]):.5f}")
    target = data(np.random.default_rng(seed+1), count)
    results = {}
    for name, predict in predictors.items():
        # Common random numbers keep this comparison paired.
        generated = sample(np.random.default_rng(seed+2), sched, predict, count)
        w1 = np.abs(np.sort(target)-np.sort(generated)).mean()
        print(f"{name:12s} right={(generated>0).mean():.3f} "
              f"near_mode={(np.abs(np.abs(generated)-CENTER)<.6).mean():.3f} W1={w1:.3f}")
        results[name] = generated
    return target, results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--samples", type=int, default=2048)
    args = p.parse_args()
    if args.steps < 2 or args.samples < 1:
        p.error("steps >= 2 and samples >= 1 required")
    run(args.steps, args.seed, args.samples)
