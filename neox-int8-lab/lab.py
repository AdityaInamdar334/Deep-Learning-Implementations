"""Reference arithmetic for a linear projection. NumPy CPU, not a timing benchmark."""
import argparse
import numpy as np


def quantize_rows(a):
    peak = np.max(np.abs(a), axis=1, keepdims=True)
    scale = np.where(peak > 0, peak / 127, 1.0)
    codes = np.clip(np.rint(a / scale), -127, 127).astype(np.int8)
    return codes, scale


def project(x, w, bias, threshold, packed_correction=False):
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    columns = np.any(np.abs(x) > threshold, axis=0) if threshold else np.zeros(x.shape[1], bool)
    ordinary = x.copy()
    ordinary[:, columns] = 0
    qx, sx = quantize_rows(ordinary)
    qw, sw = quantize_rows(w)
    accum = qx.astype(np.int32) @ qw.astype(np.int32).T
    output = accum * (sx * sw.T)
    correction_weights = (qw * sw if packed_correction else w)[:, columns]
    output += x[:, columns] @ correction_weights.T
    return output + bias, columns


def fixture(seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .7, (32, 128))
    x[::4, 3] = 24
    x[1::5, 91] = -12
    w = rng.normal(0, .12, (48, 128))
    bias = rng.normal(0, .02, 48)
    return x, w, bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0, 6, 16, 32])
    parser.add_argument("--packed-correction", action="store_true")
    args = parser.parse_args()
    if any(t < 0 or not np.isfinite(t) for t in args.threshold):
        parser.error("thresholds must be finite and nonnegative")
    x = np.array([[64., .24, .24, .24]])
    w = np.array([[0., 1., 1., 1.]])
    print("Hand example (reference 0.720000)")
    for threshold in [0, 6]:
        y, columns = project(x, w, 0, threshold)
        print(f"threshold={threshold}  output={y.item():.6f}  columns={np.flatnonzero(columns).tolist()}")
    x,w,bias = fixture(args.seed)
    reference = x @ w.T + bias
    print("Synthetic projection: 32 tokens, 128 input and 48 output features")
    for threshold in args.threshold:
        y, cols = project(x, w, bias, threshold, args.packed_correction)
        error = y - reference
        print(f"threshold={threshold:g}  columns={cols.sum():3d}  "
              f"RMSE={np.sqrt(np.mean(error**2)):.6f}  max_abs={np.abs(error).max():.6f}")
