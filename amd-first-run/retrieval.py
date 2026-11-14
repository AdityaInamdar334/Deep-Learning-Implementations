"""Measure batched exact dot-product search on fixed, synthetic float32 vectors."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def inputs(rows, queries, dim, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, dim), dtype=np.float32),
            rng.standard_normal((queries, dim), dtype=np.float32))


def search(queries, corpus, batch, argmax, concat):
    parts = []
    for start in range(0, len(queries), batch):
        scores = queries[start:start + batch] @ corpus.T
        parts.append(argmax(scores))
    return concat(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "rocm"], default="cpu")
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--queries", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", type=Path, default=Path("result.json"))
    args = p.parse_args()
    if min(args.rows, args.queries, args.dim, args.repeats, *args.batches) < 1:
        p.error("all sizes and repeats must be positive")
    corpus, queries = inputs(args.rows, args.queries, args.dim, args.seed)
    # Limit memory used by the correctness check on larger runs.
    check_n = min(32, args.queries)
    reference = (queries[:check_n].astype(np.float64) @ corpus.astype(np.float64).T).argmax(axis=1)
    metadata = dict(device=args.device, python=platform.python_version(), numpy=np.__version__,
                    rows=args.rows, queries=args.queries, dimensions=args.dim,
                    repeats=args.repeats, seed=args.seed, dtype="float32", check_queries=check_n)
    if args.device == "rocm":
        import torch
        if not (torch.cuda.is_available() and torch.version.hip):
            raise SystemExit("Requires ROCm PyTorch and a visible AMD GPU.")
        metadata.update(torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(0))
        torch.set_float32_matmul_precision("highest")
        corpus, queries = (torch.from_numpy(x).to("cuda:0") for x in (corpus, queries))
        argmax, concat = lambda x: x.argmax(dim=1), torch.cat
        sync = torch.cuda.synchronize
    else:
        argmax, concat = lambda x: x.argmax(axis=1), np.concatenate
        sync = lambda: None
    results = []
    for batch in args.batches:
        search(queries, corpus, batch, argmax, concat)
        sync()
        secs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            ids = search(queries, corpus, batch, argmax, concat)
            sync()
            secs.append(time.perf_counter() - start)
        checked = ids[:check_n]
        if args.device == "rocm":
            checked = checked.cpu().numpy()
        median = statistics.median(secs)
        results.append(dict(batch=batch, seconds=secs, median_seconds=median,
                            queries_per_second=args.queries / median,
                            top1_agreement=float(np.mean(checked == reference))))
    report = json.dumps(dict(environment=metadata, results=results), indent=2)
    args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
