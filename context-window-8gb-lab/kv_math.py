#!/usr/bin/env python3
"""Illustrative full-attention F16 KV-cache capacity calculator (not a VRAM profiler)."""
import argparse

def kv_bytes(tokens: int, layers: int, kv_heads: int, head_dim: int, bytes_per_element: float = 2.0) -> float:
    if min(tokens, layers, kv_heads, head_dim) < 1 or bytes_per_element <= 0:
        raise ValueError("all dimensions and precision must be positive")
    return tokens * layers * kv_heads * head_dim * 2 * bytes_per_element

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=28, help="Qwen2.5 7B: 28")
    p.add_argument("--kv-heads", type=int, default=4, help="Qwen2.5 7B: 4")
    p.add_argument("--head-dim", type=int, default=128, help="Qwen2.5 7B: 3584/28=128")
    p.add_argument("--bytes", type=float, default=2.0, help="F16/BF16 = 2; quantized cache requires format-specific overhead")
    p.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    args = p.parse_args()
    print("Illustrative dense KV cache (does not include weights, buffers, allocation overhead, or CPU/GPU placement)")
    for context in args.contexts:
        print(f"{context:>6} tokens: {kv_bytes(context, args.layers, args.kv_heads, args.head_dim, args.bytes)/1024**2:>8.1f} MiB")
if __name__ == "__main__":
    main()
