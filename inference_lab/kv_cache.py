"""Episode 2: Static Key-Value (KV) Caching Generator.

Eliminates repetitive prompt recomputation by preserving past key and value tensors
across autoregressive decoding steps. Reduces computational work from
O(G * P + G^2 / 2) down to O(P + G).
"""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def quantiles(values):
    ordered = sorted(values)
    return {f"p{q}": ordered[math.ceil(q / 100 * len(ordered)) - 1] if ordered else None
            for q in (50, 95, 99)}

def inspect_cache_memory(pkv):
    """Computes total elements and resident byte footprint of the KV cache."""
    total_elements = 0
    total_bytes = 0
    if pkv is None:
        return {"elements": 0, "bytes": 0, "layers": 0}
    
    # Handle transformers DynamicCache (v4.36+ / v5.x)
    if hasattr(pkv, "layers"):
        for layer in pkv.layers:
            if hasattr(layer, "keys") and layer.keys is not None:
                total_elements += layer.keys.numel() + layer.values.numel()
                total_bytes += (layer.keys.numel() * layer.keys.element_size() +
                                layer.values.numel() * layer.values.element_size())
        return {"elements": total_elements, "bytes": total_bytes, "layers": len(pkv.layers)}
    
    # Handle classic tuple-of-tuples format ((k_layer0, v_layer0), ...)
    if isinstance(pkv, (tuple, list)):
        for layer in pkv:
            k, v = layer[0], layer[1]
            total_elements += k.numel() + v.numel()
            total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
        return {"elements": total_elements, "bytes": total_bytes, "layers": len(pkv)}
    
    return {"elements": 0, "bytes": 0, "layers": 0}

@torch.inference_mode()
def generate_with_cache(model, prompt, count, eos=None):
    """Autoregressive generation utilizing Key-Value caching.
    
    Prefill pass processes the full prompt [1, P].
    Subsequent decode steps pass only 1 new token [1, 1] plus the past KV cache.
    """
    if count < 1 or prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("Use one nonempty prompt and a positive output length.")
    
    device = prompt.device
    lengths = []
    timestamps = []
    generated_tokens = []
    
    synchronize(device)
    start = time.perf_counter_ns()
    
    # --- Prefill Phase (Step 0) ---
    lengths.append(prompt.shape[1])
    out = model(input_ids=prompt, use_cache=True)
    logits = out.logits
    pkv = out.past_key_values
    token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens.append(token)
    del logits
    synchronize(device)
    timestamps.append(time.perf_counter_ns())
    
    hit_eos = (eos is not None and token.item() == eos)
    
    # --- Decode Phase (Steps 1 to count - 1) ---
    if not hit_eos:
        for _ in range(count - 1):
            lengths.append(1)  # Only 1 token processed per decode step!
            out = model(input_ids=token, past_key_values=pkv, use_cache=True)
            logits = out.logits
            pkv = out.past_key_values
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(token)
            del logits
            synchronize(device)
            timestamps.append(time.perf_counter_ns())
            if eos is not None and token.item() == eos:
                break
                
    elapsed = [(end - start) / 1e6 for end in timestamps]
    gaps = [b - a for a, b in zip(elapsed, elapsed[1:])]
    cache_info = inspect_cache_memory(pkv)
    
    output_tensor = torch.cat(generated_tokens, dim=1)
    return output_tensor, {
        "input_lengths": lengths,
        "positions_processed": sum(lengths),
        "ttft_ms": elapsed[0],
        "tpot_ms": statistics.mean(gaps) if gaps else None,
        "itl_ms": gaps,
        "total_ms": elapsed[-1],
        "output_tokens": len(elapsed),
        "cache_elements": cache_info["elements"],
        "cache_bytes": cache_info["bytes"],
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "fixture", help="Local causal-LM checkpoint")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--prompt", default="sensor reports hot")
    p.add_argument("--lengths", type=int, nargs="+", help="Synthetic prefix lengths for benchmark")
    p.add_argument("--new", type=int, default=8)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/kv_cache_run.json"))
    args = p.parse_args()

    if min(args.new, args.runs, args.threads) < 1 or args.warmup < 0:
        p.error("Use positive counts and nonnegative warmup.")
        
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=getattr(torch, args.dtype)
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    
    seed_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not seed_ids:
        p.error("Prompt must encode to at least one token.")
        
    lengths = args.lengths or [len(seed_ids)]
    limit = getattr(model.config, "max_position_embeddings", None)
    if min(lengths) < 1 or (limit and max(lengths) + args.new - 1 > limit):
        p.error(f"Every prefix must fit context limit ({limit}).")

    def prompt_for(length):
        return torch.tensor([(seed_ids * math.ceil(length / len(seed_ids)))[:length]], device=device)

    if args.lengths is None:
        prompt_tensor = prompt_for(len(seed_ids))
        output, row = generate_with_cache(model, prompt_tensor, args.new, tokenizer.eos_token_id)
        continuation = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        print("--- Episode 2: KV-Cached Generation ---")
        print("Prompt:            ", args.prompt)
        print("Continuation:      ", continuation)
        print("Forward inputs:    ", row["input_lengths"])
        print(f"Positions processed: {row['positions_processed']} (vs baseline uncached: {sum(range(len(seed_ids), len(seed_ids)+args.new))})")
        print(f"TTFT: {row['ttft_ms']:.3f} ms | TPOT: {row['tpot_ms']:.3f} ms | Total: {row['total_ms']:.3f} ms")
        print(f"KV Cache Size:      {row['cache_elements']} floats ({row['cache_bytes']} bytes)")
        return

    result = {
        "model": str(args.model.resolve().name),
        "architecture": type(model).__name__,
        "parameters": sum(w.numel() for w in model.parameters()),
        "device": args.device,
        "dtype": args.dtype,
        "threads": args.threads,
        "policy": "greedy, batch=1, use_cache=True, static KV caching",
        "groups": [],
    }

    print(f"Benchmarking Static KV Caching across prefix lengths: {lengths}")
    for length in lengths:
        prompt = prompt_for(length)
        for _ in range(args.warmup):
            generate_with_cache(model, prompt, args.new)
            
        rows = []
        for _ in range(args.runs):
            output, row = generate_with_cache(model, prompt, args.new)
            rows.append(row)
            
        baseline_positions = sum(length + i for i in range(args.new))
        cached_positions = length + (args.new - 1)
        saving_pct = 100.0 * (1.0 - cached_positions / baseline_positions)
        
        group = {
            "length": length,
            "requests": rows,
            "last_output_ids": output[0].tolist(),
            "positions_processed": cached_positions,
            "baseline_positions": baseline_positions,
            "work_reduction_pct": saving_pct,
            "ttft_ms": quantiles([r["ttft_ms"] for r in rows]),
            "tpot_ms": quantiles([r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]),
            "total_ms": quantiles([r["total_ms"] for r in rows]),
            "cache_bytes": rows[-1]["cache_bytes"],
            "service_tokens_per_s": 1000 * sum(r["output_tokens"] for r in rows) / sum(r["total_ms"] for r in rows),
        }
        result["groups"].append(group)
        print(f"prefix {length:4d} | TTFT p50 {group['ttft_ms']['p50']:.3f} ms | "
              f"TPOT p50 {group['tpot_ms']['p50']:.3f} ms | "
              f"Work reduction: {saving_pct:.1f}% | {group['service_tokens_per_s']:.1f} tokens/s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved benchmark results to {args.out}")

if __name__ == "__main__":
    main()

