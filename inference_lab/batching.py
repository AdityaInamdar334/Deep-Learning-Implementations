"""Episode 3: Dynamic Request Batching with Left-Padding and KV Caching.

Solves the multi-tenant throughput bottleneck by batching heterogeneous requests
with variable prompt lengths using left-padding, causal attention masks,
per-request position offsets, and per-sequence early stopping upon EOS.
"""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import List, Optional, Tuple

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

def collate_left_padded(prompt_token_lists: List[List[int]], pad_token_id: int, device: torch.device):
    """Collate variable-length prompt lists into left-padded tensors.
    
    Returns:
        input_ids: [batch_size, max_seq_len]
        attention_mask: [batch_size, max_seq_len] (0 for pad, 1 for real token)
        position_ids: [batch_size, max_seq_len] (0..L-1 for real tokens)
        prompt_lens: List of unpadded lengths
    """
    batch_size = len(prompt_token_lists)
    prompt_lens = [len(p) for p in prompt_token_lists]
    max_len = max(prompt_lens)

    padded_ids = []
    attn_masks = []

    for p in prompt_token_lists:
        pad_count = max_len - len(p)
        padded_ids.append([pad_token_id] * pad_count + p)
        attn_masks.append([0] * pad_count + [1] * len(p))

    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)
    # Position IDs starting at 0 for the first valid token of each request
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

    return input_ids, attention_mask, position_ids, prompt_lens

@torch.inference_mode()
def generate_batched(
    model,
    prompts: List[torch.Tensor] | List[List[int]],
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[List[int]], dict]:
    """Execute batched autoregressive generation with KV caching and early termination.
    
    Args:
        model: Causal LM
        prompts: List of 1D tensors or List of token ID lists
        max_new_tokens: Max tokens to generate per request
        eos_token_id: Optional EOS token ID
        pad_token_id: Pad token ID (defaults to eos_token_id if None)
        device: Target hardware device
    
    Returns:
        completions: List of generated token lists per request (excluding padding and prompts)
        telemetry: Detailed latency, throughput, and step tracking dictionary
    """
    if len(prompts) == 0 or max_new_tokens < 1:
        raise ValueError("Must provide at least one prompt and positive max_new_tokens.")

    if device is None:
        device = next(model.parameters()).device

    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    # Normalize prompt inputs to lists of ints
    token_lists = []
    for p in prompts:
        if isinstance(p, torch.Tensor):
            token_lists.append(p.squeeze().tolist() if p.ndim > 0 else [p.item()])
        else:
            token_lists.append(list(p))

    batch_size = len(token_lists)
    input_ids, attention_mask, position_ids, prompt_lens = collate_left_padded(token_lists, pad_token_id, device)

    completions: List[List[int]] = [[] for _ in range(batch_size)]
    active = [True] * batch_size
    timestamps = []
    lengths = []

    synchronize(device)
    start = time.perf_counter_ns()

    # --- Step 0: Batched Prefill ---
    lengths.append(input_ids.shape[1])
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = out.logits
    pkv = out.past_key_values
    next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del logits

    synchronize(device)
    timestamps.append(time.perf_counter_ns())

    for i in range(batch_size):
        tok = next_tokens[i].item()
        completions[i].append(tok)
        if eos_token_id is not None and tok == eos_token_id:
            active[i] = False

    # Current token sequence position for each request
    curr_pos = position_ids[:, -1:] + 1

    # --- Steps 1 to max_new_tokens - 1: Incremental Batched Decode ---
    for _ in range(max_new_tokens - 1):
        if not any(active):
            break

        lengths.append(1)
        # Update attention mask: append 1s for all items in batch
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
            dim=1,
        )

        out = model(
            input_ids=next_tokens,
            attention_mask=attention_mask,
            position_ids=curr_pos,
            past_key_values=pkv,
            use_cache=True,
        )
        logits = out.logits
        pkv = out.past_key_values
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del logits

        curr_pos = curr_pos + 1
        synchronize(device)
        timestamps.append(time.perf_counter_ns())

        for i in range(batch_size):
            if active[i]:
                tok = next_tokens[i].item()
                completions[i].append(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    active[i] = False

    elapsed = [(end - start) / 1e6 for end in timestamps]
    gaps = [b - a for a, b in zip(elapsed, elapsed[1:])]
    total_tokens_generated = sum(len(c) for c in completions)

    return completions, {
        "batch_size": batch_size,
        "input_lengths": lengths,
        "prompt_lens": prompt_lens,
        "ttft_ms": elapsed[0],
        "tpot_ms": statistics.mean(gaps) if gaps else None,
        "itl_ms": gaps,
        "total_ms": elapsed[-1],
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": 1000.0 * total_tokens_generated / elapsed[-1],
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "fixture", help="Local model fixture")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--new", type=int, default=8)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/batching_run.json"))
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=getattr(torch, args.dtype)
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    test_prompts = [
        "sensor reports hot",
        "sensor reports cold . fan off .",
        "sensor reports hot . fan on .",
        "sensor reports cold",
        "fan on",
        "fan off",
        "sensor reports cold .",
        "sensor reports hot .",
    ]

    print("--- Episode 3: Dynamic Request Batching ---")
    print(f"Testing batches of heterogeneous prompts with left-padding on {device.type}...")

    # Single-batch demonstration
    demo_batch = test_prompts[:3]
    demo_tokens = [tokenizer.encode(p, add_special_tokens=False) for p in demo_batch]
    demo_outputs, demo_telemetry = generate_batched(
        model, demo_tokens, args.new, eos_token_id=tokenizer.eos_token_id, device=device
    )

    print("\nDemonstration Batch (3 heterogeneous requests):")
    for i, (pr, out_ids) in enumerate(zip(demo_batch, demo_outputs)):
        decoded = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(f"  Req {i+1} [len {len(demo_tokens[i])}]: \"{pr}\" -> \"{decoded}\" ({len(out_ids)} tokens)")
    print(f"TTFT: {demo_telemetry['ttft_ms']:.2f} ms | Total: {demo_telemetry['total_ms']:.2f} ms | Throughput: {demo_telemetry['tokens_per_second']:.1f} tok/s\n")

    # Benchmarking sweep
    results = {
        "model": str(args.model.resolve().name),
        "device": args.device,
        "threads": args.threads,
        "policy": "dynamic batching with left-padding and KV cache",
        "sweep": [],
    }

    print("Benchmarking throughput scaling across batch sizes:")
    for b_size in args.batch_sizes:
        prompts_subset = (test_prompts * math.ceil(b_size / len(test_prompts)))[:b_size]
        tok_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts_subset]

        for _ in range(args.warmup):
            generate_batched(model, tok_lists, args.new, tokenizer.eos_token_id, device=device)

        batch_runs = []
        for _ in range(args.runs):
            _, trace = generate_batched(model, tok_lists, args.new, tokenizer.eos_token_id, device=device)
            batch_runs.append(trace)

        avg_throughput = statistics.mean(r["tokens_per_second"] for r in batch_runs)
        p50_total = quantiles([r["total_ms"] for r in batch_runs])["p50"]
        p50_ttft = quantiles([r["ttft_ms"] for r in batch_runs])["p50"]
        total_tokens = sum(r["total_tokens_generated"] for r in batch_runs) // len(batch_runs)

        entry = {
            "batch_size": b_size,
            "avg_throughput_tok_per_s": avg_throughput,
            "p50_total_ms": p50_total,
            "p50_ttft_ms": p50_ttft,
            "avg_tokens_per_batch": total_tokens,
        }
        results["sweep"].append(entry)
        print(f"Batch {b_size:2d} | Throughput: {avg_throughput:6.1f} tok/s | Total Latency p50: {p50_total:5.2f} ms | TTFT p50: {p50_ttft:5.2f} ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved batching benchmarks to {args.out}")

if __name__ == "__main__":
    main()

