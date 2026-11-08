"""Single-request, full-prefix greedy inference. All timing units in the JSON are ms."""
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

@torch.inference_mode()
def generate(model, prompt, count, eos=None):
    if count < 1 or prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("Use one nonempty prompt and a positive output length.")
    sequence = prompt
    device = sequence.device
    lengths, timestamps = [], []
    synchronize(device)
    start = time.perf_counter_ns()
    for _ in range(count):
        lengths.append(sequence.shape[1])
        logits = model(input_ids=sequence, use_cache=False).logits
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, token), dim=1)
        del logits
        synchronize(device)
        timestamps.append(time.perf_counter_ns())
        if eos is not None and token.item() == eos:
            break
    elapsed = [(end - start) / 1e6 for end in timestamps]
    gaps = [b - a for a, b in zip(elapsed, elapsed[1:])]
    return sequence[:, prompt.shape[1]:], {
        "input_lengths": lengths, "positions_processed": sum(lengths),
        "ttft_ms": elapsed[0], "tpot_ms": statistics.mean(gaps) if gaps else None,
        "itl_ms": gaps, "total_ms": elapsed[-1], "output_tokens": len(elapsed),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "fixture", help="Local causal-LM checkpoint")
    p.add_argument("--device", choices=["cpu","cuda","mps"], default="cpu")
    p.add_argument("--dtype", choices=["float32","float16","bfloat16"], default="float32")
    p.add_argument("--prompt", default="sensor reports hot")
    p.add_argument("--lengths", type=int, nargs="+", help="Exact synthetic prefix lengths for benchmark mode")
    p.add_argument("--new", type=int, default=8)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/my_run.json"))
    args = p.parse_args()
    if min(args.new,args.runs,args.threads) < 1 or args.warmup < 0:
        p.error("Use positive counts and nonnegative warmup.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                    torch_dtype=getattr(torch,args.dtype)).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not seed_ids:
        p.error("Prompt must encode to at least one token.")
    lengths = args.lengths or [len(seed_ids)]
    limit = getattr(model.config, "max_position_embeddings", None)
    if min(lengths) < 1 or (limit and max(lengths)+args.new-1 > limit):
        p.error(f"Every prefix must be positive and fit the context limit ({limit}) through the last forward pass.")
    def prompt_for(length):
        return torch.tensor([(seed_ids * math.ceil(length / len(seed_ids)))[:length]], device=device)
    if args.lengths is None:
        output, row = generate(model, prompt_for(len(seed_ids)), args.new, tokenizer.eos_token_id)
        print("Continuation:",tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
        print("Forward-pass input lengths:", row["input_lengths"])
        return
    result = {"model": str(args.model.resolve().name), "architecture": type(model).__name__,
              "parameters": sum(w.numel() for w in model.parameters()),
              "parameter_bytes": sum(w.numel()*w.element_size() for w in model.parameters()),
              "torch": str(torch.__version__), "transformers": transformers.__version__,
              "python": platform.python_version(), "platform": platform.platform(),
              "device": args.device, "device_name": torch.cuda.get_device_name(device) if device.type=="cuda" else platform.machine(),
              "dtype": args.dtype, "threads": args.threads, "warmup": args.warmup,
              "runs": args.runs, "new": args.new, "prompt_seed_ids": seed_ids,
              "policy": "greedy, batch=1, use_cache=False, fixed output length, synchronized per token",
              "groups": []}
    for length in lengths:
        prompt = prompt_for(length)
        for _ in range(args.warmup):
            generate(model,prompt,args.new)
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        for _ in range(args.runs):
            output, row = generate(model,prompt,args.new)
            rows.append(row)
        group = {"length":length, "requests":rows, "last_output_ids":output[0].tolist(),
                 "ttft_ms":quantiles([r["ttft_ms"] for r in rows]),
                 "tpot_ms":quantiles([r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]),
                 "total_ms":quantiles([r["total_ms"] for r in rows]),
                 "service_tokens_per_s":1000*sum(r["output_tokens"] for r in rows)/sum(r["total_ms"] for r in rows),
                 "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None}
        result["groups"].append(group)
        print(f"prefix {length:4d} | TTFT p50 {group['ttft_ms']['p50']:.3f} ms | "
              f"TPOT p50 {group['tpot_ms']['p50']} ms | {group['service_tokens_per_s']:.1f} output tokens/s")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(f"Recorded {args.out}")

if __name__ == "__main__":
    main()
