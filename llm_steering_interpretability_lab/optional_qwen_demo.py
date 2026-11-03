"""OPTIONAL, not run in the offline benchmark.

Requires: pip install -U transformers accelerate safetensors
Downloads Qwen3-0.6B from Hugging Face. This is a live model demonstration,
not a Qwen3.5-4B neuron-explorer reproduction.
"""
import argparse
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def collect_last_token_hidden(model, tokenizer, texts, layer_index, device):
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    batch = {key: val.to(device) for key, val in batch.items()}
    with torch.no_grad():
        states = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
    # states[0] is embeddings, states[i + 1] is after decoder block i
    indices = batch["attention_mask"].sum(-1) - 1
    row = torch.arange(len(texts), device=device)
    return states[layer_index + 1][row, indices].float()


@contextmanager
def add_residual_direction(model, layer_index, direction, strength):
    layer = model.model.layers[layer_index]
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            x = output[0] + float(strength) * direction.to(output[0].device, output[0].dtype)
            return (x,) + output[1:]
        return output + float(strength) * direction.to(output.device, output.dtype)
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not (0 <= args.layer < 28):
        raise ValueError("This example is specifically written for 28-layer Qwen3-0.6B")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    # Matched prompt structures. Sample size is intentionally tiny for pedagogy.
    happy = [
        "A joyful story about a gardener.",
        "A cheerful story about a traveler.",
        "A hopeful story about a student.",
        "A delighted story about a scientist.",
    ]
    sad = [
        "A gloomy story about a gardener.",
        "A sorrowful story about a traveler.",
        "A hopeless story about a student.",
        "A miserable story about a scientist.",
    ]
    d = collect_last_token_hidden(model, tok, sad, args.layer, device).mean(0)
    d -= collect_last_token_hidden(model, tok, happy, args.layer, device).mean(0)
    d = F.normalize(d, dim=0)
    prompt = "Write a short story about a traveler arriving in a new city."
    input_ids = tok(prompt, return_tensors="pt").to(device)
    def sample():
        with torch.no_grad():
            result = model.generate(**input_ids, do_sample=False,
                                    max_new_tokens=args.max_new_tokens,
                                    pad_token_id=tok.eos_token_id)
        return tok.decode(result[0, input_ids["input_ids"].shape[-1]:], skip_special_tokens=True)
    baseline = sample()
    with add_residual_direction(model, args.layer, d, args.strength):
        changed = sample()
    restored = sample()
    print("MODEL", args.model, "LAYER", args.layer, "STRENGTH", args.strength)
    print("BASELINE:\n",baseline)
    print("STEERED:\n",changed)
    print("RESTORED MATCH:",baseline == restored)
    print("NOTE: Small contrast set; do not infer that steering worked without multiple held-out prompts, controls and damage checks.")

if __name__ == "__main__":
    main()
