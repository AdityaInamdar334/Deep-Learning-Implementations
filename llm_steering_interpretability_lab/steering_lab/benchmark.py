"""A reproducible, CPU-only illustration of steering and specificity controls."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import torch
import torch.nn.functional as F
from .model import train_model, make_batch
from .interventions import (collect_neurons, contrast_direction,
    steer_residual, steer_neuron, project_output_weights, categorical_kl)


def evaluate(model, tokens, mood, topic):
    with torch.no_grad():
        a, b = model(tokens)
    return {"mood_accuracy": (a.argmax(-1) == mood).float().mean().item(),
            "topic_accuracy": (b.argmax(-1) == topic).float().mean().item(),
            "sad_probability": F.softmax(a, -1)[:, 1].mean().item()}, (a, b)


def run_seed(seed=0, steps=160, eval_size=512):
    model = train_model(seed, steps)
    test, mood, topic = make_batch(eval_size, 50_000 + seed, "test")
    baseline, (a0, t0) = evaluate(model, test, mood, topic)
    # hold out *different* samples for constructing the contrast vector.
    corpus, cmood, ctopic = make_batch(512, 70_000 + seed, "train")
    layer = 2
    vector = contrast_direction(model, corpus[cmood == 1], corpus[cmood == 0], layer)
    topic_vec = contrast_direction(model, corpus[ctopic == 1], corpus[ctopic == 0], layer)
    g = torch.Generator().manual_seed(seed + 121)
    rand = torch.randn(vector.shape, generator=g)
    rand -= (rand @ vector) * vector
    rand = F.normalize(rand, dim=0)
    # Fixed concept controls: deterministic orthogonal random direction,
    # and wrong-topic contrast direction.
    cases = {"baseline": {**baseline, "mood_kl": 0., "topic_kl": 0.}}
    for name, d, strength in [
        ("mood_0.5", vector, .5), ("mood_1", vector, 1.),
        ("mood_2", vector, 2.), ("mood_4", vector, 4.),
        ("random_2", rand, 2.), ("topic_2", topic_vec, 2.),
        ("negative_mood_2", vector, -2.)]:
        with steer_residual(model, layer, d, strength):
            metrics, (a, t) = evaluate(model, test, mood, topic)
        cases[name] = {**metrics, "mood_kl": categorical_kl(a0, a),
                       "topic_kl": categorical_kl(t0, t)}
    # A top-correlated neuron is not necessarily a pure causal feature.
    neurons = collect_neurons(model, layer, corpus)
    score = neurons[cmood == 1].mean(0) - neurons[cmood == 0].mean(0)
    neuron_id = int(score.abs().argmax())
    neuron_strength = 5. if score[neuron_id] > 0 else -5.
    with steer_neuron(model, layer, neuron_id, neuron_strength):
        met, (a, t) = evaluate(model, test, mood, topic)
    cases["top_neuron_5"] = {**met, "mood_kl": categorical_kl(a0, a),
                              "topic_kl": categorical_kl(t0, t)}
    # Projection is on the copy of the toy network and is permanent for that copy.
    altered = project_output_weights(model, layer, vector)
    met, (a, t) = evaluate(altered, test, mood, topic)
    cases["projection_edit"] = {**met, "mood_kl": categorical_kl(a0, a),
                                 "topic_kl": categorical_kl(t0, t)}
    return {"seed": seed, "layer": layer, "selected_neuron": neuron_id,
            "conditions": cases}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--eval-size", type=int, default=512)
    p.add_argument("--output", default="results/benchmark.json")
    args = p.parse_args()
    runs = [run_seed(s, args.steps, args.eval_size) for s in range(args.seeds)]
    names = runs[0]["conditions"].keys()
    summary = {}
    for name in names:
        summary[name] = {}
        for metric in runs[0]["conditions"][name]:
            vals = [r["conditions"][name][metric] for r in runs]
            summary[name][metric] = {"mean": mean(vals), "std": stdev(vals) if len(vals)>1 else 0.}
    payload = {"metadata": {"educational": True, "pretrained_llm": False,
              "target_model": "Synthetic supervised residual SwiGLU dual-head classifier",
              "seeds": args.seeds, "training_steps": args.steps, "eval_size": args.eval_size,
              "note": "These are not Qwen or Neuron Explorer measurements."},
              "summary": summary, "runs": runs}
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(payload, indent=2))
    print("condition                 mood_accuracy   topic_accuracy   p(sad)    mood_KL")
    for name, metrics in summary.items():
        print(f"{name:26} {metrics['mood_accuracy']['mean']:.3f}           {metrics['topic_accuracy']['mean']:.3f}            {metrics['sad_probability']['mean']:.3f}     {metrics['mood_kl']['mean']:.3f}")
    print(f"Saved to {out}")

if __name__ == '__main__':
    main()
