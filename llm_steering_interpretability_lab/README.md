# Activation Steering and Neuron Interpretability Lab

A small independent PyTorch experiment accompanying the article *A Single Neuron Made Qwen Talk About Tomatoes. Here Is What That Actually Tells Us About Language Models.*

**This is not Qwen, an LLM, the original Neuron Explorer, or a reproduction of any published model benchmark.** The default experiment trains a 4-block residual SwiGLU network on two artificial tasks: predicting a mood word and a topic word from a short sequence. It is designed to make the intervention mechanisms inspectable and testable on CPU.

## Features

- Mean-difference feature vectors, normalized to unit length.
- Reversible PyTorch hooks at residual blocks and gated MLP intermediate features.
- A fixed-dose sweep with matched random and off-target topic controls.
- A projection edit applied to a copied model's MLP output matrix, and its bias.
- Two independent classifier heads, measuring target shift and off-target interference.
- KL divergence against the untouched model.
- Separate training/held-out sets with disjoint nuisance token ranges.
- Reproducible five-seed benchmark and 11 tests.
- Optional, separate script for the real Qwen3-0.6B model (not run by the benchmark).

## Setup

Python 3.10 or newer recommended:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 python -m steering_lab.benchmark --seeds 5 --steps 160 --eval-size 512 --output results/benchmark.json
```

To create the two figures:

```bash
python -m pip install matplotlib
python plot_results.py
```

To try the optional real-model example (downloads model weights, can be slow and require substantial RAM):

```bash
python -m pip install -U transformers accelerate safetensors
python optional_qwen_demo.py --layer 14 --strength 1.0
```

**Important:** The optional model example does not measure a causal effect robustly by itself. Its four paired prompts are only a minimal demonstration. Add independent evaluation prompts, controls, logit/KL checks, and holdout splits before drawing conclusions. The example targets Qwen3-0.6B, not Qwen3.5-4B, and has not been exercised in our CPU-only environment because the model checkpoint and `transformers` library were unavailable. Package internals and model APIs may change with `transformers` versions.

## Interpreting the numbers

- `sad_probability`: mean predicted probability of the artificial gloomy class, **not generated text sentiment**.
- `mood_accuracy` and `topic_accuracy`: accuracy on independent held-out synthetic labels.
- `mood_kl` / `topic_kl`: categorical KL of baseline head probabilities versus modified model probabilities.
- `projection_edit`: changes only one output projection in a copied toy model; not a global removal of a semantic feature.
- `top_neuron_5`: selects a neuron by its held-out *training-corpus* activation contrast and intervenes on that one neuron. This does not prove it is monosemantic.

For reproducible numeric comparisons, use the generated `results/benchmark.json` and keep the training settings unchanged.

## Scope and research ethics

We do not implement refusal removal or uncensoring. The text explains the mechanism at a high level because it is relevant to mechanistic interpretability and model safety. All executable edits operate on benign artificial mood/topic features. Do not use steering as a substitute for access controls or safety evaluations.

## Sources

- Turner et al., *Steering Language Models With Activation Engineering*: https://arxiv.org/abs/2308.10248
- Panickssery et al., *Steering Llama 2 via Contrastive Activation Addition*: https://arxiv.org/abs/2312.06681
- Lyra, *neuron-explorer*: https://github.com/lyramakesmusic/neuron-explorer
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen3.5-4B-Base: https://huggingface.co/Qwen/Qwen3.5-4B-Base
