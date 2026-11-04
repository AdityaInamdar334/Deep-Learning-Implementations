# Benchmark Notes: Hierarchical Transformers and Causal Shortening

This document records how to interpret the saved results in `results/benchmark.json`.

## Provenance and scope

The underlying architecture comes from Nawrot et al., *Hierarchical Transformers Are More Efficient Language Models*, Findings of NAACL 2022 (https://aclanthology.org/2022.findings-naacl.117/). The educational baseline was inspired by labml.ai's annotated PyTorch implementation (https://nn.labml.ai/transformers/hour_glass/index.html).

**Our benchmark is independent and synthetic.** It is not the published paper's enwik8 or ImageNet32 benchmark, does not use the original learned attention-based resamplers, and does not train a large language model.

## Experimental protocol

- Dataset: generated sequences of 32 symbols, sampled from a seeded periodic recurrence. The main training data repeats with **period four**. The shifted condition repeats with **period seven**. The data are mathematical sequences, not natural-language text.
- Seeds: 0, 1, and 2, trained independently for each configuration. Training and evaluation use distinct random seeds.
- Optimizer: AdamW, `lr=0.002`, weight decay `0.01`, gradient clipping at norm `1.0`.
- Training: 95 optimizer updates per configuration, batch size 12, full sequence length 48, next-token cross-entropy objective.
- Model: vocabulary 32, width 32, four heads. Standard Transformer has four causal blocks. Single-level Hourglass uses one full-resolution pre block, two short-resolution middle blocks, and one full-resolution post block.
- Validation: 128 sequences per seed and condition. Evaluate familiar period four at length 48, changed period seven at length 48, and longer sequences at length 64 under period four.
- Metrics: full-sequence cross-entropy, perplexity, copy-region next-token accuracy, parameter count, locally measured training time, and causality failure score. Copy-region accuracy excludes the first `period - 1` positions where the repeated rule is not yet directly predictable from history.
- Run file: `results/benchmark.json`, 24 trials, eight configurations, three seeds. The aggregate contains mean and **sample** standard deviation.
- Runtime environment: saved metadata includes Python and PyTorch versions and Torch thread count. Timing reflects this local CPU setup only and should not be transferred to other hardware.

## Mean measured results

Accuracy percentages are rounded from the saved JSON. Higher means better for the three accuracy metrics; zero is required for the causality-difference metric.

| Configuration | Familiar accuracy | Changed-period accuracy | Longer same-period accuracy | Train seconds | Trainable params |
|---|---:|---:|---:|---:|---:|
| `vanilla_4` | 13.29% | 8.88% | 14.28% | 1.195 | 36,320 |
| `hourglass_k2` | 22.11% | 10.04% | 21.33% | 0.820 | 36,320 |
| `hourglass_k3` | 15.60% | 5.16% | 16.33% | 0.778 | 36,320 |
| `hourglass_k4` | 17.85% | 8.38% | 18.12% | 0.793 | 36,320 |
| `hourglass_linear_k3` | 55.84% | 4.39% | 54.63% | 0.981 | 42,592 |
| `hourglass_no_skip` | 23.99% | 4.25% | 24.10% | 0.751 | 36,320 |
| `hourglass_factor_dropout` | 14.95% | 9.05% | 15.19% | 0.852 | 36,320 |
| `hourglass_unshifted_UNSAFE` | 15.58% | 14.55% | 16.34% | 0.978 | 36,320 |

The `UNSAFE` row is a **negative control and does not represent a valid causal model**. Its apparent predictive scores must not be used to rank autoregressive architectures.

The measured causality difference is 0.0000000 at the saved precision for all valid configurations; it averages **0.760** for the deliberately unshifted model. This measures how much a change to input token 5 changes logits at earlier positions on a test sequence. This is evidence of a dependency violation, not an accuracy metric.

## What the measurements establish

**The basic causal invariant works.** All valid model configurations passed the tests that change later inputs and compare earlier logits. The unshifted control demonstrably fails. This is an important software invariant even when no model is fully trained.

**Shorter attention can reduce runtime in this local experiment.** The three average-pool Hourglass configurations finish 95 steps faster than the four-layer full-resolution baseline. This observation is specific to the code, CPU, PyTorch version, and shapes used, and it does not establish proportional GPU speedups. The learned linear version also shortens the middle attention but adds projection work.

**Learned resampling helps a stable narrow task.** With `k=3`, linear shortening plus linear upsampling substantially improves familiar accuracy over averaging with identical factor and training steps. Its extra 6,272 trainable parameters make this a useful *mechanism demonstration*, not an equal-capacity proof.

**The rule shift reverses that apparent advantage.** The learned version retains little of its fitted accuracy when the recurrence period changes, despite preserving accuracy when only the sequence length changes. These observations suggest learning a pattern-specific positional mechanism, but do not establish a universal failure of learned resampling on real-world sequences.

**Removing skip connections and varying factors have mixed outcomes.** The no-skip version fits this familiar toy task surprisingly well relative to average-pool `k=3`, but fails more dramatically under rule shift. Factor dropout is limited to random selection between factors two and three for average pooling in our code and should not be mistaken for the authors' exact experimental setup.

## Major caveats

1. **Not equal capacity:** all standard/average variants have 36,320 parameters; the learned linear version has 42,592. There is no equally sized vanilla control for the learned version.
2. **Not equal compute:** 95 optimizer steps is not the same floating-point budget or wall-clock budget across architectures. Times are measured and reported separately.
3. **Not converged:** learning dynamics after 95 updates may differ strongly from long-run outcomes.
4. **Artificial task:** an easily repeated pattern is not an adequate proxy for natural language or ImageNet32.
5. **Single synthetic shift:** training period four versus evaluation period seven reveals one failure mechanism, not the distribution of all generalization behavior.
6. **No uncertainty guarantee beyond these seeds:** sample standard deviation across three runs is insufficient for precise inferential claims.
7. **No specialized kernels:** PyTorch implementations, CPU threading, and memory layout materially influence measured performance.
8. **Invalid control:** the intentionally unshifted model is not an autoregressive candidate even when its loss happens to be low.
9. **No pretrained models:** the project measures neither the official Hourglass checkpoint nor any frontier language model.

## Suggested rigorous next experiments

- Match parameter count by widening the vanilla baseline and report validation loss at fixed FLOPs and fixed wall-clock budgets.
- Train longer and track the quality-versus-runtime frontier instead of comparing only one endpoint.
- Vary recurrence periods and offset independently, including periods not aligned with the shortening factor.
- Implement causal attention-based shortening and learned attention-based upsampling faithful to the paper, and compare against linear and average pooling.
- Evaluate on held-out byte-level natural language with validation/test separation and explicit bits-per-character measurement.
- Measure memory and throughput with `torch.profiler`, both for teacher-forced evaluation and true incremental generation.

## Reproduction

From `hourglass_transformer_lab/`:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m hourglass_lab.demo
python -m hourglass_lab.benchmark --seeds 3 --steps 95 --batch-size 12 --eval-size 128 --output results/reproduced.json
python plot_results.py
```

`plot_results.py` reads the supplied `results/benchmark.json`. Benchmark runs write after every completed trial to preserve partial results if interrupted. To regenerate figures from a new benchmark, update the plotting script's `DATA` path or carefully replace the reference JSON after making a backup.

## Source attribution

- Piotr Nawrot et al., [Hierarchical Transformers Are More Efficient Language Models](https://aclanthology.org/2022.findings-naacl.117/), Findings of NAACL 2022.
- labml.ai, [Hourglass annotated PyTorch implementation](https://nn.labml.ai/transformers/hour_glass/index.html).
- Google Trax, [original Hourglass code](https://github.com/google/trax/blob/master/trax/models/research/hourglass.py).

All lab measurements should be labeled **independent educational synthetic experiment**, not paper reproduction.
