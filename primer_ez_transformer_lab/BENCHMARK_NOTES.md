# Benchmark Notes: Primer EZ, Ablations, and the Period-Shift Failure

These notes are part of an independently constructed educational experiment. They do not constitute a reproduction of the Google Research NeurIPS 2021 results.

## Research questions

1. Does Squared ReLU improve a tiny causal LM trained on a periodic-copy prediction task?
2. Does causal depth-wise convolution applied to Q, K, V improve training on the same task?
3. Can a model with substantially lower in-distribution NLL become worse under a structural distribution shift?
4. Do query-only, key-only, and value-only convolution have different robustness patterns?
5. Is quality improvement merely due to more wall-clock compute or more parameters?

## Dataset construction

All data is synthetic and generated independently in each run using dedicated `torch.Generator` objects. Each sequence begins with `period` randomly drawn vocabulary IDs. Later positions copy the ID from `period` positions ago, except for a 2.5% random-substitution probability.

- Vocabulary: 16 symbols.
- Training and in-distribution evaluation period: 4.
- Shifted evaluation period: 6.
- Sequence returned: 25 tokens, split into 24 conditioning inputs and 24 next-token targets.
- Length of model context: 24.
- Evaluation excludes the prefix positions where the motif antecedent cannot yet be available.
- Training includes all token positions. This deliberate train/evaluation difference is identical for all configurations.
- Evaluation uses a fresh PRNG seed of `90_000 + run_seed`; training uses `10_000 + run_seed`.
- The evaluation sequence samples are identical across architectures at a fixed seed.

No training documents, user records, or external corpora are involved.

## Architecture and training

- Two-layer, four-head causal Transformer.
- Hidden size 48; FFN size 144.
- Token output projection tied to the token embedding matrix.
- Learned position embeddings.
- AdamW: learning rate 0.002 and weight decay 0.01.
- Global gradient clipping 1.0.
- Batch 32, 260 steps, 199,680 training tokens per run.
- Three seeds: 0, 1, 2.
- Held-out evaluation: four batches of 64 per distribution.
- Explicit PyTorch matmul attention; no fused SDPA kernels or GPU utilization.
- Convolutions: width three, causal left padding, grouped by feature channels.
- Shared-head default follows the tensor arrangement demonstrated by labml.ai.
- The `primer_per_head` variant uses distinct kernels for every head.

## Metrics

`id_nll` and `ood_nll`: mean cross-entropy over eligible next-token positions, in natural-log units. Lower is better.

`id_accuracy` and `ood_accuracy`: top-1 next-token accuracy over eligible positions, as fractions from 0 to 1. Higher is better.

`time_seconds`: observed wall-clock training duration for the run, **excluding** evaluation, plots, startup, and installation.

`tokens_per_second`: the nominal number of training input-token positions processed per second during the measured loop. Values are CPU/environment-specific.

`parameters`: unique trainable PyTorch parameter elements. Tied embeddings are counted once.

`sample_std`: sample standard deviation across the three seeds, not a confidence interval.

These values are saved in `results/benchmark.json` along with individual trials, loss snapshots, and experiment metadata.

## Three-seed means

The complete report is the canonical source. The following are compact readings, not a new experimental run:

- ReLU baseline: NLL 1.615, shifted NLL 2.143, accuracy 34.25%, shifted accuracy 21.88%, 49,248 parameters, 2.08s mean training time.
- Squared ReLU only: NLL 1.598, shifted NLL 2.112, accuracy 34.78%, shifted accuracy 21.93%, 49,248 parameters, 2.12s.
- GELU baseline: NLL 1.593, shifted NLL 2.137, accuracy 35.37%, shifted accuracy 22.61%, 49,248 parameters, 2.08s.
- QKV convolution with ReLU: NLL 0.258, shifted NLL 6.049, accuracy 95.69%, shifted accuracy 7.50%, 49,536 parameters, 3.14s.
- Primer EZ: NLL 0.251, shifted NLL 5.883, accuracy 95.74%, shifted accuracy 7.45%, 49,536 parameters, 3.09s.
- Primer EZ, per-head kernels: NLL 0.260, shifted NLL 4.351, accuracy 95.78%, shifted accuracy 13.65%, 50,400 parameters, 2.90s.
- Query-only convolution: NLL 0.553, shifted NLL 2.609, accuracy 88.95%, shifted accuracy 31.08%, 49,344 parameters, 2.66s.
- Key-only convolution: NLL 0.582, shifted NLL 0.762, accuracy 83.52%, shifted accuracy 77.17%, 49,344 parameters, 2.67s.
- Value-only convolution: NLL 0.878, shifted NLL 2.299, accuracy 74.88%, shifted accuracy 34.29%, 49,344 parameters, 2.58s.

Additional baseline-only control, 390 steps over three seeds: mean familiar NLL 1.497, shifted NLL 2.072, familiar accuracy 35.87%, shifted accuracy 23.15%, mean training time 3.30s. This is only approximately wall-clock matched and does not equate theoretical FLOPs.

The striking key-only result should be investigated as an architectural hypothesis, not marketed as a result generalizing to natural language or modern models. This is a single hand-designed data generator with a specific train/test mismatch.

## Methodological safeguards

The code implements checks for:

- No future-token influence in the convolution branch.
- No future-token influence in attention with and without MDHA.
- No future-token influence in the complete models.
- Channel independence for depth-wise groups.
- Reproducible pseudorandom data generation.
- Cross-configuration shared train/evaluation seeds.
- Separate random streams for training and evaluation.
- Consistent training schedules, parameters apart from the actual ablation, and loss definition.
- Finite gradients and sensible forward tensor shapes.

No results were selected based on the best individual seed. The three seeds are fixed in advance.

## Limitations and confounders

- No natural-language benchmark or tokenizer is used.
- No actual 500M or 1.9B parameter networks were trained.
- No original evolutionary architecture search was performed.
- Q-only, K-only, and V-only are our diagnostic ablations, not production recommendations from the paper.
- Exactly one structured distribution shift is tested, from period four to period six.
- Each configuration uses an equal step and batch budget, but not an equal parameter count or exact FLOP budget.
- Per-head convolution has additional parameters and a different inductive bias.
- CPU time is not directly comparable with the TPU compute measurements in the original paper.
- The models use learned absolute positional embeddings, a fixed context, and an explicitly coded attention implementation.
- Synthetic results should not be used to claim real-world robustness or safety.

## Reproducing

```bash
python -m unittest discover -s tests -v
python -m primer_lab.demo
python -m primer_lab.benchmark \
  --seeds 3 --steps 260 --eval-batches 4 \
  --configs baseline_relu squared_relu_only dconv_only primer_ez \
            gelu_baseline primer_per_head q_only k_only v_only \
  --output results/reproduced.json
```

For short process limits, use `--start-seed 0`, `--start-seed 1`, and `--start-seed 2` with `--seeds 1` and merge trial records with `primer_lab.benchmark.summarize`.

Reproduction of numeric accuracy/NLL should be close under the same PyTorch version. Exact wall-clock times and some float-level differences may vary by hardware, BLAS kernels, and platform.

## Primary research

So, D. R. et al. *Primer: Searching for Efficient Transformers for Language Modeling*. NeurIPS 2021. https://arxiv.org/abs/2109.08668

labml.ai. *Annotated Primer EZ PyTorch implementation*. https://nn.labml.ai/transformers/primer_ez/index.html
