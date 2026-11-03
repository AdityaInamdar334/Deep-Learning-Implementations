# Benchmark Notes: What the Working-Memory Lab Actually Demonstrates

## Protocol

- 5 pseudorandom seeds; each seed independently trains a context-only PyTorch MLP for 180 Adam steps.
- Every seed evaluates 100 newly sampled tasks, each with eight distinguishable contexts and four actions, with balanced yet randomly permuted correct actions.
- Each task generates 64 random query contexts. Targets are computed from its private mapping.
- Memory configuration has either 2, 4 or 8 fixed addressable slots, and is reset between tasks.
- At most 4, 16 or 32 support context-action pairs are sampled from the **same task** and written before querying.
- A prediction for an uncovered address defaults to the pretrained baseline. The model is not retrained during memory adaptation.
- Corruption: replace about 25% of support labels with deliberately incorrect labels; exact realized count varies across tasks.
- Shuffle: permute labels independently from support contexts to break correct associations.
- Report is unweighted mean accuracy across seeds and sample standard deviation across the five seed-level accuracies.

## Reference run

From `results/benchmark.json`:

- no memory 24.82%, sample SD 1.01 percentage points
- eight slots, support 16 91.74%, SD 0.65 pp
- eight slots, support 4 55.52%, SD 1.27 pp
- eight slots, support 32 99.12%, SD 0.50 pp
- four slots, support 16 57.32%, SD 1.39 pp
- two slots, support 16 37.87%, SD 1.16 pp
- eight slots, support 16, corruption 25% 73.55%, SD 0.85 pp
- eight slots, support 16, shuffled labels 27.07%, SD 2.27 pp

These ablations are independently constructed and should not be represented as Kang et al.'s Atari results.

## Why the task is deliberately easy

In this benchmark, context identifiers are exact categorical values. The memory keys are **preprogrammed prototypes**, so the hard representational problem of discovering useful latent addresses has been bypassed. An eight-slot memory has enough capacity to retain a value for every context. Given sufficient correct demonstrations for every context, accuracy necessarily approaches 100%. This is by construction, not an autonomous discovery of a transferable policy.

The baseline is not given the task identity or demonstrations and therefore cannot infer the new task's randomly assigned action mapping. The experiment controls the availability of relevant context. It is **not** a compute-matched comparison of two equally informed policies. That distinction is the point of the experiment: explicit task-local working memory changes what information is available at decision time.

With too few slots, contexts share memory locations and overwrite each other's action evidence. With corrupted labels, the memory reliably stores wrong information. With shuffled labels, content-addressed retrieval does not help because associations are broken. This shows that memory fidelity is essential.

## Source distinction

Kang et al. (ICML 2024) trained DT-Mem on Atari and Meta-World data, used learned query/key/value transformations, an erase/add memory update, and LoRA on memory projections. Their reported results should never be conflated with this lab.

Our tiny Transformer demonstrates causal return-state-action tokenization but is not used in the benchmark's adaptation loop. Our memory is a distinct, smaller teaching mechanism with fixed keys and no jointly learned memory policy. The LoRA demo only validates parameter-efficient adaptation of one frozen projection.

## Recommended extensions

- Replace fixed keys with trainable projections and verify generalization to never-seen observations.
- Add identically informed baselines, such as kNN and a model with in-context support examples.
- Fit an actual Transformer policy and train the memory jointly.
- Compare expected return through rollouts, not action accuracy alone.
- Evaluate noisy continuous observations, unequal action frequencies, nonstationary task mappings and interference across tasks.
- Report parameter counts, data access, compute, confidence intervals and tuning effort for every method.
