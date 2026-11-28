# New original experiment for the Medium article

## Suggested title

**I Reused Two Transformer Layers Three Times. They Nearly Matched Six Independent Layers.**

*In a controlled multi-hop reasoning pilot, sharing weights cut trainable parameters by about 65% at the same block-application count. More inference passes did not automatically make the model better.*

## Opening paragraph (optional replacement)

I trained a six-block transformer with 52,200 parameters. Then I trained another transformer that executed six blocks, but only stored two unique blocks, repeating them three times. It had 18,024 trainable parameters. On the held-out one-to-three-hop pointer-chasing task, the six-block model achieved 59.29% accuracy, while the recurrent model reached 57.16%. That is not a demonstration of frontier-model reasoning: a simple one-hop heuristic already reached 55.86%. But it exposes the distinction between parameter count and executed depth. More interestingly, executing the recurrent model for six passes instead of three reduced its accuracy to 52.34%. Reusing weights is possible; learning to use additional computation effectively is a different problem.

## Insert after Section 17: How should we train a looped transformer?

### A controlled experiment: more passes are not automatically better

The final pilot trained five architectures across three seeds (11, 22, 33), with 360 optimizer updates per run, a batch of 32, and independently generated test examples. Each input contains an eight-node directed functional graph, a starting node, and a hop count. The model predicts the terminal node after following the specified number of links. Training uses one to three hops; testing also covers four to six hops. For each seed and hop count, all architectures see the same 256 test examples. Each model uses 32-dimensional representations and four attention heads.

The two-block dense baseline contains 18,024 trainable parameters. The fixed loop model has exactly the same count, but executes its blocks three times. The six-block dense model has 52,200 parameters and the same six block applications per training example as the fixed recurrent model. This matches block applications, not measured FLOPs. The gated MoE variant has 43,441 trainable parameters and a different per-block computational cost.

Mean held-out accuracy across seeds and hop counts:

- Two-block dense: 57.34% on 1–3 hops, 46.79% on 4–6 hops.
- Six-block dense: 59.29% and 48.78%.
- Fixed loop: 57.16% and 47.74%.
- Variable loop (evaluated at three passes): 55.95% and 46.79%.
- Gated looped MoE (three passes): 58.42% and 50.56%.
- One-hop heuristic: 55.86% and 37.67%; starting-node heuristic: 19.53% and 24.91%. Exact graph traversal gives 100%.

The fixed loop model achieved 53.12%, 56.81%, 57.16%, 56.08%, 54.56%, and 52.34% accuracy over one through six inference passes on the trained hop range. More passes were not monotonically beneficial. The variable-depth model also did not show consistent gains from additional passes. These results should not be interpreted as proof that recurrent architectures cannot reason: the experiment is short, the task is synthetic, and the models are far from perfect. Nor does the small accuracy gap establish equivalence. The original pilot with symmetric edge encodings and untrained categorical hop embeddings is archived separately; the final model uses directed role embeddings and continuous hop counts.

## Figure placements and captions

- `figures/parameters.png`: Section 15, after parameter-count code. Unique trainable parameter counts for the five educational architectures.
- `figures/routing.png`: Section 16, after sparse MoE code. Pooled top-2 selected expert slots across blocks and seeds for three-hop test examples at three passes; not production Inkling routing.
- `figures/hops.png`: Section 17, after the results above. Mean held-out accuracy by number of pointer hops, with the training range ending at three hops.
- `figures/heuristic_comparison.png`: Section 17, immediately after discussing baselines. Learned models compared with simple non-neural heuristics on the same test examples.
- `figures/loop_scaling.png`: Section 17, after discussing variable-depth training. Same trained recurrent weights evaluated at different numbers of passes, averaged over one-to-three-hop tasks and seeds.
- `figures/activations.png`: Section 21, after pass-index discussion. Relative Frobenius-norm change between consecutive passes at six-pass inference on three-hop examples. Activation change alone is not evidence of reasoning or interpretability.

## Reproduction and limitations

The repository includes runnable code, recorded logs, JSON results and tests. Fifteen final CPU training runs and fifteen archived initial pilot runs were executed. Five unit tests passed. These are full-context graph classifiers, not autoregressive LLMs. No Inkling or Qwen weights were downloaded or benchmarked, no GPU/MPS runs were performed, and the Ollama benchmarking script is optional and not live-tested. The experiment does not measure exact FLOPs, real model memory usage, or code-generation correctness. The original article is included unchanged; use this document to add measured results and replace its earlier implementation-status note.
