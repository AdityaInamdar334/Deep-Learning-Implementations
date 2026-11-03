# Distributed Science Agent Lab

**How much scientific evidence do we gain by splitting investigations across independent workers?**

An offline, fully reproducible, MIT-licensed Python laboratory accompanying the Medium tutorial **"1,946 AI Agents Analyzed a Brain Atlas in 32 Minutes. The Hard Part Was Knowing Which Discoveries to Trust."**

This lab reconstructs the **work decomposition, evidence verification, controlled multiple testing, and orchestration** principles behind distributed scientific analysis. It is **not OpenAI's original pipeline**, does not access GPT-Rosalind, and does not reproduce biomedical results from the PerturbAI whole-brain CRISPR study.

## What you can investigate

- **Budgeted generalist:** inspect the top eight signals with a fixed analysis budget.
- **Same-budget specialists:** same eight signals, independent task packaging.
- **Full serial investigators:** inspect 37 features sequentially.
- **Full parallel investigators:** inspect the identical 37 features with threads.
- **Verification ablations:** disable holdout evidence, Benjamini-Hochberg correction, or both.
- **I/O sensitivity:** compare real wall-clock measurements with an artificial 20 ms remote-tool delay and no added delay.
- **Provenance:** every finding includes a feature name, effect direction, sample size, source key, raw discovery p-value, corrected p-value, and held-out p-value.

## Dataset and important limits

We use the public scikit-learn Wine dataset, **classes 0 and 1 only**, representing 130 observations and 13 chemical features. We attach **24 synthetic independent Gaussian null features** to expose false discoveries. This is not a biological dataset and the findings are not scientific novelties. We use Welch tests and an independent stratified train/holdout split.

The `science_lab` workers are **deterministic statistical Python functions**, not language-model agents. Adding threads changes schedule, not model reasoning. Artificial I/O sleep is intentionally used to represent remote tool latency; no real network calls occur. The cost ledger reports **proxy token units and hypothetical dollars**, not billable API usage.

## Quick start

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m science_lab.demo
python -m science_lab.benchmark \
  --seeds 5 \
  --null-features 24 \
  --latency-ms 20 \
  --workers 6 \
  --output results/benchmark.json
python plot_results.py
```

### No-latency control

```bash
python -m science_lab.benchmark \
  --seeds 3 \
  --null-features 24 \
  --latency-ms 0 \
  --workers 6 \
  --output results/no_latency_ablation.json
```

### Outputs

- `results/benchmark.json`: all per-seed work items, provenance, acceptance decisions, timing, and summarized metrics.
- `results/no_latency_ablation.json`: zero artificial I/O ablation.
- `figures/01_coverage.png`: independently replicated claim coverage.
- `figures/02_parallel_wall_clock.png`: serial versus parallel artificial-I/O timing.
- `figures/03_ablations.png`: accepted claims that did not independently replicate.
- `BENCHMARK_NOTES.md`: reproducibility, protocol, biases, and interpretation.
- `ARTICLE.md`: complete Medium-ready tutorial.

## Headline results from the included five-seed experiment

The limited-budget generalist accepted **7.0** independently replicated findings on average. The same-budget specialist case also accepted **7.0**. Full serial or parallel coverage accepted **8.6**. The complete serial run took **0.837 s**, while the six-worker parallel variant took **0.161 s** with a 20 ms per-task artificial delay. With **no artificial delay**, parallelism was slower: **0.071 s**, versus **0.050 s** for serial execution, averaged across three seeds.

Removing independent holdout checks accepted **2.4 extra nonreplicating claims** per five-seed run, on average. Removing both safeguards accepted **3.4 extra nonreplicating claims** and **1.6 known synthetic nulls**, on average.

These are educational, small-sample findings. Statistical association, replication on held-out data, and causal biological discovery are different things.

## Source research

- [Original 2026 distributed biological reasoning paper](https://doi.org/10.64898/2026.10.15.759436)
- [OpenAI research announcement](https://openai.com/index/scaling-ai-research-for-complex-biological-data/)
- [Official released research code and reasoning atlas](https://github.com/jinlabneurogenomics/distributed_agent)
- [Scikit-learn Wine dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- [Benjamini and Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

## Responsible reuse

Use a study-appropriate unit of replication and never assume individual biological cells are independent donor or animal replicates. Do not treat scientific literature summaries or model-generated mechanisms as experimentally confirmed causal explanations. Keep validation data and evidence checks outside the control of any investigator you are evaluating.

**License:** MIT. See `LICENSE`.
