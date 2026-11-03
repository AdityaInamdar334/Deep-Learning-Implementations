# Reproducibility and benchmark notes

## Scope

This is a **local statistical orchestration simulation**, not an evaluation of GPT-Rosalind or real scientific LLM agents. It uses Wine chemistry features and synthetic noise, not cell data. The objective is to isolate changes caused by feature coverage, verification, multiple testing control, and parallel execution.

## Fixed setup

- Five deterministic seeds (`0` through `4`) in the primary experiment.
- Wine classes 0 and 1, 130 total samples, 13 observed chemistry features.
- 24 synthetic Gaussian null features, newly seeded for each run.
- A stratified 70% discovery and 30% holdout split selected before analysis.
- Discovery feature ordering by training-set absolute standardized mean difference.
- Welch's unequal-variance t-test for each investigated feature.
- Benjamini-Hochberg correction across the tested features in that configuration.
- Claim acceptance requires adjusted discovery p below 0.05 and a holdout p below 0.05 with the same effect direction.
- Default six-worker pool for the parallel configurations.
- Artificial `time.sleep(0.02)` per investigated task to model I/O waiting.
- Hypothetical tokens: 380 fixed, 230 per task input, 175 per task output, 250 synthesis, 80 extra per specialist task. The rate of $3 per million proxy tokens is **illustrative**, not provider pricing.

## What the metrics mean

- `accepted_claims` = count passing the pipeline's acceptance decision.
- `independently_replicated_claims` = accepted claims that passed the independent holdout significance and direction check.
- `known_null_claims` = accepted features drawn from random Gaussian columns, which have no engineered relationship to the cultivar labels.
- `claims_without_replication` = accepted claims that fail the independent holdout check.
- `testable_precision` = the share of accepted claims passing the heldout rule, not scientifically validated precision.
- `runtime_seconds` = observed machine wall clock, sensitive to hardware and scheduling.
- `proxy_tokens` and `proxy_dollars` = made-up accounting assumptions for workload comparisons, never API billing.

## Observed primary benchmark (means, five seeds)

- Budgeted generalist, eight features: **7.0** replicated claims, **0.180 s**, **3,870** proxy tokens.
- Same-budget specialists, eight features: **7.0** replicated claims, **0.178 s**, **4,510** proxy tokens.
- Full serial, 37 features: **8.6** replicated claims, **0.837 s**, **18,575** proxy tokens.
- Full parallel, 37 features: **8.6** replicated claims, **0.161 s**, **18,575** proxy tokens.
- No holdout verification: **11.0** accepted, **8.6** replicated, **0.6** synthetic nulls.
- No BH correction: **9.2** accepted and replicated, **0.0** known synthetic nulls.
- Neither safeguard: **12.6** accepted, **9.2** replicated, **1.6** known synthetic nulls.

## No added I/O delay

Three seeds, six workers, same dataset and feature count:

- Full serial: **0.050 s** mean.
- Full parallel: **0.071 s** mean.

In tiny CPU-bound workloads, thread-pool overhead outweighs any overlap. These timings are machine-specific and not comparable to remote GPT-Rosalind inference.

## Interpretation and threats to validity

- Specialization alone does **not** improve the eight-feature comparison. Both configurations run the same statistical test and produce the same findings. The full system's incremental discoveries come from examining more candidate features.
- BH control is computed for the selected test family and is subject to selective-inference caveats; exploratory feature ranking based on discovery data has implications for inference.
- We do not adjust the holdout alpha for all exploratory choices. Do not interpret an independent heldout p-value as definitive replication across independent scientific cohorts.
- The original observed Wine features are not ground-truth causal effects; the synthetic independent nulls are the only known-negative controls by design.
- Synthetic null feature generation is seeded but finite-sample chance associations remain possible.
- Statistical findings from Wine cannot support claims about perturbation biology or new treatments.
- The prototype has no literature retrieval, LLM self-critique, automated hypothesis generation, or MCP tools.
- The benchmark does not model provider concurrency caps, prompt caching, model failures, token billing, or expert review time.
- The `time.sleep` is an intentional pedagogical device. It should never be described as real inference latency.

## Suggested stronger follow-up

On a public perturbation dataset, use animal- or donor-level heldout biological replicates, independently obtained validation measurements, and a domain-appropriate differential-expression model. Benchmark actual LLM token usage and grounded citation support. Report selective discovery efficiency per unit of spend and whether proposed follow-up experiments survive prospective laboratory tests.
