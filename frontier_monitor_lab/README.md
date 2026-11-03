# Frontier Agent Monitoring Lab

**A reproducible PyTorch teaching lab for layered safety monitoring of AI agents.**

This project is inspired by the public architecture described in OpenAI's October 29, 2026 research article [How we detect frontier model misalignment](https://alignment.openai.com/how-we-detect-frontier-model-misalignment/) and METR's [independent July 2026 red-team assessment](https://metr.org/blog/2026-10-22-red-teaming-openai-agent-monitoring/).

> **Research integrity:** This is an independent, synthetic, educational experiment. It does **not** run OpenAI's monitor, use OpenAI model activations, reproduce OpenAI's internal results, or show real-world safety effectiveness. PyTorch classifiers here are deliberately tiny stand-ins. The investigator is policy logic, not a frontier language-model agent. Costs are fictional relative units, not dollars or GPU measurements.

## Quick start

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m monitor_lab.experiment --seeds 3 --output results/benchmark.json
```

Run a faster smoke test:

```bash
python -m monitor_lab.experiment --seeds 1 --train 180 --val 90 --test 120 --epochs 25 --output results/smoke.json
```

Run a two-case demonstration before the benchmark:

```bash
python -m monitor_lab.demo
```

The full command runs three independent training seeds with 700 training, 300 validation, 500 held-out standard and 500 held-out shifted examples per seed. The joint shift uses a separate 500-case split. It saves a machine-readable JSON report.

## What it implements

- **L0:** A 10-dimensional `torch.nn.Linear` classifier on artificially generated activation-like feature vectors. The features are label-correlated *by construction* and are not extracted from a language model.
- **Audit:** Deterministic uniform sampling of 5 percent of probe-negative traffic using a SHA-256 hash of the seed and case ID.
- **L1:** A PyTorch bag-of-tokens neural reviewer that classifies synthetic narrative plus tool and task context. A separate text-only reviewer supports an ablation.
- **L2:** A deterministic permission-based investigator comparing observed tool actions with the *recorded* task allowance. It is not a language-model investigative agent.
- **L3:** An out-of-band telemetry comparison that can detect a simulated network action hidden from the agent transcript.
- **Escalation:** Two decisions, security review queue and urgent alert, measured independently.
- **Ablations:** No audits, all traffic to L1, every case to L2 and L3, no independent telemetry, omitted compaction checks, and a text-only reviewer.
- **Shift tests:** Alternative language in narratives, plus a joint test with a weaker label-correlated activation proxy.
- **False alerts:** Some benign examples have approvals missing from the recorded policy feed. A full investigation of every case will therefore flag legitimate actions.

## Reproducing the reported results

```bash
python -m unittest discover -s tests -v
python -m monitor_lab.experiment --seeds 3 --train 700 --val 300 --test 500 --epochs 65 --output results/benchmark.json
```

Measured in the included synthetic test fixture, mean across three seeds:

- Familiar test set, layered + 5% audit: L2 recall ~90.3%, urgent recall ~91.7%, 65.8 fictional cost units per case.
- Familiar test set, every case at L1: L2 recall 100%, 79.7 units per case.
- Familiar test set, no telemetry: L2 recall ~80.6%, 51.8 units per case.
- Familiar test set, skip compaction: L2 recall ~77.6%, 56.6 units per case.
- Wording shift, layered: L2 recall ~34.6%, urgent recall ~30.7%.
- Wording shift, text-only L1: L2 recall ~7.9%.
- Familiar test set, investigate every case: L2 recall 100%, 155.0 units, L2 false positive rate ~5.9%.

**The high familiar-set scores are largely a property of our synthetic generator.** In particular, the dataset encodes simple policy rules that an investigator can recover exactly. The held-out wording results reveal fragility instead of suggesting deployment readiness.

## Evaluation protocol

1. Train and validation data use separate deterministic RNG streams and disjoint IDs.
2. Train both models on training data only, and set thresholds on validation data targeting 90 percent positive recall at each classifier separately.
3. Freeze thresholds before computing held-out and shifted test results.
4. Use the same generated cases and model scores across ablations within each seed. This is a paired comparison.
5. Calculate end-to-end incident recall, urgent-alert recall, false alarm rate, review fraction and fictional unit cost. Print mean and sample standard deviation across three seeds.
6. Do not substitute these synthetic metrics for estimates of real deployed agent behavior.

## Code layout

```text
frontier_monitor_lab/
  monitor_lab/
    data.py          # Synthetic incidents and proxy features
    models.py        # Small PyTorch classifiers
    pipeline.py      # Layered routing and independent checking
    metrics.py       # Recall, false positives, cost
    experiment.py    # Training, validation, ablations and CLI
    demo.py          # Why external telemetry changes a decision
  tests/
    test_monitor.py  # Reproducibility and correctness tests
  results/
    benchmark.json   # Full precomputed three-seed benchmark
  ARTICLE.md         # Long-form companion tutorial
  README.md
  requirements.txt
```

## Safety and scope

There are no real attack payloads, exploit scripts, API keys, internet connections, sandbox escape attempts, or offensive capabilities. Every action and telemetry record is constructed in local memory. This lab should not be used as a production security policy engine.

For a realistic study, use independently annotated real traces under appropriate access and privacy controls, hold out entire incident families, verify the agent context matches the reviewer context, test the alert delivery path, and measure incident response time. Do not tune on the test incidents.

## Sources

- [OpenAI: How we detect frontier model misalignment](https://alignment.openai.com/how-we-detect-frontier-model-misalignment/), October 29, 2026.
- [METR: Red-Teaming OpenAI's Internal Agent Monitoring Systems](https://metr.org/blog/2026-10-22-red-teaming-openai-agent-monitoring/), October 22, 2026.
- [OpenAI: The Hugging Face incident and the road ahead](https://openai.com/index/hugging-face-incident-and-the-road-ahead/), August 26, 2026.
- [OpenAI: Beyond g-mean²: complementary metrics for monitorability](https://alignment.openai.com/beyond-gmean-squared), October 20, 2026.

## License

The educational code in this archive is released under the MIT License. See `LICENSE`.
