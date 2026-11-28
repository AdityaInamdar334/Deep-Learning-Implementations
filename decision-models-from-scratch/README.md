# Decision Models from Scratch

Build a typed decision engine for a customer-support agent, inspired by Jev, Laya and MAYA. This is an educational implementation, not a reproduction of proprietary Jev internals or the published MAYA checkpoint.

## Quick start (CPU, no downloads of model weights)

Python 3.11 or 3.12 recommended. Install the platform-appropriate PyTorch wheel if the standard installation does not work.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py
```

The command runs three seeded permutation experiments (nine neural-network fits), generates support data, trains TF-IDF baselines, evaluates a dry-run agent, simulates token costs and runs seven unit tests. JSON, logs and two figures are saved under results/ and figures/. The included results were generated on CPU; the optional Qwen path was not run.

## Learning path

1. `core.py`: shared candidate scorer, cross-entropy + Brier loss, temperature scaling and confidence gates.
2. `offline.py`: 6,000 synthetic states per seed, four candidate vectors, deliberately biased canonical ordering. Train on 4,000, calibrate on 1,000, test on 1,000. Compare position-specific scoring, shuffled training and shared scoring. Evaluate all 24 candidate permutations, mapping predictions back to stable candidate identities.
3. `support.py`: generate 768 train cases, 192 calibration cases and 192 test cases, each with three questions. Hold out whole wording templates and case IDs. Intent uses message only; action receives gold intent and trusted backend state; escalation receives a synthetic urgency flag. These last two are conditional diagnostics, not end-to-end metrics. The deterministic policy is a toy gate, not a complete business policy.
4. `agent.py`: infer intent without gold labels, then choose a deterministic action and validate it. Uncertain cases defer; no real LLM fallback or transaction is executed. The test exposes a coverage collapse under unseen phrasing.
5. `cost.py`: analytical frontier/local token budget. Example `python cost.py --coverage .72 --local-input 300`. This is not actual billing, latency or throughput.
6. `qwen.py`: optional direct-logit inference and LoRA training, stable IDs, randomized candidate order, verified hard labels and optional teacher distributions, calibration, option-mass gating, adapter persistence.
7. `tests/test_lab.py`: equivariance, gradient, split integrity, policy, soft-target mapping, abstention and token arithmetic tests.

## Train a custom Qwen model (optional)

```bash
pip install -r requirements-qwen.txt
python support.py
python qwen.py evaluate --model Qwen/Qwen3-0.6B --out runs/zero-shot --batch-size 1
python qwen.py train --model Qwen/Qwen3-0.6B --out runs/support-qwen --epochs 2 --batch-size 1
python qwen.py evaluate --model Qwen/Qwen3-0.6B --adapter runs/support-qwen --out runs/reloaded --batch-size 1
python qwen.py predict --model Qwen/Qwen3-0.6B --adapter runs/support-qwen --request sample_request.json --calibration runs/reloaded/evaluation.json
```

Model weights require internet access and disk space. The engine selects CUDA, Apple MPS or CPU; override with `--device cpu`, `--device mps` or `--device cuda`. CUDA uses BF16 when supported, other devices default to FP32. Batch size 1 is a conservative starting point, not a memory guarantee. Model weights alone require roughly 2.4 GB for nominal 0.6B FP32 parameters; training adds activations and optimizer state. Try 1.7B or 4B only if hardware permits. The Qwen implementation is syntax-checked, not runtime-verified; dependency/API and tokenization compatibility must be tested after installation. No Qwen results are claimed.

The engine reads the final vocabulary logits with no autoregressive decoding. It verifies distinct single-token A–P markers, uses left padding, disables thinking in the Qwen3 chat template, and rejects contexts over 2,048 tokens rather than silently truncating options. Check the actual prompt-boundary tokenization and compare alternative whitespace markers on calibration data before relying on the readout. It normalizes over valid options and tracks their original vocabulary probability mass. Training uses CE + 0.1 Brier plus a small option-mass penalty. A teacher distribution is used only if valid and its top answer agrees with the gold label. The gate is fitted on calibration examples with option mass >= 0.5 and at least 30 accepted examples at 98% empirical accuracy; it defers all if no threshold qualifies. These are educational defaults, not safety guarantees. The evaluate command records batch timings (including tokenization and forward pass, without warmup) and input token counts; no production latency claim is intended. Repeated evaluation of the same test set is for demonstration; reserve a fresh final test for model selection in real research.

## Your own dataset

Replace datasets/train.jsonl, calibration.jsonl and test.jsonl after running the generator. Do not rerun support.py or run_all.py afterwards unless you want to overwrite them. Each line follows this schema:

```json
{"case_id":"episode-001","state":{"message":"I was charged twice"},"question":"Which team should help?","options":[{"id":"billing","text":"Billing"},{"id":"returns","text":"Returns"}],"answer":"billing","teacher_probs":{"billing":0.9,"returns":0.1}}
```

`teacher_probs` is optional. Stable IDs must be unique and the gold answer must match one. Use 2–16 candidate options. For a Boolean decision use yes/no; for an ordinal decision use ordered semantic labels and calculate an expected score if meaningful. Keep whole customer episodes together in one split. Include failure and recovery states, ambiguity, policy changes, new phrasing and explicit deferral. Use only authorized, de-identified data. Never use customer claims as trusted authentication facts. No model may directly authorize a payment or account mutation.

## Executed results (new independent experiment)

Three seeds (7, 19, 42), mean over held-out states:

| Scorer | Canonical accuracy | All-permutation accuracy | Any-answer flip rate |
|---|---:|---:|---:|
| Position-biased | 93.30% | 74.61% | 53.27% |
| Shuffled training | 82.43% | 82.38% | 2.47% |
| Shared position-free | 82.67% | 82.67% | 0.00% |

The biased model looks stronger under the biased canonical evaluation, but is less robust. Shuffling helps but does not guarantee invariance. Shared independent scoring is permutation equivariant (subject to ties and numerical effects). These new measurements are not the previous article's original 23.1% / 0.4% / 0% experiment. Do not substitute them silently. The included `results/offline.json` contains per-seed calibration metrics, coverage and accepted accuracy.

Support baseline: intent 50.00%, conditional next action 63.02%, conditional escalation 87.50%. The calibrated dry-run agent deferred all 192 test cases. This demonstrates that a confidence gate can protect against uncertain inputs while eliminating potential cost savings. It does not establish a useful production agent. Inspect macro F1, especially for imbalanced escalation labels. For realistic evaluation, add human-verified trajectories, a real fallback, and an independent task-level evaluation.

## Figure placement (new experiment only)

- `figures/permutation.png`: Section 8, after describing the three scoring models. Caption: Any-answer flip rate across all 24 option orders, averaged over three seeds in a new synthetic experiment.
- `figures/accuracy.png`: Section 8, after the canonical-versus-permutation accuracy discussion. Caption: Canonical and permutation-average accuracy across three seeds. The canonical distribution intentionally contains positional shortcuts.

## Research reading

- Guo et al. (2017), On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
- Hinton et al. (2015), Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531
- Hu et al. (2022), LoRA: https://arxiv.org/abs/2106.09685
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Laya public implementation: https://github.com/NandhaKishorM/laya
- MAYA a1 and JevAgentHarness: see the companion article bibliography for the source claims and distinctions between published and local results.
