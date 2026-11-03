# Benchmark methodology and limitations

## Research question

Does a council of specialists make better decisions than a single generalist in a synthetic environment with known true utilities? Which benefits survive shared informational errors? Does conversation-driven agreement help in the absence of new evidence? Can a separately applied risk cap stop unconditional approval without altering option rankings?

## Workload

Synthetic three-option decisions with seven score dimensions. Each dimension is a 0-10 quantity. Exactly two dimensions have weight three, two have weight two, and the remaining have weight one. Option-level shared attractiveness is sampled, followed by dimension-level noise. Independent binary catastrophic-risk labels occur with 13% probability per option. Shared framing errors are sampled independently of the true option values.

The experiment uses five deterministic PRNG seeds (0,1,2,3,4) with 600 cases per seed. Those 3,000 cases are the same across policies. Output includes all trial metrics plus means and sample standard deviations.

## Measurement

The primary accuracy metric is argmax of the true relevance-weighted utility across options versus the simulated council's argmax. Regret is the difference in true utility between oracle and selected option. These oracle values are hidden from the seat-card interface, although the simulation generator necessarily uses ground truth to construct noisy observations.

False unconditional approval is reported as the fraction of cases where the chosen option has a hidden ruin flag and the verdict is GO. This metric is intentionally narrow. It does not measure the quality of mitigations or prevent GO WITH CONDITIONS being given to some risky cases.

## Policy definitions

- Solo: the truth-level aggregate utility is corrupted by large independent Gaussian noise, which models an imperfect generalist. The generalist does not receive each clean feature independently.
- Quick: the two highest-weight dimensions, a downside lens, and a challenger lens are chosen using a deterministic simplified heuristic.
- Independent seven: each dimension receives individual Gaussian error and a weak shared component.
- Correlated seven: the shared component is much stronger, and the rest of the process is unchanged.
- Debate: starts from exactly the same cards as correlated seven; shifts every score 52% toward the unweighted mean across seats. Its synthetic cost reflects a three-round council. This is a conformity proxy, not real debate.
- Audited: measurement noise is divided by 2.5 and risk detection improves. This is a *constructed better-evidence condition*, not a discovered property of real agents.
- Audited plus cap: exactly the same cards as the audited condition; any reported likely ruin risk on the winning option caps the positive verdict at RESHAPE.

## Numerical results

| Policy | Accuracy | Mean regret | Unconditional GO on risky option | Mean spread | Modeled calls | Fictional proxy tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Solo | 54.0% | 0.611 | 7.70% | 0.000 | 1 | 915 |
| Quick four | 73.6% | 0.196 | 4.47% | 4.752 | 4 | 3,000 |
| Seven independent | 81.7% | 0.103 | 2.83% | 6.142 | 7 | 5,085 |
| Seven correlated | 60.2% | 0.444 | 5.80% | 5.648 | 7 | 5,085 |
| Debate no evidence | 59.7% | 0.453 | 5.43% | 2.704 | 21 | 14,185 |
| Audited specialists | 86.6% | 0.045 | 2.90% | 5.494 | 14 | 9,635 |
| Audited with cap | 86.6% | 0.045 | 0.00% | 5.494 | 14 | 9,635 |

All figures rounded. Full-precision values and sample std are in `results/benchmark.json`. With only five seeds, confidence intervals for generalization to new scenario generators are not established.

## Interpretation boundaries

1. The original Roast Room's weighted average and ruin-cap ideas are represented, but the Python code is neither a fork of its application code nor a live integration with Claude Code.
2. The numeric personas do not reason or debate using language.
3. Correlated noise can dramatically alter results; the magnitude is a construction choice.
4. The audit improves information quality by definition; it does not estimate the real cost or reliability of human verification.
5. Proxy tokens use a fictional linear accounting rule (220 + 650 per modeled call + 45 per seat). It is not an estimate of actual prompt length, caching, or billing.
6. A decision that maximizes a synthetic 0-10 utility function need not be safe or desirable in the real world. Risk is evaluated separately.
7. Negative findings for simulated conformity should not be represented as negative findings about genuine multi-agent debates or Roast Room itself.

## Reproduce

```bash
python -m unittest discover -s tests -v
python -m decision_lab.benchmark --seeds 5 --cases 600 --output results/benchmark.json
python plot_results.py --input results/benchmark.json --outdir figures
```
