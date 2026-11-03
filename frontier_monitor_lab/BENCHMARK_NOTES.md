# Benchmark notes and limitations

This lab is a synthetic instructional harness. Its numbers are not evidence of OpenAI's internal monitor efficacy or safety of any deployed agent.

## Data construction

Ten synthetic scenario families share structure across the training, validation and familiar test sets. Different seeds prevent identical generated records, but templated events remain easy to learn. A neural reviewer sees structured tool and allowed-action fields. The test source contains fabricated feature vectors correlated with incident labels by construction.

Wording shift uses a separate generated sample of scenario families and alternative text. It does not hold each individual case fixed while changing only text. However, the underlying proxy activation distribution is unchanged. Joint shift additionally reduces signal in the numerical proxy features. For rigorous causal linguistic studies, generate exact matched pairs with the same tool events, labels and feature vectors.

Some legitimate cases have incomplete recorded approval metadata. That makes the rule investigator produce false alerts in the full-investigation setting. Labels represent our generator's intended simulated authorization, which is not always completely visible to the monitor.

## Model training and calibration

- Train uses 700 items per seed.
- Validation uses 300 items per seed.
- Familiar, wording shift and joint shift each use 500 independently generated items per seed.
- Separate models and calibration thresholds are fitted each seed.
- The validation threshold targets 90 percent positive recall for L0 and L1 separately.
- Thresholds are frozen when reporting test and shifted performance.
- Every ablation in a seed shares model scores and cases so comparisons are paired.

## Metric definitions

- L1 recall = positive-case fraction that *reaches* the transcript reviewer. This is a stage-routing recall and does not measure L1's own classification accuracy.
- L2 recall = incident fraction reaching the security queue after any required screening and rule/telemetry verification.
- P0 recall = urgent-labeled incident fraction receiving urgent alert.
- False positive rate = benign fraction receiving the corresponding positive decision.
- Precision = flagged positives divided by all flagged cases; returned as `null` when the denominator is zero.
- Coverage = fraction not excluded by the `skip_compaction` ablation.
- Cost = assumed costs L0=1, L1=20, L2=100, L3=35 relative units, averaged over cases. This is not latency, actual GPU time, or money.

Mean and sample standard deviation use three independent model-training seeds. They are not confidence intervals. The scenario templates are also not representative of the prevalence of real incidents.

## Methodological threats

- Generated labels strongly align with simple tool policy rules.
- No true frontier LLM, live agent, sandbox, real external network, or private activation is used.
- Policy checking depends on access to a correct authorization source.
- L2 is a strict rule, not an actual investigative AI agent.
- L3 assumes an honest and complete out-of-band telemetry record. This is rarely guaranteed in practice.
- The current generator includes only a small number of benign hard negatives and cannot provide a reliable estimate of real false-positive burden.
- Fictional cost units underweight human on-call costs, queueing, backpressure and missed alerts.
- This lab cannot support conclusions about intention, deceptive alignment or monitorability of GPT-6 Astra.

## How to improve it

Introduce human-adjudicated redacted real-world traces with appropriate consent and permissions. Hold out incident families. Use exact paired narrative rewrites. Add policy-version drift and telemetry loss. Capture every dispatch and ingestion event in independent append-only logs. Evaluate false-positive cost, tail latency and containment-before-effect rates. Repeat evaluation after monitor and monitored model updates.
