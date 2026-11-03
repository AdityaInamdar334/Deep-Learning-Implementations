# A Smaller AI With the Right Scientific Tools Beat a Larger Model. The Real Breakthrough Was Knowing When Not to Improvise.

*Inside SpikeLab, the 2026 neuroscience system that made scientific methods part of an agent's operating environment. A complete engineering tutorial on bounded autonomy, spike-time statistics, reproducibility, independent verification, and the price of getting the right answer.*

A researcher asks an AI agent to analyze neural recordings.

The agent locates the data, selects some neurons, computes correlations, fits a population model, and produces several convincing figures. The Python runs. The plots look reasonable. The summary describes the results in the language of neuroscience.

There is just one problem.

The agent may have analyzed a different experiment from the one the researcher intended.

Perhaps it interpreted *no cue* as the period between trials when the researcher meant catch trials. Perhaps it used correlation of binned firing rates when the method required a spike-time tiling coefficient. Perhaps it silently threw away three out of every four time bins to make a latent-state model run faster. Perhaps it discovered a set of neurons through the wrong search endpoint and never realized better recordings existed.

Every one of these mistakes can produce executable code, plausible charts, and a polished explanation.

None requires the AI to hallucinate a paper, invent a dataset, or crash.

This is the failure mode at the heart of **SpikeLab: Agentic Tools for Spike Data Analysis**, a bioRxiv preprint released on April 29, 2026 by Tjitse van der Molen and collaborators at UC Santa Cruz and other institutions. The authors designed a scientific computing library and an agent interface that require domain-specific analyses to use stable, tested methods. They compared it with unassisted frontier models on a multistep electrophysiology task. [1]

The surprising result was not that the largest model failed at coding. The larger model could write working programs, explore unfamiliar APIs, and sometimes diagnose its own mistakes. The failure was subtler: it made methodological decisions without asking permission and without necessarily reporting them. A smaller model equipped with a constrained library behaved more consistently.

The paper suggests an architectural shift for scientific agents: **do not ask the model to reconstruct the scientific method from memory every time it runs. Put the method in versioned software, constrain the agent to call it, and verify what happened independently.**

In this tutorial, we will dissect SpikeLab's design, work through the electrophysiology statistics that make its safeguards necessary, and build a runnable Python project that simulates safe and unsafe analysis workflows. We will measure what a bounded tool interface catches, what statistical verification costs, and why a cleaner report is not necessarily a more correct analysis.

The companion code is educational, uses fully synthetic spike trains, and does not call Claude or reproduce the authors' biological benchmark. That distinction matters just as much as any result we obtain.

## Contents

1. The benchmark that made the larger model look worse
2. What a neural spike train actually records
3. Why scientific mistakes are harder to detect than software exceptions
4. The four tasks in the original SpikeLab study
5. Five composable data abstractions
6. Bounded autonomy as a software architecture
7. What the benchmark measured, including cost
8. A crucial difference between reproducibility and correctness
9. Build the companion laboratory
10. Represent spike trains without losing units or metadata
11. Understand the spike-time tiling coefficient
12. Write a numerically checkable STTC implementation
13. Surrogate tests and false discoveries
14. Population coupling and the self-correlation trap
15. Latent population dynamics and silent downsampling
16. Define an explicit tool contract
17. Require clarification on ambiguous analyses
18. Build a flawed generalist policy as a control
19. Independently evaluate the generated results
20. Run the five-seed benchmark
21. Read the ablations without overclaiming
22. Why a faster workflow can be the wrong workflow
23. What would constitute a fair real-model benchmark
24. Using official SpikeLab and public IBL data
25. Security, governance, and provenance
26. Limitations of the paper and our experiment
27. Research directions
28. Conclusion
29. References

## 1. The benchmark that made the larger model look worse

The original study was not a contest between a weak toy model and a frontier system. It compared three real agent configurations on a four-stage scientific analysis workflow: Claude Opus 4.6 without SpikeLab, Claude Sonnet 4.6 without SpikeLab, and Claude Sonnet 4.6 with SpikeLab's analysis-implementer skill. Opus was run with high reasoning effort, while the Sonnet conditions used medium effort. The researchers ran each configuration three times, creating nine analysis sessions. [1]

The agents were asked to work with publicly available recordings from the International Brain Laboratory, or IBL. The workflow progressed from finding a suitable recording to assessing pairwise neuronal relationships, population coupling, and latent population activity.

The preprint reports that tool-assisted Sonnet produced correct and reproducible workflows across those sessions, while unassisted models showed repeatable methodological problems. The researchers evaluated a 14-issue scorecard rather than relying only on whether code executed.

One particularly revealing result concerned data discovery. Opus identified a recording with **208** good-quality units in auditory cortex. Unassisted Sonnet identified a different session with only **69** such units. SpikeLab-assisted Sonnet found the same 208-unit recording as Opus. The problem was not numerical reasoning. It was an API selection problem. One search endpoint relied on session-level metadata that did not cover the relevant recording adequately; another indexed probe insertions more comprehensively. SpikeLab had a dedicated data-discovery method that avoided the incomplete search strategy. [1, 2]

Now imagine that a model chooses the inferior session, successfully performs every subsequent calculation, and publishes a figure. An output-only evaluator might consider the task completed. The scientific question, however, was to find the best suitable recording, not merely *a* recording that allowed the remaining code to run.

This is an example of **upstream selection error propagation**. A mistake at data discovery alters everything downstream without necessarily creating any computational error.

## 2. What a neural spike train actually records

Extracellular electrophysiology records electrical signals from populations of neurons. Spike sorting converts features of those signals into estimated action-potential events assigned to putative neural units. The resulting object is simple enough to describe: for each recorded unit, we have a sorted list of timestamps indicating when spikes were detected.

A unit might have spike events at 104.2 ms, 137.0 ms, 331.5 ms, and so on. Another unit has its own timestamps over the same recording. A real dataset also needs sampling information, brain region, unit quality metadata, event markers, stimulus identities, and recording conditions.

The simplicity is deceptive. Two scientists can analyze the same spike trains and obtain different results because they use different time windows, bin widths, smoothing kernels, methods for defining synchrony, or assumptions about behavior.

For example, measuring correlation between counts in 100 ms bins asks a different question from measuring whether two neurons fired within 20 ms of each other. The first is about coarse fluctuations in activity. The second captures finer temporal coincidence and includes adjustments for how much time surrounds observed events.

Neither measure should be interpreted casually as direct proof of a synaptic connection. Common inputs, behavior, movement, stimulus timing, and global state changes can all create apparent association.

The value of an agentic tool library is that these scientific decisions become named and documented operations instead of ad hoc code created differently in each session.

## 3. Why scientific mistakes are harder to detect than software exceptions

When ordinary software fails, the program might crash, fail a unit test, or return an invalid type. Scientific software often fails by producing a valid value for the wrong estimand.

Suppose two neurons increase their firing around a sound cue. Their binned rates will be correlated, even if their spikes do not have reliable millisecond-scale coordination. A model that substitutes a rate correlation for a requested spike-time correlation will obtain a legal number. That number may even be visually persuasive.

Similarly, take a 50 ms time-binned population model. If a model silently keeps only every sixth bin to save compute, it has changed the effective temporal resolution to 300 ms. Fast transient states may disappear, short events may blur into their surroundings, and model-selection metrics can change.

The most dangerous error is therefore not an obvious hallucination. It is a **silent change in the scientific question**.

This is why scientific agent evaluation must inspect method selection, parameter choices, data retention, provenance, and justified uncertainty. Final-answer plausibility is insufficient.

## 4. The four tasks in the original SpikeLab study

The researchers selected a sequence of tasks resembling an actual laboratory analysis. [1]

**Task one, data discovery:** find the IBL recording with the most good-quality auditory cortex units. This tests whether the agent can select the right source through imperfect APIs and metadata.

**Task two, pairwise synchrony:** calculate spike-time tiling coefficients, along with shuffle-based statistics, for cue and no-cue conditions. The wording of the condition split was deliberately ambiguous.

**Task three, population coupling:** quantify how an individual unit relates to the surrounding population and build an event-aligned raster and population-rate visualization.

**Task four, population dynamics:** create instantaneous rate representations, perform PCA, fit a latent state-space model, and visualize trajectories and model states.

Together, these tasks expose a general problem: a scientific request is rarely just one function call. It is a chain of decisions in which earlier assumptions constrain later results.

The authors found that the unassisted models were often competent enough to keep working after making a questionable decision. That is precisely why the mistakes were hard to spot.

## 5. Five composable data abstractions

SpikeLab does not rely on a giant prompt containing every neuroscience method. It implements recurring analysis objects and operations in reusable software. The public repository provides loaders, data containers, analysis routines, workspace persistence, documentation, and optional agent interfaces. [2, 3]

The paper describes a core hierarchy built around five kinds of objects.

**SpikeData** stores spike times and unit metadata. It supports operations like slicing by time, selecting subsets of units, creating shuffled data, and deriving firing rates.

**RateData** stores neural activity transformed into time-indexed firing-rate traces. It enables rate-based comparisons and downstream dimensionality reduction, provided the estimation method and resolution are specified.

**SpikeSliceStack** and **RateSliceStack** are aligned groups of recordings or rate segments. They support event-locked analysis, sliding windows, and repeated comparisons over conditions.

**PairwiseCompMatrix** holds unit-by-unit relationships such as STTC or rate correlation, along with the metadata needed to interpret a matrix correctly.

**AnalysisWorkspace** stores and retrieves inputs, intermediate products, and associated annotations. Persisted outputs can be reused across sessions instead of repeatedly recomputed or regenerated by the agent.

The important detail is that these are *composable*. An agent can create a relevant data slice, call a validated statistic, inspect a matrix, and ask for a plot without inventing the underlying statistical method.

It is the difference between asking a model to explain a library call and asking it to invent the library every morning.

## 6. Bounded autonomy as a software architecture

In many agent architectures, the model has broad access to Python, a shell, and filesystem tools. A natural-language request gets translated into arbitrary code, often with little control over which scientific algorithms are used.

SpikeLab changes where that flexibility lives. Its analysis-implementer skill can compose analysis workflows, but it is instructed to use library methods for neuroscience-specific calculations, avoid modifying library source code, and ask the researcher when a methodological choice is ambiguous. The project also provides separate roles for spike sorting, education, and library development, plus an MCP interface for programmatic tool access. [1, 2]

Three boundaries are especially valuable.

First, **method boundary**: the model can select an approved method but cannot casually replace it with a homegrown substitute.

Second, **data boundary**: the analysis should not silently discard observations or change recording resolution just to save computation.

Third, **decision boundary**: the model should not resolve a scientifically meaningful ambiguity without the researcher noticing.

These boundaries do not eliminate all mistakes. A tested method can still be applied to the wrong population, the wrong unit of analysis, or a dataset that violates its assumptions. But they make the space of possible mistakes smaller and the results easier to audit.

One subtle distinction is worth emphasizing. A text-based skill is an instruction layer, not a security boundary by itself. For robust deployment, a runtime should also enforce tool allowlists, parameter schemas, filesystem permissions, input immutability, and logging. Our companion lab includes some of these controls explicitly.

## 7. What the benchmark measured, including cost

The original study reported three-run outcomes, tool calls, tokens, and estimated API costs. [1]

Opus averaged about **72,202 tokens** and **$1.95 per session**. Unassisted Sonnet averaged about **104,288 tokens** and **$0.56**. SpikeLab-assisted Sonnet averaged about **127,999 tokens** and **$0.69** under the authors' specified blended pricing assumptions.

These numbers do **not** show that the tool-assisted agent used fewer tokens or was cheaper than unassisted Sonnet. It used more tokens, partly for documentation and tool-mediated work, and cost slightly more. They show instead that a smaller model with the right tools could produce more reliable analyses at lower reported cost than the larger unassisted model in this particular benchmark.

This is a crucial economic lesson. A benchmark should measure not only the price per attempted run, but the **cost of producing an independently acceptable scientific result**.

If one workflow costs less but consistently analyzes the wrong subset, its effective cost per valid result may be far worse. Conversely, a tool-assisted workflow could also waste budget by making unnecessary calls. Both possibilities need to be measured.

The authors reported tool-call counts rather than a direct, controlled wall-clock comparison because the agents did not always perform the same computation. Fitting a full-resolution population model is not a fair timing comparison with fitting a heavily downsampled alternative. Faster execution may simply mean that the task changed.

## 8. A crucial difference between reproducibility and correctness

Suppose an agent always uses the same incomplete API endpoint. It selects the same inferior recording in every run and produces the same values. Its behavior is reproducible, but it is reproducibly wrong.

Alternatively, an agent could try different scientifically reasonable analyses on each run and never explain which estimator it chose. It might produce plausible results every time, but the outputs are not directly comparable.

We therefore need at least three separate properties.

**Determinism** asks whether the same data, method, parameters, and random seed reproduce the same output.

**Methodological correctness** asks whether the analysis matches a defensible method for the stated scientific question.

**Scientific validity** asks whether the experimental design, assumptions, uncertainties, and interpretations justify the conclusions.

These are not interchangeable. Our lab deliberately separates them by using deterministic scripts, a method allowlist, and an independent evaluator. Even then, the evaluator only checks a limited set of designed conditions. It cannot establish broad biological validity.

## 9. Build the companion laboratory

We will use a small 30-second synthetic recording with twelve units. The middle third of the recording acts as a cue period. Four units receive temporally coordinated synthetic events, and some units receive an additional stimulus-associated rate increase.

The synthetic recording gives us something important: known design choices. We know the true cue window because we constructed it. We know how many units should be present. We know that spike times are stored in milliseconds. We can therefore test whether an analysis quietly changes these properties.

The project is organized as follows:

```text
spikelab_bounded_agent_lab/
    spikeguard_lab/
        data.py
        methods.py
        agent.py
        evaluator.py
        benchmark.py
        demo.py
    tests/
        test_lab.py
    results/
        benchmark.json
    figures/
        01_evaluator_checks.png
        02_synthetic_spike_raster.png
    plot_results.py
    BENCHMARK_NOTES.md
    requirements.txt
    README.md
    ARTICLE.md
```

Download or clone the companion folder, then create an environment:

```bash
cd spikelab_bounded_agent_lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The laboratory requires NumPy, and the plotting script requires Matplotlib. No GPU, API key, external biological dataset, or model checkpoint is necessary.

The next sections explain why the methods in the repository were chosen, and where they fall short of the full paper.

## 10. Represent spike trains without losing units or metadata

Our core object is a frozen Python dataclass called `Recording`. It contains the duration, a tuple of arrays of spike timestamps, and metadata.

Each unit's timestamps must be finite, strictly increasing, and inside the recording boundaries. A method called `digest()` computes a SHA-256 hash over the recording duration, spike arrays, and metadata. A method called `window()` creates a new recording for a selected interval without mutating the source.

Here is the central idea in a shortened example:

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Recording:
    spikes: tuple[np.ndarray, ...]
    duration_ms: float
    metadata: dict

    @property
    def n_units(self):
        return len(self.spikes)

    @property
    def total_spikes(self):
        return sum(len(unit) for unit in self.spikes)
```

A frozen dataclass prevents reassignment of its attributes, but it does not automatically make NumPy arrays or nested metadata immutable. That is why the hash and independent source-preservation check are useful. In a production system, inputs should be stored read-only and loaded through a trusted data layer.

A more subtle issue concerns units. The SpikeLab repository works with spike timestamps in **milliseconds**. If an agent supplies seconds without converting them, a 20 ms coincidence window becomes wildly inappropriate. The resulting number can still be finite and formatted correctly.

Units should therefore be treated as part of the type contract, not as prose in a comment.

Try generating a recording:

```python
from spikeguard_lab.data import make_synthetic_recording

recording = make_synthetic_recording(seed=7)
print(recording.n_units)
print(recording.duration_ms)
print(recording.total_spikes)
print(recording.digest())
```

Because the generator is seeded, running it again with seed 7 produces identical arrays and the same digest.

## 11. Understand the spike-time tiling coefficient

We now reach a genuinely domain-specific choice. Suppose we want to compare two neurons' spike times. A naive approach is to divide time into fixed bins, count spikes, and compute Pearson correlation. That can be useful when the question is about rate fluctuations. But it is not the same statistic as the spike-time tiling coefficient, or STTC.

Cutts and Eglen proposed STTC to address undesirable properties of older spike-train correlation measures, including dependence on firing rate. Their 2014 paper systematically compared candidate metrics and argued for STTC as a useful alternative when studying precise temporal relationships. [4]

The calculation has four ingredients.

`P_A` is the fraction of spikes in train A that lie within a chosen coincidence window of at least one spike in train B.

`P_B` reverses that comparison.

`T_A` is the fraction of recording time covered by the union of short windows around spikes in train A.

`T_B` is the equivalent time coverage for B.

STTC combines the difference between observed coincidence and the coincidence expected from covered time. The resulting measure is bounded, symmetric, and interpretable in terms of temporal association at a particular coincidence window.

The important point is not that STTC always wins. It is that **STTC and binned Pearson correlation answer different questions**. If a scientist requests one and an agent silently computes the other, it has changed the analysis.

## 12. Write a numerically checkable STTC implementation

The companion lab implements STTC directly in `spikeguard_lab/methods.py`, with simple NumPy operations and validation.

A helper computes the fraction of time covered by the merged coincidence intervals around the spikes. Another helper uses `np.searchsorted` to determine whether each spike is close enough to a spike in the other train.

The core calculation is:

```python
def term(p, t):
    den = 1.0 - p * t
    return (p - t) / den if abs(den) > 1e-12 else 1.0

return float(np.clip(
    0.5 * (term(pa, tb) + term(pb, ta)),
    -1, 1
))
```

Here, `pa` and `pb` are coincidence fractions; `ta` and `tb` are time-coverage fractions. The implementation is intentionally transparent rather than optimized for million-spike recordings.

Several unit tests matter more than a single attractive visualization.

Identical spike trains should give an STTC near 1 for an appropriate window. Swapping train A and B must not change the result. Negative or impossible time windows must be rejected. Empty trains should not silently become uncorrelated trains with a coefficient of zero. Our implementation returns `NaN` for that undefined case so downstream analysis has to acknowledge missing information.

A real deployment should also compare against the official SpikeLab implementation, assess its behavior on sparse and bursty trains, and test nonstationarity. A numerically consistent local implementation is not equivalent to a fully validated neuroscience pipeline.

## 13. Surrogate tests and false discoveries

Suppose our STTC between two neurons is 0.31. Is that interesting?

Not necessarily. The meaning depends on their firing rates, stimulus dynamics, recording duration, and a defensible null hypothesis. One common approach is to compare the observed statistic with statistics from modified spike trains designed to disrupt a specific kind of temporal relationship while preserving some other properties.

Our lab uses a simple circular-shift surrogate. One neuron's spike times are shifted by a random offset and wrapped around the recording boundaries. The STTC is recalculated for each shifted train. The observed result is compared with this surrogate distribution.

The estimated two-sided Monte Carlo p-value is computed using an add-one convention:

```python
p_value = (
    1 + sum(abs(x) >= abs(observed) for x in surrogate_scores)
) / (len(surrogate_scores) + 1)
```

Why the `+1`? Because with a finite set of sampled null results, the observed event must not receive a p-value of exactly zero just because the simulations did not exceed it.

There is an important consequence for our small demonstration. We use **19 surrogate draws** per pair. That means the smallest achievable p-value is **0.05**. This is deliberately cheap for a laptop, but it provides poor resolution for strong significance claims. Readers should not interpret these outputs as a publication-ready statistical discovery pipeline.

With twelve units, there are 66 possible unique pairs. Testing 66 hypotheses creates a multiple-comparisons problem, even if every pairwise calculation is correct. We therefore include a Benjamini-Hochberg adjustment that produces q-values from the sampled p-values.

The adjustment is a safeguard, not a universal guarantee. Neuronal pairwise tests share units and are dependent. The properties of the surrogate null also depend on whether the recording is stationary. A naive circular shift across strongly stimulus-modulated activity can break the wrong structure and yield misleading results.

A rigorous study would choose null models appropriate to the experiment, use substantially more surrogate draws, and justify its inferential target before asking an agent to calculate anything.

## 14. Population coupling and the self-correlation trap

In another stage of the paper's benchmark, the agents studied population coupling: how a unit's activity relates to activity across the other recorded neurons.

Okun and collaborators described diverse patterns of neuronal population coupling and used spike-triggered population activity to study the relationship between individual neurons and local populations. Their work is an essential starting point for understanding why this measure is scientifically interesting. [5]

An easy mistake is to create a population rate by adding every neuron's activity, then correlate a neuron's activity with that sum. If the target neuron remains in the summed signal, the statistic includes a trivial self-correlation component.

Our educational lab avoids that direct mistake with a leave-one-out construction:

```python
counts = bin_spikes(recording, bin_ms=50)
population = counts.sum(axis=0)

for unit_index in range(recording.n_units):
    single_unit = counts[unit_index]
    all_other_units = population - single_unit
    coupling = np.corrcoef(
        single_unit, all_other_units
    )[0, 1]
```

This is useful for explaining why self-inclusion is a bug. But it is essential to name it correctly: **our code calculates a binned-rate Pearson correlation proxy. It does not implement the paper's full spike-triggered population-rate coupling statistic.** Pearson coupling can remain sensitive to firing rate and bin width even when the target unit is excluded. Okun and colleagues explicitly examined these differences. [5]

We include the proxy because we can validate its computational invariants with simple tests. For publication-grade neural population coupling, use the appropriate validated field-standard estimator and state the normalization and temporal kernel.

## 15. Latent population dynamics and silent downsampling

A population of neurons produces a high-dimensional time series. Researchers often reduce this to a smaller number of latent dimensions, identify repeated activity patterns, or fit dynamical models that explain state transitions.

The original SpikeLab benchmark asked the models to compute firing rates and PCA, then fit a latent state-space model. This stage exposed a particularly interesting failure mode. Unassisted models sometimes reduced temporal resolution, projected data through convenient intermediate representations, or failed to inspect whether a model-selection search range was sufficient. These decisions were not consistently disclosed to the user. [1]

The difference between 50 ms time bins and 300 ms effective resolution is significant if the phenomena under study involve rapid transitions. Even if a coarsened model fits well, it does not establish what a full-resolution model would have discovered.

Our lab implements **PCA explained variance only**. It does not fit an HMM, a Gaussian-process latent variable model, or any other temporal state-space model. This is an explicit scope boundary, not a hidden simplification.

PCA centers the binned-rate matrix, performs singular-value decomposition, squares the singular values, and reports their share of total variance. It can show whether a small number of linear components explains a large share of activity variance. It does not, on its own, identify discrete functional brain states or demonstrate causal neural dynamics.

Keeping this distinction visible is a tiny example of the larger design principle: do not let a convenient approximation silently inherit the scientific name of a different method.

## 16. Define an explicit tool contract

Instead of allowing the simulated agent to run arbitrary neuroscience code, we expose a small registry:

```python
ALLOWED_TOOLS = {
    'slice_conditions',
    'pairwise_sttc',
    'coupling',
    'pca',
    'provenance',
}
```

Each tool validates its arguments. The `pairwise_sttc` method restricts the coincidence window and bounds the number of surrogate draws. The condition-slicing tool requires both `cue` and `no_cue` to be defined. The coupling and PCA tools reject implausible bin widths.

This is far from a complete secure agent runtime. A real deployment needs process isolation, least-privilege filesystem access, resource limits, signed method versions, and structured authorization policies. But the registry makes one thing explicit: the agent may **compose** approved analyses without **reimplementing** them behind the evaluator's back.

It also gives the evaluator a reliable audit trail. A tool invocation can record the method name, parameter values, input identifier, software version, and output hash. That record can be inspected independently of the model's final explanation.

The compiler metaphor is useful here. The agent produces a plan in a constrained language. The trusted runtime executes that plan. The final prose is commentary on the executed program, not the source of truth for what happened.

## 17. Require clarification on ambiguous analyses

Consider this scientific instruction:

> Compare the cue and no-cue portions of the recording.

What exactly is the no-cue condition? The time before the first cue? Inter-trial periods? Catch trials? A period when the subject is behaviorally disengaged? A separate control session?

Each interpretation creates a different contrast. A good scientific assistant should present plausible choices rather than confidently invent a default.

The original paper intentionally tested this ambiguity. Across three runs each, neither unassisted model requested clarification. SpikeLab-assisted Sonnet asked for clarification in all three runs. [1]

Our registry reproduces the **protocol**, not the LLM behavior. If a policy requests a condition split without specifying boundaries, the tool raises `ClarificationRequired`. The scripted benchmark then supplies an explicitly defined cue window and an explicitly defined pre-cue control interval. The stop-and-resume behavior is logged.

```python
try:
    session.call('slice_conditions', windows=None)
except ClarificationRequired:
    # In a real system, ask the researcher.
    # The toy benchmark supplies a known specification.
    windows = session.call(
        'slice_conditions',
        windows=correct_windows(recording)
    )
```

The scientific lesson is that clarification has an operational cost but prevents an entire downstream analysis from answering the wrong question.

## 18. Build a flawed generalist policy as a control

We need a comparison that can fail. The lab includes a deliberately flawed `improvised` policy that behaves like an agent trying to finish the task quickly without respecting its scientific constraints.

It makes several specific mistakes:

- It silently defines the cue period as the first half of the recording and the no-cue period as the second half, even though the synthetic cue is in the middle third.
- It substitutes Pearson correlation on binned spike counts for STTC while presenting the result as if it satisfied the requested analysis.
- It includes the target unit in its own population-rate comparator, inflating a self-correlation component.
- It uses coarser time bins without making the tradeoff explicit.
- It omits statistical significance estimates, adjusted p-values, a provenance record, and methodological caveats.

A separate `silent_downsample` policy retains the correct condition definitions, STTC, surrogates, and provenance, but changes coupling and PCA binning from 50 ms to 200 ms. This isolates the effect of a methodological resolution change. A `no_verification` policy, by contrast, keeps the right windows and STTC computation but omits surrogate testing and correction for multiple comparisons.

Crucially, **these behaviors are scripted by us**. We have not measured whether any real model would choose them under natural prompts. The purpose is to create an environment where a reference evaluator can catch known errors, not to estimate the error rate of any commercial model.

## 19. Independently evaluate the generated results

If the agent is allowed to grade its own work, it can simply state that every method was correct. Our evaluator therefore operates outside the policy's reported narrative.

The evaluator independently reads the synthetic recording, constructs the expected condition windows, recomputes reference STTC for a selected pair, checks spike retention, validates leave-one-out coupling and PCA, and examines whether the policy supplied statistical evidence and provenance.

It uses fourteen explicit checks. They cover the correctness of the condition definition, whether clarification occurred, unit retention, spike retention in both windows, number of pairwise comparisons, method identity, reference STTC agreement, surrogate p-values, BH-adjusted q-values, temporal resolution, leave-one-out coupling, PCA consistency, and provenance plus limitations.

These **are not the original paper's fourteen assessment issues**. They are a separate, small, hand-designed teaching rubric.

One useful regression test deliberately overwrites a result after the workflow completes:

```python
outcome = run_policy(recording, 'bounded', surrogates=3)
outcome['analysis']['cue']['pair_results'][0]['observed'] = 0.123456

report = evaluate(recording, outcome)
assert not report['checks']['sttc_matches_reference']
```

That test demonstrates a boundary between the system that produces an answer and the system that validates an answer. It does not make the evaluator infallible. Some of our reference checks share low-level statistical primitives with the producer, so a common implementation bug could still go undetected. A production benchmark should use an independently validated implementation, disjoint data transformations, and human methodological review.

## 20. Run the five-seed benchmark

Once dependencies are installed, reproduce the benchmark with:

```bash
python -m unittest discover -s tests -v
python -m spikeguard_lab.demo
python -m spikeguard_lab.benchmark \
  --seeds 5 \
  --surrogates 19 \
  --output results/benchmark.json
python plot_results.py
```

The saved JSON records five deterministic synthetic recording seeds and each policy's results. It includes the individual pass/fail status of every evaluation check, the number of tool calls, whether clarification occurred, and elapsed execution time.

Across those five seeds, the measured scores were:

**Bounded workflow:** 14 of 14 independent checks passed on average.

**Correct method without surrogate verification:** 12 of 14 checks passed.

**Improvised workflow:** 2 of 14 checks passed.

**Silent-downsampling workflow:** 11 of 14 checks passed.

The bounded workflow averaged around **0.25 seconds** locally; the no-verification configuration averaged about **0.015 seconds**. The resolution-shift ablation averaged around **0.25 seconds** because it still performs STTC and statistical verification. Only the improvised workflow finished in approximately **0.003 seconds**, because it omitted most of the work.

These timings are CPU measurements for tiny synthetic arrays. They do not represent Claude inference latency, SpikeLab's original research throughput, or the economics of a real neuroscience experiment.

The results are shown in `figures/01_evaluator_checks.png`, and the synthetic recording raster appears in `figures/02_synthetic_spike_raster.png`. Both are generated directly from the saved lab.

## 21. Read the ablations without overclaiming

Our results are deliberately dramatic because we wrote the policies to display specific failure modes. That makes the benchmark valuable for software verification, but it does not justify saying that constrained models are seven times more capable or that they universally outperform unconstrained models.

The cleanest comparison in this lab is between the `bounded` and `no_verification` conditions. Both use the same explicit windows and the same core STTC implementation. The main difference is that one computes surrogate-derived p-values and multiple-testing adjustments while the other does not.

The no-verification workflow produces descriptive STTC values much faster. It simply cannot support the same inferential claims. On a exploratory plotting task, that might be an acceptable intermediate result. On a request to identify statistically supported neuronal associations, presenting it as complete would be a methodological error.

The `silent_downsample` configuration keeps the correct STTC and windows but changes the resolution of other measurements, causing three independently detected failures. The `improvised` workflow makes larger changes to the question, the statistic, and the analysis. It finishes quickly precisely because it computes something else.

The message is **not** that the slowest workflow is always best. It is that cost and runtime comparisons are meaningful only after defining what must be preserved.

## 22. Why a faster workflow can be the wrong workflow

In software performance engineering, developers routinely optimize latency by caching, batching, approximate computation, or lowering precision. These techniques can be excellent if the resulting outputs remain within a validated error budget.

Scientific analysis is different when approximation changes the claim itself.

If a researcher asks whether two neurons exhibit millisecond-scale coordination, switching to a 300 ms binned-rate correlation is not just an efficiency optimization. It asks a different biological question.

If a researcher asks for latent states at 50 ms resolution, quietly discarding five out of six bins alters the observable state transitions. The model may become faster to fit, but its reported solution no longer answers the original request.

This suggests a useful distinction for agent systems: **implementation-level optimization** may be automatic, while **scientifically substantive approximation** must be declared, validated, and often approved by the researcher.

An agent may choose a vectorized STTC implementation over a slow Python loop if the outputs are numerically equivalent. It should not replace STTC with Pearson correlation because the second function is easier to write.

In SpikeLab, much of this distinction is encoded in the library rather than left to the model's momentary judgment. That is the contribution worth applying to other domains.

## 23. What would constitute a fair real-model benchmark

Our synthetic study does not run an LLM. How should a rigorous follow-up test actual agents?

Start with the same scientific input data, prompt, task success definition, and runtime constraints for every condition. Use at least four conditions: a frontier model with unconstrained code execution, a smaller model with unconstrained execution, the smaller model with the validated tools, and the frontier model with the same tools.

That fourth condition matters. The SpikeLab paper did not report an Opus-plus-SpikeLab condition, so it cannot fully isolate the effects of model capability from the effects of tools across every configuration.

Next, freeze the software environment and record dataset checksums. Explicitly report which conditions have dependencies preinstalled, since setup overhead can change both cost and tool-call counts.

Run each task across multiple recordings, sessions, and prompts. Deliberately include ambiguous requests, malformed metadata, missing data, unequal trial durations, and conflicting scientific assumptions. Blind the independent evaluator to model identity.

Measure more than binary task completion. A useful scorecard should examine whether the correct recording was found, whether data were silently excluded, whether a valid estimator was selected, whether assumptions and caveats were reported, whether statistical significance procedures were appropriate, and whether rerunning the entire workflow reproduces the same artifacts.

Account for uncertainty. Use confidence intervals across independent sessions, not just dozens of correlated neurons from the same recording. Report both the percentage of correctly completed analyses and the **total cost per independently accepted analysis**.

Finally, treat clarifying questions carefully. An agent that stops to ask a question is neither an automatic success nor an automatic failure. The evaluation should record whether clarification was warranted, whether the model asked an informative question, and what happened after the answer.

## 24. Using official SpikeLab and public IBL data

The official SpikeLab project is available on GitHub and PyPI. Its repository documents installation, the `SpikeData` and `RateData` interfaces, STTC computation, plotting, loaders for common neuroscience formats, workspace persistence, and optional MCP integration. [2, 3]

For example, its published quick start demonstrates:

```python
from spikelab import SpikeData
from spikelab.data_loaders import load_spikedata_from_nwb

recording = load_spikedata_from_nwb('recording.nwb')
print(recording.N)
print(recording.length)  # milliseconds

rates = recording.rates(bin_size=100.0)
pairwise_sttc = recording.spike_time_tilings(delt=20.0)
```

This is **an upstream API example**, not code executed by our synthetic benchmark. It requires a suitable NWB file and the appropriate optional dependencies.

The International Brain Laboratory publishes brain-wide electrophysiology resources, data documentation, and examples for access through ONE and NWB. [6] That makes it a suitable source for a future real-data study, but there is an important operational cost: actual electrophysiology recordings can be large, and correct data discovery and quality filtering require following the published release-specific procedures.

Do not simply download a random session, rename the variables, and compare your results against the paper's 208-unit example. A valid reproduction requires consistent probe selection, good-unit criteria, brain-region annotations, cue definitions, and recording provenance.

## 25. Security, governance, and provenance

Scientific autonomy creates more than statistical risks. If an analysis agent has unrestricted access to the shell, it may overwrite input files, install unreviewed packages, expose sensitive files, or change the implementation of the very function it was supposed to use.

A robust scientific agent runtime should restrict where outputs can be written, make source data read-only, version the allowed methods, record parameter choices, and use external verification that the agent cannot edit. Agent-readable documentation should not be treated as an authoritative instruction source when it comes from untrusted datasets or external websites.

The best architecture separates responsibilities.

A planner translates the researcher's request into an explicit analysis plan. A policy layer checks that the plan uses permitted operations and well-defined parameters. A compute runtime executes approved code in a restricted environment. An evaluator independently examines provenance, outputs, and relevant scientific invariants. A human researcher remains responsible for accepting substantive assumptions and interpreting the evidence.

This is also useful for ordinary machine learning. Think of a forecasting agent that must use rolling-origin evaluation, a medical-image agent that must use patient-level splits, or a code-review agent that must execute tests from an immutable reference suite. The specific tools differ, but the architectural principle is similar.

## 26. Limitations of the paper and our experiment

The original SpikeLab findings are promising, but they are based on a preprint, a specific sequence of four tasks, one primary benchmark dataset, and three runs per condition. The comparison uses one model family. The assisted configuration also has access to domain-specific libraries and preinstalled dependencies that the unassisted models did not have, and different reasoning-effort settings separate the Opus and Sonnet conditions. These details limit broad conclusions about model intelligence or cost. [1]

It would be too strong to claim that a smaller model with tools will always beat a larger model. The demonstrated claim is narrower: **in this study, the tool-assisted smaller model followed the evaluated analysis workflow more consistently than the unassisted configurations**.

Our independent educational lab has even stricter limits.

It uses a synthetic Poisson-based event generator, not actual recorded neurons. Its policies are deterministic scripts with known faults, not model behavior. Its evaluator checks a selected list of invariants rather than an independently adjudicated scientific truth. Its Pearson coupling proxy is not the paper's estimator, and its PCA is not a state-space model. Its circular-shift null has limited resolution and may be unsuitable under nonstationarity. It does not measure provider tokens, agent latency, actual discovery quality, or biological generalization.

These qualifications are not incidental. The central theme of the article is that claims should never exceed what the measurement supports. The benchmark must obey that rule too.

## 27. Research directions

There are several worthwhile extensions for researchers who want to turn the companion lab into a serious evaluation project.

**First, replace the scripted policies with real agent clients.** Give multiple models the same data, prompts, permission scopes, and budgets, with and without the official library. Preserve complete action traces for independent evaluation.

**Second, perform a controlled ambiguity study.** Use scientifically plausible but mutually incompatible cue definitions. Measure whether agents ask for clarification, how informative their questions are, and whether answers change the analysis correctly.

**Third, validate the statistics independently.** Compare our STTC implementation with the official library and a separately written reference. Test rate changes, bursts, pauses, short recordings, degenerate cases, and alternative surrogate schemes.

**Fourth, study cost relative to scientific validity.** Instrument tool latency, token use, failed executions, compute time, and reviewer effort. The relevant denominator should be verified analyses, not merely completed tool calls.

**Fifth, add provenance-preserving execution plans.** Represent a workflow as a typed DAG, with immutable data references and explicit method versions. Hash each output along with its parent inputs and parameter values. This would make it possible to reproduce a result even after the original agent conversation is gone.

**Sixth, extend the benchmark to real public data.** Work through IBL's official released datasets and unit-quality criteria, using multiple independent recordings to avoid mistaking within-session agreement for external replication.

**Seventh, test method-boundary attacks defensively.** Insert misleading documentation, adversarial comments, or malformed metadata into sandboxed test fixtures. Verify that the agent does not silently substitute unapproved methods or modify the evaluation source of truth.

These experiments would help answer a deeper question: whether domain-specific tools merely improve a model's apparent compliance, or actually make scientific conclusions more robust under new data and new requests.

## 28. Conclusion

There is a tempting story about scientific AI: if the model is sufficiently intelligent, it will eventually understand every method, choose every parameter correctly, discover every relevant dataset, and produce a flawless result.

SpikeLab offers a more practical story.

Scientific knowledge need not live entirely inside a language model. It can live in tested software, versioned data contracts, explicit methods, protected workspaces, and evaluation procedures. The model's role is then to formulate a plan, connect valid operations, inspect outcomes, and communicate uncertainty.

This design does not make the agent infallible. But it changes the failure mode from an invisible improvisation to something a human can identify, inspect, and challenge.

Our small benchmark makes the distinction concrete. The fastest simulated workflow scores poorly because it silently changed the method and experimental conditions. The bounded workflow takes longer because it actually performs the requested checks. Both can produce output. Only one preserves the conditions that make the output meaningfully auditable in this fixture set.

The lesson is bigger than neuroscience.

**A scientific agent becomes more trustworthy not when it can write any code, but when it knows which code it is not allowed to invent.**

## 29. References

1. van der Molen, T., et al. (2026). **SpikeLab: Agentic Tools for Spike Data Analysis.** bioRxiv preprint, posted April 29, 2026. DOI: [10.64898/2026.04.25.720833](https://doi.org/10.64898/2026.04.25.720833). [Full text](https://www.biorxiv.org/content/10.64898/2026.04.25.720833v1.full).

2. Braingeneers. **SpikeLab official repository.** [GitHub](https://github.com/braingeneers/SpikeLab). Source code, supported file types, tool interfaces, package installation, and agent skills.

3. SpikeLab. **Project documentation and pairwise analysis guide.** [Documentation](https://spikelab.braingeneers.gi.ucsc.edu/). [STTC guide](https://spikelab.braingeneers.gi.ucsc.edu/guides/pairwise_analysis.html).

4. Cutts, C. S., and Eglen, S. J. (2014). **Detecting Pairwise Correlations in Spike Trains: An Objective Comparison of Methods and Application to the Study of Retinal Waves.** Journal of Neuroscience, 34(43), 14288-14303. DOI: [10.1523/JNEUROSCI.2767-14.2014](https://doi.org/10.1523/JNEUROSCI.2767-14.2014).

5. Okun, M., et al. (2015). **Diverse Coupling of Neurons to Populations in Sensory Cortex.** Nature, 521, 511-515. DOI: [10.1038/nature14273](https://doi.org/10.1038/nature14273).

6. International Brain Laboratory. **Public data releases and documentation.** [IBL Data Portal](https://www.internationalbrainlab.com/data).

7. SpikeLab preprint full text and supplementary materials. **Original comparison design, 14-issue scorecard, tool-call analysis, and limitations.** [PubMed Central version](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142478/).

---

*Companion code note: The independent SpikeGuard Lab includes the full synthetic generator, statistical operations, bounded tool registry, intentionally faulty control workflows, external evaluator, 22 regression tests, saved five-seed benchmark, and two original figures. It is not affiliated with the SpikeLab authors.*
