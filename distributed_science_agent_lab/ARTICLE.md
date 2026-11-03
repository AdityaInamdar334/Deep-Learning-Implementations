# 1,946 AI Agents Analyzed a Brain Atlas in 32 Minutes. The Hard Part Was Knowing Which Discoveries to Trust.

*Inside OpenAI, PerturbAI, and Scripps Research's distributed biological reasoning system, and a reproducible Python workshop that tests when scientific agent specialization and parallelism are actually worth the cost.*

**Research context:** OpenAI, PerturbAI, and Scripps Research, October 2026.  
**Companion project:** The downloadable `distributed_science_agent_lab` repository accompanying this article.  
**Audience:** ML researchers, data scientists, software engineers, computational biologists, and people building scientific agents.

A laboratory can measure the effects of disrupting nearly two thousand genes across millions of brain cells. The experiments run, the data are processed, and the statistical software produces a mountain of results.

Then a harder problem begins.

Which observation matters? Which one is already well understood? Which suggests a mechanism nobody has considered? Which is a statistical artifact? Which deserves another experiment, and which sounds exciting only because its explanation is written confidently?

For a long time, the answer was to put more scientists in front of more papers. That does not scale easily. A single perturbation can affect several cell types, alter hundreds of transcripts, and connect to research scattered across different disciplines. Multiply the work by a thousand and the bottleneck is no longer measurement. It is interpretation.

On October 21, 2026, OpenAI announced a collaboration with PerturbAI and Scripps Research that tried a different approach. They assigned **1,946 AI agents** to separate gene perturbations in a whole-brain CRISPR atlas. Those agents examined existing cellular-response measurements, read scientific literature, and wrote evidence-linked interpretations. A subsequent synthesis stage combined the results into a reasoning atlas.

The reported execution time was **32 minutes**. The research preprint records **39,185 paper reads**, **16,445 retained citations**, and **3,397 normalized, deduplicated cross-perturbation findings**. In blinded comparisons, biology experts preferred the distributed reports in **45 of 60 judgments**. [1, 2]

Those numbers are striking. They are also easy to misunderstand.

The agents did **not** grow eight million cells, perform CRISPR experiments, sequence the samples, or independently prove 3,397 new biological mechanisms in half an hour. They interpreted an **existing, processed** experimental atlas and prioritized hypotheses for scientists to investigate. The system used GPT-Rosalind, a specialized life-sciences model, and a research workflow that is not equivalent to spawning a few generic chatbots. [1, 2, 3]

This article is about the engineering idea hiding underneath the announcement: **when does dividing scientific work among specialized agents create better evidence, and when does it merely create more text, more tokens, and more opportunities to be wrong?**

We will read the primary research, reconstruct the architecture, design an independent verification protocol, implement a smaller version in Python, run ablations, and measure the tradeoff among coverage, quality, latency, and estimated cost. The workshop works locally without a GPU, an API key, or access to private biology data.

Our experiment uses public wine chemistry measurements and deliberately generated noise features. It does not claim to reproduce GPT-Rosalind, the original brain atlas, or the biological discoveries. It tests the orchestration principles under conditions we can inspect and verify.

---

## Contents

1. The real bottleneck in high-throughput science
2. What the original project actually did
3. Reading the evidence behind the headline
4. Why one large agent can be the wrong architecture
5. From a single prompt to a scientific work graph
6. What specialization means in scientific computation
7. The evidence contract every worker should return
8. The dataset for our independent workshop
9. A research question we can actually verify
10. Reproducible experimental design
11. Load the public dataset and generate null controls
12. Implement a statistical investigator
13. Correct for multiple comparisons
14. Build a provenance-preserving orchestrator
15. Parallelize without changing the analysis
16. Add independent verification
17. Measure tokens and costs honestly
18. Reproduce the five-seed experiment
19. What the benchmark actually found
20. Ablation 1: fewer investigations
21. Ablation 2: parallel versus serial
22. Ablation 3: remove the holdout verifier
23. Ablation 4: remove multiple-testing correction
24. Why more citations can produce worse science
25. How to evaluate actual LLM-based research agents
26. Statistical and biological failure modes
27. From correlations to follow-up experiments
28. When should you use thousands of agents?
29. How to extend the workshop to biological data
30. Reproducibility, limitations, and closing argument
31. References and research resources

---

## 1. The real bottleneck in high-throughput science

Consider a modern perturbation screen. A researcher disrupts a gene, lets the system respond, and measures changes in gene expression. Repeat this for thousands of genes across many cell types and you get something closer to a searchable map of biological responses than a conventional experiment.

This is the motivation behind **Perturb-seq**, which combines genetic perturbations with single-cell transcriptional measurements. In a brain-wide study, the results are organized not merely by which gene was disrupted but also by which population of cells experienced a particular response. A finding may be strong in one neuronal population and absent in another.

Ordinary statistical software can identify differentially expressed genes, estimate effect sizes, and compare perturbation profiles. But a list of altered transcripts does not explain what happened. Interpretation requires several additional steps:

- Recognize the affected cellular program.
- Connect that program to prior experimental work.
- Determine whether the observation is expected or surprising.
- Identify competing explanations and missing controls.
- Evaluate whether a proposed mechanism generates a falsifiable prediction.
- Choose the next experiment that has the best chance of resolving uncertainty.

The last item is particularly important. The scientific value of an agent is not the length of its report. It is whether the report changes what the team can test next.

OpenAI and its collaborators describe the bottleneck as **biological interpretation at the scale of experimental measurement**. Their design addresses that bottleneck by assigning bounded tasks to agents instead of requiring one agent to reason about the entire experiment at once. [1, 2]

## 2. What the original project actually did

The underlying system is described in *Distributed AI agents enable comprehensive discovery of biological insights at scale*, a 2026 bioRxiv preprint by Alexander Sun and collaborators. At the time of this article, it is a **preprint**, not a peer-reviewed demonstration that its proposed mechanisms have been experimentally confirmed. [2]

The input was a whole-brain, in vivo CRISPR Perturb-seq atlas. It covered approximately eight million mouse-brain cells or nuclei and nearly two thousand gene perturbations. The distributed reasoning run involved **1,946 independently scoped agents**. Each agent received cell-type-resolved differential-expression information and experimental context, with access to analytical tools and scientific literature. [2, 3]

This is the crucial architectural choice: an agent was assigned an **individual perturbation**, not an arbitrary chunk of a giant prompt.

That task boundary is meaningful because each gene disruption defines an interpretable unit of evidence. One worker can investigate a gene, compare its effects among cell types, seek relevant publications, and record what remains unexplained. The output can then be joined to outputs from other perturbations.

The pipeline has at least two levels of reasoning:

**Local interpretation:** For each perturbation, analyze its signatures, retrieve relevant literature, distinguish established findings from inferred mechanisms, and produce a structured report with caveats.

**Global synthesis:** Connect local reports across genes and cell types, identify convergent and opposing effects, normalize duplicative statements, and rank cross-perturbation patterns for follow-up.

The published repository includes the corpus tooling, runtime, analysis scripts, and released findings. Its README distinguishes querying the released atlas from recomputing the original statistics. Reproducing the data-intensive benchmark requires a **10.8 GB pinned Parquet download** and a much larger prepared data footprint. [4]

## 3. Reading the evidence behind the headline

The 32-minute result deserves attention, but it is not the only result worth measuring.

According to the preprint, the agents logged **39,185 paper reads** and retained **16,445 citations**. The researchers also classified **1,519 high-confidence findings** according to how they related to literature: 420 agreed with precedent, 636 represented inferences, 101 disagreed with retrieved precedent, and 362 had no retrieved precedent. These categories help distinguish an observation supported by prior work from an interesting interpretation that still needs checking. [2]

The synthesis process then generated **3,397 normalized relational findings** connecting perturbations, affected programs, cell types, and evidence. These findings were ranked rather than declared discoveries. [2]

There was also a human evaluation. Thirty doctoral-level biology experts assessed 60 reports across quality categories. In a separate paired comparison, 20 report pairs were each judged by three experts. Distributed-agent reports won **45 of 60 judgments**, single-agent reports won 10, and five were ties. That is what the widely quoted **75% preference** means. [2]

It does not mean 75% of generated hypotheses were biologically correct. Preference scores are human judgments about the usefulness and quality of reports, not prospective wet-lab validation.

In the reported largest-workload comparison, distributed agents retrieved far more supporting literature than a single agent. OpenAI's public summary says **more than 700 references versus 11** in the compared reporting setting. The larger aggregate citation count in the preprint refers to a different quantity across the complete distributed run. Do not interchange those two metrics. [1, 2]

Finally, the preprint gives concrete examples. It discusses differential consequences of **Tsc1** and **Tsc2** disruption in cerebellar neurons and a **Dctn1** perturbation associated with increased expression of cholesterol-biosynthesis genes. Both are research leads. Neither should be presented as proof of a drug target or therapeutic effect. [1, 2]

## 4. Why one large agent can be the wrong architecture

Imagine placing all perturbation statistics, all previous papers, and all possible mechanisms in a single conversation. There are three immediate limits.

The first is **attention allocation**. Even if a model can technically process the entire context, it has to decide where to spend its limited reasoning effort. Strong or familiar findings may crowd out low-signal but important anomalies.

The second is **tool budget**. One agent must decide which references to retrieve, which calculations to run, and how to divide a large task into subtasks. A finite time or token budget creates a competition among hypotheses.

The third is **failure coupling**. If the agent makes an early incorrect assumption about the experiment, subsequent sections can inherit it. Its errors may be correlated rather than independent.

Splitting the work changes the shape of the problem. A worker investigating one perturbation does not need to decide how to allocate attention among all 1,946 perturbations. Its job is to produce an evidence package for one well-defined unit.

There is a tradeoff. Every new agent may incur a fresh instruction prefix, retrieval overhead, model context, and synthesis cost. More workers are not free. Splitting the work can also destroy global context unless reports preserve comparable measurements and explicit links.

The real optimization objective is not the number of agents. It is **verified scientific information gained per unit of money, time, and human review**.

## 5. From a single prompt to a scientific work graph

A useful way to reconstruct the architecture is as a directed work graph:

```text
Processed experimental dataset
          |
          v
       Planner
          |
          +--> Perturbation A investigator --> Evidence package A
          |
          +--> Perturbation B investigator --> Evidence package B
          |
          +--> Perturbation C investigator --> Evidence package C
          |
          +--> ...
          |
          v
  Evidence and statistics verifier
          |
          v
      Synthesis stage
          |
          v
   Deduplicated finding graph
          |
          v
  Human review and experimental planning
```

The key is that **generation and verification have different responsibilities**.

An investigator can suggest a mechanism. A verifier checks whether the reported effect exists in the data, whether the statistical test is appropriate, whether the cited publication supports the statement, and whether the conclusion is stronger than the evidence permits.

A synthesis agent should not simply paste 1,946 reports into one long summary. It should normalize entities, merge near-duplicates, preserve conflicting evidence, and identify relationships that could not be seen by examining a single perturbation.

If there are ten workers claiming that one cellular pathway responds to perturbation, that is not automatically ten independent pieces of evidence. Their findings may depend on shared measurements, the same literature review, or the same incorrect assumption.

## 6. What specialization means in scientific computation

The word *specialization* can refer to several different things.

**Data specialization:** Give each worker a separate gene, region, time series, or experimental condition. Workers may run the same model and instructions. This is close to the perturbation-scoped design used in the original study.

**Method specialization:** Assign different analytical roles, such as statistical modeling, literature review, visualization, causal critique, or reproducibility auditing.

**Model specialization:** Use different models optimized for chemistry, sequence interpretation, code execution, mathematical analysis, or document retrieval.

**Verification specialization:** Run a second, independent process that audits the first process rather than sharing its objective of producing a compelling story.

These are not interchangeable. In particular, a distributed system can be valuable even if every worker runs exactly the same base model. Conversely, an elaborate collection of agents with different personality descriptions may provide no benefit if everyone sees the same evidence and makes the same mistakes.

In our lab, each feature gets one bounded statistical task. The roles are represented by deterministic Python functions rather than language models. This lets us measure scheduling, coverage, evidence checks, and false discovery without confounding those effects with model intelligence.

## 7. The evidence contract every worker should return

Do not let a scientific agent return only a paragraph saying it discovered something interesting.

Require an object that includes at least:

- The precise testable claim and the entity it concerns.
- The dataset release, feature name, and observation subset.
- Sample sizes and effect direction.
- The statistical method, effect size, and uncertainty.
- Raw and multiple-testing-adjusted significance values.
- Whether an independent sample replicated the result.
- Provenance for every computation and citation.
- Relevant counterevidence and unresolved assumptions.
- One concrete follow-up experiment or additional measurement.

A serialized finding in our workshop resembles:

```json
{
  "name": "proline",
  "feature_index": 12,
  "discovery_p": 0.00001,
  "holdout_p": 0.00002,
  "effect_d": 2.0,
  "holdout_effect_d": 1.8,
  "candidate": true,
  "replicated": true,
  "accepted": true,
  "provenance": "sklearn:load_wine:classes_0_1:seed_0:feature_12"
}
```

Those numbers are an **illustrative schema**, not a copied result from the benchmark. Real values are written to the saved JSON when you run the project.

Structured evidence makes independent checking possible. A fluent paragraph without raw provenance does not.

## 8. The dataset for our independent workshop

Reproducing OpenAI's exact experiment would require the original brain atlas, its processing conventions, the GPT-Rosalind research workflow, and the source study's specialized evaluation. That is not necessary for the question we want to test first.

Instead, we use the **Wine recognition dataset** distributed with scikit-learn. It contains **178 samples**, **13 measured chemical features**, and labels for three wine cultivars. The underlying dataset is an established public machine-learning benchmark. We restrict our experiment to **classes 0 and 1**, yielding **130 samples**. [5]

This dataset is not a clinical dataset, not a model of neurons, and not a source of biomedical discoveries. That is deliberate. It lets us focus on whether multiple independent investigations and a verification layer produce better evidence.

To test false discoveries, we add **24 synthetic Gaussian noise features**, generated independently of the cultivar labels. These are known-null controls by construction. They may appear correlated with outcomes in a finite sample through chance alone.

The resulting study has **37 candidate features**:

- 13 observed chemistry measurements.
- 24 synthetic null controls.

The scientific question is deliberately modest: **which feature differences between two cultivars replicate in observations held aside before discovery?**

Do not confuse a significant feature difference with a causal mechanism. There is no gene perturbation, drug discovery, or experimental intervention in our lab.

## 9. A research question we can actually verify

We want to compare four systems.

**A budgeted generalist** investigates the eight largest preliminary signals. It provides a useful baseline representing limited analytical coverage.

**Eight specialists** investigate those same eight features independently, with the same statistical method. This is our equal-feature-budget control.

**Full serial specialists** investigate all 37 features, one after another, and produce a single verified report.

**Full parallel specialists** investigate all 37 with a thread pool. The statistical algorithm and data remain identical. Only scheduling changes.

We also run three ablations that remove independent verification, multiple-testing correction, or both.

The lab separates three questions that are often blurred together:

1. Does examining **more** of the dataset produce more independently replicated findings?
2. Does **parallel scheduling** preserve results while reducing wall-clock time?
3. Do **verification safeguards** reject claims that look convincing on discovery data but fail independent checks?

This design can answer all three without pretending a small statistical program is a biomedical research assistant.

## 10. Reproducible experimental design

We predefine the study before running comparisons.

The same seeds generate the same row splits and synthetic nulls. A stratified split allocates **70% to discovery** and **30% to independent holdout**. The discovery process may inspect only the discovery partition. It must not select features using holdout statistics.

Each selected feature receives a Welch two-sample test, an effect-size estimate, and a separate holdout calculation. We use Benjamini-Hochberg correction to address the multiple-comparison problem among the tests performed in each configuration. An accepted claim must pass the discovery threshold and replicate on holdout in the **same direction**.

This is an intentionally conservative educational acceptance rule, not a general scientific law. It can reject real effects when holdout samples are small, and using repeated significance thresholds is not a substitute for a prospective experimental validation.

We benchmark **five seeds**. The same feature rankings and data are used for the serial and parallel runs within a seed.

For the latency demonstration, each worker includes a configurable **20 millisecond artificial delay** to imitate a remote API or literature lookup. That delay is not an actual model call. We also keep a hypothetical token-cost ledger, clearly separated from observed execution and actual provider billing.

## 11. Load the public dataset and generate null controls

Create a Python environment with `numpy`, `scipy`, `scikit-learn`, and `matplotlib`. The complete source is in the companion repository.

The data loader starts like this:

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
keep = wine.target != 2

x = np.asarray(wine.data[keep], dtype=np.float64)
y = np.asarray(wine.target[keep], dtype=np.int64)

rng = np.random.default_rng(90000)
null_features = rng.normal(size=(len(x), 24))
x = np.concatenate([x, null_features], axis=1)

train_ids, holdout_ids = train_test_split(
    np.arange(len(y)),
    test_size=0.30,
    stratify=y,
    random_state=0,
)

x_train, y_train = x[train_ids], y[train_ids]
x_holdout, y_holdout = x[holdout_ids], y[holdout_ids]
```

The line that matters most is the split. It occurs before we compute and rank discoveries. If you pick an impressive feature using the entire dataset and only then run a supposed holdout check, your evaluation is already contaminated.

For an authentic biological screen, the split unit may need to be **mouse**, **experimental batch**, **donor**, or **perturbation family**, not individual cell. Millions of cells sampled from the same biological replicate do not become millions of statistically independent experimental units. This is the pseudoreplication problem, and a good scientific agent needs enough domain knowledge to notice it.

## 12. Implement a statistical investigator

Each investigator receives exactly one selected feature and returns discovery and holdout statistics.

```python
import numpy as np
from scipy.stats import ttest_ind


def effect_size(group_a, group_b):
    n_a, n_b = len(group_a), len(group_b)
    pooled = np.sqrt(
        ((n_a - 1) * group_a.var(ddof=1) +
         (n_b - 1) * group_b.var(ddof=1)) /
        (n_a + n_b - 2)
    )
    return float((group_a.mean() - group_b.mean()) / pooled)


def test_feature(x, labels, feature):
    a = x[labels == 0, feature]
    b = x[labels == 1, feature]
    p = float(ttest_ind(a, b, equal_var=False).pvalue)
    return p, effect_size(a, b)
```

Welch's test accommodates unequal population variances. The effect size gives a more interpretable measure of magnitude than a p-value alone. It still does not reveal a causal explanation.

In the full implementation, every result also records the feature name, partition sizes, seed, provenance string, and whether the feature is one of our synthetic controls. We preserve all derived measurements, feature identifiers, and test statistics in the emitted run JSON for auditability. The underlying sample matrices are regenerated deterministically from the dataset and seed.

## 13. Correct for multiple comparisons

Suppose you investigate one feature at a 5% false-positive threshold. Random error is already possible.

Now investigate hundreds or thousands. Some apparently significant results will appear even if many features have no real association. If an agent presents only its top surprising results, the reader may never see how many hypotheses were tested.

We use a Benjamini-Hochberg adjustment as a useful demonstration of **false discovery rate control**.

```python
import numpy as np


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()

    order = np.argsort(p)
    adjusted_sorted = np.minimum(
        1.0,
        p[order] * m / np.arange(1, m + 1)
    )
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted
```

The complete lab validates input values and reports both raw and adjusted p-values. [6]

A subtle point: standard BH guarantees depend on assumptions about the tested hypotheses and their dependence structure. It is not an automatic protection against arbitrary model-generated hypothesis fishing. Selecting features adaptively, retrying analyses, changing thresholds after reading the results, and searching a huge literature corpus can invalidate naive calculations.

For the lab, our feature selection uses discovery data only, and the holdout evidence remains separate. That separation is more important than giving every generated paragraph a confidence score.

## 14. Build a provenance-preserving orchestrator

The controller does not need to be intelligent in the conversational sense. It needs to guarantee that a task was executed on the correct data, that results have consistent fields, and that no feature was silently dropped.

The core pattern is:

```python
from concurrent.futures import ThreadPoolExecutor


def investigate_one(study, feature_index):
    # Real source also attaches provenance and a separate holdout test.
    return inspect_feature(study, feature_index)


def investigate_many(study, indices, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(
            lambda index: investigate_one(study, index),
            indices,
        ))
```

The important property is **deterministic task identity**. Each result must map back to one known feature and one known split.

In a production scientific workflow, you would also want idempotent job identifiers, retries with backoff, durable intermediate outputs, per-worker cost limits, rate limiting, error taxonomies, and the ability to resume an interrupted run without duplicating scientific findings.

The original distributed-agent repository goes beyond a simple Python thread pool. It provides corpus releases, research-analysis tooling, and an interface for querying the reasoning atlas. Its public MCP endpoints provide read-only corpus access, while recomputing raw statistics requires local access to the underlying dataset. [4]

## 15. Parallelize without changing the analysis

Concurrency is valuable when tasks are sufficiently independent and spend meaningful time waiting for remote services. It does not automatically improve the numerical results.

Our benchmark runs the **same 37 feature analyses** with one worker and with six workers. The only scheduling change is execution order and overlap. Tests assert that both variants return the same claims, p-values, and evidence records.

This is the right control. If the parallel version reports better scientific accuracy than the serial version despite identical analysis and evidence, something in the evaluation is changing besides parallelism.

The code uses threads because the simulated literature retrieval is I/O-like work. Actual CPU-heavy numerical analysis may benefit more from multiprocessing, vectorized NumPy/SciPy operations, job scheduling, or distributed compute. On a tiny in-memory dataset, thread overhead can easily erase any advantage if you remove the artificial remote latency.

A useful throughput metric is **completed verified investigations per wall-clock second**, not merely number of concurrent sessions.

## 16. Add independent verification

The verifier should be able to reject a good story.

For each candidate feature, it checks the held-out data and requires the effect direction to agree with discovery. A candidate passes our educational rule only when the independent test reaches the specified threshold and its estimated effect is not directionally reversed.

```python
candidate = adjusted_discovery_p < 0.05
holdout_ok = (
    holdout_p < 0.05
    and discovery_effect * holdout_effect > 0
)
accepted = candidate and holdout_ok
```

Importantly, the holdout evidence must be computed by a path the investigator cannot rewrite. In an LLM deployment, simply asking the same model to reassure itself that its earlier claim was correct is weak independence.

For scientific pipelines that operate on sensitive or regulated data, independent verification also means separate access controls, data lineage, review logs, and checks for unauthorized transformations.

## 17. Measure tokens and costs honestly

A multi-agent system can be faster and more expensive at the same time.

A serial workflow may spend one instruction prefix for a large job. Thirty-seven specialists may each consume another prompt, another tool response, and another generated report. The model provider charges for total inference work, not for how elegantly the orchestration diagram is drawn.

For our lab we record a transparent **proxy token ledger**. Each task has illustrative input and output token budgets, and distributed execution adds additional per-task overhead. We apply an explicitly hypothetical rate of **$3 per million proxy tokens**. It is not live GPT-Rosalind pricing, and no actual model tokens are generated by this lab.

The baseline eight-feature generalist consumes **3,870 proxy tokens** under these assumptions. The full 37-feature distributed run consumes **18,575 proxy tokens**, regardless of whether it runs sequentially or in parallel.

This difference is expected. The distributed system does substantially more coverage. The cost cannot be interpreted as an apples-to-apples comparison of model efficiency, because work volume is not fixed.

What would a fair model-provider benchmark require?

Record real input tokens, cache writes and reads, output tokens, tool retries, reasoning usage if exposed, wall-clock latency, concurrency, provider rate limits, and cost at the actual contracted prices. Then compare **verified leads per dollar**, **false leads per hundred suggestions**, **human review minutes per useful result**, and **time to a validated hypothesis**.

## 18. Reproduce the five-seed experiment

Download and extract the companion ZIP, then run:

```bash
cd distributed_science_agent_lab
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m science_lab.demo
python -m science_lab.benchmark \
  --seeds 5 \
  --null-features 24 \
  --workers 6 \
  --latency-ms 20 \
  --output results/benchmark.json
python plot_results.py
```

No model checkpoint, API key, GPU, or external download is required for the default lab. The dataset is included with scikit-learn. The repository contains saved benchmark output, regression tests, and original figures.

The benchmark creates seven configurations per seed, including limited-budget comparisons and ablations. It emits the full finding records instead of only the aggregate scores. This lets a reader ask whether a reported success survives a closer inspection of each accepted claim.

## 19. What the benchmark actually found

Across five deterministic seeds, the eight-feature generalist examined eight features and returned an average of **7.0 independently replicated associations**.

The eight-feature specialist control also returned **7.0**. That result matters. We should not claim a quality improvement from agent specialization when the same statistical algorithm is applied to the same eight features. The lab does not produce such an improvement.

When we expanded the search to all 37 features, the full specialist workflow returned an average of **8.6 independently replicated associations**. The additional gain came from **coverage**, not a magical improvement in model intelligence.

The serial run averaged **0.837 seconds** and the six-worker parallel run averaged **0.161 seconds**, a roughly **5.2-fold wall-clock improvement**. The actual statistical results were identical. The speedup depended on our 20 millisecond per-task artificial I/O delay, so it is not a prediction of GPT-Rosalind throughput.

The same full workflow used the same **18,575 proxy tokens** in serial and parallel variants. Parallelism changed how long the work took, not how many task reports it required.

The cost-efficiency comparison is less flattering to exhaustive coverage. The eight-feature baseline produced about **1.81 replicated associations per 1,000 proxy tokens**, versus **0.46 per 1,000** for the full distributed run. A more comprehensive search found additional signals, but each extra signal cost more in this constructed scenario. That is precisely why cost-per-verified-finding belongs next to total discovery count.

That is a much narrower conclusion than the headline, but also a more dependable one: **partitioning increases coverage when the per-task budget is constrained, and scheduling makes independent I/O-bound investigations finish sooner.**

## 20. Ablation 1: fewer investigations

The generalist and equal-budget specialist configurations each examined eight features selected using only discovery data. Both returned the same average number of replicated associations, **7.0**.

Full coverage increased that count to **8.6**. We gained approximately 1.6 additional associations per run while examining 29 additional candidate features.

Does that justify the extra work? It depends on what a verified association is worth, how much the full run costs, and whether the extra findings are scientifically useful. For a low-cost diagnostic feature screen, exhaustive analysis might be reasonable. For an expensive agent that retrieves and analyzes dozens of papers per feature, you might prefer a priority queue.

A practical alternative is **adaptive allocation**: give every candidate a cheap first pass, then spend larger research budgets on high-uncertainty, high-novelty, or high-impact findings. Crucially, those policies must be evaluated using held-out research tasks so that adaptive search does not simply reward overfitting to the scoring function.

## 21. Ablation 2: parallel versus serial

The serial full-coverage pipeline averaged **0.837 seconds**. Six-way parallel scheduling averaged **0.161 seconds**.

The two modes produced exactly the same accepted findings, since they used identical calculations and input partitions.

This isolates the value of concurrency: elapsed time. It does not show that six workers are inherently better scientists than one. In a production system the overhead can reverse the result, especially when tasks are too small, a provider has tight rate limits, retrieval endpoints serialize access, or thousands of requests cause throttling.

We ran an additional **three-seed, zero-artificial-latency ablation**. With the 20 millisecond delay removed, serial execution averaged **0.050 seconds** and parallel execution averaged **0.071 seconds**. The thread pool became slower because there was almost no waiting time to overlap. These microsecond-scale measurements are machine-dependent, but the direction of the tradeoff is instructive.

A useful benchmark should therefore report a latency curve for one, two, four, eight, and more workers. Stop adding concurrency when throughput plateaus or quality drops. Do not optimize for the most dramatic number of launched agents.

## 22. Ablation 3: remove the holdout verifier

This is where the scientific danger becomes visible.

With independent verification enabled, the full pipeline accepted **8.6 claims on average**, all of which met the lab's holdout replication rule. With verification disabled, it accepted **11.0**, but only **8.6** replicated independently.

That is an average of **2.4 additional accepted claims per run that failed the independent replication rule**. Some are real effects with insufficient holdout power; others may be sampling artifacts. Our experiment cannot label all failures as false in an absolute scientific sense.

Among the accepted unverified claims, **0.6 per run** involved deliberately generated null features. That is a more concrete warning because those columns were generated to be unrelated to the label.

A system that reports only how many interesting findings it generated would score the unverified run higher. A system that measures **supported and independently replicated findings** would not.

## 23. Ablation 4: remove multiple-testing correction

Removing Benjamini-Hochberg correction while retaining independent verification increased the average number of replicated accepted claims from **8.6 to 9.2** in this sample, without admitting synthetic nulls among accepted claims across these five runs.

That is not proof that correction is unnecessary. It is a result from one small benchmark where a holdout filter also operates. Multiple-testing adjustments are especially important as the hypothesis family expands, tests become correlated, or the system optimizes its prompt toward impressive-looking findings.

When we removed **both** correction and verification, the pipeline accepted **12.6 claims per run**, but only **9.2** passed the holdout rule. It also accepted an average of **1.6 synthetic null features**.

The damage is not primarily that the model creates bad text. It is that the **decision rule is no longer aligned with evidence quality**.

A serious scientific agent must be allowed to say, *the data are insufficient to support this claim*. An expensive agent with a strong incentive to produce novelty may do exactly the opposite unless the evaluation rewards appropriate uncertainty.

## 24. Why more citations can produce worse science

The original distributed reasoning study generated many thousands of references and citations. That scale is meaningful because literature retrieval can prevent agents from reinventing existing findings. But citation count is not a quality measure by itself.

Imagine three reports:

- A report cites one highly relevant experiment and describes its limitations accurately.
- Another cites twenty reviews but makes a claim none of them directly supports.
- A third cites five primary studies, all derived from the same underlying dataset, then presents them as five independent replications.

Which report is better? Counting references cannot answer.

A scientific evidence system needs claim-level provenance. Every citation should be linked to the exact claim it allegedly supports, with information about experimental system, sample type, intervention, measurement, population, and known limitations.

For a useful benchmark, evaluate citation **precision**, not simply citation count. Sample claims and ask domain experts to decide whether their references genuinely support them. Penalize invented citations, irrelevant references, exaggerated conclusions, and unsupported cross-species extrapolation.

The original study explicitly distinguished agreement with the literature, inference, disagreement, and absence of retrieved precedent. That is an excellent practice because it prevents the system from pretending every observation has the same evidential status. [2]

## 25. How to evaluate actual LLM-based research agents

Our statistical workers isolate orchestration, but real language-model investigators introduce new failure modes. They can invent biological mechanisms, misuse a tool, read an irrelevant paper, or generate plausible explanations for chance correlations.

A stronger benchmark should keep the raw scientific question fixed and compare several configurations:

**Single agent, equal total inference budget.** Let one capable model allocate the entire budget across tasks.

**Distributed agents, equal total budget.** Divide that same amount of spend across independent, perturbation-specific investigations.

**Distributed agents, equal per-task budget.** Allow the total distributed budget to grow and measure whether additional coverage justifies the cost.

**Distributed agents with independent evidence verification.** Add statistical checks and citation audits whose results cannot be overwritten by the agents producing the claims.

**Distributed agents with selective escalation.** Use inexpensive workers for obvious or low-signal cases and larger models for complex, ambiguous questions.

Each experiment should report the same outcome metrics: independently correct findings, known-null discoveries, citation support, novel hypotheses preferred by blinded experts, failed or unsupported claims, total spend, latency, and human review effort.

Randomize the order in which reports are evaluated. Blind reviewers to whether text came from a generalist or distributed system. Treat repeated judgments of the same report as statistically dependent, rather than counting them as independent new discoveries.

For prospective validation, register a subset of hypotheses *before* new biological experiments are conducted. The strongest test of a scientific assistant is whether it predicts genuinely new measurements that later hold up.

## 26. Statistical and biological failure modes

There are several ways an impressive agent demo can mislead even technically sophisticated readers.

**Pseudoreplication.** Cells from a small number of mice may not be independent experimental units. A statistical test that counts every cell as an independent animal can dramatically overstate certainty.

**Batch effects.** The agent may interpret a sequencing batch, processing-day effect, or sample-preparation difference as a perturbation response.

**Cell-type composition changes.** Apparent changes in gene expression may arise because the fraction of cell types changes, rather than because each cell changes its transcriptional program.

**Selection bias.** The system may search thousands of possible links but disclose only the most impressive, without accounting for the search process.

**Citation laundering.** An agent may convert speculation in a review paper into a confident factual mechanism through repeated paraphrasing.

**Correlated agent errors.** Spawning more copies of the same model does not guarantee independent evidence. All agents may share pretraining priors, retrieval ranking biases, and identical reasoning mistakes.

**Data leakage.** A source publication, held-out result, or benchmark answer can slip into retrieval and contaminate the evaluation of apparent novelty.

**Human preference bias.** Reviewers may prefer fluent, detailed explanations even when the underlying evidence is weaker. Human evaluation needs carefully designed rubrics and factual audits.

**Conflating perturbation with mechanism.** Disrupting one gene and observing a response is a causal intervention on the gene, but the pathway linking that disruption to a downstream phenotype may remain uncertain. A plausible explanatory graph is not proof of every intermediate edge.

Each of these problems becomes more expensive when one system can generate thousands of confident reports in minutes.

## 27. From correlations to follow-up experiments

The Dctn1 example is a good model of what an AI-generated scientific lead should look like.

The original atlas observed that disrupting Dctn1 affected neuronal cholesterol-biosynthesis transcription. The distributed agents connected this with literature about intracellular transport and cholesterol distribution. They proposed a question about whether changed intracellular cholesterol delivery might trigger a compensatory response. [1, 2]

The correct output of that reasoning is **an experiment plan**, not the announcement of a confirmed treatment.

A rigorous next step would specify the expected observable differences under competing explanations, use independent biological replicates, control for cell-type composition, and define measurements that could falsify the hypothesis. The research team, not a generic agent, should determine whether the biological context and safety requirements make those experiments appropriate.

This is where scientific agents can be most useful: identifying a precise uncertainty and suggesting the *kind* of evidence needed to resolve it.

## 28. When should you use thousands of agents?

Use many workers when tasks are naturally separable, evidence can be normalized afterward, and each investigation has enough work to amortize its context and orchestration overhead.

Do not use thousands of workers simply because the infrastructure allows it. For a ten-feature dataset on one laptop, a vectorized statistical script is faster, cheaper, and easier to audit than a large language-model swarm.

The value proposition changes when each of thousands of perturbations requires literature search, context-dependent reasoning, numerical analysis, and a structured interpretation. Then specialized investigation can extend analytical coverage while the global synthesis layer maintains the connections.

Even there, the most useful optimization target is often **a good routing policy**. Easy findings can receive inexpensive checks. Ambiguous or high-impact cases can receive deeper analysis. Scientific uncertainty should drive compute allocation, not just whether a report sounds interesting.

## 29. How to extend the workshop to biological data

There are two responsible paths.

The first is to replace the Wine dataset with a **small, public, fully de-identified summary-level dataset**, such as publicly released differential-expression contrasts that include experimental replicates and the relevant metadata. Preserve donor- or animal-level splits when required by the study design. Adapt the worker to run the appropriate statistical model rather than treating expression measurements as generic independent samples.

The second is to use the **original authors' released repository**. It contains a hosted reasoning corpus, a read-only MCP querying interface, manuscript analysis scripts, and directions for acquiring the full prepared atlas. As documented in the repository, the full data pipeline involves large downloads and local storage. [4]

These paths answer different questions. The small public-data workshop teaches engineering principles. Running the original scripts helps verify the published analysis. Neither substitutes for prospective experimental validation.

If you bring a real language model into the loop, keep raw data and statistical verification under trusted code. The language model can propose hypotheses, translate concepts, and help interpret the outputs. Let a separate, reviewable pipeline decide what the data actually show.

## 30. Reproducibility, limitations, and closing argument

Our project uses five seeds, a public 130-sample chemistry classification subset, 24 synthetic null features, 37 feature investigations, stratified discovery and holdout partitions, and deterministic Welch-test workers. It includes 17 unit tests, independently saved results, and three generated figures.

It does **not** use GPT-Rosalind, call any LLM endpoint, retrieve actual scientific papers, or process millions of brain cells. The simulated I/O delay is an experimental control. The proxy token count is not actual usage. Statistical replication on the held-out Wine samples is not biological discovery.

The measured takeaway is narrower but actionable. When a generalist can examine only eight candidate signals, expanding to independent full-coverage investigators can find additional replicated associations. When workers wait for remote services, parallelism can reduce elapsed time without altering evidence. When we remove verification, more reports pass but more are unsupported.

That last point is the entire engineering challenge of AI-driven science.

A system that generates 10,000 scientific claims in half an hour is not necessarily better than one that generates ten. It becomes useful when the evidence attached to those claims lets researchers **discard weak ideas early, identify strong ones efficiently, and choose better experiments**.

The important milestone is not a swarm of agents typing at extraordinary speed. It is an infrastructure where the throughput of scientific interpretation rises **without allowing unsupported conclusions to rise with it**.

---

## 31. References and research resources

**[1]** OpenAI. *Using thousands of AI agents to uncover new leads in biological data.* October 21, 2026. https://openai.com/index/scaling-ai-research-for-complex-biological-data/

**[2]** Sun, A., Park, J. J., DeTomaso, D., et al. *Distributed AI agents enable comprehensive discovery of biological insights at scale.* bioRxiv, 2026. Preprint, not yet peer reviewed. https://doi.org/10.64898/2026.10.15.759436

**[3]** PerturbAI. *PerturbAI and OpenAI Unveil First-of-Its-Kind AI Reasoning Agents to Scale Biomedical Discovery.* October 21, 2026. https://perturb.ai/news/perturbai-launches-ai-reasoning-agents

**[4]** Jin Lab Neurogenomics. *Distributed Agents: Open-source research framework, released reasoning atlas and manuscript-analysis materials.* https://github.com/jinlabneurogenomics/distributed_agent

**[5]** scikit-learn. *Wine recognition dataset documentation.* https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html

**[6]** Benjamini, Y., and Hochberg, Y. *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.* Journal of the Royal Statistical Society, Series B, 1995. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

**[7]** OpenAI. *Introducing new capabilities to GPT-Rosalind.* June 3, 2026, with subsequent availability update. https://openai.com/index/introducing-new-capabilities-to-gpt-rosalind/

**[8]** PerturbAI. *Whole-brain CRISPR atlas dataset.* Public data listing and access information. https://huggingface.co/datasets/perturbai/wholebrain_crispr_atlas

**[9]** scikit-learn. *Stratified train-test split and reproducibility utilities.* https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

**[10]** SciPy. *Welch's two-sample t-test in scipy.stats.ttest_ind.* https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

**[11]** Tiwari, K., et al. *Genetic Perturbation Prediction Models Fail Out-of-Distribution: A Rigorous Benchmark and Systematic Evaluation.* bioRxiv preprint, October 2026. https://doi.org/10.64898/2026.10.16.759604

**[12]** *Evaluating agentic AI for biological discovery in autonomous and copilot settings.* bioRxiv preprint, June 2026. https://doi.org/10.64898/2026.06.04.729919

---

*Project note: The accompanying GitHub-ready Python lab is an independent educational implementation. It is not affiliated with or endorsed by OpenAI, PerturbAI, Scripps Research, or the original paper's authors.*
