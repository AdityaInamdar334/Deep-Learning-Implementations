# The AI Told the Truth. The Benchmark Marked It Wrong.
## How Evaluation Defects Distort AI Progress: An Investigation into SciCode, SWE-bench, Humanity's Last Exam, and the Rebuilt Measurement Layer

> *Imagine spending months training a better language model.*
>
> *You improve its architecture, optimize its inference pipeline, and spend thousands of GPU hours refining its reasoning. Then you submit it to a respected scientific coding benchmark.*
>
> *It scores 60%.*
>
> *Your predecessor scored 60%, too.*
>
> *The leaderboard suggests you have made no progress. Perhaps the architecture has reached its limits. Perhaps scientific reasoning requires a fundamentally different approach.*
>
> *But what if the models were never the problem?*
>
> *What if the benchmark was rejecting correct solutions because its reference answers were wrong, its numerical tolerances were unreasonable, and its instructions contradicted its own tests?*

---

## Executive Summary

In August 2026, researchers published exactly that finding: an audit of **SciCode** identified 263 defects across 65 test problems. After correcting the benchmark and re-evaluating twelve frontier-model snapshots under matched conditions, reported subproblem accuracy rose from **45–60% to 84–98%**. Whole-problem accuracy skyrocketed from **9–27% to 69–92%** [1].

**No model needed to be retrained.** The researchers changed the measuring instrument.

SciCode was not an isolated case. During 2026, systematic investigations uncovered substantial scoring defects in **Humanity's Last Exam (HLE)** (46% flawed in Epoch's stratified sample), **SWE-bench Verified** (59.4% defect rate in audited failed tasks), **SWE-bench Pro** (~30% breaking task issues), and **GPQA Diamond**. Concurrently, research on **DeepQuestion** showed that increasing cognitive complexity via Bloom's taxonomy can cause model accuracy to decline by up to 70%, even on bug-free evaluations [10]. By October 27, 2026, Artificial Analysis removed SciCode from its Intelligence Index and replaced several foundational components in Index v5 [12].

This is not evidence that benchmarks are useless. It is evidence that **benchmark quality is itself a primary variable in AI measurement**, and one that researchers, engineers, and decision-makers have often underestimated.

---

## Table of Contents

1. [The Hidden Assumption Behind Every AI Leaderboard](#1-the-hidden-assumption-behind-every-ai-leaderboard)
2. [SciCode: The Clearest Example of a Broken Measurement Instrument](#2-scicode-the-clearest-example-of-a-broken-measurement-instrument)
3. [Why Correct Code Can Fail a Benchmark](#3-why-correct-code-can-fail-a-benchmark)
4. [Humanity's Last Exam: When the Answer Key Is Wrong](#4-humanitys-last-exam-when-the-answer-key-is-wrong)
5. [GPQA Diamond: Separating Evidence From Disputed Claims](#5-gpqa-diamond-separating-evidence-from-disputed-claims)
6. [SWE-bench: How Two Generations of Coding Evaluations Developed Serious Problems](#6-swe-bench-how-two-generations-of-coding-evaluations-developed-serious-problems)
7. [DeepQuestion: What Happens When the Benchmark Is Too Easy?](#7-deepquestion-what-happens-when-the-benchmark-is-too-easy)
8. [The New Benchmark Auditing Process](#8-the-new-benchmark-auditing-process)
9. [Artificial Analysis Index v5: Rebuilding the Measurement Layer](#9-artificial-analysis-index-v5-rebuilding-the-measurement-layer)
10. [How to Evaluate AI Models Without Repeating These Mistakes](#10-how-to-evaluate-ai-models-without-repeating-these-mistakes)
11. [What These Findings Actually Mean for AI Progress](#11-what-these-findings-actually-mean-for-ai-progress)
12. [References and Further Reading](#12-references-and-further-reading)

---

## 1. The Hidden Assumption Behind Every AI Leaderboard

A benchmark score looks like an objective number.

A language model answers 100 questions. It gets 83 correct. We report 83% accuracy. Simple.

But that number quietly depends on several implicit assumptions:
1. The questions are mathematically well-defined.
2. The reference answers are correct.
3. The scoring system accepts all valid solutions.
4. The scoring system rejects invalid solutions.
5. The model has access to the tools and resources the task requires.
6. The evaluation data has not already appeared in the model's training material.
7. The benchmark actually measures the capability we care about.

Break any one of these assumptions, and the interpretation of the score collapses.

Suppose a model correctly solves a physics problem, but the benchmark's answer key contains an incorrect physical constant. The model receives **zero credit**. Suppose another model reproduces the mistaken constant because it encountered the question during training. That model receives **full credit**. The leaderboard now rewards familiarity with an erroneous answer over independent scientific reasoning.

### Three Ways a Benchmark Can Mislead Us

* **False negatives (correct work rejected):** The benchmark marks a valid answer as incorrect due to an erroneous reference answer, overly strict tests, missing assumptions, or an excessively narrow output format.
* **False positives (incorrect work accepted):** The benchmark gives credit to an incomplete or invalid solution due to poor test coverage, answer leakage, or exploitable grading mechanisms.
* **Construct mismatch (measuring the wrong capability):** A task accurately measures memorized trivia while being presented as a test of scientific reasoning. A coding evaluation measures how well an agent satisfies a fragile test suite rather than whether it produces maintainable software.

```
       [Raw Model Capability]
                 |
                 v
      +--------------------+
      |  Candidate Answer  |
      +--------------------+
                 |
                 v
      +--------------------+
      | Measurement Layer  | <--- Vulnerability Surface:
      |   - Test harness   |      * Broken reference answers
      |   - Tolerances     |      * Inconsistent conventions
      |   - Grader logic   |      * Undocumented constraints
      +--------------------+
                 |
                 v
        [Reported Metric]  <--- May diverge completely from capability!
```

---

## 2. SciCode: The Clearest Example of a Broken Measurement Instrument

SciCode was designed to measure scientific coding abilities. Unlike ordinary programming benchmarks, its tasks require models to combine scientific domain understanding with executable numerical implementations: mathematical derivations, physical simulations, and multi-step calculations.

### The Suspicious Plateau

According to the SciCode-Verified paper, leading models in 2026 had clustered around a 60% subproblem accuracy plateau. Successor models matched their predecessors rather than showing clear scaling improvements.

The researchers investigated: *what if correct solutions were systematically failing the tests?*

Their domain-expert audit covered all 65 original test problems and identified **263 defects**. Of those, **192 caused correct, instruction-following solutions to be rejected** [1]. The paper reports that these score-suppressing defects affected **91% of the main problems**, and **78% of them required specialized physics or mathematics knowledge to detect**.

### What Happened After Correction?

| Metric | Original SciCode | SciCode-Verified | Delta |
| :--- | :---: | :---: | :---: |
| **Subproblem Accuracy** | 45–60% | **84–98%** | +38–39 pp |
| **Whole-Problem Accuracy** | 9–27% | **69–92%** | **+60–65 pp** |
| **Model Snapshots Evaluated** | 12 | 12 (matched) | — |
| **Model Retraining Required** | — | **Zero** | Pure measurement fix |

### Why Whole-Problem Scores Collapsed

SciCode tasks contain multi-step dependency graphs where later steps depend on earlier outputs. A whole problem receives credit only when **all** its scored components pass:

$$\text{WholeProblemCredit} = \prod_{k=1}^{K} \mathbb{I}\left(\text{Pass}(S_k)\right)$$

If Step 5 contains an unstated physical convention or overly strict float tolerance, Step 5 fails. Even if the model correctly reasons through Steps 1–4 and Step 6, the entire multi-step problem receives zero credit. Cascading failures artificially suppressed whole-problem accuracy from 69–92% down to 9–27%.

---

## 3. Why Correct Code Can Fail a Benchmark

Automated grading follows a sequential pipeline:
$$\text{Task} \longrightarrow \text{Generate Code} \longrightarrow \text{Sandbox Exec} \longrightarrow \text{Compare Output} \longrightarrow \text{Assign Credit}$$

The breakdown occurs at the comparison stage:

### Problem A: Overly Strict & Undocumented Constraints
An instruction requests: *"Write a function that returns unique integers in sorted order."*
```python
def solve(values):
    return sorted(set(values))
```
A hidden grader requires:
```python
assert "deduplicate_and_sort" in globals()  # Rejection of valid code!
```

### Problem B: Incorrect Numerical Tolerances
In numerical simulation, machine precision and algorithm differences produce slight variations:
$$\text{Expected} = 0.665132, \quad \text{Actual} = 0.665031$$
```python
np.allclose(actual, expected, rtol=1e-5, atol=1e-8)  # Rejection!
```
Unless numerical conditioning is validated against task physics, arbitrary tolerances discard correct analytical implementations.

### Problem C: Defective Reference Implementation
When the benchmark author's reference code contains a bug, that bug generates the ground truth. A model that implements the physical law correctly is marked wrong; a model that mimics the buggy code gets full credit.

---

## 4. Humanity's Last Exam: When the Answer Key Is Wrong

Humanity's Last Exam (HLE) was curated to test frontier models on PhD-level questions across academic disciplines.

On August 11, 2026, Epoch AI published a review of HLE. Reviewers stratified 48 questions (six from each of eight subject categories):
* **22 of the 48 reviewed questions (~46%) contained substantial errors** capable of altering the score [3].
* 12 questions were mathematically impossible to answer as written (e.g. asking for a 4-point Discrete Fourier Transform on an 8-element sequence).
* 10 questions rejected valid answers due to contradictory explanations (e.g. calculating $1 / 21.3535$ in the derivation, but listing $21.35$ in the answer key).
* Epoch classified the reviewed version as **Flawed**.

Independent projects (such as HLE-Verified [6]) introduced structured repair pipelines, reporting average accuracy gains of 7–10 percentage points once defective questions were repaired.

---

## 5. GPQA Diamond: Separating Evidence From Disputed Claims

GPQA Diamond comprises 198 graduate-level science questions.

* **Forensic lower bounds:** Sammy Zeng reported an estimated 26.8% inherent error lower bound [4], though replication efforts noted that item-level verification scripts were private [5].
* **Cleaned subset releases:** The GPQA Diamond Clean project excluded 9 audited defective questions, retaining 189 items [7], shifting model rankings.

The lesson: claims of benchmark defect rates must be accompanied by item-level defect ledgers, explicit sampling methodologies, and reproducible adjudication criteria.

---

## 6. SWE-bench: How Two Generations of Coding Evaluations Developed Serious Problems

SWE-bench evaluates whether an autonomous agent can resolve GitHub issues by applying patches to real repositories.

### February 2026: SWE-bench Verified Scrutiny
OpenAI audited 138 difficult tasks in SWE-bench Verified:
* **59.4% of the audited tasks contained material defects** (overly specific tests, unstated requirements, or environment failures) [8].
* **Data Contamination:** Every frontier model tested exhibited evidence of having memorized historical PRs or issues from public GitHub training corpora.

### July 2026: SWE-bench Pro Audit
OpenAI recommended SWE-bench Pro as a harder, contamination-resistant alternative. When pass rates climbed from 23.3% to 80.3% in eight months, a second audit of 731 tasks flagged **200–249 broken tasks (~30%)**, prompting OpenAI to withdraw its recommendation [9].

---

## 7. DeepQuestion: What Happens When the Benchmark Is Too Easy?

Even if a benchmark is 100% bug-free and uncontaminated, it can still fail if its questions operate at low cognitive depth.

Khoramfar et al. (*DeepQuestion*, 2026) systematically elevated benchmark questions using **Bloom's Taxonomy**:

```
              [ Level 4: Synthesis & Evaluation ]  <-- Steep Accuracy Drop (~70% drop)
                     [ Level 3: Analysis ]
                  [ Level 2: Application ]
                    [ Level 1: Recall ]            <-- Saturated (~90%+ Accuracy)
```

Across 10 leading models, accuracy dropped by up to 70% as tasks moved from simple recall to multi-variable evaluation and synthesis [10]. A model scoring 95% on recall-heavy benchmarks may fail completely when asked to design an experimental protocol under real-world constraints.

---

## 8. The New Benchmark Auditing Process

Epoch AI established a structured public Benchmark Review registry with three formal verdicts [11]:

```
                     +----------------------------+
                     |  Benchmark Review Request  |
                     +----------------------------+
                                   |
                                   v
                     +----------------------------+
                     |  Inspectable Data & Rules? |
                     +----------------------------+
                              /          \
                        [No] /            \ [Yes]
                            v              v
                  +-----------------+    +-------------------------+
                  | Not Enough Info |    | Stratified Task Audit   |
                  +-----------------+    +-------------------------+
                                                      |
                                    +-----------------+-----------------+
                                    | Defect Rate >= 20% in Sample?     |
                                    +-----------------+-----------------+
                                           /                     \
                                     [Yes]/                       \[No]
                                         v                         v
                                +-----------------+       +-----------------+
                                |     FLAWED      |       |    VERIFIED     |
                                +-----------------+       +-----------------+
```

As of November 2026, out of 26 reviewed benchmark suites in Epoch's registry:
* **5 Verified** (~19.2%)
* **13 Flawed** (~50.0%)
* **8 Not enough info** (~30.8%)

---

## 9. Artificial Analysis Index v5: Rebuilding the Measurement Layer

On October 27, 2026, Artificial Analysis launched **Intelligence Index v5** [12]:
* **60% private test sets or held-out answers** to combat training data contamination.
* **Removed SciCode entirely** (previously 10% of Index weight) following defect disclosures.
* **Replaced HLE with HLE-Diamond** (1,000 refined reasoning and knowledge questions).
* **Introduced FrontierCode 2.0:** Evaluates whether a maintainer would merge the proposed code, rather than fragile unit-test assertions alone.
* **Introduced Terminal-Bench-Science:** Real multi-step command-line scientific workflows.

---

## 10. How to Evaluate AI Models Without Repeating These Mistakes

1. **Define the capability before choosing the benchmark.**
2. **Inspect the benchmark's construction** (author expertise, tolerance justifications).
3. **Separate model capability from evaluation setup** (tooling budgets, scaffoldings).
4. **Audit surprising failures and suspicious successes.**
5. **Report uncertainty (confidence intervals, paired tests) rather than raw point accuracy.**
6. **Build an internal workflow evaluation set** with verified domain invariants.
7. **Maintain a versioned defect ledger and evaluation audit trail.**
8. **Evaluate the evaluator:** inject synthetic perturbations to prove grader validity.

---

## 11. What These Findings Actually Mean for AI Progress

When a language model fails a benchmark, there are two distinct hypotheses:
1. *The model's reasoning or implementation is flawed.*
2. *The benchmark's question, reference answer, tolerance, or test harness is flawed.*

For years, the machine learning community defaulted to assuming the first hypothesis. The benchmark audits of 2026 demonstrated how costly that assumption was. As models become more capable, evaluation engineering—building, auditing, and maintaining trustworthy measuring instruments—is becoming the true gating factor for AI progress.

---

## 12. References and Further Reading

1. **Hu, S., Huang, L., Deng, Y., & Chen, K. (2026).** *SciCode-Verified: How Benchmark Defects Underestimated the Scientific-Coding Ability of Language Models.* arXiv:2608.04975. [https://arxiv.org/abs/2608.04975](https://arxiv.org/abs/2608.04975)
2. **SciCode-Verified Research Repository.** Public benchmark release, audit ledger, and evaluation protocol. [https://github.com/flyingwagner/scicode-verified](https://github.com/flyingwagner/scicode-verified)
3. **Epoch AI (2026).** *Humanity's Last Exam: Benchmark Review.* August 11, 2026. [https://epoch.ai/benchmarks/hle/review](https://epoch.ai/benchmarks/hle/review)
4. **Zeng, S. (2026).** *Humanity's Last Hallucination: A Forensic Audit of the Scientific Insolvency in GPQA and HLE.* Zenodo record 18293568. [https://zenodo.org/records/18293568](https://zenodo.org/records/18293568)
5. **Independent Verification of GPQA and HLE Error Claims.** [https://github.com/lhl/hle-gpqa-error-claims](https://github.com/lhl/hle-gpqa-error-claims)
6. **Zhai, W., et al. (2026).** *HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam.* arXiv:2602.13964. [https://arxiv.org/abs/2602.13964](https://arxiv.org/abs/2602.13964)
7. **Allcock, A. (2026).** *GPQA Diamond Clean and When the Answer Key Is Wrong.* [https://gpqa.ai/](https://gpqa.ai/)
8. **OpenAI (2026).** *Why SWE-bench Verified No Longer Measures Frontier Coding Capabilities.* February 23, 2026. [https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
9. **OpenAI (2026).** *Separating Signal From Noise in Coding Evaluations.* July 8, 2026. [https://openai.com/index/separating-signal-from-noise-coding-evaluations/](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)
10. **Khoramfar, A., et al. (2025, rev. 2026).** *DeepQuestion: Systematic Generation of Real-World Challenges for Evaluating LLMs Performance.* arXiv:2505.24532. [https://arxiv.org/abs/2505.24532](https://arxiv.org/abs/2505.24532)
11. **Epoch AI (2026).** *Benchmark Reviews: Methodology and Included Benchmarks.* [https://epoch.ai/data/benchmark-reviews-documentation/methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology)
12. **Artificial Analysis (2026).** *Announcing Artificial Analysis Intelligence Index v5.* October 27, 2026. [https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5)
13. **Epoch AI (2026).** *Terminal-Bench 4.0.0: Benchmark Review.* September 4, 2026. [https://epoch.ai/benchmarks/terminal-bench-4/review](https://epoch.ai/benchmarks/terminal-bench-4/review)

