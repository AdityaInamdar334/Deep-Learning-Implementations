# Reproducibility and Interpretation Notes

## What this experiment demonstrates

The experiment shows that direct, no-gradient modification of neural weights can correct a deliberately constructed network error, but that function-preserving parameter symmetries undermine naive weight interpretation and that a simple magnitude-pruning method can be just as effective as a semantic repair in some settings.

**This is not an LLM benchmark and not an official reproduction of MIT CSAIL's November 2026 Weight-Space Programming results.**

## Conditions

- Network: 4 input dimensions, 16 ReLU hidden units, 3 action logits, exactly 131 parameters including zero-valued biases.
- Three input task coordinates considered in the reading experiment: 0, 1, 2; coordinate 3 is reserved as nuisance.
- Task label: action 0 if the task coordinate is positive, otherwise action 1.
- Corruption: output connections from positive and negative nuisance ReLU units to action index 2 have weight 4.0 each; productive connections have weight 2.0.
- `semantic` repair: zero exactly the two nuisance-to-stall output connections. The editor is provided the designed semantics.
- `top_magnitude`: zero two largest-magnitude output weights, a straightforward non-LLM baseline.
- `random_two`: select two nonzero output weights uniformly at random and zero them; the edit budget matches the semantic repair's two actual changed parameters.
- `half_semantic`: zero only one of the two nuisance-to-stall output connections.
- `global_shrink`: multiply the entire output matrix by 0.5. This changes four nonzero weights and should preserve argmax, illustrating that parameter movement need not improve behavior.
- `wrong_path`: zero both productive output connections, showing that small edits can be catastrophic.
- `original`: retain the corrupted network without modification.

## Randomness and evaluation

`--seeds 5` iterates seeds 0 to 4. No training loop or gradient step is executed. For each seed and edit policy:

- Familiar test: 3,000 independent 4D Gaussian observations.
- Shifted test: 3,000 independent observations with coordinate 3 multiplied by 2.5. Labels continue to depend exclusively on coordinate 0.
- Navigation test: 300 episodes, starting positions sampled uniformly between -1.8 and 1.8, 14 maximum action selections, 0.2 step length, target tolerance 0.12. The nuisance coordinate is independently resampled as Gaussian noise at each step.
- The editing methods cannot read the labels or the evaluator-generated input samples before selecting an edit.

Reading experiment: for each of three task coordinates, 60 randomly selected permutations of 16 hidden neurons per seed are applied exactly to the incoming rows, biases, and outgoing columns. The network's function is unchanged by construction. That produces 900 variants over five seeds.

The `permutation_invariant` reader sums absolute first-layer weights across all hidden neurons separately for each input coordinate. This specifically handles neuron permutations in this constructed network. It does not solve weight interpretation under arbitrary hidden-space transformations.

## Results: five-seed means

No Markdown tables are included so the notes can also be copied into a Medium article.

**Readers:** first-neuron heuristic 38.111%; permutation-invariant aggregate 100.000%.

**Corrupted:** familiar 29.83%; shifted 12.31%; navigation 30.00%; edited parameters 0.

**Semantic:** familiar 100%; shifted 100%; navigation 100%; edited parameters 2.

**Top magnitude:** familiar 100%; shifted 100%; navigation 100%; edited parameters 2.

**Half semantic:** familiar 64.83%; shifted 55.88%; navigation 95.87%; edited parameters 1.

**Random two nonzero:** familiar 37.97%; shifted 35.81%; navigation 50.80%; edited parameters 2.

**Global shrink:** familiar 29.83%; shifted 12.31%; navigation 30.00%; edited parameters 4.

**Wrong pathway:** familiar 0%; shifted 0%; navigation 7.00%; edited parameters 2.

This result is largely engineered by design: the correct two weights are known, and those two weights are the largest ones. A realistic benchmark must use learned networks and blinded edit targets.

## Interpretation of perfect accuracy

Perfect holdout accuracy for the repaired network is not surprising. Its logits are a hand-designed, deterministic ReLU implementation of the correct target decision rule. No generalization to novel semantic tasks is involved.

The shift evaluates nuisance robustness, but not an adversarially chosen shift, a new input modality, a realistic robotics dynamics model, or language-model decoding behavior.

## Projection demonstration

`weightlab/projection.py` performs a selected-column, low-rank output-space projection. The tests verify orthogonality after `scale=1.0`, preservation of untouched columns, non-mutation of the input matrix, and reversal/amplification at `scale=6.0`.

The projection method is structurally related to targeted output-direction weight edits, but its random-tensor tests do not establish anything about real Qwen direction choice or mechanistic correctness.

## Source versus independent results

MIT CSAIL reports improvements in actual Qwen3.5-0.8B benchmark scores after a targeted edit discovered by GPT-6 Astra, with a restricted external Python harness and a forward-pass budget. Those are **source-reported results**.

Our synthetic five-seed classification and navigation scores are **independent lab results** and must never be described as measurements of Astra, Qwen3.5, MinAtar, or PPO.

A separate technical auditor examined the released Qwen checkpoint and found 130,188 bitwise-different tensor elements among 131,072 addressed locations, due in part to finite-precision rounding. Our lab's two-weight change is audited using exact numeric comparisons at PyTorch tensor precision, which is conceptually similar but far smaller and simpler.

## Commands

```bash
python -m unittest discover -s tests -v
python -m weightlab.demo
python -m weightlab.benchmark --seeds 5 --readings 60 --samples 3000 --episodes 300 --output results/benchmark.json
python plot_results.py
```

The JSON stores raw trials and mean plus sample standard deviation, allowing readers to recalculate aggregate numbers. For scientific use, inspect raw trial data rather than relying only on printed means.
