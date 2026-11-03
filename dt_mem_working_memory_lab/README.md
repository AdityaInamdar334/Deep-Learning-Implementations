# Working Memory for Decision-Making: Educational PyTorch Lab

An independently implemented, CPU-friendly workshop inspired by Jikun Kang and collaborators' ICML 2024 paper, *Think Before You Act: Decision Transformers with Working Memory*.

**This is not the authors' official DT-Mem implementation and does not reproduce their Atari or Meta-World experiments.** The official code is at https://github.com/luciferkonn/DT_Mem. The paper is at https://proceedings.mlr.press/v235/kang24b.html.

The goal is to make three mechanisms tangible:

1. A causal return-conditioned Transformer must not use the action it is currently predicting.
2. A small addressable memory can use recent task demonstrations at inference time, without updating the baseline model's parameters.
3. Memory capacity, noisy writes, and the correctness of addressing matter as much as the existence of memory.

A separate minimal LoRA example shows how a frozen weight matrix can be adapted using low-rank parameters.

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer
- No GPU, Atari ROM, Meta-World installation, API key or network services

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```bash
python -m unittest discover -s tests -v
python -m dtmem_lab.benchmark --seeds 5 --tasks 100 --queries 64 --steps 180 --output results/benchmark.json
python -m dtmem_lab.dt_train --steps 160
python -m dtmem_lab.lora_demo
```

The default benchmark uses 500 unseen tasks across five independent seeds, with 64 query decisions per task. Each task defines a fresh permutation of eight contexts onto four balanced actions. The baseline is a small MLP pre-trained on random task mappings. A task-specific mapping is **unidentifiable** from the query context alone. The memory method receives 4, 16, or 32 expert context-action pairs at test time. Therefore a performance improvement principally demonstrates **availability of information** in task-specific support examples, not a generally superior agent architecture.

## Results from the saved reference run

The results in `results/benchmark.json` were generated with PyTorch 2.10.0+cpu, five seeds, 100 tasks per seed, 64 queries per task and 180 baseline optimization steps per seed.

- Without memory: **24.82%** mean accuracy.
- Eight slots, 16 support interactions: **91.74%**.
- Eight slots, four interactions: **55.52%**.
- Eight slots, 32 interactions: **99.12%**.
- Four slots, 16 interactions: **57.32%**.
- Two slots, 16 interactions: **37.87%**.
- Eight slots, 16 interactions, 25% incorrect labels: **73.55%**.
- Eight slots, 16 interactions, shuffled context-action labels: **27.07%**.

Sample standard deviations and per-seed results are stored in the JSON. The memory is exact on previously observed contexts only when there is one independent slot per context and the demonstrations are not corrupted. No-match queries use the pretrained baseline. All benchmarks are synthetic. See `BENCHMARK_NOTES.md` for interpretation and threats to validity.

The standalone tiny DT training exercise reached 100% action accuracy on new sequences drawn from the **same fixed synthetic mapping** in the reference run (160 updates); this demonstrates supervised memorization and correct causal tokenization, not held-out-task generalization.

The LoRA mechanics demo fits an artificially constructed rank-two weight update with 40 trained parameters while freezing 96 base weights. This demonstration does not model the paper's downstream Meta-World fine-tuning.

## Code tour

- `dtmem_lab/dt_train.py`: train the tiny causal Transformer on one fixed mapping to confirm its supervised action-prediction path.
- `dtmem_lab/tiny_dt.py`: causal Transformer using return, state and action tokens. The prediction for action at time `t` is taken from the corresponding **state token**, which appears before the same-step action token. A unit test checks against target leakage.
- `dtmem_lab/memory.py`: a fixed-key, softmax-addressed slot memory, with accumulated task-specific action values and coverage-gated fallback.
- `dtmem_lab/toy_tasks.py`: balanced randomized task mappings and a context-only MLP baseline.
- `dtmem_lab/benchmark.py`: deterministic paired ablations over independent held-out tasks and multiple seeds.
- `dtmem_lab/lora_demo.py`: frozen linear projection with trainable low-rank factors.
- `tests/test_lab.py`: tests of memory, task distribution, causal masking, LoRA and benchmark determinism.
- `results/benchmark.json`: saved results from the reference CPU run.
- `ARTICLE.md`: full accompanying technical tutorial.

## Reproducibility and scientific cautions

- The benchmark's memory uses **fixed one-hot context prototypes**, not learned memory addresses or the exact learned erasure/addition equations in Kang et al.
- The tiny causal DT module is **not trained jointly with the memory** in this benchmark. It is supplied as a separate runnable, architecturally correct illustration of causal token ordering.
- Training-time speedups, agent returns in a physical simulator, long-horizon control, zero-shot generalization and catastrophic forgetting mitigation **cannot be inferred** from these results.
- Query actions are balanced across tasks by construction, so the pretrained model's expected accuracy on completely new task mappings is around 25%.
- Strong memory performance is driven by matching support examples. Increasing context count, changing observation space, omitting support, or adding conflicting demonstrations may reduce performance.
- Reproducing the original work requires the official dataset, model configuration, environment wrappers, compute budget, and evaluation protocol.

## References

1. Kang et al., *Think Before You Act: Decision Transformers with Working Memory*, ICML 2024: https://proceedings.mlr.press/v235/kang24b.html
2. Official DT-Mem: https://github.com/luciferkonn/DT_Mem
3. Chen et al., *Decision Transformer: Reinforcement Learning via Sequence Modeling*, NeurIPS 2021: https://papers.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html
4. Graves et al., *Neural Turing Machines*: https://arxiv.org/abs/1410.5401
5. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*: https://arxiv.org/abs/2106.09685

## License

The independently authored educational lab is provided under the MIT License in `LICENSE`. The original DT-Mem source and paper have their own licensing terms.
