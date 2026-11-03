# Benchmark notes

## Provenance

These are independent educational PyTorch experiments performed on a newly initialized, trained, four-block residual model. No Qwen checkpoint was downloaded or tested. No Neuron Explorer sweep was reproduced. The optional Hugging Face adapter is unexecuted in this environment.

## Configuration

- PyTorch: 2.10.0+cpu
- Seeds: 0, 1, 2, 3, 4
- Training: 160 AdamW steps per seed, batch 256, learning rate 0.007
- Hidden dimension: 48; 4 residual SwiGLU blocks with inner width 96
- Synthetic input: six tokens, independently encoded mood and topic labels plus shuffled nuisance tokens
- Training discovery: 512 balanced synthetic examples per seed
- Evaluation: 512 held-out synthetic examples per seed
- Evaluation nuisance-token range is disjoint from training nuisance-token range
- Intervention point: residual block index 2
- Direction: normalized difference between mean block outputs for gloomy and joyful examples
- Random control: normalized direction orthogonalized against the mood vector
- Topic control: normalized contrast for tomato and stone labels
- Neuron: the largest absolute gated-intermediate activation mean contrast in the discovery set
- Projection edit: removes direction from one copy of the third block's down-projection contribution and bias
- Metrics: mood accuracy, topic accuracy, mean predicted gloomy probability, KL of each head against baseline

## Five-seed results

The figures are means, with sample standard deviations also preserved in `results/benchmark.json`. All values are educational and must not be attributed to Qwen.

- Baseline: mood accuracy 0.911; topic accuracy 0.815; gloomy probability 0.468.
- Mood +0.5: gloomy probability 0.501; mood KL 0.027.
- Mood +1: gloomy probability 0.537; mood KL 0.110.
- Mood +2: gloomy probability 0.607; mood KL 0.425.
- Mood +4: gloomy probability 0.725; mood accuracy 0.787; mood KL 1.378.
- Random +2: gloomy probability 0.485; mood KL 0.033.
- Topic +2: gloomy probability 0.470; mood KL 0.022.
- Mood -2: gloomy probability 0.349.
- Neuron +5 (sign corrected): gloomy probability 0.618; topic accuracy 0.775; mood KL 0.465.
- Local one-layer projection edit: gloomy probability 0.456; mood KL 0.066.

## Interpretation cautions

A higher gloomy-class probability is not greater behavioral success or higher model capability. It may be indiscriminate movement of predictions to one class. Mood accuracy *decreases* at the largest dose. Topic accuracy is a separate interference measurement, not a full suite of off-target tasks. The random control is one orthogonal random direction per seed, not a full calibrated null distribution of 256 or more directions. The model is trained with supervised concept labels and does not have natural-language semantics. Hidden representations can be highly entangled in pretrained LLMs.

## Reproducibility

```
python -m unittest discover -s tests -v
python -m steering_lab.benchmark --seeds 5 --steps 160 --eval-size 512 --output results/benchmark.json
python -m pip install matplotlib
python plot_results.py
```

The optional Qwen3-0.6B experiment is not part of the tested benchmark. `README.md` describes its additional dependencies.
