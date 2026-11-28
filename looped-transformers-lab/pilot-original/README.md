# Looped Transformers from Scratch

A runnable companion to the Inkling 2 Mini architecture article. This is an educational recurrent-depth experiment, not a reproduction of Inkling or Qwen. It uses bidirectional attention over an entire graph and a query, with a classification head, not autoregressive language modeling.

## Quick start

Python 3.10+ recommended. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v
python lab.py --device cpu
```

On Windows activate with `.venv\Scripts\activate`. For Apple Silicon or NVIDIA, try `--device mps` or `--device cuda` (only CPU was tested). For a longer experiment use `python lab.py --steps 2000 --batch 64 --device auto`; this overwrites the supplied results and figures, so preserve them first. For MPS, operator support varies; CPU is the verified fallback.

## Scientific design

Each sample is a random directed functional graph of eight labeled nodes, a starting node, and a hop count. The target is the node reached by following that many edges. The training stream uses one to three hops; test streams include four to six. Graphs can contain cycles and repeated endpoints. Test samples are generated independently using separate RNG streams; exact duplicate graphs are not explicitly filtered. All models see the same training samples per seed; test examples are fixed across loop counts and architectures for each seed and hop. The uniform guess baseline is 12.5%, but graph cycles can create more informative heuristics (e.g., predicting the starting node); this is not a proof of algorithmic reasoning. Hop embeddings for 4–6 are untrained, making extrapolation additionally difficult.

- `dense2`: two unique transformer blocks, executed once.
- `dense6`: six unique blocks, executed once.
- `loop_fixed`: two unique blocks, executed three times in training.
- `loop_variable`: two unique blocks, one to five passes sampled uniformly per batch in training.
- `loop_moe`: two shared blocks with four experts each, top-2 sparse dispatch, a small router balancing loss, and learned scalar input reinjection. This changes several factors at once, so it is not an isolated MoE ablation.

All runs use 32-dimensional states, four heads, AdamW, learning rate 0.002, gradient clipping at 1, and the same number of optimizer updates. Two-block dense and recurrent dense models have exactly matching trainable parameter counts. Six-block dense and three-pass recurrent dense models have matching block applications per update with identical block dimensions; this is a compute proxy, not measured FLOP equality. Variable-loop work is recorded exactly as block applications; MoE has different per-block cost. Test inference loop counts are 1–6 for recurrent models; 256 samples per hop per seed. The default short run is a pilot, not a convergence study. No validation-based model selection or hyperparameter tuning was performed.

## Files

- `models.py`: dense blocks, shared recurrent depth, sparse expert dispatch and gated input reinjection.
- `lab.py`: data generation, training, held-out evaluation, activation analysis, plotting.
- `test_lab.py`: label correctness, parameter sharing, gradients, sparse dispatch equivalence, memory arithmetic.
- `memory.py`: hypothetical mixed local/global KV-cache and raw weight storage estimator. Layer counts mean cached computational applications, not physical weights. Defaults are illustrative, not confirmed model architecture settings.
- `benchmark.py`: optional streaming Ollama benchmark for the actual downloaded models; records model tags/digests, settings, elapsed time, first visible response/reasoning chunk, prompt and decoding metrics. No generated code is executed. It is not a correctness benchmark.
- `results/`: CPU run log, JSON, unit test log, memory example.
- `figures/`: actual measured pilot plots.
- `ARTICLE-UPDATE.md`: measured findings and captions.

## Benchmark actual local models (optional)

Requires a running Ollama server and compatible downloaded checkpoints. Check availability and model tags for your installation, then:

```bash
ollama pull inkling-2-mini
ollama pull qwen3.8:27b
python benchmark.py --repeats 3 --ctx 8192 --limit 2048
python benchmark.py --think false --out results/ollama-no-think.json
python memory.py --context 131072
```

The benchmark alternates model order, requests unloading between runs, and records loading time. Reasoning settings may be rejected or interpreted differently by each checkpoint. Results are not comparable across quantizations without noting them. The first received chunk is a proxy for time to first token, not GPU instrumentation. Review outputs and test solutions independently; peak memory and task correctness are not measured by this script. Ollama and GPU/MPS have not been tested here. No model downloads are required for the offline experiments.
