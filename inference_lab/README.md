# Build an inference server · Episode 1
One request, greedy output, full-prefix recomputation. The included GPT-2 checkpoint was trained on two short synthetic sentences. You can finish the lab on a CPU. Python 3.12 is the tested environment.

## 1. Install and generate
From the extracted inference_lab directory, use a virtual environment. Linux and macOS commands follow. On Windows activate with .venv\Scripts\activate.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --prompt "sensor reports hot"
python run.py --prompt "sensor reports cold"
```

Expected continuations include “fan on” and “fan off” respectively. Each forward pass receives one more token than the previous one. The printed continuation excludes the original prompt and hides the special end token. The EOS ID still counts among emitted tokens. The model directory is bundled and all model loads use local files only. Package installation can require internet access. For a CUDA environment install the compatible PyTorch 2.8 wheel for your driver before installing requirements.

## 2. Inspect one request
Run `python run.py --lengths 3 --runs 1 --warmup 0 --new 4 --out results/trace.json`. Open the JSON. Find input_lengths [3,4,5,6], positions_processed 18, output_tokens 4 and three inter-token gaps. Check that TTFT plus those gaps equals total_ms. This exercise is for inspecting the loop. Warm up for performance measurements.

## 3. Run the reference sweep
```bash
python run.py --lengths 16 128 512 --runs 100 --warmup 2 --new 8 --threads 1 --out results/my_cpu.json
python plot.py results/my_cpu.json --out figures/my_cpu.png
python -m unittest -v
```

Expect three length groups, 100 requests each. Read the article alongside the figure. Timing varies with hardware and load. The example trace is results/cpu_reference.json. TTFT starts after tokenization and input transfer. Each output becomes ready on the device before timestamping. Detokenization, network transit and queuing are outside the timer. TPOT is each request's mean of its later gaps. Percentiles are nearest-rank over requests within a group. An output of length one produces null TPOT. Service throughput divides output token count by summed in-loop request duration. GPU peak allocated memory includes the model and live tensors; CPU and MPS leave this field null.

## 4. Change one variable
Run the same command with --new 1 to inspect TTFT, or with --new 32 to inspect repeated work. For P prompt tokens and G outputs the maximum forward length is P+G-1. The fixture supports 1,024 positions. Keep a distinct JSON filename per configuration. For a local pretrained checkpoint, pass --model /path/to/model --prompt "Your prompt"; use an ordinary, locally loadable decoder-only Transformers checkpoint and tokenizer, with unpadded single requests. Exact-length benchmarking repeats the prompt IDs as needed. It counts fixed G even through EOS. Interactive generation stops at the tokenizer's single EOS token or the specified length. Models requiring custom generation processors, chat formatting, or a list of stop IDs need their own adapters.

## 5. Repeat on a device you own
Use --device cuda or --device mps with a compatible PyTorch build. Pass --dtype float16 or bfloat16 if supported by that device and checkpoint. The default is float32. Use a prompt and precision fitting available memory. If the checkpoint already approaches device capacity, begin with a short prompt. Record the exact model revision, device model, driver, power mode and runtime versions with the JSON. CUDA and MPS synchronization are implemented; the included reference run used CPU.

## What is in the ZIP?
- run.py contains the generator, timing boundaries, workload construction and raw JSON writer.
- train_fixture.py recreates the synthetic corpus, tokenizer and model. The training environment and seed appear in the code. Training is optional.
- fixture/ contains the trained local model, tokenizer and configuration.
- test_run.py checks token agreement against incremental cached forwards, causal prefix invariance, EOS handling, timing identities and percentile definitions.
- plot.py recreates the latency plot from any compatible trace.
- results/ and figures/ contain observed CPU measurements, validation output and the generated figure.
- ARTICLE.md and ARTICLE.html contain the tutorial. The HTML embeds the figure for offline reading.

FAQ: Slow or inconsistent first run? Warm up each length and keep thread count fixed. Missing checkpoint? Run from the extracted directory with fixture/ beside run.py. “Context limit” error? Reduce the largest prompt or output length. Out of memory? Reduce dimensions or use the included checkpoint. Results differ from a GPU article? Compare exact model, workload and timing boundary.
