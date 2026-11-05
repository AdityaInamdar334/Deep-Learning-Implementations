# Validation record

Environment: Python 3.12.14, PyTorch 2.15.0+cpu, two CPU threads, FP32. No GPU or MPS validation was performed.

All 15 tests in `test_generation.py` passed. They cover per-step cached/full-prefix logit parity, chunked prefill, causality, greedy sequence equivalence, cache lengths and bytes, zero budget and input guards, EOS stopping, seeded sampling, top-k-one parity, fresh and interleaved request state, UTF-8 streaming, checkpoint reload, last-only vocabulary projection, evaluation-mode enforcement, and wrong-history sensitivity. The full test output is in `results/tests.txt`.

Training completed for 200 steps on the bundled repetitive synthetic corpus. Validation byte NLL decreased from 5.6130745896 to 0.1532725796 on 1,968 target bytes. This is a pipeline exercise, not a meaningful generalization benchmark. The data hash and runtime version are recorded in `results/training.json`.

The checkpoint reloaded and completed greedy cached, greedy uncached, and sampled generation. Cached and uncached displayed continuations matched for the tested prompt. Traces and outputs are in `results/`. The training data contains no reserved EOS tokens; EOS handling was tested with a controlled stop ID instead.

The benchmark completed warmups and seven timed repeats of each path with shuffled order. It used random weights, a 64-token synthetic prompt, and 32 fixed greedy output tokens. Every timed output matched the reference. Median total latency was 56.3827 ms uncached and 23.4766 ms cached in this shared CPU environment. These are teaching-model measurements, not GPT-NeoX-20B results or GPU projections. First-token and subsequent-token timings, all raw samples, and scope are in `results/benchmark.json`.

Analytical GPT-NeoX-20B cache storage at batch one and 2,048 positions was 2.0625 GiB with two-byte elements. This is logical K/V storage, not measured peak allocation. The supplied `memory.py` produced `results/neox20b_memory.json`.

Not tested: actual 20B weight loading, pretrained tokenization, CUDA/MPS, reduced precision, concurrent serving, padded generation, cache eviction, long-context quality, or every dependency version in the requirements range. Generated checkpoints are excluded from the ZIP and can be recreated with the training command.
