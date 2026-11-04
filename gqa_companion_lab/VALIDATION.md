# Validation record

Validated with Python 3.12.14 and PyTorch 2.15.0+cpu, using FP32 and two CPU threads. No GPU validation was performed.

The twelve tests in `test_gqa.py` passed. They cover causal visibility, cached/chunked versus full inference, compact storage, contiguous head grouping, grouped attention outputs and gradients, native SDPA parity, matching initialization, invalid configurations, checkpoint reload, pooling weights and biases, conversion changing the function, and tiny-batch learning.

Three independent 30-step training runs completed on the bundled synthetic corpus with identical data ordering and evaluation blocks. The final validation byte NLL values rounded to four decimals were 3.3758 for MHA, 3.3880 for GQA, and 3.3731 for MQA. These are pipeline smoke tests, not model-quality comparisons.

An MHA checkpoint was mean-pooled from eight KV heads to two. Validation NLL changed from 3.3758 to 3.5000 immediately after conversion. Five further training steps produced 3.2337, and the saved checkpoint reloaded and generated text. This tiny repetitive-data run does not establish quality recovery on realistic tasks; a continued-MHA control is needed for a comparative claim.

A randomly initialized checkpoint from the preceding NeoX companion lab was imported, its output compared with the original implementation, pooled to GQA, and reloaded. See `results/legacy_import.json` for the observed maximum error.

Cache inspection verified 524,288, 131,072, and 65,536 bytes for MHA, GQA, and MQA with four layers, batch two, 64 tokens, head width 16, and FP32 storage.

The repeat and grouped timing scripts both completed. Raw samples, medians, settings, and runtime version are in `results/benchmark_*.json`. CPU timings were noisy in this shared environment. They validate the timing workflow and are not evidence of production speedups or a stable backend ranking. The native backend was checked for numerical parity, not benchmarked here.

The Tiny Shakespeare download path is provided but was not exercised for this validation. GPU/MPS execution, lower precision, long-context quality, serving throughput, and installation on every permitted dependency version remain unverified. The archive includes code and results, not pretrained weights. Local runs create checkpoints.
