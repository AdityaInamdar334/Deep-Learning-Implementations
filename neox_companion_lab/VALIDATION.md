# Observed validation

Executed on CPU, Python 3.12, PyTorch 2.15.0+cpu, Transformers 4.57.6.

- Eight model tests passed: causal prefix invariance; cached chunk parity;
  greedy cache parity; output shape and untied weights; RoPE norm/pass-through;
  checkpoint roundtrip; tiny-batch optimization; invalid config rejection.
- Random-config HF reference comparison passed: maximum absolute logit error
  8.940696716308594e-08. Exact GELU, parallel residuals, no dropout, FP32.
- Custom-model smoke: 30 steps, batch 2, block 32, CPU, seed 42, demo.txt.
  Initial byte NLL 5.578338123136951; final 3.4068212124609176.
- Saved custom-model checkpoint reloaded and generated a continuation.
- Pretrained Pythia-70M generation completed.
- Pythia bias-only smoke: 2 steps, batch 1, block 32, CPU, demo.txt.
  Initial token NLL 4.692547182242076; final 4.606189250946045.
- Reload reproduced token NLL 4.606189250946045.
- Whole-state comparison against base found 37 changed tensors, all biases.
  The training selection contains 34,304 scalar bias parameters.
- Base revision: a39f36b100fe8a5377810d56c3f4789b9c53ac42.
- All Python files passed parsing and ZIP integrity was checked.

The synthetic corpus is repetitive across splits. These losses are plumbing
checks, not language-model quality or generalization benchmarks. Byte and
subword token losses cannot be directly compared.

Not tested: CUDA, MPS, quantization, original 20B loading, multi-GPU execution,
long training on Tiny Shakespeare, all versions within dependency bounds.
Trained weights are excluded from the ZIP; reproduce them with README commands.
