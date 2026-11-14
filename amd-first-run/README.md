# AMD ROCm First Run: Batched Exact Dense Retrieval Benchmarking

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![ROCm](https://img.shields.io/badge/AMD-ROCm%20%7C%20HIP-ED1C24.svg?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![Tests](https://img.shields.io/badge/Tests-2%2F2%20Passing-brightgreen.svg)](test_retrieval.py)
[![Hardware](https://img.shields.io/badge/Hardware-CPU%20%7C%20AMD%20Instinct%20%7C%20Radeon-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A clean, dependency-minimal benchmarking laboratory for validating your first AMD ROCm GPU environment, measuring batched exact dense vector retrieval (GEMM + argmax), enforcing HIP device synchronization boundaries, and auditing top-1 numerical agreement against float64 CPU baselines.*

Based on AMD ROCm PyTorch documentation and hardware deployment standards:  
**[AMD ROCm Documentation: PyTorch Installation and HIP Semantics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)**.

</div>

---

## Executive Summary

When provisioning and configuring AMD GPUs (such as AMD Instinct MI250/MI300X or Radeon PRO/RX series) with ROCm (Radeon Open Compute) and PyTorch, verifying that the hardware environment is functioning correctly requires an inspectable, deterministic workload.

Dense vector retrieval (the core computational kernel powering Retrieval-Augmented Generation / RAG, vector databases, and semantic search) provides the ideal verification benchmark:
1. **Compute & Memory Stress**: Executes dense matrix-matrix multiplication ($Q \cdot C^T$) followed by row-wise reduction ($\arg\max$).
2. **Numerical Fidelity Verification**: Validates whether FP32 hardware arithmetic on AMD GPUs reproduces top-1 retrieval rankings identically against an unaccelerated 64-bit float CPU reference (`top1_agreement`).
3. **Asynchronous Execution Auditing**: Enforces explicit HIP synchronization barriers (`torch.cuda.synchronize()`) to ensure wall-clock timings measure kernel completion rather than asynchronous host dispatch latency.
4. **Batching Economics**: Demonstrates how expanding query batch size saturates compute units, accelerating throughput from 25,000 to over 64,000 queries per second on CPU, and orders of magnitude higher on accelerated silicon.

```
Dense Vector Retrieval Pipeline:
Corpus Matrix C:  [ N = 4,096 documents, D = 128 dimensions ]
Query Batch Q:    [ B = 32 queries,      D = 128 dimensions ]

1. Batched Matrix Multiply (GEMM):
   Scores S = Q @ C.T  -> Shape: [ 32, 4096 ]
2. Parallel Row-wise Argmax:
   Top-1 IDs = argmax(S, axis=1) -> Shape: [ 32 ]
3. Verification:
   Compare Top-1 IDs against float64 CPU ground truth -> top1_agreement = 1.0 (100%)
```

---

## Mathematical Foundations

### 1. Vector Scoring & Top-1 Ranking

Given a corpus of $N$ document embeddings $C \in \mathbb{R}^{N \times D}$ and a query batch $Q \in \mathbb{R}^{B \times D}$:

$$\text{Score}(i, j) = q_i \cdot c_j = \sum_{k=1}^D Q_{i, k} C_{j, k}$$
$$\text{Top-1 ID}_i = \arg\max_{j \in \{0, \dots, N-1\}} \text{Score}(i, j)$$

#### Floating-Point Flops:
$$\text{FLOPs} = 2 \cdot B \cdot N \cdot D$$
For $N = 4,096, D = 128, B = 32$:
$$\text{FLOPs} = 2 \times 32 \times 4096 \times 128 = 33,554,432 \text{ FLOPs} \approx 33.55 \text{ MFLOPs per batch}$$

---

### 2. Synchronization & Latency Profiling

In ROCm PyTorch, GPU kernels are queued asynchronously on the HIP stream. A naive Python `time.perf_counter()` call around a forward pass will return immediately while the actual GEMM kernel is still in flight in the hardware pipeline.

`retrieval.py` enforces explicit device barriers:
```python
if args.device == "rocm":
    import torch
    sync = torch.cuda.synchronize  # Invokes hipDeviceSynchronize
else:
    sync = lambda: None
```

Timing strictly isolates:
- Execution of `Q @ C.T`
- `argmax(dim=1)`
- Result tensor concatenation
- Device synchronization wait

It excludes input generation, host-to-device memory copy, and verification overhead.

---

## Empirical Benchmark & Results

### Reference Results (CPU Baseline: 4,096 docs, 64 queries, 128 dims, 5 repeats)

```bash
python3 retrieval.py --device cpu --output cpu.json
```

```json
{
  "environment": {
    "device": "cpu",
    "python": "3.12.14",
    "numpy": "2.3.5",
    "rows": 4096,
    "queries": 64,
    "dimensions": 128,
    "repeats": 5,
    "dtype": "float32"
  },
  "results": [
    {
      "batch": 1,
      "median_seconds": 0.002555,
      "queries_per_second": 25048.8,
      "top1_agreement": 1.0
    },
    {
      "batch": 8,
      "median_seconds": 0.002055,
      "queries_per_second": 31150.2,
      "top1_agreement": 1.0
    },
    {
      "batch": 32,
      "median_seconds": 0.000993,
      "queries_per_second": 64463.9,
      "top1_agreement": 1.0
    }
  ]
}
```

Throughput scales by **$2.57\times$** simply by increasing batch size from 1 to 32, while maintaining **100% top-1 agreement** with the float64 ground truth.

---

## Quickstart & Commands

### 1. Run on Local CPU (Verification & Smoke Testing)

```bash
cd amd-first-run

# Run unit tests
python3 -m unittest -v

# Run baseline retrieval benchmark
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 retrieval.py --output cpu.json
```

### 2. Run on Provisioned AMD GPU (ROCm Environment)

Follow the official [AMD ROCm PyTorch Installation Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html). Once inside the ROCm container/virtual environment:

```bash
# 1. Standard verification run (identical inputs to CPU)
python3 retrieval.py --device rocm --output rocm.json

# 2. Scaled production benchmark (65,536 corpus, 256 queries, 768 dimensions)
python3 retrieval.py --device rocm \
  --rows 65536 \
  --queries 256 \
  --dim 768 \
  --batches 1 8 32 64 \
  --output rocm_large.json
```

---

## Unit Testing & Verification

The test suite validates deterministic seed replay and exact index recovery on orthogonal identity corpora:

```bash
python3 -m unittest -v test_retrieval.py
```

```
test_known_ids_and_partial_batch (test_retrieval.RetrievalTests.test_known_ids_and_partial_batch) ... ok
test_seed_replays (test_retrieval.RetrievalTests.test_seed_replays) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.087s

OK
```

---

## Repository Structure

```
amd-first-run/
├── README.md           # This comprehensive benchmarking guide and reproduction reference
├── example-cpu.json    # Reference JSON output benchmark from a single-threaded CPU run
├── retrieval.py        # Core retrieval engine supporting NumPy CPU and PyTorch ROCm/HIP
└── test_retrieval.py   # Unit tests verifying exact ID recovery and seed determinism
```

---

## Academic & Hardware References

1. **AMD Corporation (2024)**. *AMD ROCm Information Hub & PyTorch Installation Guide*. [rocm.docs.amd.com](https://rocm.docs.amd.com/).
2. **Johnson, Jeff, Matthijs Douze, and Hervé Jégou (2019)**. *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data, 7(3), 535–547.

---

## License

This project is licensed under the [MIT License](../LICENSE).
