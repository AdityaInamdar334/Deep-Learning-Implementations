# Private Speech Recognition Lab: whisper.cpp

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-10%2F10%20Passing-brightgreen.svg)](tests/test_lab.py)
[![Engine](https://img.shields.io/badge/Engine-whisper.cpp%20%28GGML%29-181717.svg?logo=github&logoColor=white)](https://github.com/ggml-org/whisper.cpp)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline%20%2F%20No%20Uploads-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A privacy-first, zero-dependency local speech-to-text laboratory evaluating **Word Error Rate (WER)**, **Real-Time Factor (RTF)**, segment timestamp alignment, and SRT subtitle generation across local GGML Whisper models (`tiny.en`, `base.en`, `small.en`) using `whisper.cpp`.*

</div>

---

## Executive Summary

Cloud-based speech-to-text APIs present severe confidentiality liabilities when transcribing sensitive audio (e.g. executive meetings, clinical interviews, proprietary code discussions). Furthermore, running Python-heavy PyTorch pipelines introduces substantial runtime overhead and multi-gigabyte dependency environments.

This laboratory implements a **hardened, offline-first speech recognition and evaluation harness** built around **`whisper.cpp`** and Python's standard library:
1. **Zero-Telemetry Local Execution**: Audio capture, preprocessing, neural transcription, and evaluation run entirely offline on local hardware with zero external network calls.
2. **Deterministic Provenance**: SHA-256 hashes of input WAV audio, binary executables, and model checkpoints are recorded alongside all metric outputs.
3. **Formal Metric Decomposition**: Implements an auditable 2D dynamic programming Levenshtein algorithm that decomposes Word Error Rate into substitutions ($S$), deletions ($D$), and insertions ($I$).
4. **Throughput Accounting**: Computes Real-Time Factor (RTF) and speed multiplier ($X\times$ real-time) to measure hardware efficiency across CPU and Metal acceleration.

```
                  ┌──────────────────────────────────────────────┐
                  │          Local Microphone / WAV Input        │
                  │   (16 kHz Mono, 16-bit PCM Linear Format)    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  High-Performance Local Inference Engine (whisper.cpp / whisper-cli)                  │
│  - Accelerated via Apple Metal, AVX2, or CUDA                                          │
│  - GGML Quantized Checkpoints (tiny.en, base.en, small.en)                             │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
       ┌─────────────────────────┐               ┌─────────────────────────┐
       │   Timestamped Text &    │               │  Execution Telemetry    │
       │   SRT Subtitle Streams  │               │  - Audio Duration (s)   │
       │   - Offset millisecond  │               │  - Wall Compute Time (s)│
       │   - Subtitle blocks     │               │  - Real-Time Factor     │
       └────────────┬────────────┘               └────────────┬────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  Auditable Levenshtein DP Evaluator (evaluate.py / common.py)                          │
│  - Standardized Unicode NFKC Normalization & Punctuation Stripping                     │
│  - Full Word Error Rate: WER = (Substitutions + Deletions + Insertions) / N_ref        │
│  - Corpus-level aggregate WER across multi-clip manifests                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulations

### 1. Real-Time Factor (RTF)
Measures the computational throughput of the batch inference pass relative to the physical length of the audio:
$$\text{RTF} = \frac{\text{WallSeconds}}{\text{AudioDurationSeconds}}, \quad \text{SpeedMultiplier} = \frac{1}{\text{RTF}}$$
- An $\text{RTF} < 1.0$ indicates that the model transcribes faster than real time (e.g. $\text{RTF} = 0.25 \implies 4\times$ real-time).

### 2. Word Error Rate (WER)
Given a normalized reference token sequence $R = (r_1, \dots, r_N)$ and hypothesis sequence $H = (h_1, \dots, h_M)$, minimum edit distance is solved via 2D dynamic programming:
$$D[i, j] = \min \begin{cases}
D[i-1, j-1] & \text{if } r_i = h_j \\
D[i-1, j-1] + 1 \quad (\text{Substitution } S) & \text{if } r_i \ne h_j \\
D[i-1, j] + 1 \quad (\text{Deletion } D) \\
D[i, j-1] + 1 \quad (\text{Insertion } I)
\end{cases}$$

$$\text{WER} = \frac{S + D + I}{N}$$

### 3. Corpus Aggregate WER
When evaluating over an audio manifest with $K$ distinct clips:
$$\text{WER}_{\text{corpus}} = \frac{\sum_{k=1}^K (S_k + D_k + I_k)}{\sum_{k=1}^K N_k}$$

---

## Directory Structure

```text
offline-whisper-lab/
├── README.md              # Research documentation, mathematical definitions, execution guide
├── RESULTS_TEMPLATE.md    # Standardized markdown report template for hardware runs
├── data/
│   ├── manifest.example.json # Multi-clip benchmarking manifest template
│   └── reference_example.txt # Example reference transcript
├── results/               # Destination directory for runs.csv, metadata.json, SRTs
├── src/
│   ├── common.py          # Levenshtein DP WER, SRT generator, duration & hash helpers
│   ├── record.py          # Explicitly user-initiated microphone recorder (FFmpeg)
│   ├── transcribe.py      # Core whisper-cli runner, RTF calculator, artifact writer
│   ├── evaluate.py        # Standalone WER evaluator
│   └── benchmark.py       # Multi-model manifest sweep runner
└── tests/
    └── test_lab.py        # 10 unit tests covering WER invariants, SRT formatting, and mock CLI
```

---

## Step-by-Step Reproduction Guide

### 1. Build whisper.cpp

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp

# Configure and compile with Release optimizations (Metal auto-detected on macOS)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4

# Download GGML model checkpoints
sh ./models/download-ggml-model.sh tiny.en
sh ./models/download-ggml-model.sh base.en
sh ./models/download-ggml-model.sh small.en

cd ..
```

### 2. Run Test Suite

```bash
cd offline-whisper-lab
python3 -m unittest discover -s tests -v
```
*Validates exact word matching, substitutions, deletions, insertions, segment parsing, SRT formatting, and mock CLI orchestration across 10 unit tests (all pass).*

### 3. Record an Audio Clip Locally (Optional)

```bash
# macOS (AVFoundation)
python3 src/record.py --seconds 20 --device ':0' --output data/clean.wav

# Linux (PulseAudio)
python3 src/record.py --seconds 20 --device default --output data/clean.wav

# Or convert any existing recording via FFmpeg:
ffmpeg -i my_recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le data/clean.wav
```

### 4. Transcribe & Generate Subtitles

```bash
python3 src/transcribe.py \
  --cli ../whisper.cpp/build/bin/whisper-cli \
  --model ../whisper.cpp/models/ggml-base.en.bin \
  --audio data/clean.wav \
  --repeat 3 \
  --output results/clean
```
Outputs:
- `trial_01.txt`: Plaintext transcription.
- `trial_01.srt`: Timestamped subtitles formatted for video players.
- `runs.csv`: Per-trial audio seconds, wall latency, RTF, speed multiplier.
- `metadata.json`: Machine architecture, model SHA-256, audio SHA-256.

### 5. Evaluate Word Error Rate (WER)

```bash
python3 src/evaluate.py \
  --reference data/reference_example.txt \
  --hypothesis results/clean/trial_01.txt \
  --output results/clean/wer.json
```

### 6. Run Multi-Model Benchmark Sweep

Configure `data/manifest.json` with target audio files and reference ground truths, then execute:

```bash
python3 src/benchmark.py \
  --cli ../whisper.cpp/build/bin/whisper-cli \
  --models ../whisper.cpp/models/ggml-tiny.en.bin \
           ../whisper.cpp/models/ggml-base.en.bin \
           ../whisper.cpp/models/ggml-small.en.bin \
  --manifest data/manifest.json \
  --repeat 3 \
  --out results/sweep
```
*Produces `all_runs.csv` and `summary.json` containing corpus-level WER, median RTF, and wall clock latencies across models.*

---

## Privacy, Consent & Operational Boundaries

- **Local Execution Guarantee**: All audio samples and neural weights reside exclusively on local storage. No network connections are initiated.
- **Audio Exclusion**: `.gitignore` strictly excludes `*.wav`, `*.mp3`, `*.flac`, and generated result directories to prevent unintentional commits of private speech data.
- **Consent Notice**: Never record conversation participants without their prior knowledge and explicit consent.

---

## Attribution & License

- **Inference Engine**: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) by Georgi Gerganov and contributors.
- **Model Architecture**: [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356) by OpenAI.
- **Harness & Benchmark Code**: Released under the [MIT License](../LICENSE).
