# Benchmark Auditing & Measurement Integrity Report

## 1. SciCode Benchmark Defect Audit & Recovery

| Metric | Original SciCode | SciCode-Verified | Recovery Delta |
| :--- | :---: | :---: | :---: |
| **Subproblem Accuracy** | 74.8% | **100.0%** | +25.2 pp |
| **Whole-Problem Accuracy** | 29.2% | **100.0%** | **+70.8 pp** |
| **Audited Main Problems** | 65 | 65 | — |
| **Defect Prevalence** | 86.2% affected | Corrected | — |

## 2. DeepQuestion Cognitive Complexity Degradation (Bloom's Taxonomy)

| Model | Recall (L1) | Application (L2) | Analysis (L3) | Synthesis (L4) | Peak-to-Trough Drop |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Frontier-Reasoning-Model** | 100.0% | 76.0% | 76.0% | 48.0% | -52.0 pp (-52%) |
| **Standard-Chat-Model** | 100.0% | 68.0% | 64.0% | 24.0% | -76.0 pp (-76%) |
| **Memorizer-Model** | 100.0% | 64.0% | 44.0% | 4.0% | -96.0 pp (-96%) |

## 3. Epoch AI Benchmark Reviews Classification Summary

| Benchmark Suite | Sample Audited | Score-Affecting Defects | Defect Rate | Formal Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **SciCode (Original)** | 65 | 59 | 90.8% | **Flawed** |
| **Humanity's Last Exam (HLE)** | 48 | 22 | 45.8% | **Flawed** |
| **SWE-bench Verified (Audited Failures)** | 138 | 82 | 59.4% | **Flawed** |
| **SWE-bench Pro (Public Split)** | 731 | 219 | 30.0% | **Flawed** |
| **Terminal-Bench 4.0.0** | 66 | 30 | 45.5% | **Flawed** |
| **GPQA Diamond Clean** | 189 | 9 | 4.8% | **Verified** |
| **FrontierCode 2.0 (Private)** | 132 | 0 | 0.0% | **Not enough info** |
| **HLE-Diamond (1000-Task)** | 100 | 8 | 8.0% | **Verified** |
