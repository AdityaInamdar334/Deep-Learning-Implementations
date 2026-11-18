# 1,000-Document Local RAG: Retrieval Architectures & Citation Grounding

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Standard%20Library%20Default-success.svg)]()
[![Tests](https://img.shields.io/badge/Tests-10%2F10%20Passing-brightgreen.svg)](tests/test_lab.py)
[![Model](https://img.shields.io/badge/Model-Qwen2.5%203B%20Instruct-blueviolet.svg)](https://ollama.com/library/qwen2.5:3b)
[![Retrievers](https://img.shields.io/badge/Retrievers-SQLite%20FTS5%20%7C%20FAISS%20Dense%20%7C%20Cross--Encoder-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A controlled, reproducible local RAG laboratory evaluating **SQLite FTS5 lexical search**, **FAISS dense embeddings**, **cross-encoder reranking**, evidence-bearing Recall@k, citation hallucination detection, and counterfactual abstention on a 1,000-document corpus.*

</div>

---

## Executive Summary

Deploying Retrieval-Augmented Generation (RAG) on local language models presents critical evaluation traps:
1. **The Retrieval Illusion**: A retriever may return the correct document, but the returned chunk might chop off the actual supporting fact, starving the LLM of necessary context.
2. **Citation Hallucination**: Small models frequently cite plausible-looking passage IDs (`doc_12:c0`) that do not contain the answer, or invent citations entirely.
3. **The Abstention Deficit**: When posed with unanswerable or counterfactual queries where the corpus lacks evidence, models tend to hallucinate plausible answers rather than declaring `UNKNOWN`.

This laboratory establishes an **auditable, fully offline benchmark** with **1,000 synthetic documents** (20 enterprise projects, 50 records each) and **50 structured test questions** (40 answerable, 10 counterfactuals). It systematically compares:
- **Lexical Retrieval**: Pure Python standard library SQLite FTS5 (BM25 token ranking).
- **Dense Vector Retrieval**: 384-dimensional FAISS cosine-similarity index (`all-MiniLM-L6-v2`).
- **Two-Stage Reranking**: Cross-encoder re-scoring (`ms-marco-MiniLM-L-6-v2`) over broad candidate pools.

```
                         ┌──────────────────────────────────────────────┐
                         │      1,000 Synthetic Corpus Documents       │
                         └──────────────────────┬───────────────────────┘
                                                │
                         Overlapping Word-Based Chunking (e.g. 100 words, 20 overlap)
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │              Multi-Tier Indexing             │
                         │  - SQLite FTS5 Full-Text Index (Default)     │
                         │  - FAISS Cosine Index (all-MiniLM-L6-v2)     │
                         └──────────────────────┬───────────────────────┘
                                                │
                                    Query: Question String
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Candidate Retrieval & Reranking Stage                                                         │
│                                                                                                │
│  [ Stage 1: Candidate Pool ]  ──> Top-20 Chunks retrieved via Lexical or Dense Search         │
│  [ Stage 2: Cross-Encoder ]   ──> Re-scored via ms-marco-MiniLM-L-6-v2                         │
│  [ Final Selection ]          ──> Top-k Context Chunks (k=5)                                   │
└───────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Local Model Inference (Ollama: Qwen 2.5 3B)                                                   │
│  - Prompt enforces: Return JSON with "answer" and "citations": ["chunk_id", ...]               │
│  - Explicit Abstention Mandate: If evidence is absent, answer "UNKNOWN" with citations: []     │
└───────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Deterministic Evaluation & Citation Audit                                                     │
│  - Evidence-Bearing Recall@k: Returned chunk has gold doc ID AND exact answer string           │
│  - Citation Validity: Every cited chunk ID exists in provided context                          │
│  - Supported Citation Rate: At least one cited chunk actually supports the claim               │
│  - Grounded Correctness: Answer exact match + supported citation, or UNKNOWN on counterfactual │
│  - Abstention Accuracy: Fraction of unanswerable items correctly answered UNKNOWN              │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparative Retrieval Pipeline

| Pipeline | Index Storage | Query Time | Python Dependencies | Primary Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Lexical (SQLite FTS5)** | `chunks.sqlite` | $< 2\text{ ms}$ | **Standard Library Only** | Exact keyword/ID matching, zero RAM overhead, instant cold start. |
| **Dense Vector (FAISS)** | In-memory `IndexFlatIP` | $\approx 15\text{ ms}$ | `faiss-cpu`, `sentence-transformers` | Semantic synonym matching, tolerance to vocabulary mismatch. |
| **Two-Stage Reranked** | FAISS / FTS5 + Cross-Encoder | $\approx 45\text{ ms}$ | `sentence-transformers` | Joint query-passage token attention; highest precision at $k=5$. |

---

## Mathematical Formulations & Grounding Metrics

### 1. Evidence-Bearing Recall@k
Given query $q$ with gold document $d^* \in \mathcal{D}$ and exact answer string $a^*$:
$$\text{Recall}_{\text{evidence}}@k = \mathbb{I}\left( \exists c \in \text{Top}_k(q) \text{ s.t. } \text{doc\_id}(c) = d^* \land a^* \in \text{text}(c) \right)$$
*(Requires the supporting text itself to be present in the chunk, not merely the document).*

### 2. Supported Citation Rate
For model response with claimed citations $\mathcal{C} = \{c_1, \dots, c_m\}$:
$$\text{SupportedCitation} = \mathbb{I}\left( \exists c \in \mathcal{C} \cap \text{Top}_k(q) \text{ s.t. } \text{doc\_id}(c) = d^* \land a^* \in \text{text}(c) \right)$$

### 3. Grounded Correctness
Evaluates whether the system achieves true end-to-end factual reliability:
$$\text{GroundedCorrect}(q) = \begin{cases}
\mathbb{I}\left( \text{norm}(A) = \text{norm}(a^*) \land \text{SupportedCitation} = 1 \right) & \text{if } q \text{ is answerable} \\
\mathbb{I}\left( \text{norm}(A) = \text{"unknown"} \land |\mathcal{C}| = 0 \right) & \text{if } q \text{ is unanswerable}
\end{cases}$$

---

## Directory Structure

```text
local-rag-1000-lab/
├── README.md                  # Comprehensive methodology, evaluation metrics, and guide
├── RESULTS_TEMPLATE.md        # Standardized reporting template for benchmark sweeps
├── requirements-optional.txt  # Optional dependencies for FAISS and Cross-Encoder
├── rag_lab.py                 # Core corpus generator, FTS/dense indexer, evaluator
├── ask.py                     # Interactive ad-hoc query CLI with cited evidence display
├── plot_results.py            # Zero-dependency SVG visualizer for multi-run comparisons
├── data/
│   ├── corpus.jsonl           # 1,000 synthetic enterprise documents
│   └── questions.jsonl        # 50 labeled questions (40 answerable, 10 counterfactual)
├── results/                   # Destination for summary.json, runs.csv, comparison.svg
└── tests/
    └── test_lab.py            # 10 unit tests covering chunking, FTS, grading, and citations
```

---

## Step-by-Step Reproduction Guide

### 1. Standard-Library Lexical Pipeline (Zero Extra Packages)

```bash
cd local-rag-1000-lab

# 1. Run unit tests
python3 -m unittest discover -s tests -v

# 2. Generate corpus & build SQLite FTS5 index (100 words per chunk, 20 overlap)
python3 rag_lab.py generate
python3 rag_lab.py index --chunk-words 100 --overlap 20

# 3. Retrieval-only ablation (no Ollama required)
python3 rag_lab.py evaluate --no-generate --retriever lexical --k 5 --out results/retrieval-only

# 4. End-to-end generation with Ollama
ollama pull qwen2.5:3b
python3 rag_lab.py evaluate --retriever lexical --k 5 --out results/lexical-100
```

### 2. Interactive Ad-Hoc Inquiries

```bash
python3 ask.py 'According to record 03, what is the backup interval for Project-00-Aster?'
```
*Displays the model's grounded answer along with exact cited passage text and chunk IDs.*

### 3. Chunking Ablation Study

```bash
# Fine chunking (40 words, 8 overlap)
python3 rag_lab.py index --db data/chunks-40.sqlite --chunk-words 40 --overlap 8
python3 rag_lab.py evaluate --db data/chunks-40.sqlite --retriever lexical --k 5 --out results/lexical-40

# Coarse chunking (180 words, 30 overlap)
python3 rag_lab.py index --db data/chunks-180.sqlite --chunk-words 180 --overlap 30
python3 rag_lab.py evaluate --db data/chunks-180.sqlite --retriever lexical --k 5 --out results/lexical-180
```

### 4. Dense Vector Retrieval & Cross-Encoder Reranking (Optional)

```bash
# Install optional vector dependencies
pip install -r requirements-optional.txt

# Evaluate dense vector retrieval (FAISS)
python3 rag_lab.py evaluate --retriever dense --k 5 --out results/dense-100

# Evaluate two-stage cross-encoder reranking
python3 rag_lab.py evaluate --retriever dense --rerank cross --candidates 20 --k 5 --out results/dense-reranked
```

### 5. Generate Metric Visualizations

```bash
python3 plot_results.py \
  results/lexical-100/summary.json \
  results/dense-100/summary.json \
  results/dense-reranked/summary.json \
  --out results/comparison.svg
```

---

## Attribution & License

- **Foundational Research**:
  - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
  - [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (Liu et al., 2023)
  - [Correctness is not Faithfulness in RAG Attributions](https://arxiv.org/abs/2412.18004) (2024)
- **License**: Released under the [MIT License](../LICENSE).
