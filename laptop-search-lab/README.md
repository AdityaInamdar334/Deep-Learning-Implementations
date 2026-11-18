# Search Everything on Your Laptop: Local Document Retrieval Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SQLite FTS5](https://img.shields.io/badge/SQLite-FTS5%20BM25-003B57.svg)](https://www.sqlite.org/fts5.html)
[![Dense Embeddings](https://img.shields.io/badge/Dense-all--MiniLM--L6--v2-yellowgreen.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Hybrid Ranking](https://img.shields.io/badge/Hybrid-RRF%20(k=60)-purple.svg)](#reciprocal-rank-fusion-rrf)
[![PDF Extraction](https://img.shields.io/badge/PDF-PyMuPDF-red.svg)](https://pymupdf.readthedocs.io/)
[![Privacy: Local Only](https://img.shields.io/badge/Privacy-100%25%20Local%20%7C%20No%20APIs-success.svg)](#privacy-security--threat-model)
[![Tests Passing](https://img.shields.io/badge/tests-7%20passing-brightgreen.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, **read-only** local document search engine designed to run entirely on consumer laptops without cloud APIs or generative LLMs. Features exact source path and PDF page provenance, SQLite FTS5 BM25 lexical retrieval, optional local sentence-transformer dense embeddings, optional FAISS acceleration, and Reciprocal Rank Fusion (RRF) hybrid search.

---

## Architecture Overview

```
                            +-------------------------------------------------------------+
                            |                 LOCAL DOCUMENT CORPUS                       |
                            |  - Markdown files (.md, .markdown)                          |
                            |  - Plain text files (.txt)                                  |
                            |  - Multi-page PDF documents (.pdf via PyMuPDF)             |
                            +-------------------------------------------------------------+
                                                          |
                                                          | [File Ingestion & Safety Filter]
                                                          | - Symlinks skipped (sandbox escape defense)
                                                          | - File size limit (<= 12 MB)
                                                          | - Extension whitelist validation
                                                          v
                                            +----------------------------+
                                            |   SHA-256 Content Hasher   |
                                            +----------------------------+
                                                          |
                                                          +--- Unchanged? ---> [Skip Document]
                                                          |
                                                          +--- Deleted?   ---> [Prune from SQLite]
                                                          |
                                                          v New or Modified
                                            +----------------------------+
                                            | Slotted Chunking Engine    |
                                            | (140 words, 30 overlap)    |
                                            +----------------------------+
                                                          |
                             +----------------------------+----------------------------+
                             |                                                         |
                             v                                                         v
             +-------------------------------+                         +-------------------------------+
             |   LEXICAL INDEXING PIPELINE   |                         |    DENSE INDEXING PIPELINE    |
             | - FTS5 Tokenizer & Sanitizer  |                         | - SentenceTransformer         |
             | - SQLite Inverted Index       |                         |   (all-MiniLM-L6-v2)          |
             | - Native BM25 Scoring         |                         | - L2 Embedding Normalization  |
             +-------------------------------+                         | - Local Disk Vector Cache     |
                             |                                         +-------------------------------+
                             |                                                         |
                             v                                                         v
+---------------------------------------------------------------------------------------------------------------+
| UNIFIED HYBRID SEARCH RUNTIME                                                                                 |
|                                                                                                               |
|    User Query: "When does the Cedar production deployment window begin?"                                     |
|         |                                                         |                                           |
|         v                                                         v                                           |
|   [ BM25 Lexical Retrieval ]                                [ Dense Semantic Retrieval ]                      |
|   - SQLite FTS5 Match Query                                 - Query Embedding (L2-norm)                       |
|   - Top-N Candidate Documents                               - FAISS IndexFlatIP / NumPy Dot-Product           |
|         |                                                         |                                           |
|         +---------------------------+-----------------------------+                                           |
|                                     |                                                                         |
|                                     v                                                                         |
|                     [ Reciprocal Rank Fusion (RRF) ]                                                          |
|                     - Standard Constant: k = 60                                                               |
|                     - Deduplication by Document Root Path                                                     |
|                                     |                                                                         |
|                                     v                                                                         |
|                     [ Rank-Ordered Search Results ]                                                           |
|                     - Source Path & Exact PDF Page Number Provenance                                          |
|                     - Text Snippet Excerpts & Chunk IDs                                                       |
+---------------------------------------------------------------------------------------------------------------+
```

---

## Core Retrieval Algorithms

### 1. Lexical Retrieval (SQLite FTS5 BM25)
Documents and individual PDF pages are chunked into sliding windows of 140 words with 30-word overlap. Chunks are ingested into an SQLite FTS5 virtual table. Queries are sanitized and evaluated using SQLite's internal Okapi BM25 implementation:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### 2. Dense Semantic Retrieval (Local Sentence-Transformers)
Text chunks are embedded locally using `all-MiniLM-L6-v2` (384 dimensions) and L2-normalized:
$$\mathbf{e} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
Vectors are stored in `.searchlab_cache/` keyed by document ID and model name. Nearest-neighbor search is executed via inner product (equivalent to cosine similarity for normalized vectors) using FAISS `IndexFlatIP` if installed, or an exact NumPy matrix dot-product fallback:
$$\text{Sim}(q, d) = \mathbf{e}_q \cdot \mathbf{e}_d$$

### 3. Reciprocal Rank Fusion (RRF)
To unify lexical and semantic retrieval without requiring calibrated score normalization, candidate rankings from BM25 and dense retrieval are fused via RRF:
$$RRF(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$
Where $\text{rank}_m(d)$ is the 1-based ranking position of document $d$ within retriever $m$'s candidate list. Documents missing from a retriever's top candidates receive no score contribution from that mode.

### 4. Evaluation Metrics
The evaluator computes document-level metrics against gold labels:
- **Recall@k**: Fraction of relevant documents retrieved in top-$k$:
  $$\text{Recall}@k = \frac{|\mathcal{R}_k \cap \mathcal{G}|}{|\mathcal{G}|}$$
- **Mean Reciprocal Rank (MRR@k)**: Position of the first relevant document:
  $$\text{MRR}@k = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q^*} \quad (\text{if } \text{rank}_q^* \le k, \text{ else } 0)$$
- **Normalized Discounted Cumulative Gain (nDCG@k)**:
  $$\text{DCG}@k = \sum_{i=1}^k \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$$

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AdityaInamdar334/Deep-Learning-Implementations.git
cd Deep-Learning-Implementations/laptop-search-lab

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Core dependencies (Standard library + PyMuPDF for PDF support)
pip install -r requirements.txt

# Optional dependencies for Dense & Hybrid search
pip install -r requirements-optional.txt
# Optional: pip install faiss-cpu
```

### 2. Run Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The test suite validates:
1. Slotted chunking boundaries and overlap invariant.
2. Invariant assertions on invalid window parameters.
3. FTS5 query token escaping and SQL injection defense.
4. Reciprocal Rank Fusion (RRF) order fusion.
5. Exact evaluation metrics computation (Recall, MRR, nDCG).
6. Transactional document update and deletion pruning.
7. Symlink traversal avoidance (sandbox integrity).

---

## Usage Guide

### Step 1: Generate Synthetic Demo Corpus

```bash
python3 build_demo.py
```
Generates 36 structured documents across Markdown (`.md`), plain text (`.txt`), and multi-page PDFs (`.pdf`), alongside 36 gold-labeled benchmark queries in `data/queries.jsonl`.

### Step 2: Index Documents

```bash
python3 searchlab.py index data/demo --db results/demo.sqlite
```

The indexer computes SHA-256 hashes for all eligible files:
- **Unchanged files** are skipped.
- **Modified files** are re-chunked and updated transactionally.
- **Deleted files** are purged from the SQLite index.

To index your own personal documents:
```bash
python3 searchlab.py index ~/Documents/notes --db results/my-notes.sqlite
```

### Step 3: Query the Search Engine

#### Lexical BM25 Search:
```bash
python3 searchlab.py search "When does the Cedar deployment start?" --db results/demo.sqlite --mode bm25 -k 5
```

#### Dense Semantic Search (requires `sentence-transformers`):
```bash
python3 searchlab.py search "How quickly should a customer hear back?" --db results/demo.sqlite --mode dense -k 5
```

#### Hybrid RRF Search:
```bash
python3 searchlab.py search "How quickly should a customer hear back?" --db results/demo.sqlite --mode hybrid -k 5
```

---

## Quantitative Evaluation & Plotting

Run automated evaluations across all 36 gold questions to benchmark retrieval accuracy:

```bash
# Evaluate BM25
python3 searchlab.py eval --db results/demo.sqlite --queries data/queries.jsonl --mode bm25 --out results/bm25.json

# Evaluate Dense Retrieval
python3 searchlab.py eval --db results/demo.sqlite --queries data/queries.jsonl --mode dense --out results/dense.json

# Evaluate Hybrid Fusion
python3 searchlab.py eval --db results/demo.sqlite --queries data/queries.jsonl --mode hybrid --out results/hybrid.json

# Render comparison SVG chart (Zero external dependencies)
python3 plot_results.py results/bm25.json results/dense.json results/hybrid.json --out results/comparison.svg
```

### Demo Baseline Results (36 synthetic multi-format corpus)

| Metric | Lexical (BM25) | Dense (all-MiniLM-L6-v2) | Hybrid (RRF k=60) |
|:---|:---:|:---:|:---:|
| **Recall@5** | 1.000 | 0.944 | 1.000 |
| **MRR@5** | 0.986 | 0.889 | 0.986 |
| **nDCG@5** | 0.989 | 0.902 | 0.989 |

---

## Privacy, Security & Threat Model

1. **Zero External Network Calls**: Runtime search and indexing operate 100% on local hardware. No telemetry, analytics, or third-party search APIs are contacted.
2. **Symlink Boundary Protection**: The crawler explicitly rejects symlinks, preventing malicious or accidental link traversals outside designated folders (e.g., preventing exposure of `~/.ssh` or `/etc`).
3. **Strict Extension Whitelist**: Only `.md`, `.markdown`, `.txt`, and `.pdf` files are scanned. Executables, hidden dotfiles, and binary blobs are ignored.
4. **Size Cap Enforcement**: Files larger than 12MB are skipped to prevent memory exhaustion from oversized logs or dumps.
5. **No Background Daemons**: Indexing executes on-demand; no background file-system watcher or indexing service is installed.

---

## CLI Reference

### `searchlab.py index`
```
usage: searchlab.py index [-h] [--db DB] [--chunk CHUNK] [--overlap OVERLAP] path
```
- `path`: Target directory path containing documents to index.
- `--db`: Destination SQLite database path (default: `searchlab.sqlite`).
- `--chunk`: Chunk size in whitespace words (default: `140`).
- `--overlap`: Overlap between adjacent chunks (default: `30`).

### `searchlab.py search`
```
usage: searchlab.py search [-h] [--db DB] [--mode {bm25,dense,hybrid}] [-k K] query
```
- `query`: Free-text search query string.
- `--mode`: Retrieval algorithm (`bm25`, `dense`, or `hybrid`).
- `-k`: Number of top results to return (default: `5`).

### `searchlab.py eval`
```
usage: searchlab.py eval [-h] [--db DB] --queries QUERIES [--mode {bm25,dense,hybrid}] [-k K] [--out OUT]
```
- `--queries`: Path to line-delimited JSON (`.jsonl`) containing test queries and ground-truth relevant document paths.
- `--out`: Destination JSON path for computed evaluation metrics.

---

## File Structure

```
laptop-search-lab/
├── README.md                    # Architecture guide, math formulations, and CLI docs
├── RESULTS_TEMPLATE.md          # Scientific benchmark reporting guide
├── requirements.txt             # Core dependencies (pymupdf)
├── requirements-optional.txt    # Optional dependencies (sentence-transformers, faiss-cpu)
├── searchlab.py                 # Core CLI search engine, indexer, and evaluator
├── build_demo.py                # Synthetic corpus generator (36 docs + 36 queries)
├── plot_results.py              # Lightweight dependency-free SVG chart renderer
├── data/
│   ├── demo/                    # Synthetic Markdown, text, and PDF documents
│   └── queries.jsonl            # 36 relevance-labeled test queries
├── results/
│   ├── demo.sqlite              # Pre-indexed demo database
│   ├── bm25.json                # Pre-evaluated benchmark results
│   ├── bm25.svg                 # Generated retrieval quality chart
│   └── pdf-sample.png           # Visual rendering of sample synthetic PDF
└── tests/
    └── test_searchlab.py        # Automated unit test suite (7 tests)
```

---

## References

- [Robertson & Zaragoza (2009): The Probabilistic Relevance Framework: BM25 and Beyond](https://dl.acm.org/doi/10.1561/1500000019)
- [Cormack, Clarke & Büttcher (2009): Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods (SIGIR)](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [Reimers & Gurevych (2019): Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP)](https://arxiv.org/abs/1908.10084)
- [Johnson, Douze & Jégou (2019): Billion-scale similarity search with GPUs (IEEE TBD)](https://arxiv.org/abs/1702.08734)
- [SQLite FTS5 Extension Documentation](https://www.sqlite.org/fts5.html)
