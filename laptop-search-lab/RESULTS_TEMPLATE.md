# Experimental report template

## Hardware and software

- Machine / CPU / RAM / OS:
- Python version, SQLite version:
- PyMuPDF / sentence-transformers / FAISS version:
- Embedding model name and revision:
- Number of files, indexable pages, chunks:
- File-type distribution and OCR policy:
- Dataset path, query labels and annotation policy:
- Index build time and embedding generation time:

## Retrieval results

| Method | Recall@5 | MRR@5 | nDCG@5 | Cold latency (ms) | Warm median latency (ms) | Index/cache disk size |
|---|---:|---:|---:|---:|---:|---:|
| SQLite FTS5 BM25 | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense local MiniLM | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid BM25 + dense RRF | TBD | TBD | TBD | TBD | TBD | TBD |

## Error audit

Record at least ten failures or false positives, classified as exact identifier mismatch, phrase paraphrase, chunk boundary loss, PDF extraction issue, duplicate file, outdated information, or missing gold label.

## Experimental cautions

1. No need to include a language model to evaluate search quality. Do not conflate retrieval and answer-generation accuracy.
2. Report document-level and chunk-level results separately when applicable.
3. Report the same corpus and held-out questions for every method.
4. Distinguish first-call model initialization from warm retrieval latency.
5. Do not claim the baseline was evaluated on real private files or production workloads without doing so.
