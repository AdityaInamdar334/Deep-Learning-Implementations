# Local RAG Research Results Template

## Environment
- Mac/PC, CPU, RAM/GPU:
- OS, Python, Ollama versions:
- Qwen2.5 3B model digest:
- Sentence embedding model revision:
- Cross encoder revision:
- Seed, corpus SHA256, questions SHA256:
- Query warmup and trial repetitions:

## Experimental grid

| Setting | Chunks (words) | Overlap | Retrieval | Reranker | Candidates | k | Evidence Recall@5 | Answer EM | Grounded correct | Median query retrieval ms | Median generation s |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | 40 | 8 | SQLite FTS5 | none | 20 | 5 | TBD | TBD | TBD | TBD | TBD |
| B | 100 | 20 | SQLite FTS5 | none | 20 | 5 | TBD | TBD | TBD | TBD | TBD |
| C | 180 | 30 | SQLite FTS5 | none | 20 | 5 | TBD | TBD | TBD | TBD | TBD |
| D | 100 | 20 | FAISS dense | none | 20 | 5 | TBD | TBD | TBD | TBD | TBD |
| E | 100 | 20 | FAISS dense | cross | 20 | 5 | TBD | TBD | TBD | TBD | TBD |

## Failure taxonomy
1. Retrieval miss: gold evidence not retrieved.
2. Citation omission: correct answer without a valid supporting citation.
3. Citation laundering: cited chunk exists but does not support answer.
4. Reader error: evidence retrieved, model answers incorrectly.
5. Failed abstention: model invents an answer to an unanswerable question.
6. Too-aggressive abstention: gold evidence retrieved but model answers UNKNOWN.

## Reporting
- Show per-category and per-question outcomes, not only aggregate accuracy.
- Report deterministic baseline and model generation results separately.
- Cite at least three representative disagreements between retrieval conditions.
- Do not state that the model `understands` all 1,000 documents merely because it retrieves named facts in the synthetic corpus.
