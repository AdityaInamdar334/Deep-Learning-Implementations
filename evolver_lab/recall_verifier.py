"""Pure Python reference implementation of Evolver's verifyGeneRecall function.

Models the lexical matching and verification algorithm from EvoMap/evolver
(packages/evolver-core/dist/ops/recall.js at v2.0.33):
1. Extracts significant vocabulary terms (non-stopwords) from the Gene's strategy instructions.
2. Checks whether assistant turns are present in the conversation transcript.
3. Computes lexical overlap between assistant text and the extracted strategy vocabulary.
4. Classifies recall into:
   - 'unknown': when no assistant turn exists (e.g., strategy was injected in user prompt only).
   - 'used': when lexical overlap meets or exceeds the threshold (default: 0.3).
   - 'unused': when an assistant turn exists but overlap is below threshold.

Demonstrates the core verification challenge: lexical overlap marks both affirmative
declarations ("I plan to...") and explicit denials ("I did not...") as 'used' (score: 1.00)
because word overlap alone cannot distinguish execution from mention or negation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

# Common English stopwords filtered during distinctive token extraction
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "their", "our", "its", "mine", "yours",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "plan",
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-zA-Z0-9_-]+", text.lower())


def extract_keywords(strategy_list: List[str]) -> Set[str]:
    """Extract distinctive non-stopword tokens from the Gene's strategy steps."""
    keywords = set()
    for step in strategy_list:
        tokens = tokenize(step)
        for tok in tokens:
            if tok not in STOPWORDS and len(tok) > 1:
                keywords.add(tok)
    return keywords


def verify_gene_recall(
    gene: Dict[str, Any],
    turns: List[Dict[str, str]],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """Evaluate whether an agent's execution transcript recalled/used the given Gene.

    Parameters
    ----------
    gene : Dict[str, Any]
        Gene definition containing at least a 'strategy' list of strings.
    turns : List[Dict[str, str]]
        List of message turns with 'role' ('user' | 'assistant') and 'text'.
    threshold : float, default 0.3
        Minimum overlap ratio required to classify as 'used'.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'recalled' ('used' | 'unused' | 'unknown') and 'score' (float).
    """
    strategy_steps = gene.get("strategy", [])
    keywords = extract_keywords(strategy_steps)

    if not keywords:
        return {"recalled": "unused", "score": 0.0, "matched": []}

    assistant_turns = [turn for turn in turns if turn.get("role") == "assistant"]

    # If no assistant response exists, recall status cannot be evaluated
    if not assistant_turns:
        return {"recalled": "unknown", "score": 0.0, "matched": []}

    assistant_text = " ".join(turn.get("text", "") for turn in assistant_turns)
    assistant_tokens = set(tokenize(assistant_text))

    matched_keywords = keywords.intersection(assistant_tokens)
    score = len(matched_keywords) / len(keywords)

    recalled = "used" if score >= threshold else "unused"

    return {
        "recalled": recalled,
        "score": round(score, 4),
        "matched": sorted(list(matched_keywords)),
    }

