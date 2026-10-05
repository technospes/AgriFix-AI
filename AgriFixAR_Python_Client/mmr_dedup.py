from __future__ import annotations
import logging
import math
import re
from typing import Dict, List, Set, Tuple, Optional
from langchain_core.documents import Document
from utils.text_utils import tokenize as _tokenize
logger = logging.getLogger(__name__)

# ── MMR hyperparameters ────────────────────────────────────────────────────────
# λ controls relevance vs diversity tradeoff.
# v2 default: 0.50 (equal weight) for stronger diversity across source docs.
# Set higher (→ 1.0) to prioritise relevance; lower (→ 0.3) for max diversity.
_MMR_LAMBDA = float(__import__("os").environ.get("AGRIFIX_MMR_LAMBDA", "0.50"))

# Similarity threshold: chunks THIS similar to an already-selected chunk
# are treated as near-duplicates and deprioritised (but not hard-blocked).
_NEAR_DUPLICATE_THRESHOLD = 0.85

# Feature B — Hard cap: at most this many chunks from the same parent document
# may appear in the final selection, regardless of MMR score.
# Parent is derived from chunk_id prefix: "<parent>_<section>".
# Override via AGRIFIX_MMR_MAX_PER_PARENT.
_MAX_PER_PARENT: int = int(__import__("os").environ.get("AGRIFIX_MMR_MAX_PER_PARENT", "2"))

# Stop words for TF-IDF cosine (keep in sync with rag.py _STOP_WORDS)
_STOP_WORDS: Set[str] = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "from","is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall","can",
    "not","no","nor","so","yet","both","either","each","more","most","other",
    "some","such","than","that","this","these","those","it","its","my","your",
    "his","her","our","their","i","we","you","he","she","they","what","which",
    "who","when","where","how","if","as","up","out","about","into","through",
    "during","before","after","above","below","between","there","here","then",
    "any","all",
    "hai","hain","ka","ke","ki","ko","se","mein","par","aur","ek","yeh","woh",
    "kya","kab","kaise","nahi","nahin","bhi","toh","jo","jab","agar","lekin",
}


# ── Text utilities ─────────────────────────────────────────────────────────────

def _tf_vector(tokens: List[str]) -> Dict[str, float]:
    """Normalised term-frequency vector."""
    from collections import Counter
    counts = Counter(tokens)
    total  = max(sum(counts.values()), 1)
    return {t: c / total for t, c in counts.items()}


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two TF vectors."""
    if not v1 or not v2:
        return 0.0
    dot   = sum(v1.get(t, 0.0) * v2.get(t, 0.0) for t in v1)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ── Parent-group helpers (Feature A + B) ──────────────────────────────────────

def _extract_parent(doc: "Document") -> Optional[str]:
    """
    Extract the parent document identifier from chunk metadata.
    Checks (in order):
      1. meta["parent_id"]  — explicit field
      2. meta["chunk_id"] split on "_"  — e.g. "6c1dc38bc4b8_symptoms" → "6c1dc38bc4b8"
      3. meta["source"] — fallback to source filename
    Returns None if no parent can be determined.
    """
    meta = doc.metadata or {}
    if "parent_id" in meta:
        return str(meta["parent_id"])
    chunk_id = str(meta.get("chunk_id", ""))
    if chunk_id and "_" in chunk_id:
        return chunk_id.rsplit("_", 1)[0]
    source = meta.get("source")
    if source:
        return str(source)
    return None




# ── Core MMR algorithm ─────────────────────────────────────────────────────────

def mmr_deduplicate(
    candidates: List[Tuple[Document, float]],
    query_tokens: List[str],
    top_n: int = 20,
    mmr_lambda: float = _MMR_LAMBDA,
    max_per_parent: int = _MAX_PER_PARENT,
) -> List[Tuple[Document, float]]:
    """
    Apply parent-aware Maximal Marginal Relevance to a candidate list.

    Args:
        candidates      List of (Document, vector_score) from semantic retrieval.
        query_tokens    Tokenised query (from rag.py _tokenize()).
        top_n           Number of chunks to return (same as _CANDIDATE_K).
        mmr_lambda      λ — relevance vs diversity tradeoff (0.3-1.0).
                        v2 default: 0.50 for stronger cross-document diversity.
        max_per_parent  Hard cap on chunks per parent document (Feature B).
                        The single best chunk from any parent is always included
                        (Feature D), even when max_per_parent=1.

    Returns:
        Deduplicated list of (Document, score) with length ≤ top_n.
        Scores are original vector scores (unchanged — reranker sees them).

    Algorithm (v2 — parent-aware):
        parent_counts tracks how many chunks from each parent are selected.

        For iteration i:
          1. Compute standard MMR score for each remaining candidate.
          2. If candidate's parent count >= max_per_parent AND it is NOT the
             single highest-scoring candidate overall (Feature D), apply an
             additional parent-diversity penalty to its MMR score.
          3. Select argmax MMR score; increment parent_counts[parent].

    Why TF-IDF cosine for inter-doc similarity:
        Fast, embedding-free, <1ms for 20 docs. Avoids storing raw vectors.
    """
    if not candidates:
        return []
    if len(candidates) <= 1:
        return candidates

    # Pre-compute TF vectors for all candidates
    doc_vectors = [
        _tf_vector(_tokenize(doc.page_content))
        for doc, _ in candidates
    ]

    # relevance_scores[i] = original ChromaDB vector score
    relevance_scores = [score for _, score in candidates]

    # Feature D: identify the globally highest-relevance candidate index
    top_relevance_idx = max(range(len(candidates)), key=lambda i: relevance_scores[i])

    # Parent tracking for cap enforcement (Feature B)
    parent_counts: Dict[str, int] = {}

    selected_indices: List[int] = []
    remaining_indices: List[int] = list(range(len(candidates)))

    dedup_events = 0
    parent_cap_events = 0

    for iteration in range(min(top_n, len(candidates))):
        best_idx   = -1
        best_score = float("-inf")

        for i in remaining_indices:
            doc, _ = candidates[i]
            parent = _extract_parent(doc)

            # Relevance term
            relevance = relevance_scores[i]

            # Diversity term: similarity to already-selected chunks
            if selected_indices:
                max_sim_to_selected = max(
                    _cosine(doc_vectors[i], doc_vectors[j])
                    for j in selected_indices
                )
            else:
                max_sim_to_selected = 0.0

            # Log near-duplicates
            if max_sim_to_selected >= _NEAR_DUPLICATE_THRESHOLD and selected_indices:
                dedup_events += 1
                logger.debug(
                    "MMR near-duplicate (sim=%.3f): %s",
                    max_sim_to_selected,
                    doc.metadata.get("chunk_id", i),
                )

            # Standard MMR score
            mmr_score = (
                mmr_lambda * relevance
                - (1 - mmr_lambda) * max_sim_to_selected
            )

            # Feature A + B: parent-diversity penalty
            # Apply when parent cap exceeded, UNLESS this is the top-relevance
            # chunk (Feature D guarantees it always gets a fair shot).
            if parent is not None and i != top_relevance_idx:
                count = parent_counts.get(parent, 0)
                if count >= max_per_parent:
                    # Penalty scales with how far over the cap we are
                    excess = count - max_per_parent + 1
                    mmr_score -= (1 - mmr_lambda) * excess * 0.3
                    parent_cap_events += 1
                    logger.debug(
                        "Parent cap penalty (parent=%s excess=%d) on %s",
                        parent, excess, doc.metadata.get("chunk_id", i),
                    )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # Update parent count
        selected_doc = candidates[best_idx][0]
        parent = _extract_parent(selected_doc)
        if parent:
            parent_counts[parent] = parent_counts.get(parent, 0) + 1

    logger.info(
        "MMR dedup: %d → %d chunks | %d near-dupes | %d parent-cap penalties | λ=%.2f max_per_parent=%d",
        len(candidates), len(selected_indices),
        dedup_events, parent_cap_events,
        mmr_lambda, max_per_parent,
    )

    return [candidates[i] for i in selected_indices]


# ── Convenience wrapper that accepts raw query string ──────────────────────────

def mmr_deduplicate_from_query(
    candidates: List[Tuple[Document, float]],
    query: str,
    top_n: int = 20,
) -> List[Tuple[Document, float]]:
    """
    Convenience wrapper: accepts raw query string instead of pre-tokenised list.
    Use this in rag.py's retrieve_with_confidence() for the cleanest integration.

    Example in rag.py retrieve_with_confidence():

        # After semantic retrieval:
        candidates = _semantic_retrieve(vector_db, query, machine_type, ...)

        # Deduplicate BEFORE hybrid reranking:
        from mmr_dedup import mmr_deduplicate_from_query
        candidates = mmr_deduplicate_from_query(candidates, query, top_n=_CANDIDATE_K)

        # Then continue to _hybrid_rerank() as before:
        reranked, top_score, confidence_label = _hybrid_rerank(query, candidates, ...)
    """
    from rag import _tokenize  # avoids circular import — rag.py defines this already
    query_tokens = _tokenize(query.lower())
    return mmr_deduplicate(candidates, query_tokens, top_n=top_n)


# ── Deduplication statistics for logging ──────────────────────────────────────

def dedup_stats(
    before: List[Tuple[Document, float]],
    after:  List[Tuple[Document, float]],
) -> dict:
    """Return a dict of before/after statistics for logging."""
    before_ids = [d.metadata.get("chunk_id", i) for i, (d, _) in enumerate(before)]
    after_ids  = [d.metadata.get("chunk_id", i) for i, (d, _) in enumerate(after)]
    removed    = set(before_ids) - set(after_ids)
    return {
        "before_n":   len(before),
        "after_n":    len(after),
        "removed_n":  len(removed),
        "removed_ids": list(removed)[:5],
        "diversity_gain_pct": round((len(removed) / max(len(before), 1)) * 100, 1),
    }


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from langchain_core.documents import Document

    # Simulate a near-duplicate candidate pool (common in small knowledge bases)
    chunks = [
        ("Motor hums but does not rotate. Check start capacitor. Replace if bulging.", 0.91),
        ("Motor makes humming sound but shaft not spinning. Inspect capacitor first.", 0.89),  # near-dup
        ("Motor hums, shaft stationary. Capacitor may have failed. Test with multimeter.",  0.87),  # near-dup
        ("Check three-phase supply voltage at motor terminals before capacitor test.", 0.81),
        ("Impeller blocked by sand or debris. Remove pump casing and clean impeller.", 0.78),
        ("Mechanical seal leaking at shaft. Replace shaft seal kit.", 0.72),
    ]

    candidates = [
        (Document(
            page_content=text,
            metadata={"chunk_id": f"chunk_{i}", "machine_type": "water_pump"}
        ), score)
        for i, (text, score) in enumerate(chunks)
    ]

    print(f"Before MMR: {len(candidates)} chunks")
    query_tokens = _tokenize("water pump motor humming not rotating")
    result = mmr_deduplicate(candidates, query_tokens, top_n=4)
    print(f"After MMR:  {len(result)} chunks\n")

    print("Selected chunks:")
    for doc, score in result:
        print(f"  [{score:.2f}] {doc.page_content[:70]}...")

    stats = dedup_stats(candidates, result)
    print(f"\nStats: {stats}")
    print("\nExpected: chunks 0,3,4,5 selected; chunks 1,2 removed as near-duplicates.")