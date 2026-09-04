"""
Hybrid retrieval: BM25 keyword search fused with vector similarity via
Reciprocal Rank Fusion (RRF).

Why this matters for this project: pure vector search sometimes misses
exact-term queries (a specific drug name, plan code, or policy number)
because semantic embeddings blur precise tokens into a general "topic"
vector. BM25 is the opposite -- it's exact-term-sensitive but has no sense
of meaning/paraphrase. Fusing both catches what either one alone would miss.

Filtering note: because this corpus is small (~100-200 chunks), BM25 scores
the ENTIRE corpus on every query (cheap at this scale) and THEN applies the
jurisdiction/audience filter to the fully-scored list, before taking the
top N. This is deliberately different from a "retrieve top-k, then filter"
pattern -- filtering after a truncated top-k would silently under-return
results (e.g. asking for 5 results but getting 2 after filtering knocks out
3). Scoring everything first means filtering never loses information.

Run standalone to sanity-check the fusion logic:
  python hybrid_search.py
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_chunks(chunks_file: Path) -> list[dict]:
    chunks = []
    with chunks_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class BM25Index:
    """Wraps rank_bm25 with the chunk metadata needed for filtering."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        jurisdiction_values: list[str],
        audience: str,
        top_n: int = 20,
    ) -> list[dict]:
        """Scores the whole corpus, filters, then ranks -- see module
        docstring for why filtering happens after full scoring, not after
        a truncated top-k."""
        scores = self.bm25.get_scores(tokenize(query))

        scored = [
            {**self.chunks[i], "bm25_score": float(scores[i])}
            for i in range(len(self.chunks))
            if self.chunks[i]["jurisdiction"] in jurisdiction_values
            and self.chunks[i]["audience"] == audience
        ]
        scored.sort(key=lambda c: c["bm25_score"], reverse=True)
        return scored[:top_n]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "chunk_id",
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """
    Standard RRF: score(doc) = sum over lists of 1 / (k + rank_in_list),
    where rank is 1-indexed. A doc absent from a list contributes 0 for
    that list. k=60 is the commonly-used default from the original RRF
    paper (Cormack et al.) -- it dampens the impact of rank differences
    at the tail while still rewarding docs that rank well in either list.

    Returns fused chunk dicts (metadata from whichever list had them first),
    each with an added "rrf_score", sorted descending, truncated to top_k.
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk[id_key]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_lookup.setdefault(cid, chunk)

    fused = [
        {**chunk_lookup[cid], "rrf_score": score}
        for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return fused[:top_k]


# --- Standalone sanity check -------------------------------------------
if __name__ == "__main__":
    # Synthetic corpus where a specific drug name only a keyword search
    # would reliably surface, to demonstrate BM25 catching what a vague
    # semantic match might rank lower.
    test_chunks = [
        {"chunk_id": "a", "text": "Metformin is a common first-line medication for type 2 diabetes management.",
         "jurisdiction": "federal", "audience": "patient"},
        {"chunk_id": "b", "text": "Diabetes is a chronic condition affecting blood sugar regulation in the body.",
         "jurisdiction": "federal", "audience": "patient"},
        {"chunk_id": "c", "text": "California requires insurers to cover diabetic supplies under state law.",
         "jurisdiction": "CA", "audience": "patient"},
        {"chunk_id": "d", "text": "Chronic kidney disease patients require ACE inhibitor therapy per clinical guidelines.",
         "jurisdiction": "federal", "audience": "clinician"},
    ]

    index = BM25Index(test_chunks)

    print("--- BM25 search: 'metformin' (exact drug name) ---")
    results = index.search("metformin", jurisdiction_values=["federal"], audience="patient")
    for r in results:
        print(f"  score={r['bm25_score']:.3f} | {r['chunk_id']} | {r['text'][:60]}")

    print("\n--- BM25 search: 'metformin', filtered to jurisdiction=CA (should still see federal + CA) ---")
    results = index.search("metformin", jurisdiction_values=["federal", "CA"], audience="patient")
    for r in results:
        print(f"  score={r['bm25_score']:.3f} | jurisdiction={r['jurisdiction']} | {r['chunk_id']}")

    print("\n--- RRF fusion test: BM25 favors 'a' (exact term), fake vector favors 'b' (semantic) ---")
    bm25_ranked = [test_chunks[0], test_chunks[1]]  # a, b
    vector_ranked = [test_chunks[1], test_chunks[0]]  # b, a  (opposite order)
    fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked], top_k=4)
    for f in fused:
        print(f"  rrf_score={f['rrf_score']:.4f} | {f['chunk_id']}")
    print("\n(Both 'a' and 'b' ranked #1 in one list and #2 in the other -- RRF should score them equally)")
