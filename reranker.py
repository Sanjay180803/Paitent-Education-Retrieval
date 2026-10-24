"""
Cross-encoder reranking: a second, more precise relevance pass on top of
the fast bi-encoder (vector) + BM25 hybrid retrieval.

WHY THIS EXISTS: the eval harness (run_eval.py) exposed a real failure mode
no cosine-similarity threshold can fix. "What is diabetes insipidus" scored
HIGHER on vector cosine similarity (0.749) than several genuinely-answerable
baseline questions -- because a bi-encoder embeds "diabetes insipidus" and
"diabetes mellitus content" into similar topic-space vectors. It correctly
recognizes the TOPIC overlap; it cannot tell that they are different
diseases, because a bi-encoder never reads the query and the candidate
together -- it embeds each independently, so nothing in the candidate's
embedding is conditioned on the query's specific wording.

A cross-encoder is a different model shape: it takes the (query, candidate)
PAIR as joint input to one transformer pass, so it can actually attend to
the mismatch between "insipidus" and the candidate's "mellitus"/"blood
glucose"/"insulin" content. This is strictly slower (can't be
precomputed/indexed like embeddings -- it must run at query time on every
candidate), which is exactly why it's a RERANKING step over a small
candidate pool (~10-20) from cheap retrieval, not a replacement for it.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 -- a standard, well-established
choice trained on real query-passage relevance judgments (MS MARCO), small
enough (~80MB) to run on CPU at interactive latency for a candidate pool
this size.
"""

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME):
        # Imported here so modules that import Reranker (the API, tests)
        # don't pull in torch until a reranker is actually constructed.
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, candidates: list[dict], text_key: str = "text") -> list[dict]:
        """
        Scores every candidate against the question with the cross-encoder
        and returns them sorted descending by that score (added as
        'cross_encoder_score', a sigmoid-activated 0-1 relevance probability).

        Called on a small candidate pool (post-fusion, pre-final-cutoff) --
        NOT on the whole corpus. Cost scales with len(candidates), which is
        why upstream retrieval narrows to ~10-20 before this runs.
        """
        if not candidates:
            return []

        pairs = [(question, c[text_key]) for c in candidates]
        scores = self.model.predict(pairs, activation_fct=_sigmoid())

        scored = [{**c, "cross_encoder_score": float(s)} for c, s in zip(candidates, scores)]
        scored.sort(key=lambda c: c["cross_encoder_score"], reverse=True)
        return scored


def _sigmoid():
    import torch
    return torch.nn.Sigmoid()
