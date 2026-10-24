"""
Tests that need the real models (bge-small embeddings, ms-marco cross-
encoder). Skipped unless RUN_MODEL_TESTS=1; CI runs them in a separate job.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")

MELLITUS = (
    "Diabetes is a disease in which your blood glucose, or blood sugar, levels are too high. "
    "Glucose comes from the foods you eat. Insulin is a hormone that helps the glucose get into "
    "your cells to give them energy."
)


@pytest.fixture(scope="module")
def reranker():
    from reranker import Reranker

    return Reranker()


def test_cross_encoder_accepts_matching_question(reranker):
    from api.main import CROSS_ENCODER_THRESHOLD

    score = reranker.rerank("what is diabetes and how does insulin work", [{"text": MELLITUS}])[0]["cross_encoder_score"]
    assert score >= CROSS_ENCODER_THRESHOLD


def test_cross_encoder_refuses_unrelated_question(reranker):
    from api.main import CROSS_ENCODER_THRESHOLD

    score = reranker.rerank("how do I change a car tire", [{"text": MELLITUS}])[0]["cross_encoder_score"]
    assert score < CROSS_ENCODER_THRESHOLD


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: ms-marco-MiniLM-L-6-v2 scores 'diabetes insipidus' highly against "
    "diabetes mellitus text. strict=True makes this test fail loudly if a model change fixes it.",
)
def test_cross_encoder_separates_insipidus_from_mellitus(reranker):
    from api.main import CROSS_ENCODER_THRESHOLD

    score = reranker.rerank("what is diabetes insipidus", [{"text": MELLITUS}])[0]["cross_encoder_score"]
    assert score < CROSS_ENCODER_THRESHOLD


def test_embedder_ranks_exact_topic_first():
    import numpy as np
    from fastembed import TextEmbedding

    from settings import EMBEDDING_MODEL

    emb = TextEmbedding(model_name=EMBEDDING_MODEL)
    q, copd, cookies = list(emb.embed([
        "what is COPD",
        "COPD, chronic obstructive pulmonary disease, makes it hard to breathe.",
        "Preheat the oven and mix butter with sugar for the cookie dough.",
    ]))
    assert float(np.dot(q, copd)) > float(np.dot(q, cookies))
