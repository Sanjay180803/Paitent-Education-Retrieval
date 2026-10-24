# Patient Education & Policy Q&A Gateway

A retrieval-augmented backend that answers patient questions about medical
conditions, Medicare, HIPAA, and state-specific insurance rules — tailored
to the asker's audience (patient vs. clinician), scoped to their
jurisdiction, and always grounded in cited sources rather than free
generation.

The engineering focus is **retrieval infrastructure correctness**, not
prompt engineering: jurisdiction-safe filtering, hybrid search, cache
invalidation without races, and a two-stage confidence gate built from
real evaluation data rather than guessed thresholds.

## Why this exists

Naive RAG in a healthcare/policy context is dangerous in specific,
nameable ways: a jurisdiction-mismatched answer (California rules served
to a Texas resident), a stale cached answer served after a policy update,
or a confidently-wrong answer to a question the corpus doesn't actually
cover (mistaking "diabetes insipidus" for "diabetes mellitus" because both
share the word "diabetes"). This project treats each of those as a
distinct engineering problem with a specific, testable fix — not something
a bigger LLM call would paper over.

## Architecture

```
Ingestion (5 sources)          MedlinePlus, PubMed (NCBI E-utilities),
                                CMS/Medicare, HHS/HIPAA, state insurance depts
        │
        ▼
Corpus consolidation           Normalizes differing field names/schemas
        │                      into one common record shape
        ▼
Paragraph-aware chunking       Never splits mid-sentence; preserves
        │                      section structure for policy/coverage tables
        ▼
Embedding + Qdrant index       FastEmbed (ONNX, no GPU) + local/server Qdrant,
        │                      metadata-filtered at the ANN query layer
        ▼
Hybrid retrieval (RRF)         Vector (semantic) + BM25 (exact-term) fused
        │                      via Reciprocal Rank Fusion
        ▼
Cross-encoder reranking        Second precision pass on the fused candidate
        │                      pool -- catches lexical-overlap traps a
        │                      bi-encoder structurally cannot
        ▼
Two-stage confidence gate      Cheap vector floor, then cross-encoder
        │                      threshold tuned from real eval data
        ▼
Redis cache (generation-       Jurisdiction- and schema-versioned cache
  and schema-versioned)        keys -- invalidation is a single atomic
        │                      INCR, not a race-prone delete
        ▼
FastAPI /query endpoint        Returns grounded answer + full source
                                citations, or an explicit refusal
```

## Key engineering decisions

**Jurisdiction filtering happens inside the vector-DB query itself**
(`query_filter` passed to Qdrant's ANN search), not as a post-hoc filter on
already-retrieved results. A federal-only or wrong-state chunk never
enters the candidate set — this is what makes the "no California content
in a Texas-scoped answer" property structurally guaranteed rather than
merely usually-true. Confirmed with a direct test asserting `CA` never
appears in a `TX`-scoped result.

**Hybrid retrieval (BM25 + vector, fused via Reciprocal Rank Fusion)**
because a bi-encoder embedding blurs exact terms (drug names, plan codes)
into general topic vectors. Verified with a controlled experiment: pure
vector search ranked a lisinopril-specific document *below* a generic
overview (0.83 vs. 0.55 cosine similarity); BM25 fusion corrected it into
a tie for first place.

**Cache invalidation via generation counters, not a reverse document
index.** The obvious design — track which cached answers used which
source document, delete them on update — creates a real race: a request
that starts before an update but finishes writing its cache entry after
invalidation has already run leaves a permanently stale answer with no one
left to delete it. Generation counters sidestep this entirely: every cache
key embeds the jurisdiction's current generation number, so a data update
is one atomic `INCR`. After that instant, every new request computes a
*different* key — reading stale data isn't unlikely, it's structurally
impossible, without any delete-and-hope-you-caught-it-in-time logic.

**Cache keys are also schema-versioned**, not just data-versioned. A code
deploy (e.g. adding the cross-encoder reranking stage) doesn't change any
jurisdiction's underlying data, so generation counters alone don't catch
it — old cache entries would be replayed verbatim after a deploy, silently
serving pre-deploy behavior. `SCHEMA_VERSION` is baked into every key for
exactly this reason; bumping it orphans old-format entries the same
structural way a generation bump orphans stale data.

**A two-stage confidence gate**, because a single cosine-similarity
threshold provably cannot separate "topically similar" from "the correct
answer." The eval harness caught this directly: "what is diabetes
insipidus" scored *higher* on vector cosine similarity (0.749) than
several genuinely-answerable baseline questions, because a bi-encoder
embeds query and candidate independently and cannot represent that
"insipidus" and "mellitus" name different diseases. A cross-encoder, which
reads the (query, candidate) pair jointly, catches this — confirmed by an
end-to-end test where both questions produced *identical* vector scores
but were correctly told apart post-rerank.

## What the eval harness found (and didn't paper over)

23 golden questions (11 answerable, 12 adversarial), each of the
adversarial ones designed to expose a specific failure mode — not just
"obviously off-topic" but topically-adjacent-and-wrong (car insurance vs.
health insurance, GDPR vs. HIPAA, income tax vs. ACA premium tax credits,
and the diabetes name confusion above).

A full threshold sweep across the cross-encoder's confidence score:

![Threshold sweep](docs/threshold_sweep.png)

The result is an honest finding, not a clean win: **there is no broad safe
operating region.** Overall accuracy peaks sharply at threshold = 0.01
(91.3%) and immediately drops off in both directions — the optimum is a
knife's edge between two failure clusters that a single scalar cannot
separate:

- Genuine answers to vaguely-phrased or synonym-mismatched questions
  scored as low as 0.012 on the cross-encoder
- Two disease-name near-misses (diabetes insipidus / diabetes mellitus
  type 3c) scored 0.843–0.963 — the general-purpose cross-encoder
  (`ms-marco-MiniLM-L-6-v2`, trained on web search relevance, not medical
  entity disambiguation) cannot tell they're different diseases

This is documented as a known, accepted limitation rather than tuned away:
a medical-domain-specific reranker or an NLI-based entity-disambiguation
step would be the real fix, and is out of scope for this iteration.

## Quick start (Docker Compose)

```bash
docker compose up --build
curl -X POST localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "does Medicare cover insulin", "jurisdiction": "CA"}'
```

This starts Redis, a Qdrant server, a one-shot indexer that builds the
index from the committed `data/processed/` files, and the API. Runtime
settings (Qdrant/Redis URLs, gate thresholds) are environment variables
documented in `settings.py`.

## Setup (without Docker)

```powershell
pip install -r requirements.txt
```

`sentence-transformers` pulls in `torch` and is the heaviest install in
the chain — expect it to take noticeably longer than the rest.

Get a free NCBI API key (optional, raises the PubMed rate limit from 3 to
10 req/sec): https://www.ncbi.nlm.nih.gov/account/

```powershell
$env:NCBI_API_KEY="your_key_here"
```

Start Redis (used for query caching):
```powershell
docker run -p 6379:6379 redis
```

## Running the pipeline

```powershell
# 1. Ingest (each is independent; run whichever sources you need)
python fetch_medlineplus.py
python fetch_pubmed.py
python fetch_cms.py
python fetch_state_insurance.py    # pages listed in config/state_pages.toml
python load_manual_pages.py        # for sources that block scripted fetches (see script docstring)

# 2. Build the unified corpus, chunk it, embed and index
python build_corpus.py
python chunk_corpus.py
python embed_and_index.py

# 3. Serve
python -m uvicorn api.main:app --reload --port 8000

# 4. Evaluate
python run_eval.py
```

## API usage

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post `
  -ContentType "application/json" `
  -Body '{"question": "does Medicare cover insulin", "audience": "patient", "jurisdiction": "CA"}'
```

Response includes `answer_grounded`, full source citations (`url`,
`title`, `jurisdiction`, `effective_date`), `cache_hit`, and both raw
confidence signals (`top_vector_score`, `top_cross_encoder_score`) for
debugging/eval purposes.

## Project structure

```
common.py                    Shared rate limiting, disk caching, robots.txt checks
fetch_*.py                   Per-source ingestion scripts
load_manual_pages.py         Loader for sources that block scripted fetches
build_corpus.py              Consolidates all sources into one corpus.jsonl
chunk_corpus.py              Paragraph-aware chunking
embed_and_index.py           FastEmbed + Qdrant indexing
hybrid_search.py             BM25 index + Reciprocal Rank Fusion
reranker.py                  Cross-encoder reranking
query_cache.py               Redis caching with generation/schema-versioned keys
invalidate_jurisdiction.py   CLI to bump a jurisdiction's cache generation
api/main.py                  FastAPI serving layer
golden_questions.json        Eval harness golden question set
run_eval.py                  Eval harness: hit@k, MRR, violation checks, threshold sweep data
docs/threshold_sweep.png     Confidence-gate threshold sweep chart
```

## Known limitations

- Cross-encoder reranking cannot distinguish medically-distinct conditions
  that share vocabulary (see eval findings above) — would need a
  domain-specific or NLI-based reranker to fully close.
- State insurance coverage is currently CA/TX/FL only; extending to more
  states means finding and verifying real consumer-guide URLs per state
  (`fetch_state_insurance.py` has no unified API to pull from).
- No LLM generation/synthesis step — the endpoint returns grounded source
  excerpts directly rather than a paraphrased answer, a deliberate scope
  choice so "grounding" stays literal rather than something that could
  drift from the source during generation.
