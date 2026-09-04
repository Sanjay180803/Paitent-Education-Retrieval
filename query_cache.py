"""
Redis caching layer for the /query endpoint, with jurisdiction-scoped
cache invalidation on data updates.

DESIGN: generation counters, not a reverse doc-id index.

The obvious-seeming approach is: track which cached answers used which
source document, then delete those specific cache entries when the
document updates. That requires a reverse index (doc_id -> set of cache
keys) and creates a real race: a request that starts BEFORE an update but
finishes (and writes its cache entry) AFTER the invalidation step has
already run will leave a stale answer cached with no one left to delete
it -- the same class of bug as the credit-double-admission race in the
RPC admission-control gateway, just in a different shape.

Generation counters sidestep the race entirely instead of fixing it after
the fact. Each jurisdiction ("federal", "CA", "TX", "FL", ...) has an
integer generation in Redis. Every cache KEY embeds the current generation
number(s) for the jurisdictions it depends on. Updating a jurisdiction's
data is one atomic INCR -- after that instant, every new request computes
a new cache key (because it reads the bumped generation), so it is
*structurally impossible* for a request to read a stale-generation cache
entry: the key itself changes, rather than relying on finding and deleting
the old one in time. Old-generation entries are simply abandoned and
expire via TTL. No Lua script is needed here because there is no
multi-step read-modify-write to make atomic -- a single INCR already is.

Cache key shape:
  query:{audience}:{jurisdiction_values sorted}:{gen_federal}[:{gen_state}]:{question_hash}

A query scoped to CA depends on BOTH the federal generation and the CA
generation (since CA answers can include federal baseline chunks) -- so
bumping either one invalidates CA-scoped cache entries, while a federal-only
query's key is unaffected by a CA-only data update. This mirrors the same
jurisdiction precedence logic already used for retrieval filtering.

SCHEMA_VERSION: generation counters version the underlying DATA (a
jurisdiction's content changing). They do NOT version the SERVING LOGIC
itself -- if you change what the endpoint computes (e.g. adding a
cross-encoder reranking stage, changing the confidence-gating algorithm,
or altering the response shape), old cache entries are still "valid" by
the generation-counter rule, since no jurisdiction's underlying data
changed. They get replayed verbatim, silently serving pre-deploy behavior
after a deploy -- exactly what happened when the cross-encoder reranker
was added but cached vector-only answers kept being served regardless.

SCHEMA_VERSION is baked into every cache key for this reason. Bump it any
time retrieval/gating/response-shape logic in api/main.py changes. Old
keys become unreachable immediately -- same structural-impossibility
argument as the generation counters, just versioning code instead of data.
"""

import hashlib
import json
from typing import Optional

import redis

DEFAULT_TTL_SECONDS = 3600  # 1 hour; tune based on how often source data actually changes

# Bump this whenever /query's retrieval or gating LOGIC changes (not just
# when underlying data changes -- that's what jurisdiction generations are
# for). This is what would have caught the stale-cache-after-reranker-
# deploy bug automatically instead of needing a manual cache flush.
SCHEMA_VERSION = "v2"


class QueryCache:
    def __init__(self, client: Optional[redis.Redis] = None, host: str = "localhost", port: int = 6379):
        # Accepting an injected client (e.g. fakeredis in tests) keeps this
        # testable without a real Redis server, while defaulting to a real
        # connection for production use.
        self.client = client or redis.Redis(host=host, port=port, decode_responses=True)

    # --- Generation counters -------------------------------------------

    def get_generation(self, jurisdiction: str) -> int:
        val = self.client.get(f"gen:{jurisdiction}")
        return int(val) if val is not None else 0

    def bump_generation(self, jurisdiction: str) -> int:
        """Call this after re-ingesting/re-embedding data for a jurisdiction
        (e.g. after fetch_state_insurance.py + embed_and_index.py updates CA
        content). Single atomic INCR -- this is the entire invalidation
        operation; see module docstring for why nothing else is needed."""
        return self.client.incr(f"gen:{jurisdiction}")

    # --- Cache key construction ------------------------------------------

    def make_key(self, question: str, audience: str, jurisdiction_values: list[str]) -> str:
        sorted_jurisdictions = sorted(jurisdiction_values)
        generations = [self.get_generation(j) for j in sorted_jurisdictions]
        gen_str = "-".join(str(g) for g in generations)
        jurisdiction_str = "-".join(sorted_jurisdictions)
        question_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
        return f"query:{SCHEMA_VERSION}:{audience}:{jurisdiction_str}:{gen_str}:{question_hash}"

    # --- Get/set -----------------------------------------------------------

    def get(self, key: str) -> Optional[dict]:
        raw = self.client.get(key)
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: dict, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.client.set(key, json.dumps(value), ex=ttl_seconds)
