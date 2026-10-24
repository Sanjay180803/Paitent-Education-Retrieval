"""
Runtime configuration, read from environment variables so the same code
runs locally, in Docker Compose, and in CI without editing source files.

| Variable                  | Default                     | Meaning |
|---------------------------|-----------------------------|---------|
| QDRANT_URL                | (unset)                     | Qdrant server URL. If unset, an embedded on-disk store at QDRANT_PATH is used. |
| QDRANT_PATH               | ./qdrant_storage            | Embedded Qdrant directory (ignored when QDRANT_URL is set). |
| REDIS_URL                 | redis://localhost:6379/0    | Query cache. Set CACHE_ENABLED=false to run without Redis. |
| CACHE_ENABLED             | true                        | |
| CACHE_TTL_SECONDS         | 3600                        | |
| CHUNKS_FILE               | ./data_corpus/chunks.jsonl  | Chunk file used to build the BM25 index at startup. |
| VECTOR_FLOOR              | 0.4                         | Stage-1 gate on top vector cosine similarity. |
| CROSS_ENCODER_THRESHOLD   | 0.01                        | Stage-2 gate on top cross-encoder score. |
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

COLLECTION_NAME = "patient_education_policy"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    qdrant_url: str | None = field(default_factory=lambda: os.environ.get("QDRANT_URL") or None)
    qdrant_path: Path = field(
        default_factory=lambda: Path(os.environ.get("QDRANT_PATH", PROJECT_ROOT / "qdrant_storage"))
    )
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    cache_enabled: bool = field(default_factory=lambda: _bool("CACHE_ENABLED", True))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.environ.get("CACHE_TTL_SECONDS", "3600")))
    chunks_file: Path = field(
        default_factory=lambda: Path(os.environ.get("CHUNKS_FILE", PROJECT_ROOT / "data_corpus" / "chunks.jsonl"))
    )
    vector_floor: float = field(default_factory=lambda: float(os.environ.get("VECTOR_FLOOR", "0.4")))
    cross_encoder_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CROSS_ENCODER_THRESHOLD", "0.01"))
    )


def qdrant_client(settings: Settings):
    from qdrant_client import QdrantClient

    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=str(settings.qdrant_path))
