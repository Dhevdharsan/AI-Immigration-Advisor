"""
Retrieval cache (Section 4): caches retrieved *documents*, never generated
answers. A file-backed dict keyed by URL is enough for V1 -- one process,
no concurrent writers. Every entry carries a fetch timestamp and is only
served back while still inside its TTL; past that, the caller re-fetches
and the cache is overwritten with fresh content. This is what saves a
repeat retrieval call without ever letting a stale answer get served,
since the full reasoning pipeline still runs fresh on whatever content
comes out of here (cached or not).
"""

import json
import time
from pathlib import Path

from app.schemas.document import Document

_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "corpus_cache" / "documents.json"

DEFAULT_TTL_SECONDS = 48 * 60 * 60  # 2 days -- short enough to catch policy updates promptly


def _load() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    return json.loads(_CACHE_PATH.read_text())


def _save(data: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(data, indent=2))


def get(url: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> Document | None:
    entry = _load().get(url)
    if entry is None:
        return None
    if time.time() - entry["fetched_at"] > ttl_seconds:
        return None
    return Document.model_validate(entry["document"])


def put(url: str, document: Document) -> None:
    data = _load()
    data[url] = {"fetched_at": time.time(), "document": document.model_dump(mode="json")}
    _save(data)
