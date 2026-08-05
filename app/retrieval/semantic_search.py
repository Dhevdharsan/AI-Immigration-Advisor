"""
Real semantic search (Section 12) over the pgvector chunk index --
replaces the old curated category -> URL lookup table (search.py's
original approach). Matches a question against whichever chunks are
actually closest in meaning, which is what a fixed per-category keyword
list can never do for a question like "What is an I-20?" that doesn't
belong to any of the planner's six action-oriented categories.

Matched chunks are grouped back into Document-shaped objects (one per
source URL) so the rest of the pipeline (grounding loop, Contradiction
Finder, verifier) doesn't need to change its Document-based contract.
Each Document's `.text` is reconstructed from only the matched excerpts,
not the page's full original text -- a bonus over the old approach, which
sent an entire 40,000+ character chapter to the LLM every time.
"""

from app.retrieval.embeddings import embed_text
from app.retrieval.vector_store import search as vector_search
from app.schemas.chunk import ChunkSearchResult
from app.schemas.document import Document
from app.schemas.taxonomy import RetrievalSource

DEFAULT_TOP_K = 8


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K, source: RetrievalSource | None = None) -> list[Document]:
    query_embedding = embed_text(query)
    results = vector_search(query_embedding, top_k=top_k, source=source)
    return _group_into_documents(results)


def _group_into_documents(results: list[ChunkSearchResult]) -> list[Document]:
    by_url: dict[str, list[ChunkSearchResult]] = {}
    order: list[str] = []
    for r in results:
        if r.url not in by_url:
            by_url[r.url] = []
            order.append(r.url)
        by_url[r.url].append(r)

    documents = []
    for url in order:
        chunks = sorted(by_url[url], key=lambda c: c.chunk_index)  # document order reads more coherently than relevance rank
        first = chunks[0]
        text = "\n\n...\n\n".join(c.chunk_text for c in chunks)
        documents.append(
            Document(
                source=first.source,
                doc_type=first.doc_type,
                url=url,
                title=first.title,
                text=text,
                last_updated=first.last_updated,
            )
        )
    return documents
