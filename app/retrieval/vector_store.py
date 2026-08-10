"""
Postgres+pgvector connection and schema (Section 10). This is what
replaces the old curated category -> URL lookup table (app/retrieval/
search.py's original approach): instead of hand-picking which pages might
be relevant to a category, every chunk of the known-relevant corpus is
embedded once at ingestion time, and a question is matched against
whichever chunks are actually closest in meaning -- which is what fixes
questions like "What is an I-20?" that don't cleanly fit any of the
planner's six action-oriented categories.
"""

import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL
from app.schemas.chunk import ChunkRecord, ChunkSearchResult
from app.schemas.document import DocType
from app.schemas.taxonomy import RetrievalSource

EMBEDDING_DIM = 1536  # text-embedding-3-small


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_schema() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                last_updated DATE,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL,
                fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops);"
        )
        # Keyword/lexical index (Section 12's hybrid-search follow-up): a large document like
        # Publication 519 can crowd every slot of a pure vector top-K with tangentially-related
        # passages, burying a short, exact-match page like "about Form 8843" hundreds of ranks
        # down even though it's the literal right answer. Full-text search on the raw chunk_text
        # catches exact terms (form numbers, statute names) that embeddings don't specially
        # weight -- combined with vector search downstream via Reciprocal Rank Fusion (see
        # semantic_search.py), not used alone.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS document_chunks_text_idx "
            "ON document_chunks USING gin (to_tsvector('english', chunk_text));"
        )
        conn.commit()


def clear_all() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE document_chunks;")
        conn.commit()


def insert_chunks(records: list[ChunkRecord], embeddings: list[list[float]]) -> None:
    if len(records) != len(embeddings):
        raise ValueError(f"{len(records)} records but {len(embeddings)} embeddings")
    with get_connection() as conn, conn.cursor() as cur:
        for record, embedding in zip(records, embeddings):
            cur.execute(
                """
                INSERT INTO document_chunks
                    (url, title, source, doc_type, last_updated, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record.url,
                    record.title,
                    record.source.value,
                    record.doc_type.value,
                    record.last_updated,
                    record.chunk_index,
                    record.chunk_text,
                    embedding,
                ),
            )
        conn.commit()


def vector_search(
    query_embedding: list[float], top_k: int = 5, source: RetrievalSource | None = None
) -> list[ChunkSearchResult]:
    """Nearest chunks by embedding (cosine) distance -- catches paraphrase/meaning matches
    a literal keyword search would miss, e.g. "can I work" matching a passage about
    "employment authorization" with no words in common."""
    where_clause = "WHERE source = %s" if source is not None else ""
    params = [query_embedding]
    if source is not None:
        params.append(source.value)
    params.append(top_k)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT url, title, source, doc_type, last_updated, chunk_index, chunk_text,
                   embedding <=> %s::vector AS distance
            FROM document_chunks
            {where_clause}
            ORDER BY distance
            LIMIT %s
            """,
            params,
        )
        rows = cur.fetchall()

    return [
        ChunkSearchResult(
            url=row[0],
            title=row[1],
            source=RetrievalSource(row[2]),
            doc_type=DocType(row[3]),
            last_updated=row[4],
            chunk_index=row[5],
            chunk_text=row[6],
            distance=row[7],
        )
        for row in rows
    ]


def keyword_search(query: str, top_k: int = 5, source: RetrievalSource | None = None) -> list[ChunkSearchResult]:
    """Nearest chunks by full-text rank -- catches exact-term matches (form numbers, statute
    citations, named programs) that embedding similarity doesn't specially weight, and that
    can get buried under near-duplicate passages from an unrelated but much larger document.

    Deliberately OR, not AND: `plainto_tsquery` alone (or `websearch_to_tsquery`) ANDs every
    significant word in the query together, so a 6-content-word natural-language question
    requires all 6 words in the same ~500-character chunk to match anything -- confirmed by
    hand to return zero rows for a real question here. Rewriting the AND query into an OR
    query (regex-replacing `&` with `|` on its text form) instead lets a chunk match on
    partial term overlap, with `ts_rank` naturally scoring more-overlapping chunks higher --
    the same "any of these words, ranked by how many hit" behavior a real search box has.

    `distance` is set to a rank-derived placeholder purely so this fits the same
    ChunkSearchResult shape as vector_search -- it is not a real cosine distance, and
    downstream fusion (semantic_search.py) ranks by list position, not this value."""
    where_clause = "AND source = %s" if source is not None else ""
    params: list = [query]
    if source is not None:
        params.append(source.value)
    params.append(top_k)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH query AS (
                SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery AS q
            )
            SELECT url, title, source, doc_type, last_updated, chunk_index, chunk_text,
                   ts_rank(to_tsvector('english', chunk_text), (SELECT q FROM query)) AS rank
            FROM document_chunks
            WHERE to_tsvector('english', chunk_text) @@ (SELECT q FROM query)
            {where_clause}
            ORDER BY rank DESC
            LIMIT %s
            """,
            params,
        )
        rows = cur.fetchall()

    return [
        ChunkSearchResult(
            url=row[0],
            title=row[1],
            source=RetrievalSource(row[2]),
            doc_type=DocType(row[3]),
            last_updated=row[4],
            chunk_index=row[5],
            chunk_text=row[6],
            distance=1.0 - min(row[7], 1.0),
        )
        for row in rows
    ]


def get_chunks_by_keys(keys: list[tuple[str, int]]) -> list[ChunkSearchResult]:
    """Fetches specific (url, chunk_index) chunks directly -- no ranking involved, so
    `distance` is a meaningless placeholder here, same as in keyword_search. Used to pull in a
    selected chunk's immediate neighbors for context (see semantic_search.py's
    `_expand_with_neighbors`): a chunk boundary can land mid-sentence, so a chunk that reads as
    incomplete on its own (e.g. referencing "this" with the antecedent in the previous chunk)
    can still be the right chunk -- it just needs its neighbor alongside it."""
    if not keys:
        return []
    conditions = " OR ".join(["(url = %s AND chunk_index = %s)"] * len(keys))
    params = [v for pair in keys for v in pair]

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT url, title, source, doc_type, last_updated, chunk_index, chunk_text
            FROM document_chunks
            WHERE {conditions}
            """,
            params,
        )
        rows = cur.fetchall()

    return [
        ChunkSearchResult(
            url=row[0],
            title=row[1],
            source=RetrievalSource(row[2]),
            doc_type=DocType(row[3]),
            last_updated=row[4],
            chunk_index=row[5],
            chunk_text=row[6],
            distance=0.0,
        )
        for row in rows
    ]
