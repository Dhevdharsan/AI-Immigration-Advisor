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


def search(query_embedding: list[float], top_k: int = 5, source: RetrievalSource | None = None) -> list[ChunkSearchResult]:
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
