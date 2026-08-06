"""
Populates the pgvector corpus index (Section 10/12) from the known-relevant
USCIS Policy Manual chapters, all SEVP student pages, and the IRS
international-taxpayers corpus. Run this whenever the corpus needs
(re)building -- first setup, or after content may have changed. Run with:
python -m app.retrieval.ingest

Scope, matching what was hand-verified as relevant to F-1 students (see
project conversation for each domain's investigation):
  - USCIS Policy Manual Volume 2 Part F (Students and Exchange Visitors),
    all 9 chapters
  - USCIS Policy Manual Volume 1 Part E (Adjudications), all 10 chapters --
    general RFE/appeals/evidence procedures, not F-1-specific, but the
    right Part for those topics
  - Every SEVP Study in the States /students/ page (all of them, not a
    keyword-matched subset) -- this is what makes a page like
    "students-and-the-form-i-20" reachable at all
  - IRS.gov's international-taxpayers subpages matching a student/
    nonresident-alien keyword, plus Form 8843/1040-NR/W-7 and Publications
    519/901/970 (see app/retrieval/scraper.py's list_irs_urls)
"""

from app.retrieval.chunking import chunk_text
from app.retrieval.embeddings import embed_texts
from app.retrieval.scraper import (
    fetch_irs_page,
    fetch_sevp_page,
    fetch_uscis_policy_manual_page,
    list_irs_urls,
    list_sevp_urls,
)
from app.retrieval.vector_store import clear_all, init_schema, insert_chunks
from app.schemas.chunk import ChunkRecord
from app.schemas.document import Document

_USCIS_BASE = "https://www.uscis.gov/policy-manual"
_USCIS_URLS = [f"{_USCIS_BASE}/volume-2-part-f-chapter-{i}" for i in range(1, 10)] + [
    f"{_USCIS_BASE}/volume-1-part-e-chapter-{i}" for i in range(1, 11)
]

EMBED_BATCH_SIZE = 100


def _fetch_all_documents() -> list[Document]:
    documents = [fetch_uscis_policy_manual_page(url) for url in _USCIS_URLS]
    documents += [fetch_sevp_page(url, lastmod=lastmod) for url, lastmod in list_sevp_urls()]
    documents += [fetch_irs_page(url) for url in list_irs_urls()]
    return documents


def _chunk_document(doc: Document) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            url=doc.url,
            title=doc.title,
            source=doc.source,
            doc_type=doc.doc_type,
            last_updated=doc.last_updated,
            chunk_index=i,
            chunk_text=chunk,
        )
        for i, chunk in enumerate(chunk_text(doc.text))
    ]


def run_ingestion() -> None:
    init_schema()
    clear_all()

    documents = _fetch_all_documents()
    print(f"Fetched {len(documents)} documents")

    records: list[ChunkRecord] = []
    for doc in documents:
        records.extend(_chunk_document(doc))
    print(f"Split into {len(records)} chunks")

    for i in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[i : i + EMBED_BATCH_SIZE]
        embeddings = embed_texts([r.chunk_text for r in batch])
        insert_chunks(batch, embeddings)
        print(f"Embedded and inserted {min(i + EMBED_BATCH_SIZE, len(records))}/{len(records)} chunks")


if __name__ == "__main__":
    run_ingestion()
