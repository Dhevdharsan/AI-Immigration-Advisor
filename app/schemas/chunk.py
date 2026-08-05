"""
A chunk is a smaller slice of a scraped page's text (Section 12's real-search
follow-up to the V1 curated-URL shortcut). Storing embeddings per chunk,
rather than per whole page, is what lets semantic search point at the
specific paragraph that answers a question instead of just "this page is
probably relevant somewhere."
"""

from datetime import date

from pydantic import BaseModel

from app.schemas.document import DocType
from app.schemas.taxonomy import RetrievalSource


class ChunkRecord(BaseModel):
    url: str
    title: str
    source: RetrievalSource
    doc_type: DocType
    last_updated: date | None = None
    chunk_index: int
    chunk_text: str


class ChunkSearchResult(BaseModel):
    url: str
    title: str
    source: RetrievalSource
    doc_type: DocType
    last_updated: date | None = None
    chunk_index: int
    chunk_text: str
    distance: float  # cosine distance -- lower means more similar
