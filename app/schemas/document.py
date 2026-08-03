"""
A single retrieved source document (Section 4/5). This is what the retrieval
cache stores, what the grounding loop hands to generation, and what the
Contradiction Finder compares against other Documents for the same sub-question.
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel

from app.schemas.taxonomy import RetrievalSource


class DocType(str, Enum):
    """
    Document-type priority *within* a source (Section 5) -- needed because two
    documents from the same source (e.g. two USCIS pages) can still disagree,
    so tier alone isn't enough to resolve a conflict.
    """

    POLICY_MANUAL = "Policy Manual"
    REGULATIONS = "Regulations"
    OFFICIAL_FORM = "Official Form"
    OFFICIAL_FAQ = "Official FAQ"
    GUIDANCE_PAGE = "Guidance Page"


# Lower index = higher priority, within the same source.
DOC_TYPE_PRIORITY: list[DocType] = [
    DocType.POLICY_MANUAL,
    DocType.REGULATIONS,
    DocType.OFFICIAL_FORM,
    DocType.OFFICIAL_FAQ,
    DocType.GUIDANCE_PAGE,
]

# Ranked hierarchy tier (Section 4). Only tiers 1-2 are wired up so far.
SOURCE_TIER: dict[RetrievalSource, int] = {
    RetrievalSource.USCIS: 1,
    RetrievalSource.SEVP: 2,
    RetrievalSource.STATE_DEPT: 3,
}


class Document(BaseModel):
    source: RetrievalSource
    doc_type: DocType
    url: str
    title: str
    text: str
    last_updated: date | None = None

    @property
    def tier(self) -> int:
        return SOURCE_TIER[self.source]
