"""
The final structured answer object (Section 10). Every field already has a
concrete mechanism behind it from earlier in the pipeline -- this is the
assembly step, not new reasoning.

Two deliberate simplifications from Section 10's table:
  - "Summary" and "Official Rule" are listed as two separate fields there
    (a short explanation vs. the precise cited rule). V1's generation step
    produces one general-rule explanation that serves both roles, so
    they're collapsed into a single `answer` field rather than fabricating
    a second, differently-phrased version we don't actually generate.
  - "Related Documents" is left out entirely -- the brief itself flags it
    as having no backing mechanism yet.
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel

from app.schemas.document import Document
from app.schemas.verification import ClaimCheck


class Confidence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class SourceRef(BaseModel):
    """A distinct source page the final answer actually draws on -- not just the single
    citation pick (`Answer.source`). Once verification stopped narrowing to one "winning"
    document for the common no-conflict case (Section 12), a real answer can legitimately be
    built from several corroborating pages, and the user should be able to open every one of
    them, not just whichever one was chosen as the primary citation."""

    url: str
    title: str


class Answer(BaseModel):
    answer: str | None  # None only when abstaining
    evidence: list[ClaimCheck] = []
    source: Document | None = None
    source_rationale: str | None = None  # Contradiction Finder's one-line rationale, if it ran
    all_sources: list[SourceRef] = []  # every distinct source a surviving claim actually cites
    confidence: Confidence | None = None
    missing_information: list[str] = []
    document_request: str | None = None  # set when plan.needs_document -- see pipeline.py's _build_document_request
    human_follow_up: str
    last_updated: date | None = None
    abstained: bool = False
    abstain_reason: str | None = None
