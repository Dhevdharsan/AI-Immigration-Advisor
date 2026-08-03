"""
Output of the Contradiction Finder (Section 5). If `resolved` is True,
`winning_document` is the one to cite (whether or not there was a real
conflict to begin with). If a genuine conflict was found but couldn't be
confidently resolved, `winning_document` stays None and `all_documents`
is what the UI shows the user instead -- both sources, side by side,
rather than the system silently picking one.
"""

from pydantic import BaseModel

from app.schemas.document import Document


class SupportingQuote(BaseModel):
    """A verbatim excerpt the model cited as evidence of a conflict. Checked in
    Python against the actual document text -- see contradiction_finder.py --
    so a claimed conflict can't stand on a misquoted or fabricated passage."""

    url: str
    quote: str


class ContradictionResult(BaseModel):
    conflict_found: bool
    resolved: bool
    winning_document: Document | None = None
    rationale: str
    all_documents: list[Document]
    supporting_quotes: list[SupportingQuote] = []
