"""
Output of one full grounding-loop run (Section 5). `draft_answer` is the
grounding loop's own generation step (Section 5 explicitly folds "Evidence
Synthesizer" into this, rather than treating it as a separate agent) --
it is NOT yet verified. The Contradiction Finder and verifier (tasks #6/#7)
consume this object next.
"""

from pydantic import BaseModel

from app.schemas.document import Document


class GroundingResult(BaseModel):
    sufficient: bool
    documents: list[Document]
    draft_answer: str | None = None
    rounds_used: int
