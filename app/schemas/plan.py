"""
The planner's entire output contract (Section 6). The planner emits exactly
one of these and does nothing else -- it does not call retrieval, does not
phrase the user-facing clarifying question, and does not touch memory beyond
reading it as input. Every other component (grounding loop, generation step)
is a consumer of this object.
"""

from pydantic import BaseModel, Field

from app.schemas.taxonomy import Category, RetrievalSource


class Plan(BaseModel):
    category: Category
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Names of required schema fields (Section 6) not yet known from the "
        "message, memory, or an uploaded document.",
    )
    needs_document: bool = Field(
        description="True if a missing field would plausibly be supplied by an uploaded "
        "document (e.g. an I-20, denial notice, or RFE) rather than by asking the user directly."
    )
    needs_clarification: str | None = Field(
        default=None,
        description="The single specific missing field name to ask the user about, if any. "
        "Never a generic 'please provide more information' -- names the exact field.",
    )
    preferred_retrieval: RetrievalSource = Field(
        description="Starting source for the grounding loop, taken from the category's schema "
        "(Section 4) -- the ranked hierarchy is still the fallback if this source is silent."
    )
