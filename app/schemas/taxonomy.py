"""
The planner's fixed category taxonomy (Section 6) and per-category information
schema. This is the auditable, extensible half of the planner: adding a new
category means adding an entry here, not touching the planner's reasoning logic.

Each category also carries a `preferred_retrieval` source (Section 4) so the
grounding loop can pick a starting tier instead of always walking the ranked
hierarchy from the top -- e.g. a travel/re-entry question is really a State
Department matter, not a USCIS one, even though USCIS still ranks higher
overall in the hierarchy.
"""

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    WORK_AUTHORIZATION = "Work Authorization"
    STATUS_MAINTENANCE = "Status Maintenance"
    TRAVEL_REENTRY = "Travel / Re-entry"
    RFE_DOCUMENT_RESPONSE = "RFE / Document Response"
    APPEALS = "Appeals"
    DEADLINES = "Deadlines"


class RetrievalSource(str, Enum):
    USCIS = "USCIS"
    SEVP = "SEVP"
    STATE_DEPT = "State Department"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    allowed_values: tuple[str, ...] | None = None
    document_derivable: bool = True
    """
    Whether this fact is the kind that's literally printed on a document (an I-20,
    EAD card, RFE/denial notice) versus something only the person can tell us
    (e.g. which enrollment status they consider themselves in right now). This is
    fixed, auditable metadata -- not a per-question guess -- and is what lets the
    planner (Section 6) mechanically choose `needs_document` vs `needs_clarification`
    for a given missing fact, instead of the LLM having to decide that live.
    """


@dataclass(frozen=True)
class CategorySchema:
    required_fields: tuple[FieldSpec, ...]
    preferred_retrieval: RetrievalSource


CATEGORY_SCHEMAS: dict[Category, CategorySchema] = {
    Category.WORK_AUTHORIZATION: CategorySchema(
        required_fields=(
            FieldSpec(
                name="Current Status Stage",
                description="Where the student is in the F-1 work-authorization pipeline",
                allowed_values=(
                    "F-1 baseline (no work authorization yet)",
                    "CPT authorized",
                    "OPT pending",
                    "OPT approved / active",
                    "STEM OPT pending",
                    "STEM OPT approved / active",
                ),
            ),
            FieldSpec(
                name="OPT/CPT Dates",
                description="Start and/or end dates of the relevant authorization period, if any",
            ),
        ),
        preferred_retrieval=RetrievalSource.USCIS,
    ),
    Category.STATUS_MAINTENANCE: CategorySchema(
        required_fields=(
            FieldSpec(
                name="Enrollment Status",
                description="Full course of study, reduced course load (RCL), authorized leave, or withdrawn",
                document_derivable=False,
            ),
            FieldSpec(
                name="Program End Date",
                description="The program end date listed on the student's I-20",
            ),
        ),
        preferred_retrieval=RetrievalSource.SEVP,
    ),
    Category.TRAVEL_REENTRY: CategorySchema(
        required_fields=(
            FieldSpec(
                name="Visa Stamp Status",
                description="Whether the student's visa stamp is currently valid, expired, or single/multiple entry",
            ),
            FieldSpec(
                name="I-20 Travel Signature Date",
                description="Date of the most recent travel signature on the I-20",
            ),
            FieldSpec(
                name="Current Program Stage",
                description="e.g. actively enrolled, on OPT, between programs",
                document_derivable=False,
            ),
        ),
        preferred_retrieval=RetrievalSource.STATE_DEPT,
    ),
    Category.RFE_DOCUMENT_RESPONSE: CategorySchema(
        required_fields=(
            FieldSpec(
                name="RFE Form Type",
                description="Which form the RFE was issued against, e.g. I-765, I-539, I-129",
            ),
            FieldSpec(
                name="RFE Basis",
                description="The specific reason/basis USCIS cited for requesting evidence",
            ),
            FieldSpec(
                name="RFE Deadline Date",
                description="The response due date printed on the RFE notice",
            ),
        ),
        preferred_retrieval=RetrievalSource.USCIS,
    ),
    Category.APPEALS: CategorySchema(
        required_fields=(
            FieldSpec(
                name="Denial Form Type",
                description="Which form/petition was denied, e.g. I-765, I-140, I-539",
            ),
            FieldSpec(
                name="Denial Basis",
                description="The specific reason USCIS gave for the denial",
            ),
            FieldSpec(
                name="Denial Date",
                description="Date on the denial notice -- appeal/motion windows run from this date",
            ),
        ),
        preferred_retrieval=RetrievalSource.USCIS,
    ),
    Category.DEADLINES: CategorySchema(
        required_fields=(
            FieldSpec(
                name="Deadline Type",
                description="e.g. 90-day unemployment clock, grace period, extension filing window, appeal window",
                document_derivable=False,
            ),
            FieldSpec(
                name="Anchor Date",
                description="The date the relevant clock starts counting from, e.g. OPT end date or denial date",
            ),
        ),
        preferred_retrieval=RetrievalSource.SEVP,
    ),
}
