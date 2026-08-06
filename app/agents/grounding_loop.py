"""
The grounding loop (Section 5): retrieve against the hierarchy, judge
whether that's enough to answer, and either stop or retrieve from the
next tier. A real iterative loop, not a single fetch -- and capped, per
Section 9's failure-mode table, so a question the corpus can't answer
doesn't loop forever.

Retrieval is now real semantic search over a pgvector chunk index
(Section 12), not the original curated category -> URL lookup table --
that shortcut broke on questions like "What is an I-20?" that don't
belong to any of the planner's six action-oriented categories, since
retrieval only ever looked at the category, never the actual question
text. Semantic search fixes that by matching the real question.

Only ever answers the general-rule (type-1) question (Section 3): the
prompt explicitly forbids individualized recommendations. The
individualized half is never handled here -- it's routed, downstream.
"""

import json

from openai import OpenAI

from app.config import GENERATOR_MODEL, OPENAI_API_KEY
from app.retrieval.semantic_search import semantic_search
from app.schemas.document import SOURCE_TIER, Document
from app.schemas.grounding import GroundingResult
from app.schemas.plan import Plan
from app.schemas.taxonomy import RetrievalSource

MAX_ROUNDS = 4  # Section 9: hard cap so a non-converging loop routes to a human instead of spinning
SEARCH_TOP_K = 8

# Sources with real ingested content in the vector index (Section 12) -- the same ones
# app/retrieval/ingest.py populates. Grouped by domain: escalating from a tax question
# into USCIS/SEVP (or vice versa) would waste rounds searching a corpus that structurally
# can't answer it, so a domain's tiers only ever escalate within that same domain, never
# across into an unrelated one. Add a new tuple here (not just to one flat list) when a
# new domain's sources come online.
_IMMIGRATION_SOURCES: tuple[RetrievalSource, ...] = (RetrievalSource.USCIS, RetrievalSource.SEVP)
_TAX_SOURCES: tuple[RetrievalSource, ...] = (RetrievalSource.IRS,)
_DOMAIN_GROUPS: tuple[tuple[RetrievalSource, ...], ...] = (_IMMIGRATION_SOURCES, _TAX_SOURCES)
IMPLEMENTED_SOURCES: tuple[RetrievalSource, ...] = _IMMIGRATION_SOURCES + _TAX_SOURCES

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "answer": {"type": ["string", "null"]},
    },
    "required": ["sufficient", "answer"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are the grounding loop's generation step for an assistant that \
answers immigration and tax questions for F-1 international students. You are given a \
user question and one or more passages retrieved from official government sources.

Decide: do these passages contain enough information to state the GENERAL rule that \
answers the question -- a rule that applies independent of this specific person's \
situation? If yes, set sufficient=true and write a clear, plain-language explanation of \
that general rule, staying strictly to what the passages actually say. Never give an \
individualized recommendation ("you should...", "in your case...") -- state what the \
rule or process is, not what the person should personally do.

If the passages don't actually address the question, set sufficient=false and leave \
answer null. Do not guess, and do not fill gaps using outside knowledge not present in \
the passages."""


def _ordered_sources(preferred: RetrievalSource) -> list[RetrievalSource]:
    """Preferred source first (Section 4's routing refinement), then the remaining
    implemented tiers in ranked-hierarchy order (Section 4's fallback) -- but only within
    the same domain as `preferred`, never escalating into an unrelated domain's sources."""
    group = next((g for g in _DOMAIN_GROUPS if preferred in g), IMPLEMENTED_SOURCES)
    ordered = sorted(group, key=lambda s: SOURCE_TIER[s])
    if preferred in ordered:
        ordered.remove(preferred)
        ordered.insert(0, preferred)
    return ordered


def _assess_and_generate(message: str, documents: list[Document], client: OpenAI) -> tuple[bool, str | None]:
    # No routine truncation: retrieved chapters run tens of thousands of characters and
    # the relevant passage can sit anywhere in them (confirmed by hand -- see project
    # notes). GPT-4o's context window fits several full chapters easily; this cap is
    # only a safety net against a pathologically large page, not a normal-case limit.
    passages = "\n\n".join(f"[{doc.title} -- {doc.url}]\n{doc.text[:60000]}" for doc in documents)
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        temperature=0,  # sufficiency/generation should be consistent, not stylistically varied
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {message}\n\nRetrieved passages:\n{passages}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "grounding_assessment", "schema": _RESPONSE_SCHEMA, "strict": True},
        },
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed["sufficient"], parsed["answer"]


def ground(message: str, plan: Plan, client: OpenAI | None = None) -> GroundingResult:
    client = client or OpenAI(api_key=OPENAI_API_KEY)
    sources = _ordered_sources(plan.preferred_retrieval)[:MAX_ROUNDS]

    for round_num, source in enumerate(sources, start=1):
        documents = semantic_search(message, top_k=SEARCH_TOP_K, source=source)
        if not documents:
            continue
        sufficient, answer = _assess_and_generate(message, documents, client)
        if sufficient:
            return GroundingResult(sufficient=True, documents=documents, draft_answer=answer, rounds_used=round_num)

    return GroundingResult(sufficient=False, documents=[], draft_answer=None, rounds_used=len(sources))
