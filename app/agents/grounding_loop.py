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
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from app.config import GENERATOR_MODEL, MECHANICAL_MODEL, get_openai_client
from app.retrieval.semantic_search import semantic_search
from app.schemas.document import SOURCE_TIER, Document
from app.schemas.grounding import GroundingResult
from app.schemas.plan import Plan
from app.schemas.taxonomy import RetrievalSource

MAX_ROUNDS = 4  # Section 9: hard cap so a non-converging loop routes to a human instead of spinning
SEARCH_TOP_K = 8
_MAX_EXPANSIONS = 2  # sub-queries generated per question -- see _expand_queries

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

_EXPANSION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"sub_queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["sub_queries"],
    "additionalProperties": False,
}

_EXPANSION_SYSTEM_PROMPT = """You are the query-expansion step of a retrieval system for an \
assistant that answers immigration and tax questions for F-1 international students. You are \
given the user's question.

A broad question (e.g. "what are the tax exemptions for X") often has its most useful answer \
-- a specific dollar amount, form number, deadline, or statute citation -- sitting in a \
narrowly-worded passage that doesn't closely resemble the broad question itself, so a search \
for the broad question alone can miss it entirely. Write up to two additional, more specific \
search queries that are likely to surface that kind of concrete detail, if the question calls \
for one -- e.g. for "what tax exemptions apply to X", a good sub-query is "X specific \
exemption dollar amount" or "X required form number".

Confirmed by hand: this can fail on vocabulary alone, not just specificity. A real IRS \
question worded around "tax exemption" got zero results for its actual dollar figure because \
the source describes that exact benefit as a "standard deduction" -- a different official \
term for the same thing, which a sub-query that only rephrases "exemption" more narrowly \
never bridges. So specifically consider: does the informal or colloquial term in the question \
(e.g. "exempt," "tax break," "waiver") likely correspond to a different formal IRS/USCIS \
mechanism name in the actual source (e.g. "deduction," "credit," "withholding allowance," \
"waiver of inadmissibility")? If plausible, make one sub-query use that formal term \
explicitly instead of just repeating the question's own wording in a narrower form.

Each sub-query must be a genuine, more specific variant of the same underlying question -- \
never invent a new topic. If the question is already narrow and specific, return fewer \
sub-queries, or none."""


def _expand_queries(message: str, client: OpenAI) -> list[str]:
    # MECHANICAL_MODEL, not GENERATOR_MODEL: paraphrasing a question into sharper sub-queries
    # is a mechanical task, not the nuanced judgment GENERATOR_MODEL is reserved for.
    response = client.chat.completions.create(
        model=MECHANICAL_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "query_expansion", "schema": _EXPANSION_RESPONSE_SCHEMA, "strict": True},
        },
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed["sub_queries"][:_MAX_EXPANSIONS]

_SYSTEM_PROMPT = """You are the grounding loop's generation step for an assistant that \
answers immigration and tax questions for F-1 international students. You are given a \
user question and one or more passages retrieved from official government sources.

Decide: do these passages contain enough information to state the GENERAL rule that \
answers the question -- a rule that applies independent of this specific person's \
situation? If yes, set sufficient=true and write a clear, plain-language explanation of \
that general rule, staying strictly to what the passages actually say. Never give an \
individualized recommendation ("you should...", "in your case...") -- state what the \
rule or process is, not what the person should personally do.

Every sentence you write is later checked and kept or dropped independently of the others, \
so each sentence must stand completely on its own and state a specific, checkable fact by \
itself. Never write a vague summary, transition, or reference sentence that only makes \
sense together with an earlier sentence (e.g. "These provisions apply to eligible \
students," "This benefit is available in certain cases") -- if it were kept alone with the \
sentence it depends on removed, a reader would learn nothing new from it. If a sentence \
doesn't add a distinct fact beyond what a nearby sentence already states, omit it.

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


def _try_source(
    message: str, source: RetrievalSource, extra_queries: list[str], client: OpenAI
) -> tuple[list[Document], bool, str | None]:
    documents = semantic_search(message, top_k=SEARCH_TOP_K, source=source, extra_queries=extra_queries, client=client)
    if not documents:
        return [], False, None
    sufficient, answer = _assess_and_generate(message, documents, client)
    return documents, sufficient, answer


def ground(message: str, plan: Plan, client: OpenAI | None = None) -> GroundingResult:
    client = client or get_openai_client()
    sources = _ordered_sources(plan.preferred_retrieval)[:MAX_ROUNDS]
    extra_queries = _expand_queries(message, client)

    # Every tier is queried and assessed concurrently, not one-at-a-time. Found live: a lower
    # tier's retrieval+assessment often takes nearly as long as the tier that actually
    # succeeds, so trying tiers sequentially pays that full latency even on questions the
    # preferred tier was always going to answer -- concurrency cuts worst-case grounding
    # latency roughly in proportion to the number of tiers. Real trade-off, accepted
    # deliberately: every tier is now always queried, even when the preferred tier alone would
    # have succeeded and made the rest unnecessary -- previously skipped via early exit, so
    # this trades some now-unconditional cost for materially lower worst-case latency.
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = {source: pool.submit(_try_source, message, source, extra_queries, client) for source in sources}
        results = {source: future.result() for source, future in futures.items()}

    # Preferred/higher-tier source wins if it succeeded, even if a lower-tier source's future
    # happened to finish first -- `sources` is already tier-ordered by _ordered_sources, so
    # walking it in that order (not completion order) preserves the preference.
    for round_num, source in enumerate(sources, start=1):
        documents, sufficient, answer = results[source]
        if sufficient:
            return GroundingResult(sufficient=True, documents=documents, draft_answer=answer, rounds_used=round_num)

    return GroundingResult(sufficient=False, documents=[], draft_answer=None, rounds_used=len(sources))
