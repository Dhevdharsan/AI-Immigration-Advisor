"""
The Contradiction Finder (Section 5). A genuinely distinct reasoning task
from the grounding loop (which decides what to retrieve next) and the
verifier (which checks an answer's claims against evidence) -- this
compares retrieved evidence against *other* retrieved evidence, which
neither of those does.

Only meaningful with 2+ documents; with fewer there's nothing to compare,
so that case is handled without calling the LLM at all.

KNOWN LIMITATION (found during development, not yet fixed): quote
verification (below) reliably catches a *fabricated* conflict -- a claimed
quote that doesn't actually appear in the source. It does NOT catch a
conflict built from two *real, verbatim* quotes that individually check
out but describe different scenarios (e.g. SEVP's severe-economic-hardship
off-campus work page vs. the STEM OPT extension rule -- both mention "Form
I-765 pending" but answer different questions). A same-scenario check was
added to the prompt (step 1 below) but did not fully resolve this in
testing -- it's still confirmed to misfire on the "Can I work while my OPT
extension is pending?" example. Fixing this properly needs the golden eval
set (Section 7's "Contradictory sources" benchmark category, task #10),
which can measure whether a prompt change actually generalizes rather than
just fitting this one example.
"""

import json

from openai import OpenAI

from app.agents._quotes import quote_verified
from app.config import GENERATOR_MODEL, get_openai_client
from app.schemas.contradiction import ContradictionResult, SupportingQuote
from app.schemas.document import DOC_TYPE_PRIORITY, Document

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "conflict_found": {"type": "boolean"},
        "resolved": {"type": "boolean"},
        "winning_url": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
        "supporting_quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["url", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["conflict_found", "resolved", "winning_url", "rationale", "supporting_quotes"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are the Contradiction Finder for an immigration Q&A assistant \
for F-1 international students. You are given two or more passages, each retrieved \
from an official source, for the same question.

1. First confirm both passages are actually describing the SAME specific scenario -- the \
same stage of the process, the same category of employment authorization, the same visa \
type. Immigration rules are full of similarly-worded passages that govern different \
sub-cases (e.g. one passage may cover severe-economic-hardship off-campus employment, \
while another covers STEM OPT extensions -- both mention "Form I-765 pending" but answer \
completely different questions). If the passages address different scenarios, this is \
NOT a conflict, no matter how differently worded they are -- treat them as simply \
answering different questions.

2. Only once you've confirmed they address the same scenario, decide whether they \
actually state different or incompatible rules for it. One passage being more detailed, \
or covering a narrower case, than another is NOT a conflict -- only flag it if they \
genuinely disagree on the same specific scenario.

3. If they do conflict, weigh three factors to decide which should be treated as \
authoritative. Use judgment, not a fixed formula:
   a. Hierarchy tier -- a lower tier number is a generally more authoritative source.
   b. Last-updated date -- a newer source reflects more current policy, and this can \
outweigh tier: a higher-tier source that is clearly older than a conflicting lower-tier \
source is a real edge case, not an automatic win for the higher tier.
   c. Document-type priority WITHIN the same source (given per document below) -- e.g. \
a Policy Manual outranks an FAQ page from the same agency.
   If, having weighed all three, you are not confident which one should win, set \
resolved=false and leave winning_url null. Do not guess just to produce an answer.

4. Whether or not there is a real conflict, if you can confidently identify one document \
as the best primary citation for this question, set winning_url to it and resolved=true.

5. Always give a one-line rationale explaining the decision -- this is shown directly to \
the user, so write it like a real explanation (e.g. "X was chosen over Y because it is \
the higher-authority source and has a newer effective date."), not a generic phrase.

6. If you set conflict_found=true, you MUST include in supporting_quotes one short \
excerpt (under 300 characters) copied EXACTLY, word-for-word, from each of at least two \
of the documents -- the specific sentences that actually disagree. Copy them verbatim, \
do not paraphrase or summarize. If you cannot point to real, exact text in at least two \
documents that disagrees, set conflict_found=false instead -- do not report a conflict \
you can't quote directly from the source text."""


def _doc_block(doc: Document) -> str:
    priority = DOC_TYPE_PRIORITY.index(doc.doc_type)
    return (
        f"[{doc.title} -- {doc.url}]\n"
        f"Source: {doc.source.value} (hierarchy tier {doc.tier}, lower = more authoritative)\n"
        f"Document type: {doc.doc_type.value} (priority {priority} within source, lower = higher priority)\n"
        f"Last updated: {doc.last_updated or 'unknown'}\n"
        f"Text:\n{doc.text[:60000]}"
    )


def find_contradictions(
    question: str, documents: list[Document], client: OpenAI | None = None
) -> ContradictionResult:
    if len(documents) < 2:
        return ContradictionResult(
            conflict_found=False,
            resolved=True,
            winning_document=documents[0] if documents else None,
            rationale="Only one source was retrieved -- nothing to compare against.",
            all_documents=documents,
        )

    client = client or get_openai_client()
    passages = "\n\n".join(_doc_block(doc) for doc in documents)
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        temperature=0,  # this is a judgment/verification task, not creative writing -- want consistency, not variety
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved passages:\n{passages}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "contradiction_check", "schema": _RESPONSE_SCHEMA, "strict": True},
        },
    )
    parsed = json.loads(response.choices[0].message.content)
    doc_by_url = {doc.url: doc for doc in documents}
    quotes = [SupportingQuote(**q) for q in parsed["supporting_quotes"]]

    conflict_found = parsed["conflict_found"]
    resolved = parsed["resolved"]
    winning_url = parsed["winning_url"]
    rationale = parsed["rationale"]

    if conflict_found:
        # A conflict claim is only as trustworthy as the quotes backing it -- verify
        # each one actually appears in the document it's attributed to, from at least
        # two distinct documents. If verification fails, don't assert a conflict we
        # can't back up (see project notes: this caught a real hallucinated conflict
        # during development, where the model claimed two sources disagreed when they
        # actually said the same thing).
        distinct_urls = {q.url for q in quotes}
        all_verified = (
            len(distinct_urls) >= 2
            and all(q.url in doc_by_url for q in quotes)
            and all(quote_verified(q.quote, doc_by_url[q.url].text) for q in quotes)
        )
        if not all_verified:
            conflict_found = False
            resolved = False
            winning_url = None
            rationale = (
                "The model reported a conflict, but the quoted evidence for it could not be "
                "verified against the retrieved text, so no conflict is being asserted. "
                f"(Original claim: {rationale})"
            )

    winning_document = doc_by_url.get(winning_url) if winning_url else None

    return ContradictionResult(
        conflict_found=conflict_found,
        resolved=resolved,
        winning_document=winning_document,
        rationale=rationale,
        all_documents=documents,
        supporting_quotes=quotes,
    )
