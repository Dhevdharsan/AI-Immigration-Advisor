"""
The verifier / scope-safety gate (Section 5/7). Two genuinely distinct
checks packaged as one agent, per Section 5's rule that a second stage
only earns its place if it does a different job, not the same check
twice:

  1. Faithfulness -- is every claim in the draft answer actually grounded
     in the retrieved evidence? Decomposed claim-by-claim, not scored as
     one holistic "faithful, 1-5" number (Section 7).
  2. Scope/safety gate -- has the answer drifted from stating the general
     rule (Section 3, always answerable) into an individualized
     recommendation (never answered, always routed)?

Runs on JUDGE_MODEL, not GENERATOR_MODEL -- Section 7's hard rule: the
model judging an answer must not be the model that generated it, or
self-preference bias inflates scores.

Per Section 9: an ungrounded claim or a scope violation is dropped before
the answer reaches the user -- never shipped silently.

Latency note: faithfulness and scope are independent checks on the same
draft answer -- scope checks the draft's sentences directly rather than
waiting on faithfulness's claim decomposition, so there's no real
dependency forcing them to run one after another. They're run
concurrently (see `verify`). Live measurement showed these two sequential
calls were the single largest contributor to total pipeline latency
(~5.5s of ~13.5s); running them in parallel cuts that to roughly the
slower of the two. Tradeoff: the scope call now always runs even when
nothing survives faithfulness (previously skipped in that case), since
we no longer know that outcome before kicking scope off -- an acceptable
cost given a total wipeout is the rare case.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from app.agents._quotes import normalize, quote_verified
from app.config import JUDGE_MODEL, OPENAI_API_KEY
from app.schemas.document import Document
from app.schemas.verification import ClaimCheck, ScopeCheck, VerificationResult

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "url": {"type": ["string", "null"]},
                    "quote": {"type": ["string", "null"]},
                },
                "required": ["claim", "supported", "url", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_CLAIMS_SYSTEM_PROMPT = """You are the faithfulness checker for an immigration Q&A \
assistant. You are given a draft answer and the official source passages it was \
supposedly generated from.

Break the draft answer down into a list of atomic factual claims -- each claim must be a \
complete, standalone sentence stating one fact or rule (not a fragment). Together, the \
claims should cover the entire substance of the draft answer, in order.

For each claim, decide whether it is directly supported by the passages:
- If supported, set supported=true and include the exact source url plus a short \
verbatim quote (under 300 characters, copied EXACTLY, word for word, not paraphrased) \
from that passage that backs the claim.
- If you cannot point to real, exact text in the passages that supports the claim, set \
supported=false and leave url/quote null. An unverifiable claim must be marked \
unsupported even if it sounds true -- do not guess."""

_SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "passes": {"type": "boolean"},
        "flagged_sentences": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["passes", "flagged_sentences", "explanation"],
    "additionalProperties": False,
}

_SCOPE_SYSTEM_PROMPT = """You are the scope/safety gate for an immigration Q&A assistant \
for F-1 international students. Immigration advice given without authorization can \
constitute unauthorized practice of law, so this assistant may only state GENERAL rules \
-- rules that hold independent of one specific person's situation (e.g. "denials of this \
type are generally appealable within 30 days under 8 CFR 103.3(a)(2)"). It must never \
give an INDIVIDUALIZED recommendation or conclusion, e.g. "you should appeal", "in your \
case you qualify", "I'd recommend filing a motion to reopen instead".

Read the given text sentence by sentence. Flag any sentence that crosses from stating a \
general rule into telling this specific person what they personally should do, or \
concluding something about their specific situation. Stating a rule, a process, a \
deadline, or a citation directly is fine, even if phrased plainly -- only flag actual \
personalized advice or conclusions.

For each sentence you flag, copy it into flagged_sentences EXACTLY as it appears in the \
input text. Set passes=true only if no sentence needs to be flagged."""


def _check_faithfulness(draft_answer: str, documents: list[Document], client: OpenAI) -> list[ClaimCheck]:
    passages = "\n\n".join(f"[{doc.title} -- {doc.url}]\n{doc.text[:60000]}" for doc in documents)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _CLAIMS_SYSTEM_PROMPT},
            {"role": "user", "content": f"Draft answer:\n{draft_answer}\n\nSource passages:\n{passages}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "faithfulness_check", "schema": _CLAIMS_SCHEMA, "strict": True},
        },
    )
    parsed = json.loads(response.choices[0].message.content)
    doc_by_url = {doc.url: doc for doc in documents}

    checks = []
    for item in parsed["claims"]:
        supported, url, quote = item["supported"], item["url"], item["quote"]
        if supported:
            doc = doc_by_url.get(url)
            if doc is None or not quote or not quote_verified(quote, doc.text):
                supported, url, quote = False, None, None
        checks.append(ClaimCheck(claim=item["claim"], supported=supported, source_url=url, quote=quote))
    return checks


def _check_scope(text: str, client: OpenAI) -> ScopeCheck:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SCOPE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scope_check", "schema": _SCOPE_SCHEMA, "strict": True},
        },
    )
    parsed = json.loads(response.choices[0].message.content)
    return ScopeCheck(
        passes=parsed["passes"], flagged_sentences=parsed["flagged_sentences"], explanation=parsed["explanation"]
    )


def verify(draft_answer: str, documents: list[Document], client: OpenAI | None = None) -> VerificationResult:
    client = client or OpenAI(api_key=OPENAI_API_KEY)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims_future = pool.submit(_check_faithfulness, draft_answer, documents, client)
        scope_future = pool.submit(_check_scope, draft_answer, client)
        claims = claims_future.result()
        scope = scope_future.result()

    grounded_claims = [c for c in claims if c.supported]
    grounding_rate = len(grounded_claims) / len(claims) if claims else 0.0

    dropped_for_scope: list[str] = []
    remaining_claims = []
    flagged_normalized = {normalize(s) for s in scope.flagged_sentences}
    for c in grounded_claims:
        if normalize(c.claim) in flagged_normalized:
            dropped_for_scope.append(c.claim)
        else:
            remaining_claims.append(c.claim)

    return VerificationResult(
        final_answer=" ".join(remaining_claims) or None,
        grounding_rate=grounding_rate,
        claims=claims,
        scope=scope,
        dropped_for_scope=dropped_for_scope,
    )
