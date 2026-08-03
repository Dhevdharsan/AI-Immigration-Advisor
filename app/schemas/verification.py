"""
Output of the verifier / scope-safety gate (Section 5/7): two genuinely
distinct checks on a grounding loop's draft answer.

  - Faithfulness: is every claim actually grounded in the retrieved
    evidence, checked claim-by-claim rather than one holistic score
    (Section 7's LogSense-derived pattern)?
  - Scope: has the answer drifted from the general rule (always
    answerable, Section 3) into an individualized recommendation (never
    answered, always routed)?

`final_answer` is None when nothing survives both checks -- the signal
to abstain and route (Section 9) rather than ship an empty or gutted
answer.
"""

from pydantic import BaseModel


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    source_url: str | None = None
    quote: str | None = None


class ScopeCheck(BaseModel):
    passes: bool
    flagged_sentences: list[str] = []
    explanation: str


class VerificationResult(BaseModel):
    final_answer: str | None
    grounding_rate: float  # Section 8 signal: fraction of claims actually supported
    claims: list[ClaimCheck]
    scope: ScopeCheck
    dropped_for_scope: list[str] = []
