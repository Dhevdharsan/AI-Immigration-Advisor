"""
The golden eval set (task #10, "the one remaining item from the original V1
task list"). A fixed list of real questions with known-correct expected
behavior -- built directly from today's debugging session, where every
single retrieval/verification bug (the AND-only keyword query matching
nothing, Publication 519 monopolizing every retrieved slot, the verifier
narrowing to a single "winning" document and wiping out a correct answer,
the "exemption" vs. "deduction" terminology gap, the candidate-pool
truncation bug) was only found because a human happened to retype the
question and notice something looked wrong. This file exists so the next
regression gets caught by `python -m app.eval.run_eval` instead.

Each case checks the cheap, robust signals that actually broke today --
did it abstain when it shouldn't (or vice versa), does the answer contain
the one or two load-bearing facts that make it actually useful (a form
number, a specific exemption, a document name) -- not exact wording, since
the generator's phrasing genuinely varies run to run (confirmed by hand
throughout today's session) even when the underlying answer is correct.
`expected_facts` should be the smallest set of substrings that would have
caught the actual bug, not an exhaustive checklist -- a looser bar is more
robust to real, harmless wording variance and less likely to cry wolf.

Confirmed by hand, immediately: even a loose single-string bar can still
be too strict when the *concept* is right but the exact word isn't -- a
correct FICA-exemption answer said "Social Security and Medicare taxes"
without the bare acronym, and a correct answer to "can I work" said
"continue working" rather than "employment". Each entry in
`expected_facts` can be one required string, or a tuple of acceptable
alternatives (any one is enough) for exactly this situation.
"""

from dataclasses import dataclass, field

ExpectedFact = str | tuple[str, ...]  # a tuple element = accept any ONE of these alternatives


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    should_abstain: bool
    expected_facts: tuple[ExpectedFact, ...] = field(default_factory=tuple)  # ALL entries must be satisfied
    notes: str = ""


GOLDEN_SET: tuple[GoldenCase, ...] = (
    # ---------- Immigration baseline (the three example questions already surfaced in /ui) ----------
    GoldenCase(
        id="opt-pending-work-authorization",
        question="Can I work while my OPT extension is pending?",
        should_abstain=False,
        expected_facts=(("employ", "work"),),
        notes="Baseline happy-path example shown in the UI itself.",
    ),
    GoldenCase(
        id="university-closure-opt-termination",
        question="What happens to my F-1 status if my university closes down mid-semester?",
        should_abstain=False,
        expected_facts=("OPT",),
        notes="Was the UI's labeled 'abstain example' -- correct back when retrieval was a "
        "curated category->URL lookup that didn't cover this. USCIS Policy Manual Chapter 5 "
        "explicitly covers it now (real semantic search + hybrid search + re-ranking). Was "
        "also a real regression guard: the retrieved chunk started mid-sentence ('This is "
        "commonly known as a grace period...') with its antecedent in the previous chunk, so "
        "the sufficiency check inconsistently (not always) marked it insufficient until "
        "_expand_with_neighbors pulled the neighboring chunk in for context.",
    ),
    GoldenCase(
        id="eb5-out-of-scope-abstain",
        question="What are the eligibility requirements for an EB-5 investor visa?",
        should_abstain=True,
        notes="Real abstain example: EB-5 is an employment-based investor visa, structurally "
        "outside the ingested corpus (USCIS Policy Manual Volume 2 Part F is F/M students "
        "only; EB-5 lives in a different, un-ingested Part). Now the UI's abstain-example "
        "button, replacing university-closure once that became genuinely answerable.",
    ),
    GoldenCase(
        id="opt-absence-days",
        question="How many days can I be absent from the US while on OPT?",
        should_abstain=False,
        expected_facts=(),
        notes="Baseline happy-path example shown in the UI itself.",
    ),

    # ---------- Regression guards: real bugs found and fixed this session ----------
    GoldenCase(
        id="i20-definition",
        question="What is an I-20?",
        should_abstain=False,
        expected_facts=("I-20",),
        notes="Guards against the original curated category->URL lookup bug: this question "
        "doesn't fit any of the planner's action-oriented categories, so a category-keyed "
        "lookup found nothing. Real semantic search fixed it.",
    ),
    GoldenCase(
        id="fica-exemption",
        question="Am I exempt from FICA taxes on my on-campus job as an F-1 student?",
        should_abstain=False,
        expected_facts=(("FICA", "Social Security"),),
        notes="Confirms the Tax domain (IRS retrieval) works at all.",
    ),
    GoldenCase(
        id="tax-filing-forms",
        question="what are the forms i need to file for taxes as an f1 student",
        should_abstain=False,
        expected_facts=("8843",),
        notes="Guards against two real bugs: keyword_search's AND-only tsquery matched zero "
        "rows for any multi-word natural question, and Publication 519 alone could fill every "
        "retrieved slot and crowd out the actual Form 8843 page.",
    ),
    GoldenCase(
        id="india-treaty-exemptions",
        question="what are the tax exemptions for international students from india",
        should_abstain=False,
        expected_facts=("India",),
        notes="Guards against the verify_node bug where narrowing to a single 'winning' "
        "citation document wiped out every claim actually grounded in a different, "
        "non-conflicting corroborating document -- this question's answer went to a full "
        "verification_wipeout abstain before that fix.",
    ),
)
