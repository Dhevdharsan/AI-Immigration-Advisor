"""
Tests the LangGraph wiring itself (Section 10): given mocked outputs from
the four agents, does the graph route to the correct terminal node for
each of Section 9's failure modes, plus the happy path? Each agent
function is patched, so this needs no network access or API key -- it's
purely checking the conditional edges are wired correctly.

Also tests two pure helper functions directly: `_verify_node`'s
conflict-based narrowing (a real bug, caught live -- narrowing to the
single "winning" document unconditionally wiped out claims correctly
grounded in a different, non-conflicting corroborating document) and
`_collect_all_sources` (which distinct pages a final answer actually
cites, not just the primary citation pick).
"""

from unittest.mock import patch

from app.agents.pipeline import _collect_all_sources, build_graph
from app.schemas.answer import SourceRef
from app.schemas.contradiction import ContradictionResult
from app.schemas.document import DocType, Document
from app.schemas.grounding import GroundingResult
from app.schemas.plan import Plan
from app.schemas.taxonomy import Category, RetrievalSource
from app.schemas.verification import ClaimCheck, ScopeCheck, VerificationResult


def _doc(url="https://a", tier_source=RetrievalSource.USCIS) -> Document:
    return Document(source=tier_source, doc_type=DocType.POLICY_MANUAL, url=url, title=url, text="Some text.")


def _plan() -> Plan:
    return Plan(
        category=Category.WORK_AUTHORIZATION,
        missing_fields=["OPT/CPT Dates"],
        needs_document=True,
        needs_clarification=None,
        preferred_retrieval=RetrievalSource.USCIS,
    )


def _invoke():
    return build_graph().invoke({"message": "Can I work?", "memory": {}})


def test_happy_path_builds_answer():
    doc = _doc()
    with (
        patch("app.agents.pipeline.plan", return_value=_plan()),
        patch("app.agents.pipeline.ground", return_value=GroundingResult(sufficient=True, documents=[doc], draft_answer="Answer text.", rounds_used=1)),
        patch(
            "app.agents.pipeline.find_contradictions",
            return_value=ContradictionResult(conflict_found=False, resolved=True, winning_document=doc, rationale="Only source.", all_documents=[doc]),
        ),
        patch(
            "app.agents.pipeline.verify",
            return_value=VerificationResult(
                final_answer="Answer text.",
                grounding_rate=1.0,
                claims=[ClaimCheck(claim="Answer text.", supported=True, source_url=doc.url, quote="Some text.")],
                scope=ScopeCheck(passes=True, flagged_sentences=[], explanation=""),
            ),
        ),
    ):
        result = _invoke()

    answer = result["answer"]
    assert answer.abstained is False
    assert answer.answer == "Answer text."
    assert answer.source == doc
    assert answer.missing_information == ["OPT/CPT Dates"]
    # User-requested behavior: still answer immediately, but also explicitly ask for a
    # document when the plan flags needs_document -- not just a passive missing-info note.
    assert answer.document_request is not None
    assert "OPT/CPT Dates" in answer.document_request


def test_no_document_request_when_plan_does_not_need_one():
    doc = _doc()
    plan_without_document = Plan(
        category=Category.WORK_AUTHORIZATION,
        missing_fields=[],
        needs_document=False,
        needs_clarification=None,
        preferred_retrieval=RetrievalSource.USCIS,
    )
    with (
        patch("app.agents.pipeline.plan", return_value=plan_without_document),
        patch("app.agents.pipeline.ground", return_value=GroundingResult(sufficient=True, documents=[doc], draft_answer="Answer text.", rounds_used=1)),
        patch(
            "app.agents.pipeline.find_contradictions",
            return_value=ContradictionResult(conflict_found=False, resolved=True, winning_document=doc, rationale="Only source.", all_documents=[doc]),
        ),
        patch(
            "app.agents.pipeline.verify",
            return_value=VerificationResult(
                final_answer="Answer text.",
                grounding_rate=1.0,
                claims=[ClaimCheck(claim="Answer text.", supported=True, source_url=doc.url, quote="Some text.")],
                scope=ScopeCheck(passes=True, flagged_sentences=[], explanation=""),
            ),
        ),
    ):
        result = _invoke()

    assert result["answer"].document_request is None


def test_abstains_when_grounding_insufficient():
    with (
        patch("app.agents.pipeline.plan", return_value=_plan()),
        patch("app.agents.pipeline.ground", return_value=GroundingResult(sufficient=False, documents=[], draft_answer=None, rounds_used=2)),
        patch("app.agents.pipeline.find_contradictions") as mock_contradiction,
        patch("app.agents.pipeline.verify") as mock_verify,
    ):
        result = _invoke()

    answer = result["answer"]
    assert answer.abstained is True
    assert answer.abstain_reason == "insufficient_retrieval"
    mock_contradiction.assert_not_called()
    mock_verify.assert_not_called()


def test_abstains_when_contradiction_unresolved():
    doc_a, doc_b = _doc("https://a"), _doc("https://b", RetrievalSource.SEVP)
    with (
        patch("app.agents.pipeline.plan", return_value=_plan()),
        patch("app.agents.pipeline.ground", return_value=GroundingResult(sufficient=True, documents=[doc_a, doc_b], draft_answer="Answer text.", rounds_used=1)),
        patch(
            "app.agents.pipeline.find_contradictions",
            return_value=ContradictionResult(conflict_found=True, resolved=False, winning_document=None, rationale="Not confident.", all_documents=[doc_a, doc_b]),
        ),
        patch("app.agents.pipeline.verify") as mock_verify,
    ):
        result = _invoke()

    answer = result["answer"]
    assert answer.abstained is True
    assert answer.abstain_reason == "unresolved_contradiction"
    assert doc_a.title in answer.human_follow_up
    assert doc_b.title in answer.human_follow_up
    mock_verify.assert_not_called()


def test_abstains_when_verification_wipes_out_everything():
    doc = _doc()
    with (
        patch("app.agents.pipeline.plan", return_value=_plan()),
        patch("app.agents.pipeline.ground", return_value=GroundingResult(sufficient=True, documents=[doc], draft_answer="Answer text.", rounds_used=1)),
        patch(
            "app.agents.pipeline.find_contradictions",
            return_value=ContradictionResult(conflict_found=False, resolved=True, winning_document=doc, rationale="Only source.", all_documents=[doc]),
        ),
        patch(
            "app.agents.pipeline.verify",
            return_value=VerificationResult(
                final_answer=None,
                grounding_rate=0.0,
                claims=[ClaimCheck(claim="Answer text.", supported=False)],
                scope=ScopeCheck(passes=True, flagged_sentences=[], explanation=""),
            ),
        ),
    ):
        result = _invoke()

    answer = result["answer"]
    assert answer.abstained is True
    assert answer.abstain_reason == "verification_wipeout"


def test_verify_sees_all_retrieved_documents_when_no_conflict():
    doc_a, doc_b = _doc("https://a"), _doc("https://b", RetrievalSource.SEVP)
    with (
        patch("app.agents.pipeline.plan", return_value=_plan()),
        patch("app.agents.pipeline.ground", return_value=GroundingResult(sufficient=True, documents=[doc_a, doc_b], draft_answer="Answer text.", rounds_used=1)),
        patch(
            "app.agents.pipeline.find_contradictions",
            return_value=ContradictionResult(conflict_found=False, resolved=True, winning_document=doc_a, rationale="Best citation.", all_documents=[doc_a, doc_b]),
        ),
        patch(
            "app.agents.pipeline.verify",
            return_value=VerificationResult(
                final_answer="Answer text.",
                grounding_rate=1.0,
                claims=[ClaimCheck(claim="Answer text.", supported=True, source_url=doc_b.url, quote="Some text.")],
                scope=ScopeCheck(passes=True, flagged_sentences=[], explanation=""),
            ),
        ) as mock_verify,
    ):
        _invoke()

    # Real bug guard: narrowing to just the citation pick (doc_a) here would have wrongly
    # failed a claim actually grounded in doc_b, a different but equally-valid, non-conflicting
    # retrieved document -- verify() must see the full retrieved set when nothing conflicts.
    called_documents = mock_verify.call_args.args[1]
    assert called_documents == [doc_a, doc_b]


def test_verify_narrowed_to_winning_document_when_conflict_resolved():
    doc_a, doc_b = _doc("https://a"), _doc("https://b", RetrievalSource.SEVP)
    with (
        patch("app.agents.pipeline.plan", return_value=_plan()),
        patch("app.agents.pipeline.ground", return_value=GroundingResult(sufficient=True, documents=[doc_a, doc_b], draft_answer="Answer text.", rounds_used=1)),
        patch(
            "app.agents.pipeline.find_contradictions",
            return_value=ContradictionResult(conflict_found=True, resolved=True, winning_document=doc_a, rationale="Newer source wins.", all_documents=[doc_a, doc_b]),
        ),
        patch(
            "app.agents.pipeline.verify",
            return_value=VerificationResult(
                final_answer="Answer text.",
                grounding_rate=1.0,
                claims=[ClaimCheck(claim="Answer text.", supported=True, source_url=doc_a.url, quote="Some text.")],
                scope=ScopeCheck(passes=True, flagged_sentences=[], explanation=""),
            ),
        ) as mock_verify,
    ):
        _invoke()

    # The losing (contradicted) document's claims must not leak into the answer -- narrowing
    # to only winning_document is correct once a real conflict was found and resolved.
    called_documents = mock_verify.call_args.args[1]
    assert called_documents == [doc_a]


def test_collect_all_sources_orders_primary_source_first_then_claim_urls():
    doc_a, doc_b = _doc("https://a"), _doc("https://b", RetrievalSource.SEVP)
    evidence = [
        ClaimCheck(claim="From b.", supported=True, source_url="https://b", quote="q"),
        ClaimCheck(claim="From a again.", supported=True, source_url="https://a", quote="q"),
    ]

    sources = _collect_all_sources(doc_a, evidence, [doc_a, doc_b])

    assert [s.url for s in sources] == ["https://a", "https://b"]


def test_collect_all_sources_falls_back_to_url_as_title_when_document_unknown():
    evidence = [ClaimCheck(claim="From somewhere.", supported=True, source_url="https://unknown", quote="q")]

    sources = _collect_all_sources(None, evidence, [])

    assert sources == [SourceRef(url="https://unknown", title="https://unknown")]


def test_collect_all_sources_deduplicates_repeated_urls():
    doc_a = _doc("https://a")
    evidence = [
        ClaimCheck(claim="First.", supported=True, source_url="https://a", quote="q"),
        ClaimCheck(claim="Second.", supported=True, source_url="https://a", quote="q"),
    ]

    sources = _collect_all_sources(doc_a, evidence, [doc_a])

    assert len(sources) == 1
