"""
Tests the Contradiction Finder in isolation (Section 5): does it skip the
LLM entirely with a single source, correctly surface a resolved pick when
sources agree or when a conflict is confidently resolvable, correctly
refuse to guess when a conflict can't be confidently resolved, and --
critically -- refuse to trust a claimed conflict whose "supporting" quotes
don't actually appear in the source text? The LLM call is mocked, so these
need no network access or API key.
"""

import json
from datetime import date
from unittest.mock import MagicMock

from app.agents.contradiction_finder import find_contradictions
from app.schemas.document import DocType, Document
from app.schemas.taxonomy import RetrievalSource


def _doc(source, doc_type, url, text="Some passage text about the rule.", last_updated=None) -> Document:
    return Document(
        source=source,
        doc_type=doc_type,
        url=url,
        title=url,
        text=text,
        last_updated=last_updated,
    )


def _llm_response(conflict_found: bool, resolved: bool, winning_url: str | None, rationale: str, quotes=()):
    payload = json.dumps(
        {
            "conflict_found": conflict_found,
            "resolved": resolved,
            "winning_url": winning_url,
            "rationale": rationale,
            "supporting_quotes": [{"url": url, "quote": quote} for url, quote in quotes],
        }
    )
    return MagicMock(choices=[MagicMock(message=MagicMock(content=payload))])


def test_single_document_skips_llm_entirely():
    client = MagicMock()
    doc = _doc(RetrievalSource.USCIS, DocType.POLICY_MANUAL, "https://a")

    result = find_contradictions("Some question", [doc], client=client)

    assert result.conflict_found is False
    assert result.resolved is True
    assert result.winning_document == doc
    client.chat.completions.create.assert_not_called()


def test_no_conflict_still_picks_a_primary_with_rationale():
    uscis_doc = _doc(RetrievalSource.USCIS, DocType.POLICY_MANUAL, "https://uscis", last_updated=date(2025, 6, 1))
    sevp_doc = _doc(RetrievalSource.SEVP, DocType.GUIDANCE_PAGE, "https://sevp", last_updated=date(2025, 1, 1))
    client = MagicMock()
    client.chat.completions.create.return_value = _llm_response(
        False, True, "https://uscis", "Both agree; USCIS chosen as the higher-authority primary source."
    )

    result = find_contradictions("Some question", [uscis_doc, sevp_doc], client=client)

    assert result.conflict_found is False
    assert result.resolved is True
    assert result.winning_document == uscis_doc
    assert "USCIS" in result.rationale


def test_resolvable_conflict_with_verified_quotes_picks_a_winner():
    old_uscis = _doc(
        RetrievalSource.USCIS,
        DocType.OFFICIAL_FAQ,
        "https://uscis-faq",
        text="Students may not work at all while an extension is pending.",
        last_updated=date(2020, 1, 1),
    )
    new_uscis = _doc(
        RetrievalSource.USCIS,
        DocType.POLICY_MANUAL,
        "https://uscis-pm",
        text="Students may continue working for up to 180 days while an extension is pending.",
        last_updated=date(2025, 6, 1),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _llm_response(
        True,
        True,
        "https://uscis-pm",
        "Policy Manual was chosen over the older FAQ because it has both higher document-type "
        "priority and a newer effective date.",
        quotes=[
            ("https://uscis-faq", "Students may not work at all while an extension is pending."),
            ("https://uscis-pm", "Students may continue working for up to 180 days while an extension is pending."),
        ],
    )

    result = find_contradictions("Some question", [old_uscis, new_uscis], client=client)

    assert result.conflict_found is True
    assert result.resolved is True
    assert result.winning_document == new_uscis
    assert len(result.supporting_quotes) == 2


def test_unresolvable_conflict_returns_no_winner_and_keeps_both():
    higher_tier_old = _doc(
        RetrievalSource.USCIS,
        DocType.GUIDANCE_PAGE,
        "https://uscis",
        text="Students may not work while an extension is pending.",
        last_updated=date(2019, 1, 1),
    )
    lower_tier_new = _doc(
        RetrievalSource.SEVP,
        DocType.GUIDANCE_PAGE,
        "https://sevp",
        text="Students may keep working for 180 days while an extension is pending.",
        last_updated=date(2025, 6, 1),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _llm_response(
        True,
        False,
        None,
        "Tier favors USCIS but recency favors SEVP; not confident enough to pick.",
        quotes=[
            ("https://uscis", "Students may not work while an extension is pending."),
            ("https://sevp", "Students may keep working for 180 days while an extension is pending."),
        ],
    )

    result = find_contradictions("Some question", [higher_tier_old, lower_tier_new], client=client)

    assert result.conflict_found is True
    assert result.resolved is False
    assert result.winning_document is None
    assert result.all_documents == [higher_tier_old, lower_tier_new]


def test_claimed_conflict_with_unverifiable_quotes_is_not_trusted():
    """The regression case: the model asserts a conflict but the "quotes" it
    gives don't actually appear in the source text -- exactly what happened
    live in development, where the model claimed SEVP said something it
    never actually said. This must not be trusted as a real conflict."""
    doc_a = _doc(RetrievalSource.USCIS, DocType.POLICY_MANUAL, "https://a", text="The real text says X.")
    doc_b = _doc(RetrievalSource.SEVP, DocType.GUIDANCE_PAGE, "https://b", text="The real text also says X.")
    client = MagicMock()
    client.chat.completions.create.return_value = _llm_response(
        True,
        True,
        "https://a",
        "These sources disagree.",
        quotes=[
            ("https://a", "This sentence does not appear anywhere in doc_a."),
            ("https://b", "Nor does this one appear in doc_b."),
        ],
    )

    result = find_contradictions("Some question", [doc_a, doc_b], client=client)

    assert result.conflict_found is False
    assert result.resolved is False
    assert result.winning_document is None
    assert "could not be verified" in result.rationale


def test_conflict_with_quotes_from_only_one_document_is_not_trusted():
    """A conflict needs evidence from at least two distinct documents --
    two quotes from the same page isn't evidence that two sources disagree."""
    doc_a = _doc(RetrievalSource.USCIS, DocType.POLICY_MANUAL, "https://a", text="Sentence one. Sentence two.")
    doc_b = _doc(RetrievalSource.SEVP, DocType.GUIDANCE_PAGE, "https://b", text="Something unrelated.")
    client = MagicMock()
    client.chat.completions.create.return_value = _llm_response(
        True,
        True,
        "https://a",
        "These disagree.",
        quotes=[("https://a", "Sentence one."), ("https://a", "Sentence two.")],
    )

    result = find_contradictions("Some question", [doc_a, doc_b], client=client)

    assert result.conflict_found is False
    assert result.resolved is False
