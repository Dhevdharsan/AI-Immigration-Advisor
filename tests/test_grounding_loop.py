"""
Tests the grounding loop's control flow in isolation (Section 5): does it
stop as soon as a source is sufficient, escalate tiers when a source isn't,
and give up cleanly at the round cap? Both retrieval and the LLM assessment
call are mocked, so these need no network access or API key.
"""

import json
from unittest.mock import MagicMock, patch

from app.agents.grounding_loop import ground
from app.schemas.document import DocType, Document
from app.schemas.plan import Plan
from app.schemas.taxonomy import Category, RetrievalSource


def _doc(source: RetrievalSource) -> Document:
    return Document(
        source=source,
        doc_type=DocType.POLICY_MANUAL if source == RetrievalSource.USCIS else DocType.GUIDANCE_PAGE,
        url=f"https://example.com/{source.value}",
        title="Test Page",
        text="Some retrieved passage text.",
    )


def _plan(preferred: RetrievalSource) -> Plan:
    return Plan(
        category=Category.WORK_AUTHORIZATION,
        missing_fields=[],
        needs_document=False,
        needs_clarification=None,
        preferred_retrieval=preferred,
    )


def _llm_response(sufficient: bool, answer: str | None):
    payload = json.dumps({"sufficient": sufficient, "answer": answer})
    return MagicMock(choices=[MagicMock(message=MagicMock(content=payload))])


def test_stops_on_first_sufficient_source():
    client = MagicMock()
    client.chat.completions.create.return_value = _llm_response(True, "General rule text.")

    with patch("app.agents.grounding_loop.retrieve", return_value=[_doc(RetrievalSource.USCIS)]):
        result = ground("Can I work during OPT?", _plan(RetrievalSource.USCIS), client=client)

    assert result.sufficient is True
    assert result.rounds_used == 1
    assert result.draft_answer == "General rule text."
    client.chat.completions.create.assert_called_once()


def test_escalates_to_next_tier_when_first_is_insufficient():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _llm_response(False, None),
        _llm_response(True, "Found it in tier two."),
    ]

    def fake_retrieve(category, source):
        return [_doc(source)]

    with patch("app.agents.grounding_loop.retrieve", side_effect=fake_retrieve):
        result = ground("Some question", _plan(RetrievalSource.USCIS), client=client)

    assert result.sufficient is True
    assert result.rounds_used == 2
    assert result.draft_answer == "Found it in tier two."


def test_skips_tier_with_no_documents_without_calling_llm():
    client = MagicMock()
    client.chat.completions.create.return_value = _llm_response(True, "Answer from SEVP.")

    def fake_retrieve(category, source):
        return [] if source == RetrievalSource.USCIS else [_doc(source)]

    with patch("app.agents.grounding_loop.retrieve", side_effect=fake_retrieve):
        result = ground("Some question", _plan(RetrievalSource.USCIS), client=client)

    assert result.sufficient is True
    client.chat.completions.create.assert_called_once()  # never called for the empty USCIS round


def test_abstains_when_no_tier_is_sufficient():
    client = MagicMock()
    client.chat.completions.create.return_value = _llm_response(False, None)

    with patch("app.agents.grounding_loop.retrieve", return_value=[_doc(RetrievalSource.USCIS)]):
        result = ground("An unanswerable question", _plan(RetrievalSource.USCIS), client=client)

    assert result.sufficient is False
    assert result.documents == []
    assert result.draft_answer is None
    assert result.rounds_used == 2  # both implemented tiers (USCIS, SEVP) tried
