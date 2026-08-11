"""
Tests the grounding loop's control flow in isolation (Section 5): does the
higher-priority tier win when it succeeds, does a lower tier's answer get
used when the higher one doesn't succeed, and does it abstain cleanly when
nothing does? Both retrieval (semantic_search) and the LLM assessment call
are mocked, so these need no network access, API key, or running database.

All tiers are now queried and assessed CONCURRENTLY (see grounding_loop.py's
`ground`), so the shared mock client dispatches on request content --
which schema is being requested (query_expansion vs grounding_assessment),
and for the latter, which source's document text is in the passages --
rather than on call order, since thread scheduling order isn't
guaranteed. Same pattern as test_verifier.py's already-concurrent checks.
"""

import json
from unittest.mock import MagicMock, patch

from app.agents.grounding_loop import _MAX_EXPANSIONS, _expand_queries, ground
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


def _expansion_response(sub_queries: list[str] | None = None):
    payload = json.dumps({"sub_queries": sub_queries or []})
    return MagicMock(choices=[MagicMock(message=MagicMock(content=payload))])


def _client(sufficiency_by_source: dict[RetrievalSource, tuple[bool, str | None]]):
    """Dispatches on the request itself: the expansion call has a distinct schema name, and a
    sufficiency-check call's passages embed the source's name (via `_doc`'s url), so which
    source a given concurrent call is about can be told apart without relying on order."""

    def side_effect(*args, **kwargs):
        schema_name = kwargs["response_format"]["json_schema"]["name"]
        if schema_name == "query_expansion":
            return _expansion_response()
        assert schema_name == "grounding_assessment"
        user_content = kwargs["messages"][1]["content"]
        for source, (sufficient, answer) in sufficiency_by_source.items():
            if source.value in user_content:
                return _llm_response(sufficient, answer)
        raise AssertionError(f"unexpected call content: {user_content!r}")

    client = MagicMock()
    client.chat.completions.create.side_effect = side_effect
    return client


def test_higher_tier_wins_when_it_succeeds():
    client = _client({RetrievalSource.USCIS: (True, "General rule text."), RetrievalSource.SEVP: (False, None)})

    with patch("app.agents.grounding_loop.semantic_search", side_effect=lambda query, top_k=8, source=None, extra_queries=None, client=None: [_doc(source)]):
        result = ground("Can I work during OPT?", _plan(RetrievalSource.USCIS), client=client)

    assert result.sufficient is True
    assert result.rounds_used == 1  # USCIS is tier position 1
    assert result.draft_answer == "General rule text."


def test_lower_tier_used_when_higher_tier_is_insufficient():
    client = _client({RetrievalSource.USCIS: (False, None), RetrievalSource.SEVP: (True, "Found it in tier two.")})

    def fake_semantic_search(query, top_k=8, source=None, extra_queries=None, client=None):
        return [_doc(source)]

    with patch("app.agents.grounding_loop.semantic_search", side_effect=fake_semantic_search):
        result = ground("Some question", _plan(RetrievalSource.USCIS), client=client)

    assert result.sufficient is True
    assert result.rounds_used == 2  # SEVP is tier position 2
    assert result.draft_answer == "Found it in tier two."


def test_skips_tier_with_no_documents_without_calling_llm():
    client = _client({RetrievalSource.SEVP: (True, "Answer from SEVP.")})

    def fake_semantic_search(query, top_k=8, source=None, extra_queries=None, client=None):
        return [] if source == RetrievalSource.USCIS else [_doc(source)]

    with patch("app.agents.grounding_loop.semantic_search", side_effect=fake_semantic_search):
        result = ground("Some question", _plan(RetrievalSource.USCIS), client=client)

    assert result.sufficient is True
    # expansion + one sufficiency call (never called for the empty USCIS tier)
    assert client.chat.completions.create.call_count == 2


def test_abstains_when_no_tier_is_sufficient():
    client = _client({RetrievalSource.USCIS: (False, None), RetrievalSource.SEVP: (False, None)})

    def fake_semantic_search(query, top_k=8, source=None, extra_queries=None, client=None):
        return [_doc(source)]

    with patch("app.agents.grounding_loop.semantic_search", side_effect=fake_semantic_search):
        result = ground("An unanswerable question", _plan(RetrievalSource.USCIS), client=client)

    assert result.sufficient is False
    assert result.documents == []
    assert result.draft_answer is None
    assert result.rounds_used == 2  # both implemented tiers (USCIS, SEVP) tried


def test_expand_queries_truncates_to_max_expansions():
    client = MagicMock()
    client.chat.completions.create.return_value = _expansion_response(["q1", "q2", "q3", "q4"])

    result = _expand_queries("Some question", client)

    assert len(result) == _MAX_EXPANSIONS
    assert result == ["q1", "q2", "q3", "q4"][:_MAX_EXPANSIONS]
