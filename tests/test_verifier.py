"""
Tests the verifier's two checks in isolation (Section 5/7): does
faithfulness checking correctly distrust an unverifiable quote (same
paranoid pattern as the Contradiction Finder), does the scope gate
correctly drop an individualized-recommendation sentence, and does the
whole thing correctly signal "abstain" (final_answer=None) when nothing
survives? The LLM calls are mocked, so these need no network access or
API key.

Faithfulness and scope now run concurrently (see verifier.py's latency
note), so their relative call order on the shared mock client isn't
guaranteed -- the mock dispatches on which schema is being requested
(faithfulness_check vs scope_check) rather than on call order, so these
tests stay deterministic regardless of thread scheduling.
"""

import json
from unittest.mock import MagicMock

from app.agents.verifier import verify
from app.schemas.document import DocType, Document
from app.schemas.taxonomy import RetrievalSource


def _doc(url: str, text: str) -> Document:
    return Document(source=RetrievalSource.USCIS, doc_type=DocType.POLICY_MANUAL, url=url, title=url, text=text)


def _mock_response(payload: dict):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))])


def _client(claims: list[dict], scope: dict):
    claims_payload = {"claims": claims}

    def side_effect(*args, **kwargs):
        schema_name = kwargs["response_format"]["json_schema"]["name"]
        if schema_name == "faithfulness_check":
            return _mock_response(claims_payload)
        if schema_name == "scope_check":
            return _mock_response(scope)
        raise AssertionError(f"unexpected schema: {schema_name}")

    client = MagicMock()
    client.chat.completions.create.side_effect = side_effect
    return client


def test_all_claims_supported_and_scope_passes():
    doc = _doc("https://a", "Students may work up to 20 hours per week on campus.")
    client = _client(
        claims=[
            {
                "claim": "Students may work up to 20 hours per week on campus.",
                "supported": True,
                "url": "https://a",
                "quote": "Students may work up to 20 hours per week on campus.",
            }
        ],
        scope={"passes": True, "flagged_sentences": [], "explanation": "No individualized advice found."},
    )

    result = verify("Students may work up to 20 hours per week on campus.", [doc], client=client)

    assert result.grounding_rate == 1.0
    assert result.final_answer == "Students may work up to 20 hours per week on campus."
    assert result.dropped_for_scope == []


def test_unverifiable_quote_is_downgraded_to_unsupported():
    doc = _doc("https://a", "The real policy text says something else entirely.")
    client = _client(
        claims=[
            {
                "claim": "A fabricated claim not actually in the source.",
                "supported": True,
                "url": "https://a",
                "quote": "This exact phrase does not appear in the document.",
            }
        ],
        scope={"passes": True, "flagged_sentences": [], "explanation": ""},
    )

    result = verify("A fabricated claim not actually in the source.", [doc], client=client)

    assert result.claims[0].supported is False
    assert result.grounding_rate == 0.0
    assert result.final_answer is None
    # Both calls now always run (they're kicked off concurrently, before either result is
    # known) -- unlike the old sequential version, scope is no longer skipped just because
    # nothing survived faithfulness. See verifier.py's latency note for the tradeoff.
    assert client.chat.completions.create.call_count == 2


def test_scope_violation_drops_the_offending_sentence():
    doc = _doc("https://a", "Denials of this type are appealable within 30 days. You should appeal your denial.")
    client = _client(
        claims=[
            {
                "claim": "Denials of this type are appealable within 30 days.",
                "supported": True,
                "url": "https://a",
                "quote": "Denials of this type are appealable within 30 days.",
            },
            {
                "claim": "You should appeal your denial.",
                "supported": True,
                "url": "https://a",
                "quote": "You should appeal your denial.",
            },
        ],
        scope={
            "passes": False,
            "flagged_sentences": ["You should appeal your denial."],
            "explanation": "This sentence gives an individualized recommendation rather than stating the general rule.",
        },
    )

    result = verify("...", [doc], client=client)

    assert result.final_answer == "Denials of this type are appealable within 30 days."
    assert result.dropped_for_scope == ["You should appeal your denial."]


def test_everything_dropped_signals_abstain():
    doc = _doc("https://a", "You should appeal your denial.")
    client = _client(
        claims=[
            {
                "claim": "You should appeal your denial.",
                "supported": True,
                "url": "https://a",
                "quote": "You should appeal your denial.",
            }
        ],
        scope={"passes": False, "flagged_sentences": ["You should appeal your denial."], "explanation": "Individualized recommendation."},
    )

    result = verify("...", [doc], client=client)

    assert result.final_answer is None
