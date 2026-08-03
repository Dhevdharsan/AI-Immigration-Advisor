"""
Tests the verifier's two checks in isolation (Section 5/7): does
faithfulness checking correctly distrust an unverifiable quote (same
paranoid pattern as the Contradiction Finder), does the scope gate
correctly drop an individualized-recommendation sentence, and does the
whole thing correctly signal "abstain" (final_answer=None) when nothing
survives? The LLM calls are mocked, so these need no network access or
API key.
"""

import json
from unittest.mock import MagicMock

from app.agents.verifier import verify
from app.schemas.document import DocType, Document
from app.schemas.taxonomy import RetrievalSource


def _doc(url: str, text: str) -> Document:
    return Document(
        source=RetrievalSource.USCIS,
        doc_type=DocType.POLICY_MANUAL,
        url=url,
        title=url,
        text=text,
    )


def _claims_response(claims: list[dict]):
    payload = json.dumps({"claims": claims})
    return MagicMock(choices=[MagicMock(message=MagicMock(content=payload))])


def _scope_response(passes: bool, flagged: list[str], explanation: str):
    payload = json.dumps({"passes": passes, "flagged_sentences": flagged, "explanation": explanation})
    return MagicMock(choices=[MagicMock(message=MagicMock(content=payload))])


def test_all_claims_supported_and_scope_passes():
    doc = _doc("https://a", "Students may work up to 20 hours per week on campus.")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _claims_response(
            [
                {
                    "claim": "Students may work up to 20 hours per week on campus.",
                    "supported": True,
                    "url": "https://a",
                    "quote": "Students may work up to 20 hours per week on campus.",
                }
            ]
        ),
        _scope_response(True, [], "No individualized advice found."),
    ]

    result = verify("Students may work up to 20 hours per week on campus.", [doc], client=client)

    assert result.grounding_rate == 1.0
    assert result.final_answer == "Students may work up to 20 hours per week on campus."
    assert result.dropped_for_scope == []


def test_unverifiable_quote_is_downgraded_to_unsupported():
    doc = _doc("https://a", "The real policy text says something else entirely.")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _claims_response(
            [
                {
                    "claim": "A fabricated claim not actually in the source.",
                    "supported": True,
                    "url": "https://a",
                    "quote": "This exact phrase does not appear in the document.",
                }
            ]
        ),
    ]

    result = verify("A fabricated claim not actually in the source.", [doc], client=client)

    assert result.claims[0].supported is False
    assert result.grounding_rate == 0.0
    assert result.final_answer is None
    # scope check should never run -- nothing survived faithfulness to check
    assert client.chat.completions.create.call_count == 1


def test_scope_violation_drops_the_offending_sentence():
    doc = _doc("https://a", "Denials of this type are appealable within 30 days. You should appeal your denial.")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _claims_response(
            [
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
            ]
        ),
        _scope_response(
            False,
            ["You should appeal your denial."],
            "This sentence gives an individualized recommendation rather than stating the general rule.",
        ),
    ]

    result = verify("...", [doc], client=client)

    assert result.final_answer == "Denials of this type are appealable within 30 days."
    assert result.dropped_for_scope == ["You should appeal your denial."]


def test_everything_dropped_signals_abstain():
    doc = _doc("https://a", "You should appeal your denial.")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _claims_response(
            [
                {
                    "claim": "You should appeal your denial.",
                    "supported": True,
                    "url": "https://a",
                    "quote": "You should appeal your denial.",
                }
            ]
        ),
        _scope_response(False, ["You should appeal your denial."], "Individualized recommendation."),
    ]

    result = verify("...", [doc], client=client)

    assert result.final_answer is None
