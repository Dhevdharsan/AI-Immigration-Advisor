"""
Tests the planner's mechanical half in isolation (Section 6): given a fixed,
mocked LLM classification, does the plan/document/clarification logic come out
right? The LLM call itself is mocked out -- this is exactly the "testable in
isolation" property the plan-object boundary is meant to give us, so these
tests need no network access or API key.
"""

import json
from unittest.mock import MagicMock

from app.agents.planner import plan
from app.schemas.taxonomy import Category, RetrievalSource


def _mock_client(category: str, known_fields: dict[str, str], all_field_names: list[str]):
    fields = [
        {"field_name": name, "known": name in known_fields, "value": known_fields.get(name)}
        for name in all_field_names
    ]
    payload = json.dumps({"category": category, "fields": fields})

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=payload))]
    )
    return client


_WORK_AUTH_FIELDS = ["Current Status Stage", "OPT/CPT Dates"]


def test_all_fields_known_proceeds_straight_to_research():
    client = _mock_client(
        Category.WORK_AUTHORIZATION.value,
        {"Current Status Stage": "OPT approved / active", "OPT/CPT Dates": "2025-06-01 to 2026-06-01"},
        _WORK_AUTH_FIELDS,
    )

    result = plan("Can I work right now?", client=client)

    assert result.category == Category.WORK_AUTHORIZATION
    assert result.missing_fields == []
    assert result.needs_document is False
    assert result.needs_clarification is None
    assert result.preferred_retrieval == RetrievalSource.USCIS


def test_missing_document_derivable_field_sets_needs_document():
    # Both Work Authorization fields are document-derivable (Section 6 / taxonomy.py).
    client = _mock_client(Category.WORK_AUTHORIZATION.value, {}, _WORK_AUTH_FIELDS)

    result = plan("Can I work while my OPT extension is pending?", client=client)

    assert set(result.missing_fields) == set(_WORK_AUTH_FIELDS)
    assert result.needs_document is True
    assert result.needs_clarification is None


def test_missing_ask_only_field_sets_needs_clarification():
    # Deadline Type is ask-only; Anchor Date is document-derivable.
    client = _mock_client(
        Category.DEADLINES.value, {"Anchor Date": "2026-08-01"}, ["Deadline Type", "Anchor Date"]
    )

    result = plan("When's my deadline?", client=client)

    assert result.missing_fields == ["Deadline Type"]
    assert result.needs_document is False
    assert result.needs_clarification == "Deadline Type"


def test_missing_both_kinds_prefers_clarification_and_still_flags_document():
    client = _mock_client(Category.DEADLINES.value, {}, ["Deadline Type", "Anchor Date"])

    result = plan("When's my deadline?", client=client)

    assert set(result.missing_fields) == {"Deadline Type", "Anchor Date"}
    assert result.needs_document is True
    assert result.needs_clarification == "Deadline Type"


def test_memory_fills_a_field_even_if_llm_does_not_report_it():
    # LLM only reports "Anchor Date" as known; memory already has "Deadline Type".
    client = _mock_client(Category.DEADLINES.value, {"Anchor Date": "2026-08-01"}, ["Anchor Date"])

    result = plan(
        "When's my deadline?",
        memory={"Deadline Type": "90-day unemployment clock"},
        client=client,
    )

    assert result.missing_fields == []
    assert result.needs_document is False
    assert result.needs_clarification is None
