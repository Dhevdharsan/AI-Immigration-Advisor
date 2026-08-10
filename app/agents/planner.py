"""
The planner (Section 6). Produces exactly one `Plan` and does nothing else --
no retrieval, no user-facing text, no memory writes. Two halves:

  1. Rules half (already fixed in app.schemas.taxonomy): the category list,
     each category's required facts, and whether each fact is the kind that's
     printed on a document or only knowable by asking the person.
  2. Judgment half (this file's one LLM call): read the question + whatever's
     already remembered, pick the best-matching category, and mark which of
     that category's required facts are already known.

Everything after the LLM call -- turning "which facts are missing" into
needs_document / needs_clarification / preferred_retrieval -- is plain,
deterministic Python. No AI judgment happens there, which is what keeps the
planner testable in isolation (Section 6).
"""

import json

from openai import OpenAI

from app.config import GENERATOR_MODEL, get_openai_client
from app.schemas.plan import Plan
from app.schemas.taxonomy import CATEGORY_SCHEMAS, Category

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": [c.value for c in Category]},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "known": {"type": "boolean"},
                    "value": {"type": ["string", "null"]},
                },
                "required": ["field_name", "known", "value"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["category", "fields"],
    "additionalProperties": False,
}


def _taxonomy_prompt() -> str:
    """Serialize every category + its required facts so the model can both
    classify and schema-check in a single call."""
    lines = []
    for category, schema in CATEGORY_SCHEMAS.items():
        lines.append(f"## {category.value}")
        for field in schema.required_fields:
            allowed = f" (one of: {', '.join(field.allowed_values)})" if field.allowed_values else ""
            lines.append(f"- {field.name}: {field.description}{allowed}")
    return "\n".join(lines)


_SYSTEM_PROMPT = f"""You are the classification step of a Q&A planner for F-1 \
international students, covering both immigration topics (OPT/CPT/STEM OPT/I-20/RFE/ \
appeals/deadlines) and tax topics (residency status, filing requirements, income/ \
withholding, tax treaty benefits, FICA exemption, tax deadlines).

Given the user's question and any facts already known from earlier in the conversation \
("memory"), do exactly two things:
1. Pick the single best-matching category from the list below.
2. For every required fact listed under that category, decide whether it is already \
known -- either stated directly in the question, or present in memory -- and if so, \
extract its value. If not stated or implied, mark it not known. Do not guess or infer \
a value that was not actually given.

Categories and their required facts:
{_taxonomy_prompt()}

Only report fields for the one category you picked."""


def _classify(message: str, memory: dict[str, str], client: OpenAI) -> tuple[Category, dict[str, str | None]]:
    memory_text = "\n".join(f"- {k}: {v}" for k, v in memory.items()) or "(none)"
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        temperature=0,  # classification/schema-checking should be consistent, not stylistically varied
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {message}\n\nAlready known (memory):\n{memory_text}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "planner_classification", "schema": _RESPONSE_SCHEMA, "strict": True},
        },
    )
    parsed = json.loads(response.choices[0].message.content)
    category = Category(parsed["category"])
    known_values = {f["field_name"]: f["value"] for f in parsed["fields"] if f["known"]}
    return category, known_values


def plan(message: str, memory: dict[str, str] | None = None, client: OpenAI | None = None) -> Plan:
    memory = memory or {}
    client = client or get_openai_client()

    category, known_from_llm = _classify(message, memory, client)
    schema = CATEGORY_SCHEMAS[category]

    missing_fields: list[str] = []
    for field in schema.required_fields:
        known = field.name in memory or field.name in known_from_llm
        if not known:
            missing_fields.append(field.name)

    if not missing_fields:
        return Plan(
            category=category,
            missing_fields=[],
            needs_document=False,
            needs_clarification=None,
            preferred_retrieval=schema.preferred_retrieval,
        )

    field_by_name = {f.name: f for f in schema.required_fields}
    document_derivable_missing = [f for f in missing_fields if field_by_name[f].document_derivable]
    ask_only_missing = [f for f in missing_fields if not field_by_name[f].document_derivable]

    return Plan(
        category=category,
        missing_fields=missing_fields,
        needs_document=bool(document_derivable_missing),
        needs_clarification=ask_only_missing[0] if ask_only_missing else None,
        preferred_retrieval=schema.preferred_retrieval,
    )
