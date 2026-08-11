"""
Tests the FastAPI endpoints (Section 10/11) -- /health, /ask, and the new
/ask/stream SSE endpoint that had zero coverage since it was added. The
graph itself is mocked (patched onto the module-level `_graph` instance
`main.py` builds at import time), so these need no network access, API
key, or running database. Building the real graph at import time is cheap
(just wires the LangGraph StateGraph, no API calls), so importing
app.api.main is itself safe without any of that.
"""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import main as main_module
from app.schemas.answer import Answer
from app.schemas.document import DocType, Document
from app.schemas.grounding import GroundingResult
from app.schemas.plan import Plan
from app.schemas.taxonomy import Category, RetrievalSource

client = TestClient(main_module.app)


def _doc() -> Document:
    return Document(source=RetrievalSource.USCIS, doc_type=DocType.POLICY_MANUAL, url="https://a", title="A", text="t")


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ask_returns_full_result_shape():
    doc = _doc()
    fake_result = {
        "plan": Plan(category=Category.WORK_AUTHORIZATION, missing_fields=[], needs_document=False, needs_clarification=None, preferred_retrieval=RetrievalSource.USCIS),
        "grounding": GroundingResult(sufficient=True, documents=[doc], draft_answer="Answer.", rounds_used=1),
        "contradiction": None,
        "verification": None,
        "answer": Answer(answer="Answer.", source=doc, human_follow_up="Ask your DSO."),
        "timings": {"plan": 0.1, "ground": 0.2},
    }

    with patch.object(main_module._graph, "invoke", return_value=fake_result):
        res = client.post("/ask", json={"message": "Can I work?"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"]["answer"] == "Answer."
    assert body["plan"]["category"] == "Work Authorization"
    assert body["contradiction"] is None
    assert "total" in body["timings"]  # /ask adds its own wall-clock total on top of per-node timings


def test_ask_stream_emits_one_node_event_per_update_then_a_done_event():
    doc = _doc()
    plan_obj = Plan(category=Category.WORK_AUTHORIZATION, missing_fields=[], needs_document=False, needs_clarification=None, preferred_retrieval=RetrievalSource.USCIS)
    grounding = GroundingResult(sufficient=True, documents=[doc], draft_answer="Answer.", rounds_used=1)
    answer = Answer(answer="Answer.", source=doc, human_follow_up="Ask your DSO.")

    # A trimmed-down fake trace (not the full 5-node graph) -- this test is about the
    # streaming wrapper's own logic (one SSE event per update, correct final assembly), not
    # re-verifying the graph's routing, which test_pipeline.py already covers.
    fake_updates = [
        {"run_planner": {"plan": plan_obj, "timings": {"plan": 0.1}}},
        {"ground": {"grounding": grounding, "timings": {"plan": 0.1, "ground": 0.2}}},
        {"build_answer": {"answer": answer}},
    ]

    with patch.object(main_module._graph, "stream", return_value=iter(fake_updates)):
        res = client.get("/ask/stream", params={"message": "Can I work?"})

    assert res.status_code == 200
    body = res.text
    assert body.count("event: node") == 3
    assert body.count("event: done") == 1

    done_payload = body.split("event: done\ndata: ")[1].strip()
    final = json.loads(done_payload)
    assert final["answer"]["answer"] == "Answer."
    assert final["plan"]["category"] == "Work Authorization"
    assert final["grounding"]["sufficient"] is True
    assert "total" in final["timings"]  # /ask/stream's done event matches /ask's timings shape


def test_ask_stream_node_event_carries_only_that_nodes_own_data():
    plan_obj = Plan(category=Category.WORK_AUTHORIZATION, missing_fields=[], needs_document=False, needs_clarification=None, preferred_retrieval=RetrievalSource.USCIS)
    fake_updates = [{"run_planner": {"plan": plan_obj, "timings": {"plan": 0.1}}}]

    with patch.object(main_module._graph, "stream", return_value=iter(fake_updates)):
        res = client.get("/ask/stream", params={"message": "Can I work?"})

    node_payload = res.text.split("event: node\ndata: ")[1].split("\n\n")[0]
    parsed = json.loads(node_payload)
    assert parsed["node"] == "run_planner"
    assert set(parsed["data"].keys()) == {"plan"}  # "timings" is surfaced separately, not duplicated into data
    assert parsed["data"]["plan"]["category"] == "Work Authorization"
    assert parsed["timings"] == {"plan": 0.1}
