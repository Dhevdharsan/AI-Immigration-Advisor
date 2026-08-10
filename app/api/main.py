"""
FastAPI entry point (Section 10/11). POST /ask runs the full graph
(planner -> grounding loop -> Contradiction Finder -> verifier) and
returns every stage's output, not just the final answer -- this is exactly
what the execution-trace view (task #9) needs, so returning it now avoids
throwing away information the graph already computed.

GET /ask/stream is the same underlying graph, but as Server-Sent Events
instead of one blocking response. A single question can take 15-75+
seconds, and /ask makes the UI show one uninterrupted blank wait for all
of it -- /ask/stream sends one event per node as LangGraph's own
`.stream(..., stream_mode="updates")` yields it, so the trace view can
light up each step (Planner, Ground, ...) the moment it actually
completes instead of only at the very end. GET (not POST) specifically so
the browser's native EventSource can be used -- EventSource only supports
GET, and hand-rolling SSE parsing over a fetch() body stream is more
failure-prone than using the browser's own implementation.

The trace view itself (app/static/index.html) is a single static page --
no build step, no separate frontend project (task #9 is an inspection
tool first). Mounted at /ui rather than "/" to keep it clearly separate
from the API routes.
"""

import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents.pipeline import build_graph

app = FastAPI(title="Immigration Navigator")
_graph = build_graph()


class AskRequest(BaseModel):
    message: str
    memory: dict[str, str] = {}


def _serialize(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    start = time.perf_counter()
    result = _graph.invoke({"message": request.message, "memory": request.memory})
    total_seconds = round(time.perf_counter() - start, 3)
    return {
        "plan": result["plan"].model_dump(mode="json"),
        "grounding": result["grounding"].model_dump(mode="json"),
        "contradiction": result["contradiction"].model_dump(mode="json") if result.get("contradiction") else None,
        "verification": result["verification"].model_dump(mode="json") if result.get("verification") else None,
        "answer": result["answer"].model_dump(mode="json"),
        "timings": {**result.get("timings", {}), "total": total_seconds},
    }


@app.get("/ask/stream")
def ask_stream(message: str) -> StreamingResponse:
    def generate():
        start = time.perf_counter()
        state: dict = {}
        # stream_mode="updates" yields {node_name: node_return_dict} after each node completes
        # -- node_return_dict is exactly what that node's function returned (e.g. {"plan":
        # ...} for the planner), the same per-node output /ask assembles all at once at the end.
        for update in _graph.stream({"message": message, "memory": {}}, stream_mode="updates"):
            for node_name, node_output in update.items():
                state.update(node_output)
                payload = {
                    "node": node_name,
                    "data": {k: _serialize(v) for k, v in node_output.items() if k != "timings"},
                    "timings": state.get("timings", {}),
                }
                yield f"event: node\ndata: {json.dumps(payload)}\n\n"

        total_seconds = round(time.perf_counter() - start, 3)
        final = {
            "plan": _serialize(state.get("plan")),
            "grounding": _serialize(state.get("grounding")),
            "contradiction": _serialize(state.get("contradiction")) if state.get("contradiction") else None,
            "verification": _serialize(state.get("verification")) if state.get("verification") else None,
            "answer": _serialize(state.get("answer")),
            "timings": {**state.get("timings", {}), "total": total_seconds},
        }
        yield f"event: done\ndata: {json.dumps(final)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")
