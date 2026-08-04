"""
FastAPI entry point (Section 10/11). One endpoint: POST /ask runs the full
graph (planner -> grounding loop -> Contradiction Finder -> verifier) and
returns every stage's output, not just the final answer -- this is exactly
what the execution-trace view (task #9) will need, so returning it now
avoids throwing away information the graph already computed.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.pipeline import build_graph

app = FastAPI(title="Immigration Navigator")
_graph = build_graph()


class AskRequest(BaseModel):
    message: str
    memory: dict[str, str] = {}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    result = _graph.invoke({"message": request.message, "memory": request.memory})
    return {
        "plan": result["plan"].model_dump(mode="json"),
        "grounding": result["grounding"].model_dump(mode="json"),
        "contradiction": result["contradiction"].model_dump(mode="json") if result.get("contradiction") else None,
        "verification": result["verification"].model_dump(mode="json") if result.get("verification") else None,
        "answer": result["answer"].model_dump(mode="json"),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
