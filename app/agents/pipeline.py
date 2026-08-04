"""
Wires the planner, grounding loop, Contradiction Finder, and verifier into
one real decision graph (Section 10). Every fork in the flowchart is a real
conditional edge here, not just a sequence of function calls, so a later
execution-trace view (task #9) can show which branch a given question
actually took.

Behavior note carried over from the grounding loop (task #5): the graph
always attempts the general-rule answer regardless of the plan's
missing_fields -- Section 3's principle is that the general rule is
answerable independent of a person's specific facts (the brief's own
worked example, "Can I work during OPT-pending", goes straight to
research with no document gate). Missing fields are surfaced in the final
Answer's `missing_information`, not used to block retrieval.

Three distinct abstain reasons, matching Section 9's failure-mode table:
  - "insufficient_retrieval": no source tier had enough to answer.
  - "unresolved_contradiction": sources disagreed and it couldn't be
    confidently resolved -- both are shown rather than one being picked.
  - "verification_wipeout": every claim was either ungrounded or an
    individualized recommendation, so nothing survived to answer with.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.contradiction_finder import find_contradictions
from app.agents.grounding_loop import ground
from app.agents.planner import plan
from app.agents.verifier import verify
from app.schemas.answer import Answer, Confidence
from app.schemas.contradiction import ContradictionResult
from app.schemas.document import Document
from app.schemas.grounding import GroundingResult
from app.schemas.plan import Plan
from app.schemas.taxonomy import CATEGORY_SCHEMAS
from app.schemas.verification import VerificationResult


class PipelineState(TypedDict, total=False):
    message: str
    memory: dict[str, str]
    plan: Plan
    grounding: GroundingResult
    contradiction: ContradictionResult | None
    verification: VerificationResult
    answer: Answer


def _build_human_follow_up(current_plan: Plan, source: Document | None) -> str:
    """Section 3: the individualized part is never answered, always routed --
    concretely, not with a generic disclaimer."""
    lines = ["For advice specific to your situation, bring the following to your DSO or an accredited immigration representative:"]
    if source is not None:
        lines.append(f'- This guidance: "{source.title}" ({source.url})')
    if current_plan.missing_fields:
        lines.append(f"- Your specific facts: {', '.join(current_plan.missing_fields)}")
    if current_plan.needs_clarification:
        lines.append(f"- Specifically: {current_plan.needs_clarification}")
    return "\n".join(lines)


def _build_document_request(current_plan: Plan) -> str | None:
    """User-requested behavior (not the flowchart's literal gate, Section 10): the general
    rule still answers immediately regardless of needs_document -- but a document is also
    explicitly requested alongside it, rather than only appearing as a passive
    'missing info' note. Re-derives which specific missing fields are document-derivable
    (an I-20/EAD/notice could supply them) using the same per-category schema the planner
    itself used, so this doesn't ask for a document to resolve a fact only the person can
    state (Section 6's needs_clarification territory instead)."""
    if not current_plan.needs_document:
        return None
    schema = CATEGORY_SCHEMAS[current_plan.category]
    field_by_name = {f.name: f for f in schema.required_fields}
    document_derivable_missing = [f for f in current_plan.missing_fields if field_by_name[f].document_derivable]
    fields_text = ", ".join(document_derivable_missing) or "the details relevant to your question"
    return (
        "To give you more specific guidance, you can upload your I-20, EAD card, or the "
        f"relevant USCIS/SEVIS notice -- this would help confirm: {fields_text}."
    )


def _compute_confidence(source: Document, corroborated: bool, conflict_found: bool, grounding_rate: float) -> Confidence:
    """Section 8's composite, collapsed into clear bands rather than a false-precision
    scalar. Thresholds are a provisional V1 heuristic -- calibrating them properly needs
    the golden eval set (task #10), not more guessing."""
    if conflict_found or grounding_rate < 0.7:
        return Confidence.LOW
    if source.tier == 1 and corroborated and grounding_rate == 1.0:
        return Confidence.HIGH
    return Confidence.MEDIUM


def _plan_node(state: PipelineState) -> dict:
    return {"plan": plan(state["message"], state.get("memory", {}))}


def _ground_node(state: PipelineState) -> dict:
    return {"grounding": ground(state["message"], state["plan"])}


def _contradiction_node(state: PipelineState) -> dict:
    grounding = state["grounding"]
    return {"contradiction": find_contradictions(state["message"], grounding.documents)}


def _verify_node(state: PipelineState) -> dict:
    grounding = state["grounding"]
    contradiction = state["contradiction"]
    documents = [contradiction.winning_document] if contradiction.winning_document else grounding.documents
    return {"verification": verify(grounding.draft_answer, documents)}


def _build_answer_node(state: PipelineState) -> dict:
    current_plan = state["plan"]
    grounding = state["grounding"]
    contradiction = state["contradiction"]
    verification = state["verification"]

    source = contradiction.winning_document or (grounding.documents[0] if grounding.documents else None)
    confidence = _compute_confidence(
        source=source,
        corroborated=len(grounding.documents) >= 2,
        conflict_found=contradiction.conflict_found,
        grounding_rate=verification.grounding_rate,
    )

    answer = Answer(
        answer=verification.final_answer,
        evidence=[c for c in verification.claims if c.supported],
        source=source,
        source_rationale=contradiction.rationale if contradiction.conflict_found else None,
        confidence=confidence,
        missing_information=current_plan.missing_fields,
        document_request=_build_document_request(current_plan),
        human_follow_up=_build_human_follow_up(current_plan, source),
        last_updated=source.last_updated if source else None,
    )
    return {"answer": answer}


def _abstain_node(state: PipelineState, reason: str) -> dict:
    current_plan = state["plan"]
    grounding = state.get("grounding")
    contradiction = state.get("contradiction")

    source = None
    if reason == "unresolved_contradiction" and contradiction:
        # Nothing is picked -- both conflicting sources are shown, not silently resolved.
        evidence_note = "; ".join(doc.title for doc in contradiction.all_documents)
        human_follow_up = (
            f"Sources disagree on this and it couldn't be confidently resolved: {evidence_note}. "
            f"{_build_human_follow_up(current_plan, None)}"
        )
    else:
        human_follow_up = _build_human_follow_up(current_plan, None)

    answer = Answer(
        answer=None,
        source=source,
        missing_information=current_plan.missing_fields,
        document_request=_build_document_request(current_plan),
        human_follow_up=human_follow_up,
        abstained=True,
        abstain_reason=reason,
    )
    return {"answer": answer}


def _route_after_ground(state: PipelineState) -> str:
    return "check_contradiction" if state["grounding"].sufficient else "abstain_insufficient_retrieval"


def _route_after_contradiction(state: PipelineState) -> str:
    contradiction = state["contradiction"]
    if contradiction.conflict_found and not contradiction.resolved:
        return "abstain_unresolved_contradiction"
    return "verify"


def _route_after_verify(state: PipelineState) -> str:
    return "build_answer" if state["verification"].final_answer else "abstain_verification_wipeout"


def build_graph():
    # Node names must not collide with PipelineState keys (langgraph rejects that), hence
    # "run_planner"/"check_contradiction" instead of "plan"/"contradiction".
    graph = StateGraph(PipelineState)
    graph.add_node("run_planner", _plan_node)
    graph.add_node("ground", _ground_node)
    graph.add_node("check_contradiction", _contradiction_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("build_answer", _build_answer_node)
    graph.add_node("abstain_insufficient_retrieval", lambda s: _abstain_node(s, "insufficient_retrieval"))
    graph.add_node("abstain_unresolved_contradiction", lambda s: _abstain_node(s, "unresolved_contradiction"))
    graph.add_node("abstain_verification_wipeout", lambda s: _abstain_node(s, "verification_wipeout"))

    graph.set_entry_point("run_planner")
    graph.add_edge("run_planner", "ground")
    graph.add_conditional_edges(
        "ground",
        _route_after_ground,
        {"check_contradiction": "check_contradiction", "abstain_insufficient_retrieval": "abstain_insufficient_retrieval"},
    )
    graph.add_conditional_edges(
        "check_contradiction",
        _route_after_contradiction,
        {"verify": "verify", "abstain_unresolved_contradiction": "abstain_unresolved_contradiction"},
    )
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {"build_answer": "build_answer", "abstain_verification_wipeout": "abstain_verification_wipeout"},
    )
    graph.add_edge("build_answer", END)
    graph.add_edge("abstain_insufficient_retrieval", END)
    graph.add_edge("abstain_unresolved_contradiction", END)
    graph.add_edge("abstain_verification_wipeout", END)

    return graph.compile()
