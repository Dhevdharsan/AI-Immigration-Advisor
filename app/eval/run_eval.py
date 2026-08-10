"""
Runs the golden set (app/eval/golden_set.py) against the real pipeline --
live OpenAI calls and a live Postgres/pgvector index, not mocks. This is
deliberately separate from `pytest tests/`, which stays fast, offline, and
free to run on every save (see tests/test_grounding_loop.py's own
docstring); this script costs real API money and takes real wall-clock
time per question, so it's a "run it before/after a retrieval or
verification change" tool, not a pre-commit check.

Usage:
    python -m app.eval.run_eval              # run every case
    python -m app.eval.run_eval india forms   # run only cases whose id contains "india" or "forms"
"""

import sys
import time

from app.agents.pipeline import build_graph
from app.eval.golden_set import GOLDEN_SET, GoldenCase


def _check(case: GoldenCase, answer_abstained: bool, answer_text: str | None) -> list[str]:
    """Returns a list of human-readable failure reasons; empty means the case passed."""
    failures = []
    if answer_abstained != case.should_abstain:
        failures.append(f"expected should_abstain={case.should_abstain}, got abstained={answer_abstained}")

    if not answer_abstained:
        haystack = (answer_text or "").lower()
        for fact in case.expected_facts:
            alternatives = (fact,) if isinstance(fact, str) else fact
            if not any(alt.lower() in haystack for alt in alternatives):
                failures.append(f"missing expected fact: {alternatives!r} (need at least one)")

    return failures


def run(filter_terms: list[str] | None = None) -> bool:
    cases = GOLDEN_SET
    if filter_terms:
        cases = tuple(c for c in cases if any(t.lower() in c.id.lower() for t in filter_terms))
        if not cases:
            print(f"No golden cases match filter {filter_terms!r}")
            return False

    graph = build_graph()
    # Keeps the actual answer text per case (not just pass/fail) so a failure is diagnosable
    # from this run's output alone -- found live: a case can fail because that specific run's
    # answer genuinely lacked an expected fact, and without the real text there's no way to
    # tell that apart from a stale/wrong expectation without re-running the question by hand.
    results: list[tuple[GoldenCase, list[str], float, str | None]] = []

    for case in cases:
        print(f"Running [{case.id}] {case.question!r} ...", flush=True)
        start = time.perf_counter()
        result = graph.invoke({"message": case.question, "memory": {}})
        elapsed = time.perf_counter() - start
        answer = result["answer"]
        failures = _check(case, answer.abstained, answer.answer)
        results.append((case, failures, elapsed, answer.answer))
        status = "PASS" if not failures else "FAIL"
        print(f"  -> {status} ({elapsed:.1f}s)")
        for f in failures:
            print(f"     - {f}")

    passed = sum(1 for _, failures, _, _ in results if not failures)
    total = len(results)
    total_time = sum(elapsed for _, _, elapsed, _ in results)

    print()
    print(f"{'=' * 50}")
    print(f"{passed}/{total} passed ({total_time:.1f}s total)")
    if passed < total:
        print()
        print("Failures:")
        for case, failures, _, answer_text in results:
            if failures:
                print(f"  [{case.id}] {case.question!r}")
                for f in failures:
                    print(f"    - {f}")
                print(f"    actual answer: {answer_text!r}")
                if case.notes:
                    print(f"    note: {case.notes}")
    print(f"{'=' * 50}")

    return passed == total


if __name__ == "__main__":
    ok = run(sys.argv[1:] or None)
    sys.exit(0 if ok else 1)
