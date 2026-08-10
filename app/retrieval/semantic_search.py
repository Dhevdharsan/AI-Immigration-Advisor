"""
Real hybrid search (Section 12) over the pgvector chunk index -- replaces
the old curated category -> URL lookup table (search.py's original
approach). Matches a question against whichever chunks are actually
closest in meaning, which is what a fixed per-category keyword list can
never do for a question like "What is an I-20?" that doesn't belong to
any of the planner's six action-oriented categories.

Pure vector search has its own real failure mode, found live: a large
document (e.g. Publication 519, 500+ chunks) can crowd every slot of the
top-K with tangentially-related passages, burying a short, exact-match
page like "about Form 8843" hundreds of ranks down even though it's the
literal right answer -- embeddings don't specially weight a literal term
like a form number the way a keyword search does. So this runs vector
search AND keyword search as two independent candidate lists, then fuses
them with Reciprocal Rank Fusion (`_reciprocal_rank_fusion`) rather than
trusting either ranking alone.

A second, distinct failure mode, also found live: a *broad* question
("what are the tax exemptions for X") can rank the one chunk with the
actual concrete answer (a specific dollar figure, a specific form number)
hundreds of ranks down on BOTH vector and keyword search, because that
chunk -- e.g. a dense numeric worksheet -- just doesn't read as similar to
the broad question, even though it's the literal answer. Widening top_k
doesn't fix this; the chunk isn't in the top few hundred, let alone top
40. The fix is query expansion (see grounding_loop.py's `_expand_queries`):
the caller can pass additional, more specific sub-queries via
`extra_queries`, and every query variant (original + expansions) gets its
own vector + keyword search, all fused together into one ranking -- so a
sub-query like "India tax treaty standard deduction amount" can surface a
chunk the original broad question never would have.

Matched chunks are grouped back into Document-shaped objects (one per
source URL) so the rest of the pipeline (grounding loop, Contradiction
Finder, verifier) doesn't need to change its Document-based contract.
Each Document's `.text` is reconstructed from only the matched excerpts,
not the page's full original text -- a bonus over the old approach, which
sent an entire 40,000+ character chapter to the LLM every time.

A third failure mode, found live, survives even query expansion: a chunk
can score well on exactly ONE of the up-to-6 vector/keyword lists (say,
rank 61 of 40-per-list) and nowhere else, so its fused RRF score loses to
chunks that rank moderately across several lists -- genuinely, not as a
bug, since that chunk really is a weaker STATISTICAL match even though a
human reading it would call it the obviously right answer (a terse,
plainly-worded worksheet number just doesn't share much surface similarity
with a natural-language question). No amount of widening the candidate
pool or loosening the per-document cap fixes this; a wider net still
ranks it low by the same scoring rule. `_llm_rerank` is the fix: pull a
much larger, loosely-capped candidate pool (`_RERANK_POOL_SIZE`), then
have an LLM actually read every candidate and judge relevance directly,
instead of trusting the vector/keyword scores that got them into the
pool. This is a genuinely different, more expensive technique -- not
another scoring-formula tweak -- because it substitutes reading
comprehension for statistical similarity.
"""

import json

from openai import OpenAI

from app.config import MECHANICAL_MODEL, get_openai_client
from app.retrieval.embeddings import embed_text
from app.retrieval.vector_store import get_chunks_by_keys, keyword_search, vector_search
from app.schemas.chunk import ChunkSearchResult
from app.schemas.document import Document
from app.schemas.taxonomy import RetrievalSource

DEFAULT_TOP_K = 8

# Each of vector/keyword search pulls this many candidates before fusion -- wider than the
# final top_k so a chunk that's mediocre by one method but strong by the other still has room
# to surface (Form 8843's vector rank was ~279th; the candidate pool needs to comfortably
# exceed the final top_k for fusion to matter at all, not just re-sort the same short list).
#
# Must be raised alongside `_RERANK_POOL_SIZE`, confirmed by hand -- this constant, not the
# fusion pool size, was the real bottleneck for the LLM re-ranker: a chunk that only scores
# well on ONE list (e.g. rank 61 of a specific vector search) gets truncated away before fusion
# ever sees it if this is smaller than that rank, no matter how large `_RERANK_POOL_SIZE` is
# set. The re-ranker can only rank what actually reaches it.
_CANDIDATE_POOL = 100

# Standard Reciprocal Rank Fusion damping constant (Cormack et al.). Larger values flatten the
# difference between e.g. rank 1 and rank 2 so neither list dominates purely by being a tight
# cluster of close scores at the top; 60 is the conventional default and untuned here.
_RRF_K = 60

# Two different caps for two different jobs. `_POOL_MAX_CHUNKS_PER_DOCUMENT` (loose) governs
# the wide net handed to the LLM re-ranker -- still capped, so one pathological document with
# hundreds of near-duplicate chunks can't fill the entire pool before the re-ranker even sees
# anything else, but loose enough that a document with several genuinely distinct relevant
# chunks (e.g. a worksheet split across separate title/eligibility/figure chunks) can have more
# than 2 make the pool together. `_FINAL_MAX_CHUNKS_PER_DOCUMENT` (tight) is unchanged from
# before and still applies to the final selection when re-ranking is skipped (empty pool).
_POOL_MAX_CHUNKS_PER_DOCUMENT = 5
_FINAL_MAX_CHUNKS_PER_DOCUMENT = 2

# Candidate pool size handed to the LLM re-ranker. Lowered from the "100+" rule-of-thumb
# starting point to 50 to cut the re-ranking prompt's token cost/latency roughly in half --
# still comfortably large enough that a chunk which only scores well on one of several
# vector/keyword lists is very likely still included, even though its fused RRF score alone
# wouldn't put it in the final top_k.
_RERANK_POOL_SIZE = 50

# Below this many pooled candidates, re-ranking adds a full LLM call for marginal benefit --
# there's little left to sort through, so the fused RRF order (already a real signal, not
# arbitrary) is used directly instead of paying re-ranking's latency/cost for it.
_RERANK_SKIP_THRESHOLD = 15

_RERANK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"relevant_ids": {"type": "array", "items": {"type": "integer"}}},
    "required": ["relevant_ids"],
    "additionalProperties": False,
}

_RERANK_SYSTEM_PROMPT = """You are the re-ranking step of a retrieval system for an assistant \
that answers immigration and tax questions for F-1 international students. You are given a \
question and a numbered list of candidate passages, each pulled by a first-pass keyword/\
similarity search that is fast but imprecise -- it often ranks a passage highly just because \
it shares surface-level wording with the question, while a short, plainly-worded passage \
elsewhere in the list is actually the concrete answer (e.g. a specific dollar figure or form \
number sitting in a terse worksheet passage that doesn't closely resemble the question's own \
phrasing at all).

Read every passage and select the ones that most directly and concretely help answer the \
question, ordered from most to least useful. A passage that states a specific fact -- a \
number, a form name, a deadline, an eligibility rule -- is more useful than one that is merely \
topically related. Return only the IDs of passages that are genuinely useful; it is fine to \
return fewer than requested if most candidates aren't relevant."""


def _llm_rerank(query: str, candidates: list[ChunkSearchResult], top_k: int, client: OpenAI) -> list[ChunkSearchResult]:
    if not candidates:
        return []
    numbered = "\n\n".join(f"[{i}] ({c.title})\n{c.chunk_text[:600]}" for i, c in enumerate(candidates))
    # MECHANICAL_MODEL, not GENERATOR_MODEL: sorting passages by relevance is a mechanical
    # task, not the nuanced judgment GENERATOR_MODEL is reserved for.
    response = client.chat.completions.create(
        model=MECHANICAL_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nCandidate passages:\n{numbered}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "rerank_selection", "schema": _RERANK_RESPONSE_SCHEMA, "strict": True},
        },
    )
    parsed = json.loads(response.choices[0].message.content)

    selected: list[ChunkSearchResult] = []
    seen: set[int] = set()
    for idx in parsed["relevant_ids"]:
        if idx in seen or not (0 <= idx < len(candidates)):
            continue
        seen.add(idx)
        selected.append(candidates[idx])
        if len(selected) >= top_k:
            break
    return selected


def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    source: RetrievalSource | None = None,
    extra_queries: list[str] | None = None,
    client: OpenAI | None = None,
) -> list[Document]:
    """`extra_queries` (see grounding_loop.py's `_expand_queries`) are additional, more
    specific phrasings of the same underlying question -- each gets its own vector + keyword
    search, and every list is fused together, so a narrow sub-query can surface a chunk the
    original broad question's search never would have.

    Retrieval is now two stages, not one: a wide, loosely-capped candidate pool
    (`_RERANK_POOL_SIZE`) from fused vector+keyword search, then an LLM re-ranking pass
    (`_llm_rerank`) that reads every candidate and picks the true top_k -- see module
    docstring for why RRF's score alone isn't enough for a chunk that only one list favors."""
    result_lists: list[list[ChunkSearchResult]] = []
    for q in [query, *(extra_queries or [])]:
        query_embedding = embed_text(q)
        result_lists.append(vector_search(query_embedding, top_k=_CANDIDATE_POOL, source=source))
        result_lists.append(keyword_search(q, top_k=_CANDIDATE_POOL, source=source))

    pool = _reciprocal_rank_fusion(result_lists, top_k=_RERANK_POOL_SIZE, max_per_document=_POOL_MAX_CHUNKS_PER_DOCUMENT)
    if not pool:
        return []

    if len(pool) <= _RERANK_SKIP_THRESHOLD:
        reranked = pool[:top_k]
    else:
        client = client or get_openai_client()
        reranked = _llm_rerank(query, pool, top_k=top_k, client=client)
    expanded = _expand_with_neighbors(reranked)
    return _group_into_documents(expanded)


def _chunk_key(chunk: ChunkSearchResult) -> tuple[str, int]:
    return (chunk.url, chunk.chunk_index)


def _expand_with_neighbors(chunks: list[ChunkSearchResult]) -> list[ChunkSearchResult]:
    """Pulls in each selected chunk's immediate previous/next chunk from the same page, even
    though the neighbor wasn't independently selected. Found live: a selected chunk can start
    mid-thought -- e.g. "This is commonly known as a grace period..." with the sentence that
    actually defines the grace period sitting in the chunk right before the boundary. Without
    that neighbor, an otherwise-sufficient, correct passage reads as incomplete, and the
    grounding loop's sufficiency check inconsistently (not always, which is what made this hard
    to pin down) marked it insufficient because of it, even though retrieval had already found
    the right page every time."""
    existing_keys = {_chunk_key(c) for c in chunks}
    needed_keys: list[tuple[str, int]] = []
    for c in chunks:
        for neighbor_index in (c.chunk_index - 1, c.chunk_index + 1):
            if neighbor_index < 0:
                continue
            key = (c.url, neighbor_index)
            if key not in existing_keys:
                needed_keys.append(key)
                existing_keys.add(key)  # avoid duplicate fetches if two selected chunks share a neighbor
    if not needed_keys:
        return chunks
    return chunks + get_chunks_by_keys(needed_keys)


def _reciprocal_rank_fusion(
    result_lists: list[list[ChunkSearchResult]], top_k: int, max_per_document: int = _FINAL_MAX_CHUNKS_PER_DOCUMENT
) -> list[ChunkSearchResult]:
    """Merges any number of independently-ranked chunk lists (one vector + one keyword search
    per query variant) using each chunk's *position* in each list, not its raw score -- cosine
    distance and text-search rank live on incomparable scales, but rank position is always
    comparable. A chunk found by only one list still counts, using just that list's
    contribution; a chunk found by several (e.g. both the original query and an expansion)
    accumulates score from each, so genuinely relevant chunks still tend to rise regardless of
    how many query variants happen to be in play.

    Selection walks the fused ranking greedily but skips a chunk once its source URL has
    already contributed `max_per_document` chunks, so one dominant document can't fill every
    slot -- lower-ranked chunks from other pages get pulled in instead."""
    scores: dict[tuple[str, int], float] = {}
    chunk_by_key: dict[tuple[str, int], ChunkSearchResult] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            chunk_by_key.setdefault(key, chunk)

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    selected: list[ChunkSearchResult] = []
    chunks_used_per_url: dict[str, int] = {}
    for key in ranked_keys:
        chunk = chunk_by_key[key]
        if chunks_used_per_url.get(chunk.url, 0) >= max_per_document:
            continue
        selected.append(chunk)
        chunks_used_per_url[chunk.url] = chunks_used_per_url.get(chunk.url, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def _group_into_documents(results: list[ChunkSearchResult]) -> list[Document]:
    by_url: dict[str, list[ChunkSearchResult]] = {}
    order: list[str] = []
    for r in results:
        if r.url not in by_url:
            by_url[r.url] = []
            order.append(r.url)
        by_url[r.url].append(r)

    documents = []
    for url in order:
        chunks = sorted(by_url[url], key=lambda c: c.chunk_index)  # document order reads more coherently than relevance rank
        first = chunks[0]
        # A "..." gap marker is only honest between chunks that actually skip content --
        # neighbor-expanded chunks (see _expand_with_neighbors) are genuinely consecutive, and
        # showing "..." between them would misleadingly suggest missing text that isn't missing.
        parts = [chunks[0].chunk_text]
        for prev, curr in zip(chunks, chunks[1:]):
            parts.append("\n\n" if curr.chunk_index == prev.chunk_index + 1 else "\n\n...\n\n")
            parts.append(curr.chunk_text)
        text = "".join(parts)
        documents.append(
            Document(
                source=first.source,
                doc_type=first.doc_type,
                url=url,
                title=first.title,
                text=text,
                last_updated=first.last_updated,
            )
        )
    return documents
