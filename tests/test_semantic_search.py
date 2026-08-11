"""
Tests the hybrid-search internals in isolation (Section 12) -- the part of
today's work that had zero automated coverage and was only ever verified by
hand against the live API and database. Covers: does RRF fusion merge
multiple ranked lists and enforce the per-document cap correctly, does
neighbor-expansion fetch the right adjacent chunks without re-fetching
already-selected ones, does document-grouping use a real "..." gap marker
only where content is actually skipped (not between genuinely consecutive
chunks), does the LLM re-ranker map returned IDs back to candidates safely,
and does semantic_search skip re-ranking entirely for a small pool. The
database (vector_search/keyword_search/get_chunks_by_keys) and the
re-ranker's OpenAI client are mocked, so these need no network access, API
key, or running database.
"""

import json
from unittest.mock import MagicMock, patch

from app.retrieval.semantic_search import (
    _RERANK_SKIP_THRESHOLD,
    _expand_with_neighbors,
    _group_into_documents,
    _llm_rerank,
    _reciprocal_rank_fusion,
    semantic_search,
)
from app.schemas.chunk import ChunkSearchResult
from app.schemas.document import DocType
from app.schemas.taxonomy import RetrievalSource


def _chunk(url: str, chunk_index: int, text: str = "text", distance: float = 0.1) -> ChunkSearchResult:
    return ChunkSearchResult(
        url=url,
        title=url,
        source=RetrievalSource.USCIS,
        doc_type=DocType.POLICY_MANUAL,
        chunk_index=chunk_index,
        chunk_text=text,
        distance=distance,
    )


# ---------- _reciprocal_rank_fusion ----------


def test_fusion_ranks_chunk_found_in_multiple_lists_above_single_list_chunks():
    # b appears near the top of both lists (should win); a and c each appear in only one.
    list1 = [_chunk("https://b", 0), _chunk("https://a", 0)]
    list2 = [_chunk("https://b", 0), _chunk("https://c", 0)]

    result = _reciprocal_rank_fusion([list1, list2], top_k=3)

    assert result[0].url == "https://b"
    assert {r.url for r in result} == {"https://b", "https://a", "https://c"}


def test_fusion_enforces_per_document_cap():
    chunks = [_chunk("https://a", i) for i in range(5)]  # all 5 from the same URL

    result = _reciprocal_rank_fusion([chunks], top_k=5, max_per_document=2)

    assert len(result) == 2
    assert all(r.url == "https://a" for r in result)


def test_fusion_respects_top_k():
    chunks = [_chunk(f"https://{i}", 0) for i in range(10)]

    result = _reciprocal_rank_fusion([chunks], top_k=3, max_per_document=10)

    assert len(result) == 3


# ---------- _expand_with_neighbors ----------


def test_expand_with_neighbors_fetches_adjacent_chunks():
    selected = [_chunk("https://a", 5)]
    with patch("app.retrieval.semantic_search.get_chunks_by_keys") as mock_fetch:
        mock_fetch.return_value = [_chunk("https://a", 4), _chunk("https://a", 6)]
        result = _expand_with_neighbors(selected)

    mock_fetch.assert_called_once_with([("https://a", 4), ("https://a", 6)])
    assert {(c.url, c.chunk_index) for c in result} == {("https://a", 4), ("https://a", 5), ("https://a", 6)}


def test_expand_with_neighbors_skips_negative_index_and_already_selected():
    # chunk_index 0's "previous" neighbor would be -1 (must be skipped); chunk_index 1 is
    # already selected, so it must not be re-requested as chunk 0's "next" neighbor either.
    selected = [_chunk("https://a", 0), _chunk("https://a", 1)]
    with patch("app.retrieval.semantic_search.get_chunks_by_keys") as mock_fetch:
        mock_fetch.return_value = [_chunk("https://a", 2)]
        _expand_with_neighbors(selected)

    mock_fetch.assert_called_once_with([("https://a", 2)])


def test_expand_with_neighbors_dedupes_a_neighbor_shared_by_two_selected_chunks():
    # Three consecutive selected chunks (0, 1, 2): chunk 0's "previous" is negative (skipped),
    # and each chunk's neighbor toward the middle is already in the selected set -- so despite
    # 3 chunks each having 2 neighbor slots, the only genuinely missing key is chunk 3, fetched
    # exactly once, not requested redundantly by both chunk 2's request and any other chunk.
    selected = [_chunk("https://a", 0), _chunk("https://a", 1), _chunk("https://a", 2)]
    with patch("app.retrieval.semantic_search.get_chunks_by_keys") as mock_fetch:
        mock_fetch.return_value = [_chunk("https://a", 3)]
        _expand_with_neighbors(selected)

    mock_fetch.assert_called_once_with([("https://a", 3)])


# ---------- _group_into_documents ----------


def test_group_into_documents_merges_by_url_and_orders_by_chunk_index():
    chunks = [_chunk("https://a", 2, "third"), _chunk("https://a", 0, "first"), _chunk("https://b", 0, "other doc")]

    docs = _group_into_documents(chunks)

    by_url = {d.url: d for d in docs}
    assert by_url["https://a"].text.index("first") < by_url["https://a"].text.index("third")
    assert by_url["https://b"].text == "other doc"


def test_group_into_documents_uses_plain_gap_for_genuinely_adjacent_chunks():
    chunks = [_chunk("https://a", 0, "first"), _chunk("https://a", 1, "second")]

    doc = _group_into_documents(chunks)[0]

    # No "..." here -- these chunks are truly consecutive (e.g. via neighbor expansion), and a
    # "..." would falsely suggest skipped content that isn't actually missing.
    assert doc.text == "first\n\nsecond"


def test_group_into_documents_uses_gap_marker_for_nonadjacent_chunks():
    chunks = [_chunk("https://a", 0, "first"), _chunk("https://a", 5, "sixth")]

    doc = _group_into_documents(chunks)[0]

    assert doc.text == "first\n\n...\n\nsixth"


# ---------- _llm_rerank ----------


def _rerank_response(relevant_ids: list[int]):
    payload = json.dumps({"relevant_ids": relevant_ids})
    return MagicMock(choices=[MagicMock(message=MagicMock(content=payload))])


def test_llm_rerank_maps_ids_back_to_candidates_in_returned_order():
    candidates = [_chunk("https://a", 0), _chunk("https://b", 0), _chunk("https://c", 0)]
    client = MagicMock()
    client.chat.completions.create.return_value = _rerank_response([2, 0])

    result = _llm_rerank("some question", candidates, top_k=8, client=client)

    assert [r.url for r in result] == ["https://c", "https://a"]


def test_llm_rerank_ignores_out_of_range_and_duplicate_ids():
    candidates = [_chunk("https://a", 0), _chunk("https://b", 0)]
    client = MagicMock()
    client.chat.completions.create.return_value = _rerank_response([0, 0, 5, 1])

    result = _llm_rerank("some question", candidates, top_k=8, client=client)

    assert [r.url for r in result] == ["https://a", "https://b"]


def test_llm_rerank_respects_top_k():
    candidates = [_chunk(f"https://{i}", 0) for i in range(5)]
    client = MagicMock()
    client.chat.completions.create.return_value = _rerank_response([0, 1, 2, 3, 4])

    result = _llm_rerank("some question", candidates, top_k=2, client=client)

    assert len(result) == 2


def test_llm_rerank_skips_client_call_when_no_candidates():
    client = MagicMock()

    result = _llm_rerank("some question", [], top_k=8, client=client)

    assert result == []
    client.chat.completions.create.assert_not_called()


# ---------- semantic_search: skip-reranking-when-small-pool ----------


def test_semantic_search_skips_reranking_for_small_pool():
    small_pool = [_chunk("https://a", 0)]  # well under _RERANK_SKIP_THRESHOLD
    client = MagicMock()
    with (
        patch("app.retrieval.semantic_search.embed_text", return_value=[0.1]),
        patch("app.retrieval.semantic_search.vector_search", return_value=small_pool),
        patch("app.retrieval.semantic_search.keyword_search", return_value=[]),
        patch("app.retrieval.semantic_search.get_chunks_by_keys", return_value=[]),
    ):
        docs = semantic_search("some question", top_k=8, source=RetrievalSource.USCIS, client=client)

    client.chat.completions.create.assert_not_called()  # re-ranking skipped entirely
    assert len(docs) == 1


def test_semantic_search_reranks_for_pool_above_threshold():
    large_pool = [_chunk(f"https://{i}", 0) for i in range(_RERANK_SKIP_THRESHOLD + 5)]
    client = MagicMock()
    client.chat.completions.create.return_value = _rerank_response([0])
    with (
        patch("app.retrieval.semantic_search.embed_text", return_value=[0.1]),
        patch("app.retrieval.semantic_search.vector_search", return_value=large_pool),
        patch("app.retrieval.semantic_search.keyword_search", return_value=[]),
        patch("app.retrieval.semantic_search.get_chunks_by_keys", return_value=[]),
    ):
        semantic_search("some question", top_k=8, source=RetrievalSource.USCIS, client=client)

    client.chat.completions.create.assert_called_once()  # re-ranking actually ran
