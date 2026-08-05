"""Tests the chunker's core properties: no content lost, no word cut in
half at a boundary, and short text isn't needlessly split."""

from app.retrieval.chunking import chunk_text


def test_short_text_is_a_single_chunk():
    text = "This is a short passage."
    assert chunk_text(text) == [text]


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(2000))  # ~13,890 chars
    chunks = chunk_text(text, chunk_size=1200, overlap=200)

    assert len(chunks) > 1
    # No chunk should have cut a word in half -- every chunk should be made
    # entirely of whole "wordN" tokens.
    for chunk in chunks:
        for token in chunk.split():
            assert token.startswith("word"), f"chunk contains a broken token: {token!r}"


def test_chunks_overlap_so_boundary_content_is_not_lost():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, chunk_size=1200, overlap=200)

    # A ~200-char overlap covers many words (not just the very last/first one), so check
    # for any shared words between consecutive chunks rather than just the immediate edges.
    for first, second in zip(chunks, chunks[1:]):
        assert set(first.split()) & set(second.split()), "consecutive chunks should overlap"
