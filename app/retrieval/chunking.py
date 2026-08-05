"""
Splits a scraped page's full text into overlapping passages small enough to
embed meaningfully (a whole 40,000-character policy manual chapter embedded
as one vector would blur together too many unrelated topics to be useful
for similarity search). The overlap means a sentence that would otherwise
land right on a chunk boundary still shows up whole in at least one chunk.
"""

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Break at the last whitespace before `end` so we don't cut a word in half.
            break_point = text.rfind(" ", start, end)
            if break_point > start:
                end = break_point
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break

        next_start = max(end - overlap, start + 1)  # ensure forward progress even on tiny chunk_size/overlap combos
        # `next_start` is just an arithmetic offset, not aligned to a word boundary -- without
        # this, the next chunk could begin mid-word (this was a real bug, caught by a test
        # asserting no chunk contains a broken token).
        space_idx = text.find(" ", next_start)
        start = space_idx + 1 if space_idx != -1 else next_start
    return chunks
