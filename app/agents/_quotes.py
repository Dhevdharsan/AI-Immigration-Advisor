"""
Shared text-matching helpers for verifying a model-claimed quote against
real source text. Used by both the Contradiction Finder and the verifier:
any time a model claims "this text supports my claim," we check the quote
actually appears in the source rather than trusting it outright.
"""

import difflib

# Found in development: a smaller judge model reliably finds the right passage
# but doesn't always transcribe it perfectly verbatim (e.g. it swapped "their"
# for "the student's" in an otherwise-correct quote). An exact-substring check
# rejected that as unsupported even though the claim was genuinely grounded --
# a false-negative failure mode (needless abstaining) that's as damaging to
# usefulness as a fabricated quote is to safety. This threshold allows
# minor paraphrase-level edits while still rejecting a quote that's
# substantially made up, which won't have anywhere near this much real overlap.
_FUZZY_MATCH_THRESHOLD = 0.85


def normalize(text: str) -> str:
    return " ".join(text.split())


def quote_verified(quote: str, source_text: str, threshold: float = _FUZZY_MATCH_THRESHOLD) -> bool:
    normalized_quote = normalize(quote)
    normalized_source = normalize(source_text)
    if not normalized_quote:
        return False
    if normalized_quote in normalized_source:
        return True
    matcher = difflib.SequenceMatcher(None, normalized_quote, normalized_source, autojunk=False)
    matched_chars = sum(block.size for block in matcher.get_matching_blocks())
    return (matched_chars / len(normalized_quote)) >= threshold
