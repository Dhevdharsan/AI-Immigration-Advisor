"""
Tests the shared quote-verification helper (app/agents/_quotes.py), used
by both the Contradiction Finder and the verifier. Locks in the two
properties that matter: minor paraphrase-level edits (a model transcribing
"their" as "the student's") should still verify, but a substantially
fabricated quote should not.
"""

from app.agents._quotes import quote_verified

SOURCE = (
    "Students who have timely and properly filed a Form I-765 for the 24-month OPT "
    "extension may continue working until the date of the USCIS written decision on the "
    "current Form I-765 or for up to 180 days after their current post-completion OPT "
    "expires, whichever is earlier."
)


def test_exact_quote_verifies():
    assert quote_verified(SOURCE, SOURCE) is True


def test_minor_paraphrase_still_verifies():
    quote = SOURCE.replace("their current post-completion", "the student's current post-completion")
    assert quote_verified(quote, SOURCE) is True


def test_fabricated_quote_does_not_verify():
    fabricated = "Please note that you cannot begin to work while the Form I-765 is pending with USCIS."
    assert quote_verified(fabricated, SOURCE) is False


def test_empty_quote_does_not_verify():
    assert quote_verified("", SOURCE) is False
