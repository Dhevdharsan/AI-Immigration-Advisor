"""
Parser tests run against saved HTML fixtures (real pages, captured by hand) so
they don't depend on network access or on USCIS/SEVP staying up. The fixtures
themselves are what proved the sites are scrapable at all (see project notes).
"""

from datetime import date
from pathlib import Path

from app.retrieval.scraper import parse_sevp_page, parse_uscis_policy_manual_page
from app.schemas.document import DocType
from app.schemas.taxonomy import RetrievalSource

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_uscis_policy_manual_page():
    html = (FIXTURES / "uscis_volume2_partf_chapter2.html").read_text()
    doc = parse_uscis_policy_manual_page(
        html, "https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-2"
    )

    assert doc.source == RetrievalSource.USCIS
    assert doc.doc_type == DocType.POLICY_MANUAL
    assert doc.title == "Chapter 2 - Eligibility Requirements"
    assert "full course of study" in doc.text
    assert doc.last_updated == date(2025, 2, 26)
    assert len(doc.text) > 500


def test_parse_sevp_page():
    html = (FIXTURES / "sevp_maintaining_status.html").read_text()
    doc = parse_sevp_page(
        html,
        "https://studyinthestates.dhs.gov/students/maintaining-status",
        lastmod=date(2025, 4, 29),
    )

    assert doc.source == RetrievalSource.SEVP
    assert doc.doc_type == DocType.GUIDANCE_PAGE
    assert doc.title == "Maintaining Status"
    assert "maintain your F or M student status" in doc.text
    assert doc.last_updated == date(2025, 4, 29)
    assert len(doc.text) > 500
