"""
Parser tests run against saved HTML fixtures (real pages, captured by hand) so
they don't depend on network access or on USCIS/SEVP staying up. The fixtures
themselves are what proved the sites are scrapable at all (see project notes).
"""

from datetime import date
from pathlib import Path

from app.retrieval.scraper import _irs_doc_type, parse_irs_page, parse_sevp_page, parse_uscis_policy_manual_page
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


def test_parse_irs_page():
    html = (FIXTURES / "irs_substantial_presence_test.html").read_text()
    doc = parse_irs_page(
        html, "https://www.irs.gov/individuals/international-taxpayers/substantial-presence-test"
    )

    assert doc.source == RetrievalSource.IRS
    assert doc.doc_type == DocType.GUIDANCE_PAGE
    assert doc.title == "Substantial presence test"
    assert "substantial presence test" in doc.text.lower()
    assert doc.last_updated == date(2026, 3, 14)
    assert len(doc.text) > 500


def test_irs_doc_type_classification():
    assert _irs_doc_type("https://www.irs.gov/publications/p519") == DocType.POLICY_MANUAL
    assert _irs_doc_type("https://www.irs.gov/forms-pubs/about-form-8843") == DocType.OFFICIAL_FORM
    assert (
        _irs_doc_type("https://www.irs.gov/individuals/international-taxpayers/substantial-presence-test")
        == DocType.GUIDANCE_PAGE
    )
