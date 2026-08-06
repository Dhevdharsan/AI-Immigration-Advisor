"""
Immigration-domain scrapers -- Tier 1 (USCIS) and Tier 2 (SEVP) -- plus the
Tax-domain scraper (IRS). Section 4/12.

Two things confirmed by hand before writing the USCIS/SEVP scrapers (see
project conversation):
  - USCIS.gov sits behind Akamai bot detection that fingerprints the TLS
    handshake itself, not just HTTP headers: `curl` and browsers pass, but
    Python's `httpx`/`requests` get a 403 "Access Denied" (Akamai block page)
    even with identical headers, because their TLS ClientHello doesn't match
    a real browser's. `curl_cffi` (impersonate="chrome120") solves this by
    using an actual browser TLS fingerprint. No JS execution is needed --
    content is server-rendered HTML either way.
  - The Study in the States sitemap lists URLs on `edit-studyinthestates.dhs.gov`
    (a staging host that itself 403s); they must be rewritten to the public
    `studyinthestates.dhs.gov` host before fetching.

IRS.gov runs the same Drupal-based platform (same sitemap-index structure,
same `<article>` main-content pattern) and needed no bot-detection
workaround when checked by hand -- but its "last reviewed" date is in a
different format ("10-Jan-2026", not "January 10, 2026"), so it gets its
own date regex.
"""

import re
import time
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

from app.schemas.document import Document, DocType
from app.schemas.taxonomy import RetrievalSource

_MONTH_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4}\b"
)

_IRS_DATE_RE = re.compile(r"Page Last Reviewed or Updated:?\s*(\d{1,2}-[A-Za-z]{3}-\d{4})")

_MAX_FETCH_RETRIES = 3


def _fetch(url: str) -> str:
    """Retries transient network failures (connection resets, timeouts) with backoff.
    Added after IRS.gov started intermittently resetting connections partway through a
    burst of sitemap-page requests during development -- general connectivity and USCIS.gov
    were both unaffected at the same time, pointing to IRS-specific rate limiting rather
    than a real outage or a bug in the request itself."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_FETCH_RETRIES):
        try:
            resp = cffi_requests.get(url, impersonate="chrome120", timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: retry any transient fetch failure
            last_exc = exc
            if attempt < _MAX_FETCH_RETRIES - 1:
                time.sleep(2**attempt)  # 1s, 2s
    raise last_exc


def _clean_text(tag) -> str:
    for junk in tag.select("script, style, nav, footer"):
        junk.decompose()
    text = tag.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _latest_date_in(text: str) -> date | None:
    found = []
    for m in _MONTH_DATE_RE.finditer(text):
        try:
            found.append(datetime.strptime(m.group(0), "%B %d, %Y").date())
        except ValueError:
            continue
    return max(found) if found else None


# ---------- USCIS Policy Manual (Tier 1) ----------


def list_uscis_policy_manual_urls() -> list[str]:
    """Walk the 5-page sitemap index and return stable Policy Manual chapter URLs."""
    urls: list[str] = []
    for page in range(1, 6):
        xml = _fetch(f"https://www.uscis.gov/sitemap.xml?page={page}")
        soup = BeautifulSoup(xml, "xml")
        for loc in soup.find_all("loc"):
            u = loc.text.strip()
            if u.startswith("https://www.uscis.gov/policy-manual/volume-"):
                urls.append(u)
    return urls


def parse_uscis_policy_manual_page(html: str, url: str) -> Document:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    title = title.split(" | USCIS")[0].strip()

    guidance_div = soup.select_one("div.tabcontent--guidance")
    text = _clean_text(guidance_div) if guidance_div else _clean_text(soup)

    updates_div = soup.select_one("div.tabcontent--updates")
    last_updated = _latest_date_in(_clean_text(updates_div)) if updates_div else None

    return Document(
        source=RetrievalSource.USCIS,
        doc_type=DocType.POLICY_MANUAL,
        url=url,
        title=title,
        text=text,
        last_updated=last_updated,
    )


def fetch_uscis_policy_manual_page(url: str) -> Document:
    return parse_uscis_policy_manual_page(_fetch(url), url)


# ---------- Study in the States / SEVP (Tier 2) ----------

_SEVP_STAGING_HOST = "edit-studyinthestates.dhs.gov"
_SEVP_PUBLIC_HOST = "studyinthestates.dhs.gov"


def list_sevp_urls(path_prefix: str = "/students/") -> list[tuple[str, date | None]]:
    """Return (url, lastmod) pairs from the SEVP sitemap, rewritten to the public host."""
    xml = _fetch(f"https://{_SEVP_PUBLIC_HOST}/sitemap.xml")
    soup = BeautifulSoup(xml, "xml")
    results: list[tuple[str, date | None]] = []
    for url_tag in soup.find_all("url"):
        loc = url_tag.find("loc")
        if loc is None:
            continue
        u = loc.text.strip().replace(_SEVP_STAGING_HOST, _SEVP_PUBLIC_HOST)
        if httpx.URL(u).path.startswith(path_prefix):
            lastmod_tag = url_tag.find("lastmod")
            lastmod = None
            if lastmod_tag is not None:
                try:
                    lastmod = date.fromisoformat(lastmod_tag.text.strip()[:10])
                except ValueError:
                    pass
            results.append((u, lastmod))
    return results


def parse_sevp_page(html: str, url: str, lastmod: date | None = None) -> Document:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    title = title.split(" | Study in the States")[0].strip()

    main = soup.select_one("main") or soup.select_one("#main-content")
    text = _clean_text(main) if main else _clean_text(soup)

    return Document(
        source=RetrievalSource.SEVP,
        doc_type=DocType.GUIDANCE_PAGE,
        url=url,
        title=title,
        text=text,
        last_updated=lastmod,
    )


def fetch_sevp_page(url: str, lastmod: date | None = None) -> Document:
    return parse_sevp_page(_fetch(url), url, lastmod)


# ---------- IRS (Tax domain, Tier 1) ----------

_IRS_LANGUAGE_PREFIXES = ("/es/", "/zh-hans/", "/zh-hant/", "/ko/", "/ru/", "/vi/", "/ht/", "/ja/", "/pt/")

# Keywords confirmed by hand against the real sitemap to match international-taxpayers
# subpages actually relevant to F-1 students (residency status, filing, treaties, FICA
# exemption) -- not the whole 436-page international-taxpayers section, which also covers
# unrelated populations (foreign artists/athletes on tour, US citizens living abroad, etc.).
_INTL_TAXPAYER_KEYWORDS = (
    "student",
    "scholarship",
    "nonresident",
    "resident-alien",
    "residency",
    "substantial-presence",
    "exempt-individual",
    "social-security",
    "medicare",
    "totalization",
    "tax-treat",
    "figuring-your-tax",
    "alien",
    "taxation-of",
)

# Verified by hand (see project conversation) -- the specific forms/publications an F-1
# student's tax questions actually turn on, beyond the international-taxpayers topic pages.
_IRS_FORM_AND_PUB_URLS = (
    "https://www.irs.gov/forms-pubs/about-form-8843",
    "https://www.irs.gov/forms-pubs/about-form-1040-nr",
    "https://www.irs.gov/forms-pubs/about-form-w-7",
    "https://www.irs.gov/publications/p519",  # U.S. Tax Guide for Aliens
    "https://www.irs.gov/publications/p901",  # U.S. Tax Treaties
    "https://www.irs.gov/publications/p970",  # Tax Benefits for Education (scholarships/fellowships)
)


def list_irs_urls() -> list[str]:
    """Walk the 12-page sitemap index, keeping English international-taxpayers subpages
    that match a student/nonresident-alien keyword, plus the verified forms/publications.

    IRS.gov doesn't 404 past the last real page: requesting page=13+ returns 200 with a
    fallback index whose <loc> entries just point back at sitemap.xml?page=1..12 --
    confirmed by hand. Those self-referential entries are filtered out before checking
    for emptiness, so that fallback page (not a real network failure) is what ends the
    walk, instead of looping on it forever."""
    urls: list[str] = []
    page = 1
    while True:
        xml = _fetch(f"https://www.irs.gov/sitemap.xml?page={page}")
        soup = BeautifulSoup(xml, "xml")
        locs = [loc.text.strip() for loc in soup.find_all("loc") if "sitemap.xml" not in loc.text]
        if not locs:
            break
        urls.extend(locs)
        page += 1
        time.sleep(0.5)  # pace the burst of sitemap-page requests -- see _fetch's retry note

    intl = [
        u
        for u in urls
        if "/individuals/international-taxpayers/" in u and not any(p in u for p in _IRS_LANGUAGE_PREFIXES)
    ]
    matched = sorted(set(u for u in intl for k in _INTL_TAXPAYER_KEYWORDS if k in u.lower()))
    return matched + list(_IRS_FORM_AND_PUB_URLS)


def _irs_doc_type(url: str) -> DocType:
    if "/publications/" in url:
        return DocType.POLICY_MANUAL  # comprehensive official guide -- highest doc-type priority within IRS
    if "/forms-pubs/about-form-" in url:
        return DocType.OFFICIAL_FORM
    return DocType.GUIDANCE_PAGE


def _irs_last_updated_in(html: str) -> date | None:
    m = _IRS_DATE_RE.search(html)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d-%b-%Y").date()
    except ValueError:
        return None


def parse_irs_page(html: str, url: str) -> Document:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    title = title.split(" | Internal Revenue Service")[0].strip()

    article = soup.find("article")
    text = _clean_text(article) if article else _clean_text(soup)

    return Document(
        source=RetrievalSource.IRS,
        doc_type=_irs_doc_type(url),
        url=url,
        title=title,
        text=text,
        last_updated=_irs_last_updated_in(html),
    )


def fetch_irs_page(url: str) -> Document:
    return parse_irs_page(_fetch(url), url)
