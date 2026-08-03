"""
Tier 1 (USCIS) and Tier 2 (SEVP) scrapers (Section 4).

Two things confirmed by hand before writing this (see project conversation):
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
"""

import re
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


def _fetch(url: str) -> str:
    resp = cffi_requests.get(url, impersonate="chrome120", timeout=20)
    resp.raise_for_status()
    return resp.text


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
