"""
V1's stand-in for a real search index (Section 4/12). We haven't built
full-text/semantic search over the whole USCIS site -- that's chunking +
embeddings + a vector DB, a bigger lift than V1 needs. Instead:

  - USCIS: a curated map from category -> specific Policy Manual chapter
    URLs, verified by hand against the real sitemap (see project notes --
    Volume 2 Part F is "Students and Exchange Visitors"; Volume 1 Part E
    is "Adjudications", which is where RFEs/appeals actually live, since
    those are general USCIS procedures, not F-1-specific ones).
  - SEVP: keyword matching against Study in the States' URL slugs, which
    (unlike USCIS's numbered chapters) are already descriptive
    (e.g. /work/applying-for-practical-training).

Both only ever return real URLs pulled from the real sitemap -- nothing
here is a guessed or fabricated citation. Swapping this module for real
semantic search later shouldn't require the grounding loop to change.
"""

from app.retrieval.cache import get as cache_get
from app.retrieval.cache import put as cache_put
from app.retrieval.scraper import fetch_sevp_page, fetch_uscis_policy_manual_page, list_sevp_urls
from app.schemas.document import Document
from app.schemas.taxonomy import Category, RetrievalSource

_USCIS_BASE = "https://www.uscis.gov/policy-manual"

USCIS_CANDIDATE_URLS: dict[Category, list[str]] = {
    Category.WORK_AUTHORIZATION: [
        f"{_USCIS_BASE}/volume-2-part-f-chapter-5",  # Practical Training
        f"{_USCIS_BASE}/volume-2-part-f-chapter-6",  # Employment
    ],
    Category.STATUS_MAINTENANCE: [
        f"{_USCIS_BASE}/volume-2-part-f-chapter-3",  # Courses/Enrollment/Full Course of Study/RCL
        f"{_USCIS_BASE}/volume-2-part-f-chapter-2",  # Eligibility Requirements
    ],
    Category.TRAVEL_REENTRY: [
        f"{_USCIS_BASE}/volume-2-part-f-chapter-7",  # Absences From the United States
    ],
    Category.RFE_DOCUMENT_RESPONSE: [
        f"{_USCIS_BASE}/volume-1-part-e-chapter-6",  # Evidence
    ],
    Category.APPEALS: [
        f"{_USCIS_BASE}/volume-1-part-e-chapter-10",  # Post-Decision Actions
    ],
    Category.DEADLINES: [
        f"{_USCIS_BASE}/volume-2-part-f-chapter-7",  # Absences (re-entry windows)
        f"{_USCIS_BASE}/volume-2-part-f-chapter-8",  # Change of Status/Extension/Length of Stay (grace periods)
    ],
}

# Keywords matched against SEVP's descriptive URL slugs. Empty list means
# "no good SEVP-specific page for this category" -- the loop just won't
# find documents here and will fall through to the next tier.
SEVP_KEYWORDS: dict[Category, list[str]] = {
    Category.WORK_AUTHORIZATION: ["practical-training", "working-in-the-united-states"],
    Category.STATUS_MAINTENANCE: ["maintaining-status", "full-course-of-study"],
    Category.TRAVEL_REENTRY: ["travel"],
    Category.RFE_DOCUMENT_RESPONSE: [],
    Category.APPEALS: [],
    Category.DEADLINES: ["change-of-status", "extension"],
}

_FETCHERS = {
    RetrievalSource.USCIS: fetch_uscis_policy_manual_page,
    RetrievalSource.SEVP: fetch_sevp_page,
}

# Only sources with a working scraper (Section 4: Tiers 1-2 wired up so far).
IMPLEMENTED_SOURCES: tuple[RetrievalSource, ...] = tuple(_FETCHERS)


def candidate_urls(category: Category, source: RetrievalSource) -> list[str]:
    if source is RetrievalSource.USCIS:
        return USCIS_CANDIDATE_URLS.get(category, [])
    if source is RetrievalSource.SEVP:
        keywords = SEVP_KEYWORDS.get(category, [])
        if not keywords:
            return []
        return [url for url, _ in list_sevp_urls() if any(k in url for k in keywords)]
    return []


def fetch_cached(url: str, source: RetrievalSource) -> Document:
    cached = cache_get(url)
    if cached is not None:
        return cached
    document = _FETCHERS[source](url)
    cache_put(url, document)
    return document


def retrieve(category: Category, source: RetrievalSource) -> list[Document]:
    return [fetch_cached(url, source) for url in candidate_urls(category, source)]
