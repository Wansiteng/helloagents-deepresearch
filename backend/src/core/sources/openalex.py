"""OpenAlex knowledge source.

`OpenAlex <https://openalex.org/>`_ is a fully-open scholarly graph covering
~250M works across every discipline (papers, datasets, books, theses…).
No API key is required, but supplying an email pushes calls into the
"polite pool" with higher rate limits and lower latency.

Each chunk's content is the paper abstract (reconstructed from OpenAlex's
``abstract_inverted_index``) plus a small bibliographic header so the
downstream summarizer prompt has authors + venue + year + citation count
to anchor on. Metadata carries enough for the citation formatter
(:ref:`improvement-roadmap` item #5) to emit proper IEEE/APA refs and for
the upcoming PDF ingest pipeline (item #2) to find an open-access PDF.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx

from core.knowledge import KnowledgeChunk, KnowledgeQuery

if TYPE_CHECKING:
    from config import Configuration

logger = logging.getLogger(__name__)

_OPENALEX_API = "https://api.openalex.org/works"
_TIMEOUT = httpx.Timeout(15.0)


class OpenAlexSource:
    """A :class:`~core.knowledge.KnowledgeSource` backed by OpenAlex."""

    name = "openalex"

    def __init__(self, config: "Configuration") -> None:
        self._config = config
        email = (config.openalex_email or "").strip()
        # OpenAlex's polite pool: identify yourself in the UA OR pass
        # ?mailto=. Doing both is harmless.
        self._email = email
        self._user_agent = (
            f"helloagents-deepresearch/0.1 (mailto:{email})"
            if email
            else "helloagents-deepresearch/0.1"
        )

    @property
    def is_local(self) -> bool:
        """OpenAlex is queried over the public internet."""
        return False

    async def query(self, q: KnowledgeQuery) -> list[KnowledgeChunk]:
        """Search OpenAlex for works matching ``q``."""
        params: dict[str, str] = {
            "search": q.text,
            "per-page": str(max(1, min(q.max_results, 25))),  # API max 200; 25 plenty
            "sort": "relevance_score:desc",
        }
        if self._email:
            params["mailto"] = self._email

        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT, headers={"User-Agent": self._user_agent}
            ) as client:
                resp = await client.get(_OPENALEX_API, params=params)
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as exc:
            logger.warning("OpenAlexSource: API call failed: %s", exc)
            return []

        chunks: list[KnowledgeChunk] = []
        for work in data.get("results", []):
            chunk = self._build_chunk(work)
            if chunk is not None:
                chunks.append(chunk)

        logger.info("OpenAlexSource: returned %d chunks", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _build_chunk(self, work: dict[str, Any]) -> KnowledgeChunk | None:
        title = (work.get("title") or "").strip()
        if not title:
            return None

        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

        year = work.get("publication_year") or ""
        venue = ""
        host = work.get("host_venue") or work.get("primary_location") or {}
        if isinstance(host, dict):
            venue = (
                host.get("display_name")
                or host.get("source", {}).get("display_name")
                or ""
            )

        authors: list[str] = []
        for authorship in work.get("authorships", [])[:8]:
            author = authorship.get("author") or {}
            name = author.get("display_name", "").strip()
            if name:
                authors.append(name)
        total_authors = len(work.get("authorships", []))
        author_block = "; ".join(authors)
        if total_authors > len(authors):
            author_block += " et al."

        cited_by = work.get("cited_by_count", 0)
        doi = work.get("doi") or ""
        openalex_id = work.get("id") or ""
        # Prefer DOI URL for citations; fall back to OpenAlex landing.
        canonical_url = doi if doi else openalex_id

        # Open-access PDF (if available) — fed to #2 PDF ingest later.
        pdf_url = ""
        oa = work.get("open_access") or {}
        if isinstance(oa, dict):
            pdf_url = (oa.get("oa_url") or "").strip()
        if not pdf_url:
            primary = work.get("primary_location") or {}
            if isinstance(primary, dict):
                pdf_url = (primary.get("pdf_url") or "").strip()

        header_lines = [f"Title: {title}"]
        if author_block:
            header_lines.append(f"Authors: {author_block}")
        if year:
            header_lines.append(f"Year: {year}")
        if venue:
            header_lines.append(f"Venue: {venue}")
        if cited_by:
            header_lines.append(f"Cited by: {cited_by}")

        if abstract:
            content = "\n".join(header_lines) + "\n\nAbstract:\n" + abstract
        else:
            # No abstract published — header alone still useful for citation,
            # but skip if there's literally nothing for the summarizer to chew on.
            content = "\n".join(header_lines)
            if not authors and not year:
                return None

        return KnowledgeChunk(
            source="openalex",
            title=title,
            url_or_path=canonical_url,
            content=content,
            score=float(work.get("relevance_score") or 0) or None,
            metadata={
                "authors": authors,
                "year": str(year) if year else "",
                "venue": venue,
                "doi": doi,
                "openalex_id": openalex_id,
                "cited_by_count": cited_by,
                "pdf_url": pdf_url,
                "kind": "academic",
            },
        )


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Rebuild a readable abstract from OpenAlex's inverted index.

    OpenAlex stores abstracts as ``{word: [positions...]}`` (a license workaround).
    We invert that back into a plain string by placing each word at every
    listed position, then joining.
    """
    if not inverted or not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            try:
                positions.append((int(i), word))
            except (TypeError, ValueError):
                continue
    if not positions:
        return ""
    positions.sort(key=lambda t: t[0])
    return " ".join(word for _, word in positions)
