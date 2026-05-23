"""arXiv knowledge source.

Queries arXiv's public Atom API and returns each hit as a
:class:`~core.knowledge.KnowledgeChunk`. No API key required — arXiv asks
only that callers identify themselves with a meaningful ``User-Agent`` and
avoid hammering the endpoint.

The chunk content is the paper abstract (which is what arXiv ships in the
search response). Title, authors, primary category, year and the canonical
abs page URL are preserved in ``metadata`` so downstream agents (summarizer,
verifier, citation formatter) can produce real bibliographic references.
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import httpx

from core.knowledge import KnowledgeChunk, KnowledgeQuery

if TYPE_CHECKING:
    from config import Configuration

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_TIMEOUT = httpx.Timeout(15.0)


class ArxivSource:
    """A :class:`~core.knowledge.KnowledgeSource` backed by arXiv's API."""

    name = "arxiv"

    def __init__(self, config: "Configuration") -> None:
        self._config = config
        # Polite UA so arXiv can contact us if our traffic ever misbehaves.
        contact = (config.openalex_email or "").strip() or "anonymous"
        self._user_agent = (
            f"helloagents-deepresearch/0.1 (mailto:{contact})"
        )

    @property
    def is_local(self) -> bool:
        """arXiv lives on the public internet."""
        return False

    async def query(self, q: KnowledgeQuery) -> list[KnowledgeChunk]:
        """Search arXiv for papers matching ``q`` and return paper chunks."""
        params = {
            "search_query": f"all:{q.text}",
            "start": "0",
            "max_results": str(max(1, q.max_results)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT, headers={"User-Agent": self._user_agent}
            ) as client:
                resp = await client.get(_ARXIV_API, params=params)
                resp.raise_for_status()
                xml_text = resp.text
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            logger.warning("ArxivSource: API call failed: %s", exc)
            return []

        try:
            return self._parse_atom(xml_text)
        except ET.ParseError as exc:
            logger.warning("ArxivSource: Atom parse failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _parse_atom(self, xml_text: str) -> list[KnowledgeChunk]:
        root = ET.fromstring(xml_text)
        chunks: list[KnowledgeChunk] = []

        for entry in root.findall("atom:entry", _NS):
            title = _text(entry, "atom:title").strip().replace("\n", " ")
            summary = _text(entry, "atom:summary").strip()
            entry_id = _text(entry, "atom:id").strip()  # canonical abs URL
            published = _text(entry, "atom:published").strip()  # e.g. 2024-03-12T...
            year = published[:4] if len(published) >= 4 else ""

            authors = [
                _text(a, "atom:name").strip()
                for a in entry.findall("atom:author", _NS)
            ]
            authors = [a for a in authors if a]

            # arXiv tags every entry with a primary category like "cs.CL"
            primary = entry.find("arxiv:primary_category", _NS)
            primary_cat = primary.get("term", "") if primary is not None else ""

            # PDF link is the <link title="pdf" ...> sibling — useful later for #2.
            pdf_url = ""
            for link in entry.findall("atom:link", _NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            if not (title and summary):
                continue

            # Compose a one-line author block ("Smith, J.; Lee, A.; ...") so the
            # summarizer prompt has bibliographic context, not just the abstract.
            author_block = "; ".join(authors[:6])
            if len(authors) > 6:
                author_block += " et al."

            header_lines = [f"Title: {title}"]
            if author_block:
                header_lines.append(f"Authors: {author_block}")
            if year:
                header_lines.append(f"Year: {year}")
            if primary_cat:
                header_lines.append(f"Category: {primary_cat}")
            content = "\n".join(header_lines) + "\n\nAbstract:\n" + summary

            chunks.append(
                KnowledgeChunk(
                    source="arxiv",
                    title=title,
                    url_or_path=entry_id,
                    content=content,
                    metadata={
                        "authors": authors,
                        "year": year,
                        "published": published,
                        "primary_category": primary_cat,
                        "pdf_url": pdf_url,
                        "kind": "academic",
                    },
                )
            )

        logger.info("ArxivSource: returned %d chunks", len(chunks))
        return chunks


def _text(node: ET.Element, path: str) -> str:
    """Look up a single child element's text; return '' when missing."""
    el = node.find(path, _NS)
    return (el.text or "") if el is not None else ""
