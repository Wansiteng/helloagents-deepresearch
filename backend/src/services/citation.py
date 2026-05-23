"""Citation infrastructure for inline ``[^N]`` references + bibliography.

Two responsibilities, both small:

1. ``assign_cite_ids`` — give each :class:`KnowledgeChunk` a stable sequential
   ``cite_id`` ("1", "2", …) scoped to the entire research run. Already-assigned
   chunks keep their id; new ones pick up from the highest existing.
2. ``format_bibliography`` — render an IEEE-style markdown footnote section
   listing every chunk actually referenced in the final report. Academic
   chunks (arxiv / openalex) use real bibliographic fields; web chunks fall
   back to "Title — site (accessed …). URL".

The summarizer / reporter prompts (see :mod:`prompts`) instruct the agent to
emit ``[^N]`` after each factual claim, with ``N`` matching ``chunk.cite_id``.
The reporter post-processes the model output to attach this bibliography to
the bottom of the report.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Iterable

from core.knowledge import KnowledgeChunk

logger = logging.getLogger(__name__)

# Match [^N] footnote markers anywhere in a string.
_CITE_RE = re.compile(r"\[\^(\d+)\]")


def assign_cite_ids(
    chunks: list[KnowledgeChunk], existing: Iterable[KnowledgeChunk] = (),
) -> None:
    """Mutate ``chunks`` in place: every chunk without a ``cite_id`` gets one.

    ``existing`` is consulted to avoid colliding with ids already assigned in
    prior tasks within the same research run. Pass ``state.all_chunks``.

    Idempotent: chunks that already have a ``cite_id`` are left alone.
    """
    used = {int(c.cite_id) for c in existing if c.cite_id and c.cite_id.isdigit()}
    next_id = (max(used) + 1) if used else 1

    for chunk in chunks:
        if chunk.cite_id:
            continue
        chunk.cite_id = str(next_id)
        next_id += 1


def collect_used_ids(text: str) -> set[str]:
    """Return the set of ``[^N]`` markers actually present in ``text``."""
    return {match.group(1) for match in _CITE_RE.finditer(text or "")}


def format_bibliography(
    chunks: list[KnowledgeChunk],
    *,
    only_used: set[str] | None = None,
    style: str = "IEEE",
) -> str:
    """Render the chunks as a markdown footnote block — one ``[^N]: ...`` per
    line, sorted by numeric ``cite_id``.

    Args:
        chunks: All chunks gathered during the research run.
        only_used: If given, drop chunks whose ``cite_id`` is not in this set
            (i.e. the report body never referenced them). When ``None``, every
            chunk with a ``cite_id`` is rendered.
        style: ``"IEEE"`` is the only supported style for now; the slot lets
            future work add APA / Chicago without breaking callers.

    Returns:
        Either an empty string (no chunks to render) or a markdown block
        starting with the ``## 参考文献`` header.
    """
    if style != "IEEE":
        raise ValueError(f"Unsupported citation style: {style!r}")

    pairs: list[tuple[int, KnowledgeChunk]] = []
    for c in chunks:
        if not c.cite_id or not c.cite_id.isdigit():
            continue
        if only_used is not None and c.cite_id not in only_used:
            continue
        pairs.append((int(c.cite_id), c))

    if not pairs:
        return ""

    pairs.sort(key=lambda t: t[0])
    accessed = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # No explicit ``## 参考文献`` header here — the frontend's
    # ``marked-footnote`` extension auto-renders the section with our
    # configured heading. Emitting our own would duplicate the title.
    lines = [
        f"[^{cite_int}]: {_format_one(c, accessed)}" for cite_int, c in pairs
    ]
    return "\n".join(lines) + "\n"


def _format_one(chunk: KnowledgeChunk, accessed: str) -> str:
    """Render a single chunk in IEEE-ish form, choosing detail based on what
    metadata the source provided."""
    md = chunk.metadata or {}
    title = (chunk.title or "(无标题)").strip()
    url = (chunk.url_or_path or "").strip()
    source = chunk.source or ""
    kind = (md.get("kind") or "").strip()
    authors: list[str] = list(md.get("authors") or [])
    year = (md.get("year") or "").strip() if isinstance(md.get("year"), str) else str(md.get("year") or "")
    venue = (md.get("venue") or "").strip() if isinstance(md.get("venue"), str) else ""
    doi = (md.get("doi") or "").strip() if isinstance(md.get("doi"), str) else ""

    # Academic (arxiv / openalex) chunks get a real-looking citation.
    if kind == "academic" or source in {"arxiv", "openalex"}:
        author_part = _format_authors(authors) if authors else ""
        bits: list[str] = []
        if author_part:
            bits.append(author_part)
        bits.append(f'"{title},"')
        if venue:
            bits.append(f"_{venue}_,")
        elif source == "arxiv":
            cat = (md.get("primary_category") or "").strip()
            arxiv_id = url.rstrip("/").rsplit("/", 1)[-1]
            tag = f"arXiv:{arxiv_id}" + (f" [{cat}]" if cat else "")
            bits.append(f"_{tag}_,")
        if year:
            bits.append(f"{year}.")
        else:
            # Ensure the line ends cleanly even without year.
            bits[-1] = bits[-1].rstrip(",") + "."
        if doi:
            bits.append(f"doi: [{doi.replace('https://doi.org/', '')}]({doi}).")
        elif url:
            bits.append(f"[{url}]({url}).")
        return " ".join(bits)

    # Web / vault / vector — fall back to a simpler "title, site, accessed" line.
    domain = _domain_of(url) if url.startswith(("http://", "https://")) else source
    if url.startswith(("http://", "https://")):
        return f'"{title}," {domain}. [{url}]({url}) (accessed {accessed}).'
    if url:
        return f'"{title}," {source} — `{url}`.'
    return f'"{title}," {source}.'


def _format_authors(authors: list[str]) -> str:
    """Format an author list IEEE-style: ``J. Smith, A. Lee, et al.``"""
    if not authors:
        return ""
    formatted = [_initialize(a) for a in authors[:3] if a]
    if len(authors) > 3:
        return ", ".join(formatted) + ", et al.,"
    return ", ".join(formatted) + ","


def _initialize(name: str) -> str:
    """``"John Smith"`` → ``"J. Smith"``. Leaves CJK names alone."""
    name = name.strip()
    if not name:
        return ""
    # CJK heuristic: any ideographic char → return as-is
    if any("一" <= ch <= "鿿" for ch in name):
        return name
    parts = name.split()
    if len(parts) <= 1:
        return name
    return f"{parts[0][0]}. {parts[-1]}"


def _domain_of(url: str) -> str:
    """Extract a human-readable site label from a URL."""
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc
        if host.startswith("www."):
            host = host[4:]
        return host or url
    except Exception:
        return url
