"""Knowledge-source abstraction for the rebuilt research engine.

A :class:`KnowledgeSource` is anything the research engine can query for
information: web search, an Obsidian vault, a local PDF folder, a code repo,
the vector memory of past research. They all expose the same async ``query``
interface so the executor can fan out to every enabled source uniformly.

Concrete implementations live in :mod:`core.sources`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(kw_only=True)
class KnowledgeQuery:
    """A request issued to a knowledge source."""

    text: str
    intent: str = ""
    """The research intent behind the query — sources may use it to rewrite."""
    max_results: int = 5


@dataclass(kw_only=True)
class KnowledgeChunk:
    """A single retrieved piece of information, normalised across sources."""

    source: str
    """Origin label, e.g. ``web:duckduckgo`` or ``vector_memory``."""
    title: str
    url_or_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    """Relevance score; higher is more relevant. ``None`` when unranked."""
    cite_id: str | None = None
    """Stable per-research citation handle, e.g. ``"1"`` matches ``[^1]``
    markers in the summary / report. Assigned by ``services.citation`` once
    the chunk is admitted into the run."""


@runtime_checkable
class KnowledgeSource(Protocol):
    """A queryable source of research information."""

    name: str

    @property
    def is_local(self) -> bool:
        """Whether this source keeps all data on the local machine.

        Used to filter sources out when the user runs in fully-local mode.
        """
        ...

    async def query(self, q: KnowledgeQuery) -> list[KnowledgeChunk]:
        """Retrieve chunks relevant to ``q``."""
        ...


# Per-chunk content cap when assembling a prompt context, to keep the
# summariser prompt within the model's context window.
_CHUNK_CONTENT_LIMIT = 2000


def build_context(chunks: list[KnowledgeChunk]) -> tuple[str, str]:
    """Format merged knowledge chunks into ``(sources_summary, context)``.

    Args:
        chunks: Retrieved chunks from one or more knowledge sources.

    Returns:
        A pair ``(sources_summary, context)``:
        - ``sources_summary``: a bullet list of titles/origins for the
          ``sources`` SSE event.
        - ``context``: a prompt-ready block fed to the summariser.
    """
    if not chunks:
        return "暂无来源信息", ""

    summary_lines: list[str] = []
    context_blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.title or chunk.url_or_path or f"来源 {index}"
        cite = chunk.cite_id or str(index)
        summary_lines.append(
            f"* [^{cite}] [{chunk.source}] {title} : {chunk.url_or_path}"
        )

        content = chunk.content or ""
        if len(content) > _CHUNK_CONTENT_LIMIT:
            content = f"{content[:_CHUNK_CONTENT_LIMIT]}... [截断]"
        # CITE_ID line makes the citation handle the very first thing the
        # summarizer / writer reads — they must reuse it verbatim as [^N].
        context_blocks.append(
            f"CITE_ID: [^{cite}]\n"
            f"信息来源: {title}（{chunk.source}）\n"
            f"路径/URL: {chunk.url_or_path}\n"
            f"信息内容: {content}\n"
        )

    return "\n".join(summary_lines), "\n\n".join(context_blocks)
