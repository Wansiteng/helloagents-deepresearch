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
