"""Web search knowledge source.

Wraps the existing synchronous ``services.search`` helpers behind the async
:class:`~core.knowledge.KnowledgeSource` interface. The wrapper holds a legacy
:class:`Configuration` because ``dispatch_search_with_retry`` needs it — an
accepted transitional coupling until the new orchestrator owns config.
"""

from __future__ import annotations

import asyncio
import logging

from config import Configuration
from core.knowledge import KnowledgeChunk, KnowledgeQuery
from services.search import dispatch_search_with_retry

logger = logging.getLogger(__name__)


class WebSearchSource:
    """A :class:`~core.knowledge.KnowledgeSource` backed by web search."""

    name = "web"

    def __init__(self, config: Configuration) -> None:
        """Build the source.

        Args:
            config: Runtime configuration; selects the search backend and
                carries options ``services.search`` needs.
        """
        self._config = config

    @property
    def is_local(self) -> bool:
        """Web search always reaches the public internet."""
        return False

    async def query(self, q: KnowledgeQuery) -> list[KnowledgeChunk]:
        """Run a web search and normalise the hits to :class:`KnowledgeChunk`."""
        search_result, notices, _answer, backend = await asyncio.to_thread(
            dispatch_search_with_retry,
            q.text,
            self._config,
            0,
        )
        for notice in notices:
            logger.info("WebSearchSource notice: %s", notice)

        if not search_result:
            return []

        chunks: list[KnowledgeChunk] = []
        for item in search_result.get("results", []):
            url = item.get("url", "")
            if not url:
                continue
            chunks.append(
                KnowledgeChunk(
                    source=f"web:{backend}",
                    title=item.get("title", "") or url,
                    url_or_path=url,
                    content=item.get("content", ""),
                )
            )
        return chunks
