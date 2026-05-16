"""Vector-memory knowledge source.

Wraps the existing synchronous :class:`services.vector_store.VectorStore`
(ChromaDB + local Ollama embeddings) behind the async
:class:`~core.knowledge.KnowledgeSource` interface, exposing past research as a
retrievable source.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from core.knowledge import KnowledgeChunk, KnowledgeQuery
from services.vector_store import VectorStore

if TYPE_CHECKING:
    from config import Configuration

logger = logging.getLogger(__name__)


class VectorMemorySource:
    """A :class:`~core.knowledge.KnowledgeSource` backed by the vector store."""

    name = "vector_memory"

    def __init__(self, vector_store: VectorStore) -> None:
        """Build the source.

        Args:
            vector_store: An already-constructed :class:`VectorStore` instance.
        """
        self._vector_store = vector_store

    @classmethod
    def from_config(cls, config: Configuration) -> VectorMemorySource:
        """Build the source by constructing a :class:`VectorStore` from config.

        Mirrors the ``VectorStore`` construction in ``agent.py``. Use this when
        you do not already hold a store instance.
        """
        store = VectorStore(
            workspace=config.vector_store_path,
            embedding_model=config.embedding_model,
            ollama_base_url=config.ollama_base_url,
            chunk_size=config.vector_chunk_size,
            chunk_overlap=config.vector_chunk_overlap,
        )
        return cls(store)

    @property
    def is_local(self) -> bool:
        """Vector memory (ChromaDB + local embeddings) never leaves the machine."""
        return True

    async def query(self, q: KnowledgeQuery) -> list[KnowledgeChunk]:
        """Run a semantic search and normalise hits to :class:`KnowledgeChunk`."""
        hits = await asyncio.to_thread(
            self._vector_store.query,
            q.text,
            q.max_results,
        )

        chunks: list[KnowledgeChunk] = []
        for hit in hits:
            metadata = hit.get("metadata", {}) or {}
            distance = hit.get("distance")
            # Cosine distance (0 = identical) -> similarity score (higher = better).
            score = 1.0 - distance if isinstance(distance, (int, float)) else None
            chunks.append(
                KnowledgeChunk(
                    source="vector_memory",
                    title=metadata.get("task_title", "") or metadata.get("topic", ""),
                    url_or_path=metadata.get("doc_id", ""),
                    content=hit.get("text", ""),
                    metadata=metadata,
                    score=score,
                )
            )
        return chunks
