"""Obsidian vault knowledge source.

Indexes a local Obsidian vault into a dedicated Chroma collection (via the
existing :class:`services.vector_store.VectorStore`) and exposes it through the
async :class:`~core.knowledge.KnowledgeSource` interface — semantic search over
the user's private notes.

This is the first source that ChatGPT / Claude Deep Research structurally
cannot offer: the vault never leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from core.knowledge import KnowledgeChunk, KnowledgeQuery
from services.vector_store import VectorStore

if TYPE_CHECKING:
    from config import Configuration

logger = logging.getLogger(__name__)

_OBSIDIAN_COLLECTION = "obsidian_vault"
_MANIFEST_NAME = "manifest.json"
_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block, if present."""
    return _FRONTMATTER_RE.sub("", text, count=1)


class ObsidianVaultSource:
    """A :class:`~core.knowledge.KnowledgeSource` backed by an Obsidian vault."""

    name = "obsidian"

    def __init__(
        self,
        vault_path: str,
        vector_store: VectorStore,
        index_workspace: str,
    ) -> None:
        """Build the source.

        Args:
            vault_path: Path to the Obsidian vault directory.
            vector_store: A :class:`VectorStore` bound to a dedicated collection.
            index_workspace: Directory for the mtime manifest sidecar.
        """
        self._vault_path = Path(vault_path).expanduser()
        self._vector_store = vector_store
        self._manifest_path = Path(index_workspace).expanduser() / _MANIFEST_NAME
        self._indexed = False
        self._index_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: Configuration) -> ObsidianVaultSource:
        """Build the source and its dedicated vector store from config."""
        store = VectorStore(
            workspace=config.obsidian_index_path,
            embedding_model=config.embedding_model,
            ollama_base_url=config.ollama_base_url,
            chunk_size=config.vector_chunk_size,
            chunk_overlap=config.vector_chunk_overlap,
            collection_name=_OBSIDIAN_COLLECTION,
        )
        return cls(
            vault_path=config.obsidian_vault_path or "",
            vector_store=store,
            index_workspace=config.obsidian_index_path,
        )

    @property
    def is_local(self) -> bool:
        """An Obsidian vault is read entirely from the local filesystem."""
        return True

    async def query(self, q: KnowledgeQuery) -> list[KnowledgeChunk]:
        """Semantic-search the vault, indexing it first if needed."""
        await self._ensure_indexed()

        hits = await asyncio.to_thread(
            self._vector_store.query, q.text, q.max_results
        )
        chunks: list[KnowledgeChunk] = []
        for hit in hits:
            metadata = hit.get("metadata", {}) or {}
            distance = hit.get("distance")
            score = 1.0 - distance if isinstance(distance, (int, float)) else None
            chunks.append(
                KnowledgeChunk(
                    source="obsidian",
                    title=metadata.get("title", ""),
                    url_or_path=metadata.get("path", ""),
                    content=hit.get("text", ""),
                    metadata=metadata,
                    score=score,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    async def _ensure_indexed(self) -> None:
        """Index the vault once per process (incremental on mtime)."""
        if self._indexed:
            return
        async with self._index_lock:
            if self._indexed:
                return
            await asyncio.to_thread(self._reindex)
            self._indexed = True

    def _load_manifest(self) -> dict[str, float]:
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_manifest(self, manifest: dict[str, float]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _reindex(self) -> None:
        """Walk the vault and (re)embed new or modified Markdown files.

        Deleted files leave stale chunks behind — ``VectorStore`` has no delete
        API yet; this is a known limitation tracked for a later PR.
        """
        if not self._vault_path.is_dir():
            logger.warning("Obsidian vault path not found: %s", self._vault_path)
            return

        manifest = self._load_manifest()
        updated = dict(manifest)
        indexed_count = 0

        for md_file in sorted(self._vault_path.rglob("*.md")):
            rel_path = str(md_file.relative_to(self._vault_path))
            try:
                mtime = md_file.stat().st_mtime
            except OSError:
                continue

            if manifest.get(rel_path) == mtime:
                continue  # unchanged since last index

            try:
                raw = md_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("跳过无法读取的笔记 %s: %s", rel_path, exc)
                continue

            text = _strip_frontmatter(raw).strip()
            if not text:
                updated[rel_path] = mtime
                continue

            self._vector_store.add_document(
                text=text,
                metadata={"path": rel_path, "title": md_file.stem},
                doc_id=rel_path,
            )
            updated[rel_path] = mtime
            indexed_count += 1

        self._save_manifest(updated)
        logger.info(
            "Obsidian vault 索引完成: %s 个文件更新 (vault=%s)",
            indexed_count,
            self._vault_path,
        )
