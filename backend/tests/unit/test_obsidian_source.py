"""Unit tests for core.sources.obsidian — fake VectorStore, no network."""
from __future__ import annotations

import pytest

from core.knowledge import KnowledgeQuery, KnowledgeSource
from core.sources.obsidian import ObsidianVaultSource, _strip_frontmatter


class _FakeVectorStore:
    """Records add_document calls; returns canned query hits."""

    def __init__(self, hits=None):
        self._hits = hits or []
        self.added: list[dict] = []

    def add_document(self, text, metadata=None, doc_id=None):
        self.added.append({"text": text, "metadata": metadata, "doc_id": doc_id})
        return 1

    def query(self, query_text, n_results=5, filter_metadata=None):
        return self._hits


def _make_vault(tmp_path):
    """Create a small vault: two plain notes + one with YAML frontmatter."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note_a.md").write_text("Note A content about asyncio.", encoding="utf-8")
    sub = vault / "sub"
    sub.mkdir()
    (sub / "note_b.md").write_text("Note B content about typing.", encoding="utf-8")
    (vault / "with_fm.md").write_text(
        "---\ntags: [x]\ntitle: FM\n---\nReal body after frontmatter.",
        encoding="utf-8",
    )
    return vault


# ──────────────────────────────────────────────────────────────────────────────
# Frontmatter stripping
# ──────────────────────────────────────────────────────────────────────────────
def test_strip_frontmatter_removes_yaml_block():
    raw = "---\ntags: [a]\n---\nbody text"
    assert _strip_frontmatter(raw) == "body text"


def test_strip_frontmatter_noop_without_block():
    assert _strip_frontmatter("just body") == "just body"


# ──────────────────────────────────────────────────────────────────────────────
# Indexing
# ──────────────────────────────────────────────────────────────────────────────
async def test_indexes_every_markdown_file(tmp_path):
    vault = _make_vault(tmp_path)
    store = _FakeVectorStore()
    source = ObsidianVaultSource(str(vault), store, str(tmp_path / "idx"))

    await source.query(KnowledgeQuery(text="anything"))

    assert len(store.added) == 3  # all three .md files indexed
    doc_ids = {entry["doc_id"] for entry in store.added}
    assert doc_ids == {"note_a.md", "sub/note_b.md", "with_fm.md"}


async def test_frontmatter_stripped_before_indexing(tmp_path):
    vault = _make_vault(tmp_path)
    store = _FakeVectorStore()
    source = ObsidianVaultSource(str(vault), store, str(tmp_path / "idx"))

    await source.query(KnowledgeQuery(text="anything"))

    fm_entry = next(e for e in store.added if e["doc_id"] == "with_fm.md")
    assert fm_entry["text"] == "Real body after frontmatter."
    assert "tags:" not in fm_entry["text"]


async def test_mtime_cache_skips_unchanged_files(tmp_path):
    vault = _make_vault(tmp_path)
    index_ws = str(tmp_path / "idx")

    # First source indexes everything and writes the manifest.
    store1 = _FakeVectorStore()
    await ObsidianVaultSource(str(vault), store1, index_ws).query(
        KnowledgeQuery(text="q")
    )
    assert len(store1.added) == 3

    # A fresh source over the same (unchanged) vault re-indexes nothing.
    store2 = _FakeVectorStore()
    await ObsidianVaultSource(str(vault), store2, index_ws).query(
        KnowledgeQuery(text="q")
    )
    assert store2.added == []


async def test_missing_vault_indexes_nothing(tmp_path):
    store = _FakeVectorStore()
    source = ObsidianVaultSource(
        str(tmp_path / "does_not_exist"), store, str(tmp_path / "idx")
    )
    chunks = await source.query(KnowledgeQuery(text="q"))
    assert store.added == []
    assert chunks == []


# ──────────────────────────────────────────────────────────────────────────────
# Query mapping
# ──────────────────────────────────────────────────────────────────────────────
async def test_query_maps_hits_to_knowledge_chunks(tmp_path):
    vault = _make_vault(tmp_path)
    hits = [
        {
            "text": "Note A content about asyncio.",
            "metadata": {"path": "note_a.md", "title": "note_a"},
            "distance": 0.25,
        }
    ]
    source = ObsidianVaultSource(
        str(vault), _FakeVectorStore(hits), str(tmp_path / "idx")
    )
    chunks = await source.query(KnowledgeQuery(text="asyncio"))

    assert len(chunks) == 1
    assert chunks[0].source == "obsidian"
    assert chunks[0].title == "note_a"
    assert chunks[0].url_or_path == "note_a.md"
    assert chunks[0].score == pytest.approx(0.75)
    assert source.is_local is True
    assert source.name == "obsidian"


def test_obsidian_source_satisfies_protocol(tmp_path):
    source = ObsidianVaultSource(
        str(tmp_path), _FakeVectorStore(), str(tmp_path / "idx")
    )
    assert isinstance(source, KnowledgeSource)
