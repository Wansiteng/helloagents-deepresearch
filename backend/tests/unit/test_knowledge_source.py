"""Unit tests for core.knowledge and core.sources — no network, no ChromaDB."""
from __future__ import annotations

import pytest

from config import Configuration
from core.knowledge import KnowledgeChunk, KnowledgeQuery, KnowledgeSource
from core.sources import web as web_module
from core.sources.vector_memory import VectorMemorySource
from core.sources.web import WebSearchSource


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
def test_knowledge_query_defaults():
    q = KnowledgeQuery(text="quantum computing")
    assert q.text == "quantum computing"
    assert q.intent == ""
    assert q.max_results == 5


def test_knowledge_chunk_construction():
    chunk = KnowledgeChunk(
        source="web:duckduckgo",
        title="T",
        url_or_path="https://x",
        content="body",
    )
    assert chunk.metadata == {}
    assert chunk.score is None


# ──────────────────────────────────────────────────────────────────────────────
# WebSearchSource
# ──────────────────────────────────────────────────────────────────────────────
async def test_web_search_source_maps_results(monkeypatch):
    def fake_dispatch(query, config, loop_count, fallback_queries=None):
        payload = {
            "results": [
                {"title": "Async in Python", "url": "https://a", "content": "c1"},
                {"title": "Asyncio guide", "url": "https://b", "content": "c2"},
                {"title": "No URL", "url": "", "content": "skipped"},
            ],
        }
        return payload, ["a notice"], None, "duckduckgo"

    monkeypatch.setattr(web_module, "dispatch_search_with_retry", fake_dispatch)

    source = WebSearchSource(Configuration())
    chunks = await source.query(KnowledgeQuery(text="asyncio"))

    assert len(chunks) == 2  # empty-URL item dropped
    assert chunks[0].source == "web:duckduckgo"
    assert chunks[0].title == "Async in Python"
    assert chunks[0].url_or_path == "https://a"
    assert chunks[0].content == "c1"
    assert source.is_local is False
    assert source.name == "web"


async def test_web_search_source_empty_result(monkeypatch):
    monkeypatch.setattr(
        web_module,
        "dispatch_search_with_retry",
        lambda *a, **k: (None, [], None, "duckduckgo"),
    )
    source = WebSearchSource(Configuration())
    assert await source.query(KnowledgeQuery(text="nothing")) == []


# ──────────────────────────────────────────────────────────────────────────────
# VectorMemorySource
# ──────────────────────────────────────────────────────────────────────────────
class _FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits
        self.last_query = None
        self.last_n = None

    def query(self, query_text, n_results=5, filter_metadata=None):
        self.last_query = query_text
        self.last_n = n_results
        return self._hits


async def test_vector_memory_source_maps_hits():
    hits = [
        {
            "text": "past research on entanglement",
            "metadata": {"task_title": "Quantum entanglement", "doc_id": "task_3"},
            "distance": 0.2,
        },
    ]
    source = VectorMemorySource(_FakeVectorStore(hits))
    chunks = await source.query(KnowledgeQuery(text="entanglement", max_results=3))

    assert len(chunks) == 1
    assert chunks[0].source == "vector_memory"
    assert chunks[0].title == "Quantum entanglement"
    assert chunks[0].url_or_path == "task_3"
    assert chunks[0].content == "past research on entanglement"
    assert chunks[0].score == pytest.approx(0.8)  # 1.0 - distance
    assert source.is_local is True
    assert source.name == "vector_memory"


async def test_vector_memory_source_passes_max_results():
    store = _FakeVectorStore([])
    source = VectorMemorySource(store)
    await source.query(KnowledgeQuery(text="q", max_results=7))
    assert store.last_n == 7
    assert store.last_query == "q"


async def test_vector_memory_source_handles_missing_distance():
    hits = [{"text": "t", "metadata": {}, "distance": None}]
    source = VectorMemorySource(_FakeVectorStore(hits))
    chunks = await source.query(KnowledgeQuery(text="q"))
    assert chunks[0].score is None
    assert chunks[0].title == ""


# ──────────────────────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────────────────────
def test_sources_satisfy_protocol():
    assert isinstance(WebSearchSource(Configuration()), KnowledgeSource)
    assert isinstance(VectorMemorySource(_FakeVectorStore([])), KnowledgeSource)
