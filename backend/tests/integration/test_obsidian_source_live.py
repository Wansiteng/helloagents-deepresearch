"""Live smoke test for ObsidianVaultSource against a real Ollama embedder.

Marked ``integration`` so the default ``pytest`` run skips it. Opt in with::

    pytest -m integration

Builds a tiny throwaway vault, indexes it with real embeddings, and checks
semantic recall. Auto-skips when Ollama is unreachable.
"""
from __future__ import annotations

import json
import urllib.request

import pytest

from core.knowledge import KnowledgeQuery
from core.sources.obsidian import ObsidianVaultSource
from services.vector_store import VectorStore

pytestmark = pytest.mark.integration

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def _ollama_has_embed_model() -> bool:
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL}/api/tags", timeout=5
        ) as response:
            payload = json.loads(response.read())
    except Exception:
        return False
    names = [m.get("name", "") for m in payload.get("models", [])]
    return any(EMBED_MODEL in name for name in names)


@pytest.fixture
def live_vault(tmp_path):
    if not _ollama_has_embed_model():
        pytest.skip(f"Ollama / '{EMBED_MODEL}' not available at {OLLAMA_BASE_URL}")

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "asyncio.md").write_text(
        "Python 的 asyncio 提供事件循环与协程，用于编写并发 I/O 代码。",
        encoding="utf-8",
    )
    (vault / "cooking.md").write_text(
        "番茄炒蛋：先炒蛋，盛出，再炒番茄，最后混合调味。",
        encoding="utf-8",
    )
    return vault, tmp_path / "idx"


async def test_obsidian_semantic_recall(live_vault) -> None:
    """Indexing a vault and querying it returns the semantically relevant note."""
    vault, index_ws = live_vault
    store = VectorStore(
        workspace=str(index_ws),
        embedding_model=EMBED_MODEL,
        ollama_base_url=OLLAMA_BASE_URL,
        collection_name="obsidian_vault_test",
    )
    source = ObsidianVaultSource(str(vault), store, str(index_ws))

    chunks = await source.query(
        KnowledgeQuery(text="Python 并发编程 协程", max_results=2)
    )

    assert chunks, "expected at least one recalled chunk"
    # The asyncio note should rank above the cooking note.
    assert "asyncio" in chunks[0].url_or_path
    assert chunks[0].source == "obsidian"
