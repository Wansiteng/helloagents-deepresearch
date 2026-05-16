"""Live smoke test for core.llm against a real Ollama server.

Marked ``integration`` so the default ``pytest`` run skips it. Opt in with::

    pytest -m integration

Auto-skips when Ollama is unreachable or has no model loaded.
"""
from __future__ import annotations

import json
import urllib.request

import pytest

from core.llm import LLMMessage, OpenAICompatibleClient

pytestmark = pytest.mark.integration

OLLAMA_BASE_URL = "http://localhost:11434"


def _first_ollama_model() -> str | None:
    """Return any loaded Ollama model name, or None if unreachable/empty."""
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL}/api/tags", timeout=5
        ) as response:
            payload = json.loads(response.read())
    except Exception:
        return None
    models = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
    return models[0] if models else None


@pytest.fixture(scope="module")
def live_model() -> str:
    model = _first_ollama_model()
    if model is None:
        pytest.skip(f"Ollama not reachable / no model loaded at {OLLAMA_BASE_URL}")
    return model


async def test_chat_round_trip(live_model: str) -> None:
    """A real non-streaming chat() call returns an assistant message."""
    client = OpenAICompatibleClient(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        model=live_model,
        timeout=120,
    )
    reply = await client.chat(
        [LLMMessage(role="user", content="Reply with the single word: OK")]
    )
    assert reply.role == "assistant"
    assert reply.content.strip(), "expected non-empty content"


async def test_chat_stream_round_trip(live_model: str) -> None:
    """A real streaming chat_stream() call yields at least one content delta."""
    client = OpenAICompatibleClient(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        model=live_model,
        timeout=120,
    )
    received = 0
    async for piece in client.chat_stream(
        [LLMMessage(role="user", content="Count: 1 2 3")]
    ):
        assert isinstance(piece, str)
        received += 1
        if received >= 3:
            break
    assert received >= 1
