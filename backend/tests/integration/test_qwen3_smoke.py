"""Qwen3 / local-LLM smoke tests.

These tests require a running Ollama instance with a Qwen3.5-family model and
will hit the network (LLM inference). They are marked ``integration`` so the
default ``pytest`` invocation skips them; opt-in with::

    pytest -m integration
"""
from __future__ import annotations

import json
import time
import urllib.request

import pytest

pytestmark = pytest.mark.integration

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_HINT = "qwen3.5"


def _ollama_available_models() -> list[str]:
    """Return loaded Ollama model names, or [] if Ollama isn't reachable."""
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL}/api/tags", timeout=5
        ) as response:
            payload = json.loads(response.read())
    except Exception:
        return []
    return [m.get("name", "") for m in payload.get("models", [])]


@pytest.fixture(scope="module")
def qwen_model_name() -> str:
    """Skip the suite if no Qwen3.5 model is loaded into Ollama."""
    models = _ollama_available_models()
    if not models:
        pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

    qwen_models = [m for m in models if OLLAMA_MODEL_HINT in m.lower()]
    if not qwen_models:
        pytest.skip(
            f"No '{OLLAMA_MODEL_HINT}*' model loaded in Ollama "
            f"(available: {models})"
        )
    return qwen_models[0]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Streaming inference reaches first tokens within a reasonable budget.
# ──────────────────────────────────────────────────────────────────────────────
def test_ollama_streaming_responds(qwen_model_name: str) -> None:
    """Hit Ollama's OpenAI-compatible streaming endpoint and read >=3 tokens.

    Qwen3.5 chain-of-thought can run 2000+ tokens; waiting for full completion
    would routinely exceed test timeouts. We declare success once the stream
    proves the model is producing tokens (content or reasoning deltas).
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        timeout=30,
    )

    received = 0
    first_token = ""
    stream = client.chat.completions.create(
        model=qwen_model_name,
        messages=[{"role": "user", "content": "你好"}],
        stream=True,
    )
    try:
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            piece = (delta.content or "") or (getattr(delta, "reasoning", "") or "")
            if piece:
                if not first_token:
                    first_token = piece
                received += 1
                if received >= 3:
                    break
    finally:
        stream.close()

    assert received >= 1, "Stream produced no tokens"
    assert first_token, "Stream produced no first token"


# ──────────────────────────────────────────────────────────────────────────────
# 2. strip_thinking_tokens handles every tag variant we've seen in the wild.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("description", "raw", "forbidden", "expected"),
    [
        ("standard <think>", "<think>thinking</think>answer", "<think>", "answer"),
        (
            "<|think|> closing form A",
            "<|think|>cot</|think|>output A",
            "<|think|>",
            "output A",
        ),
        (
            "<|think|> closing form B",
            "<|think|>cot<|/think|>output B",
            "<|think|>",
            "output B",
        ),
        (
            "unclosed <think> truncates",
            "prefix<think>tail-was-cut",
            "<think>",
            None,
        ),
        ("no tags pass through", "plain answer", None, "plain answer"),
        (
            "trailing newline preserved",
            "<think>cot</think>\nfinal",
            "<think>",
            "final",
        ),
    ],
)
def test_strip_thinking_tokens(
    description: str, raw: str, forbidden: str | None, expected: str | None
) -> None:
    from utils import strip_thinking_tokens

    out = strip_thinking_tokens(raw)
    if forbidden is not None:
        assert forbidden not in out, f"{description}: forbidden token leaked: {out!r}"
    if expected is not None:
        assert expected in out, f"{description}: expected text missing: {out!r}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. DeepResearchAgent constructs without errors under the live config.
# ──────────────────────────────────────────────────────────────────────────────
def test_agent_initialises() -> None:
    from agent import DeepResearchAgent
    from config import Configuration

    config = Configuration.from_env()
    agent = DeepResearchAgent(config=config)

    assert agent.llm is not None
    assert agent.planner is not None
    assert agent.summarizer is not None
    assert agent.writer is not None


# ──────────────────────────────────────────────────────────────────────────────
# 4. PlannerAgent decomposes a topic into at least one TodoItem.
# ──────────────────────────────────────────────────────────────────────────────
def test_planner_produces_todo_items() -> None:
    from agent import DeepResearchAgent
    from config import Configuration
    from models import SummaryState

    config = Configuration.from_env().model_copy(update={"max_web_research_loops": 1})
    agent = DeepResearchAgent(config=config)

    state = SummaryState(research_topic="Python 异步编程 asyncio 最佳实践")
    started = time.time()
    todos = agent.planner.plan_todo_list(state)
    elapsed = time.time() - started

    assert len(todos) >= 1, f"PlannerAgent returned no tasks (took {elapsed:.1f}s)"
    for todo in todos:
        assert todo.title, "TodoItem missing title"
        assert todo.query, "TodoItem missing query"
