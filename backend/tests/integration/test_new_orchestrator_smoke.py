"""Live smoke test for the new ResearchSession orchestrator.

Marked ``integration`` so the default ``pytest`` run skips it. Opt in with::

    pytest -m integration

Runs a real research flow end to end; requires a running Ollama with a model
loaded and network access for web search. Auto-skips when Ollama is unreachable.
"""
from __future__ import annotations

import json
import urllib.request

import pytest

from config import Configuration
from core.session import ResearchSession

pytestmark = pytest.mark.integration

OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL}/api/tags", timeout=5
        ) as response:
            payload = json.loads(response.read())
    except Exception:
        return False
    return bool(payload.get("models"))


@pytest.fixture(scope="module")
def live_config() -> Configuration:
    if not _ollama_reachable():
        pytest.skip(f"Ollama not reachable / no model at {OLLAMA_BASE_URL}")
    # Keep the run small: one research loop.
    return Configuration.from_env().model_copy(
        update={"max_web_research_loops": 1, "enable_reflection": False}
    )


async def test_research_session_end_to_end(live_config: Configuration) -> None:
    """A real ResearchSession.run() reaches a final report and done event."""
    session = ResearchSession("Python asyncio 简介", live_config)

    types: list[str] = []
    final_report: str | None = None
    async for event in session.run():
        types.append(event["type"])
        if event["type"] == "final_report":
            final_report = event["report"]
        if event["type"] == "error":
            pytest.fail(f"ResearchSession emitted error: {event.get('detail')}")

    assert "todo_list" in types
    assert "final_report" in types
    assert types[-1] == "done"
    assert final_report and final_report.strip()
