"""Unit tests for core.session.ResearchSession — fake services, no network."""
from __future__ import annotations

import pytest

from config import Configuration
from core.session import ResearchSession
from core.steps import execute as execute_step
from models import SummaryState, TodoItem
from services.factory import ResearchServices


# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────
class _FakePlanner:
    def __init__(self, tasks):
        self._tasks = tasks

    def plan_todo_list(self, state: SummaryState):
        return list(self._tasks)

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        return TodoItem(id=1, title="fallback", intent="i", query="q")


class _FakeSummarizer:
    def summarize_task(self, state, task, context) -> str:
        return f"summary for {task.title}"


class _FakeWriter:
    def generate_report(self, state: SummaryState) -> str:
        return "# Final Report\n\nsynthesised content"


def _services(config, tasks) -> ResearchServices:
    return ResearchServices(
        config=config,
        llm=None,
        tool_registry=None,
        vector_store=None,
        tool_tracker=None,
        planner=_FakePlanner(tasks),
        summarizer=_FakeSummarizer(),
        writer=_FakeWriter(),
        critic=None,
        llm_concurrency=1,
    )


def _two_tasks() -> list[TodoItem]:
    return [
        TodoItem(id=1, title="Task A", intent="ia", query="qa"),
        TodoItem(id=2, title="Task B", intent="ib", query="qb"),
    ]


@pytest.fixture
def patched_search(monkeypatch):
    """Patch the search helpers used by ExecuteStep with canned data."""

    def fake_dispatch(query, config, loop_count, fallback_queries=None):
        payload = {
            "results": [{"title": "T", "url": "https://x", "content": "c"}],
            "backend": "duckduckgo",
        }
        return payload, [], None, "duckduckgo"

    def fake_prepare(search_result, answer_text, config):
        return "sources summary", "context text"

    monkeypatch.setattr(execute_step, "dispatch_search_with_retry", fake_dispatch)
    monkeypatch.setattr(execute_step, "prepare_research_context", fake_prepare)


async def _collect(session: ResearchSession) -> list[dict]:
    return [event async for event in session.run()]


# ──────────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────────
async def test_run_emits_expected_event_sequence(patched_search):
    config = Configuration()
    session = ResearchSession("test topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    types = [e["type"] for e in events]

    assert types[0] == "status"
    assert types[1] == "status"           # PlanStep "拆解任务"
    assert types[2] == "todo_list"
    assert types[-1] == "done"
    assert types[-2] == "final_report"

    # Each of the 2 tasks goes in_progress -> sources -> summary chunk -> completed
    assert types.count("sources") == 2
    assert types.count("task_summary_chunk") == 2
    statuses = [e["status"] for e in events if e["type"] == "task_status"]
    assert statuses == ["in_progress", "completed", "in_progress", "completed"]


async def test_todo_list_event_shape(patched_search):
    config = Configuration()
    session = ResearchSession("topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    todo = next(e for e in events if e["type"] == "todo_list")

    assert len(todo["tasks"]) == 2
    assert todo["step"] == 0
    keys = set(todo["tasks"][0])
    assert {"id", "title", "intent", "query", "status", "stream_token"} <= keys


async def test_final_report_carries_report(patched_search):
    config = Configuration()
    session = ResearchSession("topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    final = next(e for e in events if e["type"] == "final_report")

    assert final["report"].startswith("# Final Report")
    assert session.state.structured_report == final["report"]


async def test_summary_chunk_carries_task_summary(patched_search):
    config = Configuration()
    session = ResearchSession("topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    chunks = [e for e in events if e["type"] == "task_summary_chunk"]

    assert chunks[0]["content"] == "summary for Task A"
    assert chunks[0]["task_id"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Skipped path — search returns no results
# ──────────────────────────────────────────────────────────────────────────────
async def test_task_skipped_on_empty_search(monkeypatch):
    monkeypatch.setattr(
        execute_step,
        "dispatch_search_with_retry",
        lambda *a, **k: (None, [], None, "duckduckgo"),
    )
    config = Configuration()
    session = ResearchSession("topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    statuses = [e["status"] for e in events if e["type"] == "task_status"]

    assert statuses == ["in_progress", "skipped", "in_progress", "skipped"]
    assert [e["type"] for e in events][-1] == "done"


# ──────────────────────────────────────────────────────────────────────────────
# Fallback — planner returns no tasks
# ──────────────────────────────────────────────────────────────────────────────
async def test_fallback_task_when_planner_empty(patched_search):
    config = Configuration()
    session = ResearchSession("topic", config, services=_services(config, []))
    events = await _collect(session)
    todo = next(e for e in events if e["type"] == "todo_list")

    assert len(todo["tasks"]) == 1
    assert todo["tasks"][0]["title"] == "fallback"


# ──────────────────────────────────────────────────────────────────────────────
# Error path — a service raises
# ──────────────────────────────────────────────────────────────────────────────
async def test_run_emits_error_event_on_failure():
    class _BoomPlanner:
        def plan_todo_list(self, state):
            raise RuntimeError("planner exploded")

    config = Configuration()
    services = _services(config, [])
    services.planner = _BoomPlanner()
    session = ResearchSession("topic", config, services=services)
    events = await _collect(session)

    assert events[-1]["type"] == "error"
    assert "planner exploded" in events[-1]["detail"]
