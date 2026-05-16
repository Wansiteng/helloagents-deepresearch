"""Unit tests for core.session.ResearchSession — fake services, no network."""
from __future__ import annotations

from config import Configuration
from core.knowledge import KnowledgeChunk
from core.session import ResearchSession
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


class _FakeSource:
    """A minimal KnowledgeSource for tests."""

    def __init__(self, name, chunks, *, is_local=False, fail=False):
        self.name = name
        self._chunks = chunks
        self._is_local = is_local
        self._fail = fail

    @property
    def is_local(self) -> bool:
        return self._is_local

    async def query(self, q):
        if self._fail:
            raise RuntimeError(f"source {self.name} boom")
        return list(self._chunks)


def _chunk(source="web", title="T", content="c"):
    return KnowledgeChunk(
        source=source, title=title, url_or_path="u", content=content
    )


def _services(config, tasks, sources=None) -> ResearchServices:
    if sources is None:
        sources = [_FakeSource("web", [_chunk()])]
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
        knowledge_sources=sources,
    )


def _two_tasks() -> list[TodoItem]:
    return [
        TodoItem(id=1, title="Task A", intent="ia", query="qa"),
        TodoItem(id=2, title="Task B", intent="ib", query="qb"),
    ]


async def _collect(session: ResearchSession) -> list[dict]:
    return [event async for event in session.run()]


# ──────────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────────
async def test_run_emits_expected_event_sequence():
    config = Configuration()
    session = ResearchSession("test topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    types = [e["type"] for e in events]

    assert types[0] == "status"
    assert types[1] == "status"           # PlanStep "拆解任务"
    assert types[2] == "todo_list"
    assert types[-1] == "done"
    assert types[-2] == "final_report"

    assert types.count("sources") == 2
    assert types.count("task_summary_chunk") == 2
    statuses = [e["status"] for e in events if e["type"] == "task_status"]
    assert statuses == ["in_progress", "completed", "in_progress", "completed"]


async def test_todo_list_event_shape():
    config = Configuration()
    session = ResearchSession("topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    todo = next(e for e in events if e["type"] == "todo_list")

    assert len(todo["tasks"]) == 2
    assert todo["step"] == 0
    keys = set(todo["tasks"][0])
    assert {"id", "title", "intent", "query", "status", "stream_token"} <= keys


async def test_final_report_carries_report():
    config = Configuration()
    session = ResearchSession("topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    final = next(e for e in events if e["type"] == "final_report")

    assert final["report"].startswith("# Final Report")
    assert session.state.structured_report == final["report"]


async def test_summary_chunk_carries_task_summary():
    config = Configuration()
    session = ResearchSession("topic", config, services=_services(config, _two_tasks()))
    events = await _collect(session)
    chunks = [e for e in events if e["type"] == "task_summary_chunk"]

    assert chunks[0]["content"] == "summary for Task A"
    assert chunks[0]["task_id"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Multi-source fan-out
# ──────────────────────────────────────────────────────────────────────────────
async def test_execute_step_merges_multiple_sources():
    config = Configuration()
    sources = [
        _FakeSource("web", [_chunk(source="web:duckduckgo", title="W")]),
        _FakeSource("obsidian", [_chunk(source="obsidian", title="O")], is_local=True),
    ]
    session = ResearchSession(
        "topic", config, services=_services(config, _two_tasks(), sources)
    )
    events = await _collect(session)
    sources_events = [e for e in events if e["type"] == "sources"]

    # backend field aggregates both source labels.
    assert "web:duckduckgo" in sources_events[0]["backend"]
    assert "obsidian" in sources_events[0]["backend"]
    # Both sources' chunks appear in the merged context.
    assert "W" in sources_events[0]["raw_context"]
    assert "O" in sources_events[0]["raw_context"]


async def test_one_failing_source_does_not_abort_task():
    config = Configuration()
    sources = [
        _FakeSource("web", [], fail=True),
        _FakeSource("obsidian", [_chunk(source="obsidian", title="O")], is_local=True),
    ]
    session = ResearchSession(
        "topic", config, services=_services(config, _two_tasks(), sources)
    )
    events = await _collect(session)
    statuses = [e["status"] for e in events if e["type"] == "task_status"]

    # The working source still completes the task.
    assert statuses == ["in_progress", "completed", "in_progress", "completed"]


# ──────────────────────────────────────────────────────────────────────────────
# Skipped path — every source returns nothing
# ──────────────────────────────────────────────────────────────────────────────
async def test_task_skipped_when_no_chunks():
    config = Configuration()
    sources = [_FakeSource("web", [])]
    session = ResearchSession(
        "topic", config, services=_services(config, _two_tasks(), sources)
    )
    events = await _collect(session)
    statuses = [e["status"] for e in events if e["type"] == "task_status"]

    assert statuses == ["in_progress", "skipped", "in_progress", "skipped"]
    assert [e["type"] for e in events][-1] == "done"


# ──────────────────────────────────────────────────────────────────────────────
# Fallback — planner returns no tasks
# ──────────────────────────────────────────────────────────────────────────────
async def test_fallback_task_when_planner_empty():
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
