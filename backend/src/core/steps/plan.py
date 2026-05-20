"""Plan step — decompose the research topic into a TODO list."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from models import SummaryState, TodoItem
from services.factory import ResearchServices

logger = logging.getLogger(__name__)


def serialize_task(task: TodoItem) -> dict[str, Any]:
    """Serialise a :class:`TodoItem` for the frontend (matches legacy shape)."""
    return {
        "id": task.id,
        "title": task.title,
        "intent": task.intent,
        "query": task.query,
        "status": task.status,
        "summary": task.summary,
        "sources_summary": task.sources_summary,
        "note_id": task.note_id,
        "note_path": task.note_path,
        "stream_token": task.stream_token,
        "section_draft": task.section_draft,
    }


class PlanStep:
    """Decomposes the topic into TODO items via the planner service."""

    def __init__(self, services: ResearchServices) -> None:
        """Store the service bundle."""
        self._services = services

    async def run(self, state: SummaryState) -> AsyncIterator[dict[str, Any]]:
        """Run planning, populate ``state.todo_items``, yield a ``todo_list`` event."""
        yield {"type": "status", "message": "正在拆解研究任务..."}

        todo_items = await asyncio.to_thread(
            self._services.planner.plan_todo_list, state
        )
        if not todo_items:
            logger.info("Planner produced no tasks; using multi-angle fallback")
            todo_items = self._services.planner.create_fallback_tasks(state)

        for index, task in enumerate(todo_items, start=1):
            task.stream_token = f"task_{task.id}"

        state.todo_items = todo_items

        yield {
            "type": "todo_list",
            "tasks": [serialize_task(t) for t in todo_items],
            "step": 0,
        }
