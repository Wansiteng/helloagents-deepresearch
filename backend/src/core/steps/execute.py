"""Execute step — run search + summarization for each TODO item.

PR-2 minimal flow: tasks run sequentially and summaries are non-streaming.
Parallel execution and token-level streaming are deferred (see plan).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from models import SummaryState, TodoItem
from services.factory import ResearchServices
from services.search import dispatch_search_with_retry, prepare_research_context

logger = logging.getLogger(__name__)


class ExecuteStep:
    """Searches and summarises each task, mutating it in place."""

    def __init__(self, services: ResearchServices) -> None:
        """Store the service bundle."""
        self._services = services
        self._config = services.config

    async def run(self, state: SummaryState) -> AsyncIterator[dict[str, Any]]:
        """Execute every task in ``state.todo_items`` sequentially."""
        for index, task in enumerate(state.todo_items, start=1):
            async for event in self._run_task(state, task, index):
                yield event

    async def _run_task(
        self, state: SummaryState, task: TodoItem, step: int
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one task; never raises — failures become a ``failed`` event."""
        task.status = "in_progress"
        yield {
            "type": "task_status",
            "task_id": task.id,
            "status": "in_progress",
            "title": task.title,
            "intent": task.intent,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "step": step,
        }

        try:
            fallback_queries: list[str] = []
            if self._config.search_retry_on_empty:
                fallback_queries = [
                    f"{task.title} {state.research_topic}",
                    task.title,
                ]

            search_result, notices, answer_text, backend = await asyncio.to_thread(
                dispatch_search_with_retry,
                task.query,
                self._config,
                0,
                fallback_queries,
            )
            task.notices = notices
            for notice in notices:
                if notice:
                    yield {"type": "status", "message": notice, "step": step}

            if not search_result or not search_result.get("results"):
                task.status = "skipped"
                yield {
                    "type": "task_status",
                    "task_id": task.id,
                    "status": "skipped",
                    "title": task.title,
                    "intent": task.intent,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                    "step": step,
                }
                return

            sources_summary, context = prepare_research_context(
                search_result, answer_text, self._config
            )
            task.sources_summary = sources_summary
            state.web_research_results.append(context)
            state.sources_gathered.append(sources_summary)
            state.research_loop_count += 1

            yield {
                "type": "sources",
                "task_id": task.id,
                "latest_sources": sources_summary,
                "raw_context": context,
                "step": step,
                "backend": backend,
                "note_id": task.note_id,
                "note_path": task.note_path,
            }

            summary = await asyncio.to_thread(
                self._services.summarizer.summarize_task, state, task, context
            )
            task.summary = summary.strip() if summary else "暂无可用信息"

            yield {
                "type": "task_summary_chunk",
                "task_id": task.id,
                "content": task.summary,
                "note_id": task.note_id,
                "step": step,
            }

            task.status = "completed"
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "completed",
                "summary": task.summary,
                "sources_summary": task.sources_summary,
                "note_id": task.note_id,
                "note_path": task.note_path,
                "step": step,
            }

        except Exception as exc:  # noqa: BLE001 — isolate per-task failure
            logger.exception("Task %d execution failed", task.id)
            task.status = "failed"
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "failed",
                "detail": str(exc),
                "title": task.title,
                "intent": task.intent,
                "note_id": task.note_id,
                "note_path": task.note_path,
                "step": step,
            }
