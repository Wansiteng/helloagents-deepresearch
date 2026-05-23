"""Execute step — run multi-source retrieval + summarization for each TODO item.

PR-3: the search phase fans out across every enabled knowledge source
(web, Obsidian vault, ...) concurrently and merges the results.

PR-2 minimal-flow constraints still apply: tasks run sequentially and summaries
are non-streaming. Parallel task execution and token-level streaming are
deferred.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from core.knowledge import KnowledgeChunk, KnowledgeQuery, build_context
from models import SummaryState, TodoItem
from services.citation import assign_cite_ids
from services.factory import ResearchServices

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

    async def _gather_chunks(self, query: KnowledgeQuery) -> list[KnowledgeChunk]:
        """Query every enabled knowledge source concurrently and merge results.

        A single source failing (or returning nothing) never aborts the task —
        the remaining sources still contribute.
        """
        sources = self._services.knowledge_sources
        results = await asyncio.gather(
            *(source.query(query) for source in sources),
            return_exceptions=True,
        )
        chunks: list[KnowledgeChunk] = []
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.warning("知识源 '%s' 查询失败: %s", source.name, result)
                continue
            chunks.extend(result)
        return chunks

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
            query = KnowledgeQuery(
                text=task.query,
                intent=task.intent,
                max_results=self._config.search_results_per_task,
            )
            chunks = await self._gather_chunks(query)

            if not chunks:
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

            # Assign run-scoped cite_ids so the summarizer / reporter can attach
            # [^N] markers to each claim it derives from a chunk. ``state.all_chunks``
            # is the single source of truth that the reporter's bibliography step
            # iterates over later.
            assign_cite_ids(chunks, existing=state.all_chunks)
            state.all_chunks.extend(chunks)

            sources_summary, context = build_context(chunks)
            task.sources_summary = sources_summary
            state.web_research_results.append(context)
            state.sources_gathered.append(sources_summary)
            state.research_loop_count += 1

            backends = ", ".join(sorted({chunk.source for chunk in chunks}))
            yield {
                "type": "sources",
                "task_id": task.id,
                "latest_sources": sources_summary,
                "raw_context": context,
                "step": step,
                "backend": backends,
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
