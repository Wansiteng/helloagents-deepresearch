"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Semaphore, Thread
from typing import Any, Callable, Iterator, Optional

from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry

from agents.robust_agent import RobustToolAwareAgent

from config import Configuration
from prompts import (
    open_source_model_constraint_prompt,
    report_writer_instructions,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from models import SummaryState, SummaryStateOutput, TodoItem
from services.planner import PlannerAgent
from services.reporter import WriterAgent
from services.search import dispatch_search, prepare_research_context
from services.summarizer import SummarizerAgent
from services.tool_events import ToolCallTracker
from services.vector_store import VectorStore
from tool_registry import AgentToolRegistry

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Coordinator orchestrating TODO-based research workflow using HelloAgents.

    架构概览
    --------
    本 Orchestrator 将深度研究流程解耦为三个单一职责 Agent：

    - :class:`~services.planner.PlannerAgent` — 意图理解与任务拆解
    - :class:`~services.summarizer.SummarizerAgent` — 网页阅读与信息浓缩
    - :class:`~services.reporter.WriterAgent` — 逻辑组织与长文生成

    三个 Agent 共享同一个 :class:`~tool_registry.AgentToolRegistry` 实例，
    通过统一接口声明和调用工具（笔记、搜索等），便于后续扩展。

    SSE 流式机制
    -----------
    调用 :meth:`run_stream` 时，各 Agent 在执行规划、搜索、总结等节点时，
    通过 :class:`~services.tool_events.ToolCallTracker` 将状态与中间结果实时推送，
    前端通过 ``/research/stream`` SSE 端点接收，避免长链接超时。
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise the coordinator with configuration and shared tools."""
        self.config = config or Configuration.from_env()
        self.llm = self._init_llm()

        # ── 统一工具注册表：替代分散的工具初始化代码 ───────────────────
        self.tool_registry = AgentToolRegistry(self.config)

        # ── RAG 向量记忆层（可选）──────────────────────────────────────
        self.vector_store: Optional[VectorStore] = None
        if self.config.use_vector_store:
            try:
                self.vector_store = VectorStore(
                    workspace=self.config.vector_store_path,
                    embedding_model=self.config.embedding_model,
                    ollama_base_url=self.config.ollama_base_url,
                    chunk_size=self.config.vector_chunk_size,
                    chunk_overlap=self.config.vector_chunk_overlap,
                )
                logger.info(
                    "VectorStore 已启用: path=%s model=%s",
                    self.config.vector_store_path,
                    self.config.embedding_model,
                )
            except Exception as exc:
                logger.warning("VectorStore 初始化失败，将禁用向量记忆: %s", exc)
                self.vector_store = None

        self._tool_tracker = ToolCallTracker(
            self.config.notes_workspace if self.config.enable_notes else None
        )
        self._tool_event_sink_enabled = False
        self._state_lock = Lock()
        # 本地 LLM（Ollama）同时只能处理一个请求，用信号量串行化任务执行
        self._llm_semaphore = Semaphore(1)

        # ── 三个专职 Agent（共享 tool_registry）───────────────────────
        planner_agent = self._create_tool_aware_agent(
            name="研究规划专家",
            system_prompt=todo_planner_system_prompt.strip(),
        )
        writer_agent = self._create_tool_aware_agent(
            name="报告撰写专家",
            system_prompt=report_writer_instructions.strip(),
        )
        summarizer_factory: Callable[[], ToolAwareSimpleAgent] = (
            lambda: self._create_tool_aware_agent(
                name="任务总结专家",
                system_prompt=task_summarizer_instructions.strip(),
            )
        )

        self.planner = PlannerAgent(planner_agent, self.config)
        self.summarizer = SummarizerAgent(summarizer_factory, self.config)
        self.writer = WriterAgent(writer_agent, self.config, vector_store=self.vector_store)
        # 向后兼容：旧代码可能通过 self.reporting 访问
        self.reporting = self.writer
        self._last_search_notices: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _init_llm(self) -> HelloAgentsLLM:
        """Instantiate HelloAgentsLLM following configuration preferences."""
        llm_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "timeout": self.config.llm_timeout,
        }

        model_id = self.config.llm_model_id or self.config.local_llm
        if model_id:
            llm_kwargs["model"] = model_id

        provider = (self.config.llm_provider or "").strip()
        if provider:
            llm_kwargs["provider"] = provider

        if provider == "ollama":
            llm_kwargs["base_url"] = self.config.sanitized_ollama_url()
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif provider == "lmstudio":
            llm_kwargs["base_url"] = self.config.lmstudio_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
        else:
            if self.config.llm_base_url:
                llm_kwargs["base_url"] = self.config.llm_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key

        return HelloAgentsLLM(**llm_kwargs)

    def _create_tool_aware_agent(self, *, name: str, system_prompt: str) -> ToolAwareSimpleAgent:
        """Instantiate a RobustToolAwareAgent sharing the unified tool registry.

        使用 :class:`~agents.robust_agent.RobustToolAwareAgent` 替代默认的
        ``ToolAwareSimpleAgent``，在 LLM 生成的 JSON 参数不合法时（如 content
        字段含未转义引号），通过 JSON 修复与正则兜底提取保证工具调用不失败。

        当 ``config.use_open_source_mode`` 为 True 时（针对 Qwen/Llama 等本地模型），
        会自动在 System Prompt 末尾追加强约束格式规则与 Few-Shot 示例；同时启用
        自我纠错重试闭环，最大重试次数由 ``config.open_source_model_max_retries`` 控制。
        """
        ha_registry = self.tool_registry.hello_agents_registry

        # ── 开源模型强约束 Prompt 补丁 ──────────────────────────────────
        effective_prompt = system_prompt
        if self.config.use_open_source_mode:
            effective_prompt = system_prompt + "\n" + open_source_model_constraint_prompt
            logger.debug(
                "open_source_mode: injecting constraint prompt for agent '%s'", name
            )

        max_retries = (
            self.config.open_source_model_max_retries
            if self.config.use_open_source_mode
            else 0
        )

        return RobustToolAwareAgent(
            name=name,
            llm=self.llm,
            system_prompt=effective_prompt,
            enable_tool_calling=ha_registry is not None,
            tool_registry=ha_registry,
            tool_call_listener=self._tool_tracker.record,
            self_correction_max_retries=max_retries,
        )

    def _set_tool_event_sink(self, sink: Callable[[dict[str, Any]], None] | None) -> None:
        """Enable or disable immediate tool event callbacks."""
        self._tool_event_sink_enabled = sink is not None
        self._tool_tracker.set_event_sink(sink)

    def run(self, topic: str) -> SummaryStateOutput:
        """Execute the research workflow and return the final report."""
        state = SummaryState(research_topic=topic)
        state.todo_items = self.planner.plan_todo_list(state)
        self._drain_tool_events(state)

        if not state.todo_items:
            logger.info("No TODO items generated; falling back to single task")
            state.todo_items = [self.planner.create_fallback_task(state)]

        for task in state.todo_items:
            self._execute_task(state, task, emit_stream=False)

        report = self.writer.generate_report(state)
        self._drain_tool_events(state)
        state.structured_report = report
        state.running_summary = report
        self._persist_final_report(state, report)

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=state.todo_items,
        )

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """Execute the workflow yielding incremental progress events."""
        state = SummaryState(research_topic=topic)
        logger.debug("Starting streaming research: topic=%s", topic)
        yield {"type": "status", "message": "初始化研究流程"}

        state.todo_items = self.planner.plan_todo_list(state)
        for event in self._drain_tool_events(state, step=0):
            yield event
        if not state.todo_items:
            state.todo_items = [self.planner.create_fallback_task(state)]

        channel_map: dict[int, dict[str, Any]] = {}
        for index, task in enumerate(state.todo_items, start=1):
            token = f"task_{task.id}"
            task.stream_token = token
            channel_map[task.id] = {"step": index, "token": token}

        yield {
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in state.todo_items],
            "step": 0,
        }

        event_queue: Queue[dict[str, Any]] = Queue()

        def enqueue(
            event: dict[str, Any],
            *,
            task: TodoItem | None = None,
            step_override: int | None = None,
        ) -> None:
            payload = dict(event)
            target_task_id = payload.get("task_id")
            if task is not None:
                target_task_id = task.id
                payload["task_id"] = task.id

            channel = channel_map.get(target_task_id) if target_task_id is not None else None
            if channel:
                payload.setdefault("step", channel["step"])
                payload["stream_token"] = channel["token"]
            if step_override is not None:
                payload["step"] = step_override
            event_queue.put(payload)

        def tool_event_sink(event: dict[str, Any]) -> None:
            enqueue(event)

        self._set_tool_event_sink(tool_event_sink)

        threads: list[Thread] = []

        def worker(task: TodoItem, step: int) -> None:
            try:
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "in_progress",
                        "title": task.title,
                        "intent": task.intent,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                    },
                    task=task,
                )

                # 串行化：等待信号量，确保同一时刻只有一个任务占用 Ollama
                with self._llm_semaphore:
                    for event in self._execute_task(state, task, emit_stream=True, step=step):
                        enqueue(event, task=task)
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Task execution failed", exc_info=exc)
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "failed",
                        "detail": str(exc),
                        "title": task.title,
                        "intent": task.intent,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                    },
                    task=task,
                )
            finally:
                enqueue({"type": "__task_done__", "task_id": task.id})

        for task in state.todo_items:
            step = channel_map.get(task.id, {}).get("step", 0)
            thread = Thread(target=worker, args=(task, step), daemon=True)
            threads.append(thread)
            thread.start()

        active_workers = len(state.todo_items)
        finished_workers = 0

        try:
            while finished_workers < active_workers:
                event = event_queue.get()
                if event.get("type") == "__task_done__":
                    finished_workers += 1
                    continue
                yield event

            while True:
                try:
                    event = event_queue.get_nowait()
                except Empty:
                    break
                if event.get("type") != "__task_done__":
                    yield event
        finally:
            self._set_tool_event_sink(None)
            for thread in threads:
                thread.join()

        report = self.writer.generate_report(state)
        final_step = len(state.todo_items) + 1
        for event in self._drain_tool_events(state, step=final_step):
            yield event
        state.structured_report = report
        state.running_summary = report

        note_event = self._persist_final_report(state, report)
        if note_event:
            yield note_event

        yield {
            "type": "final_report",
            "report": report,
            "note_id": state.report_note_id,
            "note_path": state.report_note_path,
        }
        yield {"type": "done"}

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        emit_stream: bool,
        step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run search + summarization for a single task."""
        task.status = "in_progress"

        search_result, notices, answer_text, backend = dispatch_search(
            task.query,
            self.config,
            state.research_loop_count,
        )
        self._last_search_notices = notices
        task.notices = notices

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
        else:
            self._drain_tool_events(state)

        if notices and emit_stream:
            for notice in notices:
                if notice:
                    yield {
                        "type": "status",
                        "message": notice,
                        "task_id": task.id,
                        "step": step,
                    }

        if not search_result or not search_result.get("results"):
            task.status = "skipped"
            if emit_stream:
                for event in self._drain_tool_events(state, step=step):
                    yield event
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
            else:
                self._drain_tool_events(state)
            return
        else:
            if not emit_stream:
                self._drain_tool_events(state)

        sources_summary, context = prepare_research_context(
            search_result,
            answer_text,
            self.config,
        )

        # ── RAG 检索：从向量库取回历史相关研究补充上下文 ─────────────────
        if self.vector_store is not None:
            rag_query = f"{state.research_topic} {task.title} {task.query}"
            try:
                rag_hits = self.vector_store.query(
                    rag_query, n_results=self.config.vector_top_k
                )
                if rag_hits:
                    rag_snippets = "\n\n---\n\n".join(
                        f"[历史研究片段 {i + 1}]\n{hit['text']}"
                        for i, hit in enumerate(rag_hits)
                    )
                    context = (
                        f"## 向量记忆库检索结果（历史相关研究）\n\n"
                        f"{rag_snippets}\n\n"
                        f"---\n\n"
                        f"## 本轮搜索结果\n\n{context}"
                    )
                    logger.info(
                        "RAG 检索: task_id=%s 命中 %d 条历史片段",
                        task.id,
                        len(rag_hits),
                    )
            except Exception as exc:
                logger.warning("RAG 查询失败，跳过历史记忆增强: %s", exc)

        task.sources_summary = sources_summary

        with self._state_lock:
            state.web_research_results.append(context)
            state.sources_gathered.append(sources_summary)
            state.research_loop_count += 1

        summary_text: str | None = None

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
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

            summary_stream, summary_getter = self.summarizer.stream_task_summary(state, task, context)
            try:
                for event in self._drain_tool_events(state, step=step):
                    yield event
                for chunk in summary_stream:
                    if chunk:
                        yield {
                            "type": "task_summary_chunk",
                            "task_id": task.id,
                            "content": chunk,
                            "note_id": task.note_id,
                            "step": step,
                        }
                    for event in self._drain_tool_events(state, step=step):
                        yield event
            finally:
                summary_text = summary_getter()
        else:
            summary_text = self.summarizer.summarize_task(state, task, context)
            self._drain_tool_events(state)

        task.summary = summary_text.strip() if summary_text else "暂无可用信息"
        task.status = "completed"

        # ── RAG 写入：将本轮摘要向量化存入记忆库 ─────────────────────────
        if self.vector_store is not None and task.summary and task.summary != "暂无可用信息":
            vs_metadata: dict[str, Any] = {
                "task_id": task.id,
                "task_title": task.title,
                "topic": state.research_topic,
                "query": task.query,
            }
            try:
                n_chunks = self.vector_store.add_document(
                    text=task.summary,
                    metadata=vs_metadata,
                    doc_id=f"task_{task.id}",
                )
                logger.info(
                    "RAG 写入: task_id=%s 写入 %d 个 chunk",
                    task.id,
                    n_chunks,
                )
            except Exception as exc:
                logger.warning("RAG 写入失败，摘要未持久化到向量库: %s", exc)

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
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
        else:
            self._drain_tool_events(state)

    def _drain_tool_events(
        self,
        state: SummaryState,
        *,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        """Proxy to the shared tool call tracker."""
        events = self._tool_tracker.drain(state, step=step)
        if self._tool_event_sink_enabled:
            return []
        return events

    @property
    def _tool_call_events(self) -> list[dict[str, Any]]:
        """Expose recorded tool events for legacy integrations."""
        return self._tool_tracker.as_dicts()

    def _serialize_task(self, task: TodoItem) -> dict[str, Any]:
        """Convert task dataclass to serializable dict for frontend."""
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
        }

    def _persist_final_report(self, state: SummaryState, report: str) -> dict[str, Any] | None:
        # ── 将最终报告存入向量库 ──────────────────────────────────────────
        if self.vector_store is not None and report and report.strip():
            try:
                n_chunks = self.vector_store.add_document(
                    text=report.strip(),
                    metadata={
                        "type": "final_report",
                        "topic": state.research_topic,
                    },
                    doc_id=f"report_{state.research_topic[:32]}",
                )
                logger.info("RAG 写入最终报告: %d 个 chunk", n_chunks)
            except Exception as exc:
                logger.warning("最终报告向量化失败: %s", exc)

        note_tool = self.tool_registry.note_tool
        if not note_tool or not report or not report.strip():
            return None

        note_title = f"研究报告：{state.research_topic}".strip() or "研究报告"
        tags = ["deep_research", "report"]
        content = report.strip()

        note_id = self._find_existing_report_note_id(state)
        response = ""

        if note_id:
            response = note_tool.run(
                {
                    "action": "update",
                    "note_id": note_id,
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            if response.startswith("❌"):
                note_id = None

        if not note_id:
            response = note_tool.run(
                {
                    "action": "create",
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            note_id = self._extract_note_id_from_text(response)

        if not note_id:
            return None

        state.report_note_id = note_id
        if self.config.notes_workspace:
            note_path = Path(self.config.notes_workspace) / f"{note_id}.md"
            state.report_note_path = str(note_path)
        else:
            note_path = None

        payload = {
            "type": "report_note",
            "note_id": note_id,
            "title": note_title,
            "content": content,
        }
        if note_path:
            payload["note_path"] = str(note_path)

        return payload

    def _find_existing_report_note_id(self, state: SummaryState) -> str | None:
        if state.report_note_id:
            return state.report_note_id

        for event in reversed(self._tool_tracker.as_dicts()):
            if event.get("tool") != "note":
                continue

            parameters = event.get("parsed_parameters") or {}
            if not isinstance(parameters, dict):
                continue

            action = parameters.get("action")
            if action not in {"create", "update"}:
                continue

            note_type = parameters.get("note_type")
            if note_type != "conclusion":
                title = parameters.get("title")
                if not (isinstance(title, str) and title.startswith("研究报告")):
                    continue

            note_id = parameters.get("note_id")
            if not note_id:
                note_id = self._tool_tracker._extract_note_id(event.get("result", ""))  # type: ignore[attr-defined]

            if note_id:
                return note_id

        return None

    @staticmethod
    def _extract_note_id_from_text(response: str) -> str | None:
        if not response:
            return None

        match = re.search(r"ID:\s*([^\n]+)", response)
        if not match:
            return None

        return match.group(1).strip()


def run_deep_research(topic: str, config: Configuration | None = None) -> SummaryStateOutput:
    """Convenience function mirroring the class-based API."""
    agent = DeepResearchAgent(config=config)
    return agent.run(topic)
