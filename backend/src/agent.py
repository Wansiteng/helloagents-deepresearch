"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import copy
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
from services.reflection import CriticAgent
from services.reporter import WriterAgent
from services.search import dispatch_search_with_retry, prepare_research_context
from services.summarizer import SummarizerAgent
from services.tool_events import ToolCallTracker
from services.vector_store import VectorStore
from tool_registry import AgentToolRegistry

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Coordinator orchestrating TODO-based research workflow using HelloAgents.

    架构概览
    --------
    本 Orchestrator 将深度研究流程解耦为三个单一职责 Agent，并叠加四项优化：

    - :class:`~services.planner.PlannerAgent` — 意图理解与任务拆解
    - :class:`~services.summarizer.SummarizerAgent` — 网页阅读与信息浓缩
    - :class:`~services.reporter.WriterAgent` — 逻辑组织与长文生成
    - :class:`~services.reflection.CriticAgent` — 报告质量反思评审

    四项优化
    --------
    1. **搜索并行化**：所有任务搜索在独立线程中同时执行（I/O 密集），
       仅 LLM 摘要调用受 ``_llm_semaphore`` 串行控制。
    2. **动态规划**：初始任务完成后评估覆盖度，按需补充最多
       ``config.max_dynamic_tasks`` 个新任务。
    3. **反思评审**：报告生成后由 CriticAgent 评分，低分时触发补充研究。
    4. **渐进式报告**：每任务完成后立即推送格式化章节草稿，无额外 LLM 调用。
    5. **上下文压缩**：传给 Writer 前自动裁剪过长摘要，防止超出上下文窗口。
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise the coordinator with configuration and shared tools."""
        self.config = config or Configuration.from_env()
        self.llm = self._init_llm()

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

        # ── LLM 并发信号量（默认 1，可通过 llm_concurrency 配置）────────
        # Ollama 默认单请求；配合 OLLAMA_NUM_PARALLEL=N 可提升为 N。
        self._llm_semaphore = Semaphore(max(1, self.config.llm_concurrency))

        # ── 三个核心 Agent + CriticAgent ───────────────────────────────
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
        self.reporting = self.writer  # 向后兼容

        # ── 反思 Agent（可选）────────────────────────────────────────────
        if self.config.enable_reflection:
            from prompts import reflection_instructions  # noqa: F401 – 验证提示词可导入
            critic_factory: Callable[[], ToolAwareSimpleAgent] = (
                lambda: self._create_tool_aware_agent(
                    name="报告质量评审专家",
                    system_prompt=(
                        "你是一名专业的研究质量评审专家，擅长识别研究报告中的不足与空白。"
                        "请对给定的研究报告进行客观评审，输出结构化 JSON 格式的评估结果。"
                    ),
                )
            )
            self.critic: Optional[CriticAgent] = CriticAgent(critic_factory, self.config)
        else:
            self.critic = None

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
            llm_kwargs["api_key"] = self.config.llm_api_key or "ollama"
        elif provider == "lmstudio":
            llm_kwargs["base_url"] = self.config.lmstudio_base_url
            llm_kwargs["api_key"] = self.config.llm_api_key or "lm-studio"
        else:
            if self.config.llm_base_url:
                llm_kwargs["base_url"] = self.config.llm_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key

        return HelloAgentsLLM(**llm_kwargs)

    def _create_tool_aware_agent(self, *, name: str, system_prompt: str) -> ToolAwareSimpleAgent:
        """Instantiate a RobustToolAwareAgent sharing the unified tool registry."""
        ha_registry = self.tool_registry.hello_agents_registry

        effective_prompt = system_prompt
        if self.config.use_open_source_mode:
            effective_prompt = system_prompt + "\n" + open_source_model_constraint_prompt
            logger.debug("open_source_mode: injecting constraint prompt for agent '%s'", name)

        if self.config.no_think_mode:
            effective_prompt = "/no_think\n" + effective_prompt
            logger.debug("no_think_mode: injecting /no_think directive for agent '%s'", name)

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

    # ------------------------------------------------------------------
    # Synchronous API (legacy / non-streaming)
    # ------------------------------------------------------------------
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

        # Dynamic planning (sync mode)
        if self.config.enable_dynamic_planning:
            try:
                new_tasks = self.planner.assess_gaps(state)
                if new_tasks:
                    state.todo_items.extend(new_tasks)
                    for task in new_tasks:
                        self._execute_task(state, task, emit_stream=False)
            except Exception as exc:
                logger.warning("动态规划评估失败，跳过: %s", exc)

        writer_state = self._compress_state_for_writer(state)
        report = self.writer.generate_report(writer_state)
        self._drain_tool_events(state)
        state.structured_report = report
        state.running_summary = report
        self._persist_final_report(state, report)

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=state.todo_items,
        )

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------
    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """Execute the workflow yielding incremental progress events.

        改进后的流程
        -----------
        1. Planner 拆解任务
        2. 并行搜索（所有任务同时发起 I/O 请求）
        3. 串行 LLM 摘要（Semaphore 控制）
        4. 渐进式章节草稿推送（无额外 LLM 调用）
        5. 动态规划：检查空白，按需追加新任务
        6. WriterAgent 生成最终报告（上下文压缩后）
        7. 反思评审：低分时追加补充研究并重新生成报告
        """
        state = SummaryState(research_topic=topic)
        logger.debug("Starting streaming research: topic=%s", topic)
        yield {"type": "status", "message": "初始化研究流程"}

        # ── 规划阶段 ──────────────────────────────────────────────────
        state.todo_items = self.planner.plan_todo_list(state)
        for event in self._drain_tool_events(state, step=0):
            yield event
        if not state.todo_items:
            state.todo_items = [self.planner.create_fallback_task(state)]

        # ── 建立任务到 SSE 通道的映射 ─────────────────────────────────
        channel_map: dict[int, dict[str, Any]] = {}
        all_threads: list[Thread] = []
        event_queue: Queue[dict[str, Any]] = Queue()

        def assign_channel(task: TodoItem, step: int) -> None:
            token = f"task_{task.id}"
            task.stream_token = token
            channel_map[task.id] = {"step": step, "token": token}

        for index, task in enumerate(state.todo_items, start=1):
            assign_channel(task, index)

        yield {
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in state.todo_items],
            "step": 0,
        }

        def enqueue(
            event: dict[str, Any],
            *,
            task: TodoItem | None = None,
        ) -> None:
            payload = dict(event)
            target_id = task.id if task is not None else payload.get("task_id")
            if target_id is not None:
                payload["task_id"] = target_id
            channel = channel_map.get(target_id) if target_id is not None else None
            if channel:
                payload.setdefault("step", channel["step"])
                payload["stream_token"] = channel["token"]
            event_queue.put(payload)

        def tool_event_sink(event: dict[str, Any]) -> None:
            enqueue(event)

        self._set_tool_event_sink(tool_event_sink)

        # ── Worker：两阶段执行（搜索并行 + LLM 串行）─────────────────
        def worker(task: TodoItem, step: int) -> None:  # noqa: C901 – complexity justified
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

                # ── Phase 1: SEARCH（无信号量 — 全并行 I/O）──────────
                fallback_queries: list[str] = []
                if self.config.search_retry_on_empty:
                    fallback_queries = [
                        f"{task.title} {state.research_topic}",
                        task.title,
                    ]
                try:
                    search_result, notices, answer_text, backend = dispatch_search_with_retry(
                        task.query,
                        self.config,
                        state.research_loop_count,
                        fallback_queries=fallback_queries,
                    )
                except Exception as exc:
                    logger.exception("Task %d search failed: %s", task.id, exc)
                    task.status = "failed"
                    enqueue(
                        {
                            "type": "task_status",
                            "task_id": task.id,
                            "status": "failed",
                            "detail": f"搜索失败: {exc}",
                            "title": task.title,
                            "intent": task.intent,
                        },
                        task=task,
                    )
                    return

                task.notices = notices
                for notice in notices:
                    if notice:
                        enqueue({"type": "status", "message": notice}, task=task)

                if not search_result or not search_result.get("results"):
                    task.status = "skipped"
                    enqueue(
                        {
                            "type": "task_status",
                            "task_id": task.id,
                            "status": "skipped",
                            "title": task.title,
                            "intent": task.intent,
                            "note_id": task.note_id,
                            "note_path": task.note_path,
                        },
                        task=task,
                    )
                    return

                sources_summary, context = prepare_research_context(
                    search_result, answer_text, self.config
                )

                # RAG 检索（向量查询快速，无需信号量）
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
                                f"{rag_snippets}\n\n---\n\n"
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

                enqueue(
                    {
                        "type": "sources",
                        "task_id": task.id,
                        "latest_sources": sources_summary,
                        "raw_context": context,
                        "step": step,
                        "backend": backend,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                    },
                    task=task,
                )

                # ── Phase 2: LLM SUMMARIZE（信号量串行）────────────────
                with self._llm_semaphore:
                    summary_stream, summary_getter = self.summarizer.stream_task_summary(
                        state, task, context
                    )
                    try:
                        for event in self._drain_tool_events(state, step=step):
                            enqueue(event)
                        for chunk in summary_stream:
                            if chunk:
                                enqueue(
                                    {
                                        "type": "task_summary_chunk",
                                        "task_id": task.id,
                                        "content": chunk,
                                        "note_id": task.note_id,
                                        "step": step,
                                    },
                                    task=task,
                                )
                            for event in self._drain_tool_events(state, step=step):
                                enqueue(event)
                    finally:
                        summary_text = summary_getter()

                    task.summary = summary_text.strip() if summary_text else "暂无可用信息"

                    # RAG 写入
                    if (
                        self.vector_store is not None
                        and task.summary
                        and task.summary != "暂无可用信息"
                    ):
                        try:
                            n_chunks = self.vector_store.add_document(
                                text=task.summary,
                                metadata={
                                    "task_id": task.id,
                                    "task_title": task.title,
                                    "topic": state.research_topic,
                                    "query": task.query,
                                },
                                doc_id=f"task_{task.id}",
                            )
                            logger.info(
                                "RAG 写入: task_id=%s 写入 %d 个 chunk", task.id, n_chunks
                            )
                        except Exception as exc:
                            logger.warning("RAG 写入失败，摘要未持久化到向量库: %s", exc)

                    # ── 渐进式报告：格式化章节草稿（无 LLM 调用）───────
                    if self.config.enable_progressive_report and task.summary:
                        section_draft = self._format_section_draft(task)
                        if section_draft:
                            task.section_draft = section_draft
                            enqueue(
                                {
                                    "type": "section_draft",
                                    "task_id": task.id,
                                    "title": task.title,
                                    "content": section_draft,
                                    "step": step,
                                },
                                task=task,
                            )

                    for event in self._drain_tool_events(state, step=step):
                        enqueue(event)

                task.status = "completed"
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "completed",
                        "summary": task.summary,
                        "sources_summary": task.sources_summary,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                        "step": step,
                    },
                    task=task,
                )

            except Exception as exc:
                logger.exception("Task %d execution failed", task.id, exc_info=exc)
                task.status = "failed"
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

        def run_batch(tasks_batch: list[TodoItem]) -> Iterator[dict[str, Any]]:
            """启动一批任务 worker 并 yield 所有事件，直到本批全部完成。"""
            if not tasks_batch:
                return

            batch_threads: list[Thread] = []
            for t in tasks_batch:
                step = channel_map.get(t.id, {}).get("step", 0)
                thread = Thread(target=worker, args=(t, step), daemon=True)
                batch_threads.append(thread)
                all_threads.append(thread)
                thread.start()

            finished = 0
            n = len(tasks_batch)
            while finished < n:
                event = event_queue.get()
                if event.get("type") == "__task_done__":
                    finished += 1
                    continue
                yield event

            # 清空本批残留事件
            while True:
                try:
                    event = event_queue.get_nowait()
                except Empty:
                    break
                if event.get("type") != "__task_done__":
                    yield event

        try:
            # ── 执行初始任务批次 ──────────────────────────────────────
            yield from run_batch(state.todo_items)

            # ── 动态规划：评估覆盖度，补充任务 ───────────────────────
            if self.config.enable_dynamic_planning:
                try:
                    yield {"type": "status", "message": "正在评估研究覆盖度..."}
                    new_tasks = self.planner.assess_gaps(state)
                    if new_tasks:
                        # 分配新 ID 与通道
                        next_id = max(t.id for t in state.todo_items) + 1
                        next_step = (
                            max(ch["step"] for ch in channel_map.values()) + 1
                            if channel_map
                            else len(state.todo_items) + 1
                        )
                        for i, task in enumerate(new_tasks):
                            task.id = next_id + i
                            assign_channel(task, next_step + i)

                        state.todo_items.extend(new_tasks)
                        yield {
                            "type": "dynamic_tasks",
                            "tasks": [self._serialize_task(t) for t in new_tasks],
                            "message": f"发现 {len(new_tasks)} 个研究空白，补充调研中...",
                        }
                        yield from run_batch(new_tasks)
                except Exception as exc:
                    logger.warning("动态规划评估失败，跳过: %s", exc)

            # ── 生成最终报告（上下文压缩后传入 Writer）──────────────
            final_step = (
                max(ch["step"] for ch in channel_map.values()) + 1
                if channel_map
                else len(state.todo_items) + 1
            )
            yield {"type": "status", "message": "正在生成研究报告..."}
            writer_state = self._compress_state_for_writer(state)
            report = self.writer.generate_report(writer_state)
            for event in self._drain_tool_events(state, step=final_step):
                yield event
            state.structured_report = report
            state.running_summary = report

            # ── 反思评审：质量检查，低分时补充研究 ────────────────────
            if self.config.enable_reflection and self.critic is not None:
                try:
                    yield {"type": "status", "message": "正在进行报告质量评审..."}
                    assessment = self.critic.assess_report(state, report)
                    yield {
                        "type": "reflection",
                        "score": assessment.get("score", 10),
                        "needs_more_research": assessment.get("needs_more_research", False),
                        "gaps": assessment.get("gaps", []),
                    }

                    if assessment.get("needs_more_research") and assessment.get(
                        "additional_queries"
                    ):
                        queries: list[str] = assessment["additional_queries"][:2]
                        extra_tasks: list[TodoItem] = []
                        next_id = max(t.id for t in state.todo_items) + 1
                        next_step = (
                            max(ch["step"] for ch in channel_map.values()) + 1
                            if channel_map
                            else final_step + 1
                        )
                        for i, q in enumerate(queries):
                            et = TodoItem(
                                id=next_id + i,
                                title=f"补充调研 {i + 1}",
                                intent=f"弥补研究不足：{q}",
                                query=q,
                            )
                            assign_channel(et, next_step + i)
                            extra_tasks.append(et)

                        state.todo_items.extend(extra_tasks)
                        score_val = assessment.get("score", "?")
                        yield {
                            "type": "dynamic_tasks",
                            "tasks": [self._serialize_task(t) for t in extra_tasks],
                            "message": (
                                f"报告质量评分 {score_val}/10，"
                                f"补充 {len(extra_tasks)} 项研究..."
                            ),
                        }
                        yield from run_batch(extra_tasks)

                        # 基于补充研究重新生成报告
                        yield {"type": "status", "message": "基于补充研究重新生成报告..."}
                        writer_state = self._compress_state_for_writer(state)
                        report = self.writer.generate_report(writer_state)
                        state.structured_report = report
                        state.running_summary = report
                except Exception as exc:
                    logger.warning("反思评审失败，跳过: %s", exc)

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

        finally:
            self._set_tool_event_sink(None)
            for thread in all_threads:
                thread.join()

    # ------------------------------------------------------------------
    # Execution helpers (sync path)
    # ------------------------------------------------------------------
    def _execute_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        emit_stream: bool,
        step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run search + summarization for a single task (sync/legacy path)."""
        task.status = "in_progress"

        fallback_queries: list[str] = []
        if self.config.search_retry_on_empty:
            fallback_queries = [f"{task.title} {state.research_topic}", task.title]

        search_result, notices, answer_text, backend = dispatch_search_with_retry(
            task.query,
            self.config,
            state.research_loop_count,
            fallback_queries=fallback_queries,
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
                        f"{rag_snippets}\n\n---\n\n"
                        f"## 本轮搜索结果\n\n{context}"
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

            summary_stream, summary_getter = self.summarizer.stream_task_summary(
                state, task, context
            )
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

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _format_section_draft(self, task: TodoItem) -> str:
        """将任务摘要格式化为渐进式章节草稿（无 LLM 调用）。"""
        if not task.summary or task.summary == "暂无可用信息":
            return ""
        lines = [f"## {task.title}", ""]
        lines.append(task.summary)
        if task.sources_summary:
            lines += ["", "**来源**", task.sources_summary]
        return "\n".join(lines)

    def _compress_state_for_writer(self, state: SummaryState) -> SummaryState:
        """返回摘要已压缩的 state 副本，防止 Writer 提示词超长。

        若总摘要字符数未超过 ``context_max_chars``，直接返回原 state。
        """
        max_total = self.config.context_max_chars
        card_max = self.config.summary_card_max_chars

        if max_total <= 0 or card_max <= 0:
            return state

        total_chars = sum(len(t.summary or "") for t in state.todo_items)
        if total_chars <= max_total:
            return state

        logger.info(
            "上下文压缩：总摘要 %d 字符超出限制 %d，按 %d 字符截断各任务摘要",
            total_chars,
            max_total,
            card_max,
        )

        compressed = copy.copy(state)
        compressed_tasks: list[TodoItem] = []
        for task in state.todo_items:
            task_copy = copy.copy(task)
            if task_copy.summary and len(task_copy.summary) > card_max:
                task_copy.summary = task_copy.summary[:card_max] + "…（已截断）"
            compressed_tasks.append(task_copy)
        compressed.todo_items = compressed_tasks
        return compressed

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
            "section_draft": task.section_draft,
        }

    def _persist_final_report(self, state: SummaryState, report: str) -> dict[str, Any] | None:
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
