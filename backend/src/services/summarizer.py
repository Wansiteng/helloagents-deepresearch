"""Summarizer Agent — 专精网页阅读与信息浓缩的单一职责 Agent。

该 Agent 接收搜索结果上下文，为单个研究子任务提炼关键发现，
支持同步调用和流式（逐 token）输出两种模式。
输出结果供 WriterAgent 汇总为最终报告。

**单一职责**：仅负责网页内容阅读与信息浓缩，不执行搜索或规划。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Tuple

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState, TodoItem
from config import Configuration
from utils import strip_thinking_tokens
from services.notes import build_note_guidance
from services.text_processing import strip_tool_calls


class SummarizerAgent:
    """Summarizer Agent：对搜索结果进行网页阅读与信息浓缩。

    工具依赖
    --------
    - ``note``（可选）：读取 PlannerAgent 创建的任务笔记，并写回摘要结果，
      实现跨 Agent 的进度共享。

    使用方式
    --------
    同步模式::

        summary = summarizer.summarize_task(state, task, context)

    流式模式（SSE）::

        stream_gen, get_full = summarizer.stream_task_summary(state, task, context)
        for chunk in stream_gen:
            yield chunk  # 逐 token 推送给前端
        full_text = get_full()  # 获取完整摘要文本
    """

    def __init__(
        self,
        summarizer_factory: Callable[[], ToolAwareSimpleAgent],
        config: Configuration,
    ) -> None:
        self._agent_factory = summarizer_factory
        self._config = config

    def summarize_task(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Generate a task-specific summary using the summarizer agent."""

        prompt = self._build_prompt(state, task, context)

        agent = self._agent_factory()
        try:
            response = agent.run(prompt)
        finally:
            agent.clear_history()

        summary_text = response.strip()
        if self._config.strip_thinking_tokens:
            summary_text = strip_thinking_tokens(summary_text)

        summary_text = strip_tool_calls(summary_text).strip()

        return summary_text or "暂无可用信息"

    def stream_task_summary(
        self, state: SummaryState, task: TodoItem, context: str
    ) -> Tuple[Iterator[str], Callable[[], str]]:
        """Stream the summary text for a task while collecting full output."""

        prompt = self._build_prompt(state, task, context)
        remove_thinking = self._config.strip_thinking_tokens
        raw_buffer = ""
        visible_output = ""
        emit_index = 0
        agent = self._agent_factory()

        def flush_visible() -> Iterator[str]:
            nonlocal emit_index, raw_buffer
            while True:
                start = raw_buffer.find("<think>", emit_index)
                if start == -1:
                    if emit_index < len(raw_buffer):
                        segment = raw_buffer[emit_index:]
                        emit_index = len(raw_buffer)
                        if segment:
                            yield segment
                    break

                if start > emit_index:
                    segment = raw_buffer[emit_index:start]
                    emit_index = start
                    if segment:
                        yield segment

                end = raw_buffer.find("</think>", start)
                if end == -1:
                    break
                emit_index = end + len("</think>")

        def generator() -> Iterator[str]:
            nonlocal raw_buffer, visible_output, emit_index
            try:
                for chunk in agent.stream_run(prompt):
                    raw_buffer += chunk
                    if remove_thinking:
                        for segment in flush_visible():
                            visible_output += segment
                            if segment:
                                yield segment
                    else:
                        visible_output += chunk
                        if chunk:
                            yield chunk
            finally:
                if remove_thinking:
                    for segment in flush_visible():
                        visible_output += segment
                        if segment:
                            yield segment
                agent.clear_history()

        def get_summary() -> str:
            if remove_thinking:
                cleaned = strip_thinking_tokens(visible_output)
            else:
                cleaned = visible_output

            return strip_tool_calls(cleaned).strip()

        return generator(), get_summary

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_prompt(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """构造 Summarizer Agent 的完整用户提示（同步/流式共用）。"""

        return (
            f"任务主题：{state.research_topic}\n"
            f"任务名称：{task.title}\n"
            f"任务目标：{task.intent}\n"
            f"检索查询：{task.query}\n"
            f"任务上下文：\n{context}\n"
            f"{build_note_guidance(task)}\n"
            "请按照以上协作要求先同步笔记，然后返回一份面向用户的 Markdown 总结（仍遵循任务总结模板）。"
        )


# ── 向后兼容别名 ────────────────────────────────────────────────────────────
#: ``SummarizationService`` 已重命名为 ``SummarizerAgent``；此别名保持导入向后兼容。
SummarizationService = SummarizerAgent
