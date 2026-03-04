"""Planner Agent — 专精意图理解与任务拆解的单一职责 Agent。

该 Agent 接收用户研究主题，通过 LLM 将其分解为 3~5 个互补的子任务（TodoItem），
每个任务包含明确意图（intent）和可执行的检索查询（query）。
输出结果供 SummarizerAgent 和 WriterAgent 消费。

**单一职责**：仅负责意图理解与任务拆解，不执行搜索或摘要。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import SummaryState, TodoItem
from prompts import get_current_date, todo_planner_instructions
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)

TOOL_CALL_PATTERN = re.compile(
    r"\[TOOL_CALL:(?P<tool>[^:]+):(?P<body>[^\]]+)\]",
    re.IGNORECASE,
)


class PlannerAgent:
    """Planner Agent：将研究主题拆解为结构化 TodoItem 列表。

    该 Agent 通过以下流程工作：

    1. 使用 ``todo_planner_system_prompt`` 初始化 LLM。
    2. 接收研究主题，通过 ``todo_planner_instructions`` 格式化用户消息。
    3. 解析 LLM 输出的 JSON 结构，生成 :class:`~models.TodoItem` 列表。
    4. （可选）通过 NoteTool 将任务状态同步到本地笔记，供后续 Agent 查阅。

    工具依赖
    --------
    - ``note``（可选）：持久化任务初始状态到本地 Markdown 笔记。
    """

    def __init__(self, planner_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = planner_agent
        self._config = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def plan_todo_list(self, state: SummaryState) -> List[TodoItem]:
        """解析研究主题，拆解为可执行的 TodoItem 列表。

        参数
        ----
        state:
            当前研究状态，``research_topic`` 字段为必填。

        返回
        ----
        解析完成的 :class:`~models.TodoItem` 列表；若解析失败则返回空列表。
        """
        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()

        logger.info("Planner raw output (truncated): %s", response[:500])

        tasks_payload = self._extract_tasks(response)
        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"任务{idx}").strip()
            intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
            query = str(item.get("query") or state.research_topic).strip()

            if not query:
                query = state.research_topic

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            )
            todo_items.append(task)

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """当规划失败时生成一个兜底任务。"""
        return TodoItem(
            id=1,
            title="基础背景梳理",
            intent="收集主题的核心背景与最新动态",
            query=f"{state.research_topic} 最新进展" if state.research_topic else "基础背景梳理",
        )

    # ------------------------------------------------------------------
    # 内部解析辅助
    # ------------------------------------------------------------------

    def _extract_tasks(self, raw_response: str) -> List[dict[str, Any]]:
        """将 Planner 原始输出解析为任务字典列表。"""
        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: List[dict[str, Any]] = []

        if isinstance(json_payload, dict):
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        if not tasks:
            tool_payload = self._extract_tool_payload(text)
            if tool_payload and isinstance(tool_payload.get("tasks"), list):
                for item in tool_payload["tasks"]:
                    if isinstance(item, dict):
                        tasks.append(item)

        return tasks

    def _extract_json_payload(self, text: str) -> Optional[dict[str, Any] | list]:
        """在文本中定位并解析 JSON 对象或数组。"""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start: end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start: end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None

    def _extract_tool_payload(self, text: str) -> Optional[dict[str, Any]]:
        """解析输出中首个 TOOL_CALL 表达式。"""
        match = TOOL_CALL_PATTERN.search(text)
        if not match:
            return None

        body = match.group("body")

        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        parts = [segment.strip() for segment in body.split(",") if segment.strip()]
        payload: dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            payload[key.strip()] = value.strip().strip('"').strip("'")

        return payload or None


# ── 向后兼容别名 ───────────────────────────────────────────────────────
#: ``PlanningService`` 已重命名为 ``PlannerAgent``；此别名保持导入向后兼容。
PlanningService = PlannerAgent
