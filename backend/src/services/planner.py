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
from prompts import gap_assessment_instructions, get_current_date, todo_planner_instructions
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

            query = self._anchor_query(state.research_topic, query)

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

    def assess_gaps(self, state: SummaryState) -> list[TodoItem]:
        """评估已完成任务的覆盖度，返回补充任务列表。

        参数
        ----
        state:
            包含已完成子任务的研究状态。

        返回
        ----
        最多 ``config.max_dynamic_tasks`` 个新 :class:`~models.TodoItem`；
        若覆盖度足够或解析失败则返回空列表。
        """
        completed = [t for t in state.todo_items if t.status == "completed"]
        if not completed:
            return []

        completed_block = "\n".join(
            f"- 任务 {t.id}《{t.title}》：{t.intent}" for t in completed
        )
        max_tasks = self._config.max_dynamic_tasks

        prompt = gap_assessment_instructions.format(
            research_topic=state.research_topic or "",
            completed_tasks_block=completed_block,
            max_tasks=max_tasks,
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()
        logger.info("Gap assessment raw output (truncated): %s", response[:400])

        tasks_payload = self._extract_tasks_from_gap_response(response)
        if not tasks_payload:
            logger.info("Gap assessment: 无需补充任务")
            return []

        existing_ids = {t.id for t in state.todo_items}
        next_id = max(existing_ids) + 1 if existing_ids else 100

        new_tasks: list[TodoItem] = []
        for i, item in enumerate(tasks_payload[:max_tasks]):
            title = str(item.get("title") or f"补充任务{i+1}").strip()
            intent = str(item.get("intent") or "覆盖研究空白").strip()
            query = str(item.get("query") or state.research_topic or "").strip()
            if not query:
                continue
            query = self._anchor_query(state.research_topic or "", query)
            new_tasks.append(TodoItem(id=next_id + i, title=title, intent=intent, query=query))

        logger.info("Gap assessment: 补充 %d 个任务: %s", len(new_tasks), [t.title for t in new_tasks])
        return new_tasks

    def _extract_tasks_from_gap_response(self, raw_response: str) -> list[dict[str, Any]]:
        """解析 gap assessment 的 JSON 响应。"""
        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            from utils import strip_thinking_tokens
            text = strip_thinking_tokens(text)

        payload = self._extract_json_payload(text)
        if isinstance(payload, dict):
            if not payload.get("has_gaps", False):
                return []
            candidate = payload.get("additional_tasks", [])
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]

        return []

    @staticmethod
    def _anchor_query(topic: str, query: str) -> str:
        """Ensure the topic's distinctive tokens survive into the search query.

        Small local models occasionally produce a generic ``query`` that
        strips the proper noun out of the topic — e.g. topic ``宁德时代 BMS``
        becomes query ``BMS systems architecture``, which then searches the
        web for *Building* Management Systems, not battery management.

        Rules:
        - If the topic contains any multi-char CJK substring (≥ 2 Chinese
          characters in a row — usually a brand or proper noun), **every
          one of those substrings must appear verbatim in the query**.
          Otherwise prepend the topic.
        - Otherwise (pure Latin topic), require at least one Capitalized or
          all-caps token (≥ 3 chars) from the topic to appear in the query
          (case-insensitive). Otherwise prepend the topic.
        """
        if not topic or not query:
            return query

        topic = topic.strip()

        cjk_anchors = re.findall(r"[一-鿿]{2,}", topic)
        if cjk_anchors:
            if not all(a in query for a in cjk_anchors):
                return f"{topic} {query}"
            return query

        latin_anchors = re.findall(r"[A-Z][A-Za-z0-9]{2,}|[A-Z]{2,}", topic)
        if not latin_anchors:
            return query
        query_lower = query.lower()
        if any(t.lower() in query_lower for t in latin_anchors):
            return query
        return f"{topic} {query}"

    @staticmethod
    def create_fallback_tasks(state: SummaryState) -> List[TodoItem]:
        """当规划失败时生成一组多角度兜底任务。

        早期版本只回退到单条「基础背景梳理」，对宏大主题来说调研角度太薄。
        现在按 5 个角度展开：背景概念 / 现状脉络 / 关键技术 / 应用案例 / 挑战展望。
        每条任务都根据 ``research_topic`` 拼出可执行的查询，避免空 query。
        """
        topic = (state.research_topic or "").strip()
        base = topic or "研究主题"

        angles: List[dict[str, str]] = [
            {
                "title": "背景与基本概念",
                "intent": "梳理主题的定义、范畴与核心概念",
                "query": f"{base} 定义 基本概念 概述",
            },
            {
                "title": "发展脉络与现状",
                "intent": "了解该方向的历史演变、关键里程碑与当前进展",
                "query": f"{base} 发展历程 最新进展 现状",
            },
            {
                "title": "关键技术与方法",
                "intent": "识别关键算法 / 架构 / 方法的原理与权衡",
                "query": f"{base} 关键技术 核心方法 实现原理",
            },
            {
                "title": "典型应用与案例",
                "intent": "了解实际部署、代表性产品与典型场景",
                "query": f"{base} 应用 案例 落地",
            },
            {
                "title": "挑战、争议与展望",
                "intent": "识别现有瓶颈、争议点与未来方向",
                "query": f"{base} 挑战 局限 未来",
            },
        ]

        return [
            TodoItem(id=idx, title=a["title"], intent=a["intent"], query=a["query"])
            for idx, a in enumerate(angles, start=1)
        ]

    # 向后兼容：旧代码可能仍调用单数版本（仅供测试桩或外部插件）
    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        return PlannerAgent.create_fallback_tasks(state)[0]

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
        """在文本中定位并解析 JSON 对象或数组。

        宽容地处理模型常见的“边角输出”：被 ```json ... ``` 包裹的代码块、
        前后多余的解释性文字、半角/全角混排引号等。
        """
        cleaned = text.strip()

        # 1) 优先尝试 ```json ... ``` / ``` ... ``` 代码块中的内容
        fence_match = re.search(
            r"```(?:json|JSON)?\s*([\s\S]+?)```",
            cleaned,
        )
        if fence_match:
            inner = fence_match.group(1).strip()
            for candidate in self._candidate_substrings(inner):
                parsed = self._safe_json_loads(candidate)
                if parsed is not None:
                    return parsed

        # 2) 直接在全文中寻找最外层 {...} 或 [...]
        for candidate in self._candidate_substrings(cleaned):
            parsed = self._safe_json_loads(candidate)
            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _candidate_substrings(text: str) -> List[str]:
        """生成可能的 JSON 候选片段（按出现顺序，先对象后数组）。"""
        candidates: List[str] = []
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end != -1 and end > start:
                candidates.append(text[start : end + 1])
        return candidates

    @staticmethod
    def _safe_json_loads(candidate: str) -> Optional[dict[str, Any] | list]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
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
