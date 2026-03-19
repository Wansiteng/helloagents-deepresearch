"""Reflection Agent — 报告质量评审专家。

在 WriterAgent 生成最终报告后，对报告进行独立质量评审，
返回评分与研究空白列表，触发选择性补充研究。

**单一职责**：仅负责已完成报告的质量评估，不执行搜索或写作。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any, Optional

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import SummaryState
from prompts import reflection_instructions
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)


class CriticAgent:
    """Reflection Agent：对最终研究报告进行质量评审。

    工作流程
    --------
    1. 接收研究状态与最终报告文本。
    2. 调用 LLM 对报告进行多维度评审。
    3. 解析 JSON 输出，返回评分、空白列表与补充查询。

    返回格式
    --------
    ::

        {
            "score": 7,
            "needs_more_research": False,
            "gaps": ["..."],
            "additional_queries": []
        }

    评分低于 ``config.reflection_score_threshold`` 且
    ``additional_queries`` 非空时，``needs_more_research`` 为 ``True``。
    """

    def __init__(
        self,
        agent_factory: Callable[[], ToolAwareSimpleAgent],
        config: Configuration,
    ) -> None:
        self._agent_factory = agent_factory
        self._config = config

    def assess_report(self, state: SummaryState, report: str) -> dict[str, Any]:
        """评审最终报告，返回质量评估结果。

        参数
        ----
        state:
            研究状态（仅使用 ``research_topic``）。
        report:
            WriterAgent 生成的完整 Markdown 报告文本。

        返回
        ----
        包含 ``score``、``needs_more_research``、``gaps``、
        ``additional_queries`` 的字典。若 LLM 解析失败则返回合格默认值。
        """
        if not report or not report.strip():
            return self._default_result()

        prompt = reflection_instructions.format(
            research_topic=state.research_topic or "未知主题",
            report=report[:6000],  # 避免超长报告撑爆上下文
        )

        agent = self._agent_factory()
        try:
            response = agent.run(prompt)
        except Exception as exc:
            logger.warning("CriticAgent LLM 调用失败: %s", exc)
            return self._default_result()
        finally:
            agent.clear_history()

        text = response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        result = self._parse_assessment(text)
        threshold = self._config.reflection_score_threshold
        score = result.get("score", 10)
        has_queries = bool(result.get("additional_queries"))
        result["needs_more_research"] = (score < threshold) and has_queries

        logger.info(
            "CriticAgent 评审完成: score=%s needs_more_research=%s gaps=%s",
            score,
            result["needs_more_research"],
            result.get("gaps", []),
        )
        return result

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _parse_assessment(self, text: str) -> dict[str, Any]:
        """解析 LLM 输出的 JSON 评审结果。"""
        payload = self._extract_json(text)
        if not isinstance(payload, dict):
            logger.warning("CriticAgent: JSON 解析失败，使用默认值。原始输出: %s", text[:200])
            return self._default_result()

        score = payload.get("score", 10)
        try:
            score = max(1, min(10, int(score)))
        except (TypeError, ValueError):
            score = 10

        gaps = payload.get("gaps", [])
        if not isinstance(gaps, list):
            gaps = []

        queries = payload.get("additional_queries", [])
        if not isinstance(queries, list):
            queries = []

        return {
            "score": score,
            "needs_more_research": False,  # 由调用方结合 threshold 决定
            "gaps": [str(g) for g in gaps[:3]],
            "additional_queries": [str(q) for q in queries[:2]],
        }

    @staticmethod
    def _extract_json(text: str) -> Optional[dict[str, Any]]:
        """从文本中提取第一个 JSON 对象。"""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start: end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 尝试修复尾部截断
        for i in range(len(candidate) - 1, -1, -1):
            if candidate[i] == "}":
                try:
                    return json.loads(candidate[: i + 1])
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _default_result() -> dict[str, Any]:
        return {
            "score": 10,
            "needs_more_research": False,
            "gaps": [],
            "additional_queries": [],
        }
