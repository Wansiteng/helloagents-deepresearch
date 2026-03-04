"""鲁棒工具调用 Agent — 对 LLM 生成的 JSON 参数做容错解析。

问题背景
--------
``ToolAwareSimpleAgent`` 中的 ``_parse_tool_parameters`` 使用标准
``json.loads`` 解析 ``[TOOL_CALL:note:{...}]`` 内的 JSON 参数。当 LLM 在
``content`` 字段中写入了含有**未转义双引号、换行符**的长摘要文本时，
``json.loads`` 会抛出 ``JSONDecodeError``，导致解析结果为空字典，
``action`` 字段因此丢失 → ``NoteTool.validate_parameters`` 返回 ``False``
→ 工具返回 ``❌ 参数验证失败``。

修复策略
--------
覆盖 ``_parse_tool_parameters``，增加两层兜底：

1. **JSON 修复**：截断到最后一个能合法闭合的 ``}``，再次尝试解析。
2. **正则提取**：当 JSON 无法修复时，用正则从原始文本中逐字段提取
   ``action``、``note_id``、``task_id``、``title``、``note_type``、
   ``tags``、``content`` 等已知字段，确保至少能获得 ``action``。

同时覆盖 ``_execute_tool_call`` 来增强日志，便于后续排查。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from hello_agents import ToolAwareSimpleAgent

logger = logging.getLogger(__name__)


class RobustToolAwareAgent(ToolAwareSimpleAgent):
    """ToolAwareSimpleAgent 子类，具备容错的工具参数解析能力。

    所有工具调用均与父类相同；唯一的差异在于 ``_parse_tool_parameters``
    在 ``json.loads`` 失败时会尝试 JSON 修复与正则逐字段提取，从而避免
    由 LLM 生成的 ``content`` 含有特殊字符导致的「参数验证失败」问题。
    """

    # ------------------------------------------------------------------
    # 覆盖：容错参数解析
    # ------------------------------------------------------------------

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict[str, Any]:  # type: ignore[override]
        """带容错的工具参数解析。

        解析流程
        --------
        1. 标准 ``json.loads``（快速路径）。
        2. JSON 修复后再次 ``json.loads``（截断到最后合法 ``}``）。
        3. 正则逐字段提取（最后兜底）。
        4. 若以上全部失败，降级到父类 key=value 解析。
        """
        stripped = parameters.strip()

        if stripped.startswith("{"):
            # ── 快速路径：标准 JSON ───────────────────────────────────
            try:
                param_dict = json.loads(stripped)
                logger.debug(
                    "Tool %s: JSON parsed OK, action=%s",
                    tool_name, param_dict.get("action"),
                )
                return self._convert_parameter_types(tool_name, param_dict)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Tool %s: json.loads failed (%s), trying JSON repair …",
                    tool_name, exc,
                )

            # ── 修复路径：截断到最后合法闭合位置 ────────────────────
            repaired = _repair_json(stripped)
            if repaired:
                try:
                    param_dict = json.loads(repaired)
                    logger.info(
                        "Tool %s: JSON repaired successfully, action=%s",
                        tool_name, param_dict.get("action"),
                    )
                    return self._convert_parameter_types(tool_name, param_dict)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Tool %s: repaired JSON still invalid (%s), falling back to regex",
                        tool_name, exc,
                    )

            # ── 正则兜底：逐字段提取已知字段 ─────────────────────────
            param_dict = _extract_fields_regex(stripped)
            if param_dict:
                logger.info(
                    "Tool %s: regex extraction produced %d fields: %s",
                    tool_name, len(param_dict), list(param_dict.keys()),
                )
                return self._convert_parameter_types(tool_name, param_dict)

            logger.error(
                "Tool %s: all JSON parsing strategies failed, "
                "raw parameters (truncated): %.200s",
                tool_name, stripped,
            )

        # ── 降级到父类 key=value 解析（非 JSON 格式）────────────────
        return super()._parse_tool_parameters(tool_name, parameters)

    # ------------------------------------------------------------------
    # 覆盖：增强 execute 日志
    # ------------------------------------------------------------------

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:  # type: ignore[override]
        """执行工具调用，出错时记录详细日志。"""
        result = super()._execute_tool_call(tool_name, parameters)
        if "参数验证失败" in result or "❌" in result:
            logger.error(
                "Tool call FAILED — tool=%s result=%r raw_params(truncated)=%.300s",
                tool_name, result, parameters,
            )
        else:
            logger.debug("Tool call OK — tool=%s result_prefix=%.80s", tool_name, result)
        return result


# ---------------------------------------------------------------------------
# 模块级辅助函数（不依赖任何 Agent 状态）
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str | None:
    """尝试修复常见的 LLM JSON 错误（截断法）。

    策略：扫描字符串，在最后一个能合法闭合的 ``}`` 处截断，再返回候选串。
    仅处理最外层对象为 ``{...}`` 的情况。
    """
    try:
        depth = 0
        in_string = False
        escape_next = False
        quote_char: str | None = None
        last_closed_pos = -1

        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch in ('"', "'"):
                if not in_string:
                    in_string = True
                    quote_char = ch
                elif ch == quote_char:
                    in_string = False
                    quote_char = None
                continue
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        last_closed_pos = i
                        break  # 找到最外层闭合即可

        if last_closed_pos > 0:
            candidate = text[: last_closed_pos + 1]
            # 验证候选串
            json.loads(candidate)
            return candidate
    except Exception:
        pass
    return None


_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "action":    re.compile(r'"action"\s*:\s*"([^"]+)"'),
    "note_id":   re.compile(r'"note_id"\s*:\s*"([^"]+)"'),
    "note_type": re.compile(r'"note_type"\s*:\s*"([^"]+)"'),
    "title":     re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    "tags":      re.compile(r'"tags"\s*:\s*(\[[^\]]*\])'),
    # content 用贪婪截断——取第一个 `"` 到 下一个明确边界（field 名或 `}`）
    "content":   re.compile(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"'),
}
_TASK_ID_PATTERN = re.compile(r'"task_id"\s*:\s*(\d+)')


def _extract_fields_regex(text: str) -> dict[str, Any]:
    """从原始 JSON 字符串中用正则逐字段提取已知字段。"""
    result: dict[str, Any] = {}

    for field, pattern in _FIELD_PATTERNS.items():
        m = pattern.search(text)
        if m:
            if field == "tags":
                try:
                    result[field] = json.loads(m.group(1))
                except json.JSONDecodeError:
                    # 降级：逗号分割
                    raw = m.group(1).strip("[] ")
                    result[field] = [t.strip().strip('"') for t in raw.split(",") if t.strip()]
            else:
                result[field] = m.group(1)

    m = _TASK_ID_PATTERN.search(text)
    if m:
        try:
            result["task_id"] = int(m.group(1))
        except ValueError:
            pass

    return result
