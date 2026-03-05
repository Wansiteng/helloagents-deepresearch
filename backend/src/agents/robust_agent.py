"""鲁棒工具调用 Agent — 针对本地开源模型的三层防御解析 + 自我纠错闭环。

功能概述
--------
本模块实现了专为 Qwen、Llama 等本地开源模型调优的 ``RobustToolAwareAgent``，
解决其在结构化输出场景下常见的以下问题：

问题一：JSON 格式错误
    LLM 在 ``content`` 字段中写入了含有未转义双引号、换行符的长摘要文本，
    或错误地使用了单引号（``'``），导致 ``json.loads`` 抛出 ``JSONDecodeError``。

问题二：工具调用验证失败
    ``action`` 字段因解析失败而丢失，触发 ``NoteTool.validate_parameters``
    返回 ``False``，最终工具返回 ``❌ 参数验证失败``。

三层防御解析策略（``_parse_tool_parameters``）
----------------------------------------------
1. **标准 JSON**（快速路径）：直接 ``json.loads``。
2a. **JSON 修复**（截断法）：截断到最后一个合法闭合的 ``}``，再次尝试。
2b. **单引号规范化**：将单引号替换为双引号后再次尝试（含截断组合）。
3. **正则逐字段提取**（最终兜底）：从非结构化文本中提取已知字段，
   确保至少能获得 ``action`` 等关键字段。

LLM 自我纠错闭环（``_execute_tool_call``）
------------------------------------------
当所有解析策略均失败，或工具执行后返回验证失败时，
系统不直接放弃，而是将错误原因转化为自然语言反馈，
通过独立的纠错请求让模型自我修正，最多重试
``self_correction_max_retries`` 次（默认 0，即 open_source_mode 下为 2）。
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any

from hello_agents import ToolAwareSimpleAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 自我纠错反馈消息模板
# ---------------------------------------------------------------------------

_SELF_CORRECTION_HEADER = (
    "❌ 工具调用格式错误，请按以下提示修正后重新输出完整的 [TOOL_CALL:...] 调用。\n\n"
)

_CORRECTION_HINT_INVALID_JSON = (
    "你输出的 JSON 参数无法被解析。常见原因：\n"
    "  • 使用了单引号（'）而非双引号（\"）\n"
    "  • content 字段中包含了未转义的双引号或换行符\n"
    "  • JSON 对象未正确闭合（缺少 }} 或 ]）\n"
    "  • 在 [TOOL_CALL:...] 外层包裹了 ```json ... ``` 代码块\n\n"
    "请重新输出严格符合以下格式的单行工具调用（不要道歉或解释）：\n"
    "  [TOOL_CALL:<工具名>:{{\"action\":\"<动作>\", ... 其他参数 ...}}]\n\n"
    "原始错误信息：{error}"
)

_CORRECTION_HINT_MISSING_PARAMS = (
    "工具调用的 JSON 格式正确，但缺少必要字段：{missing_fields}。\n\n"
    "请在重新输出时确保这些字段都存在，例如：\n"
    "  [TOOL_CALL:{tool_name}:{{\"action\":\"create\", \"task_id\":1, "
    "\"title\":\"任务标题\", \"note_type\":\"task_state\", "
    "\"tags\":[\"deep_research\",\"task_1\"], \"content\":\"状态描述\"}}]\n\n"
    "请重新输出完整的合法工具调用（不要道歉或解释）。"
)

_CORRECTION_HINT_TOOL_FAILED = (
    "工具执行失败，返回：{result}\n\n"
    "请检查你的参数是否符合工具规范，然后重新输出完整的合法 [TOOL_CALL:...] 调用。"
)


class RobustToolAwareAgent(ToolAwareSimpleAgent):
    """ToolAwareSimpleAgent 子类，具备三层防御 JSON 解析与 LLM 自我纠错能力。

    Parameters
    ----------
    self_correction_max_retries:
        工具调用失败时，最多允许向 LLM 发起自我纠错请求的次数。
        设为 0 则禁用自我纠错（与旧行为兼容）。
    *args, **kwargs:
        其余参数透传给 ``ToolAwareSimpleAgent``。
    """

    def __init__(self, *args: Any, self_correction_max_retries: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._self_correction_max_retries = self_correction_max_retries

    # ------------------------------------------------------------------
    # 覆盖 1：三层防御性工具参数解析
    # ------------------------------------------------------------------

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict[str, Any]:  # type: ignore[override]
        """带容错的工具参数解析（三层防御）。

        解析流程
        --------
        1. 标准 ``json.loads``（快速路径）。
        2a. JSON 截断修复 — 截断到最后合法闭合 ``}``。
        2b. 单引号规范化 — 替换单引号后再次尝试（含截断组合）。
        3. 正则逐字段提取（最终兜底）。
        4. 若以上全部失败，降级到父类 key=value 解析。
        """
        stripped = parameters.strip()

        if stripped.startswith("{"):
            # ── 层 1：标准 JSON ─────────────────────────────────────────
            try:
                return self._convert_parameter_types(
                    tool_name, json.loads(stripped)
                )
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Tool %s: json.loads failed (%s), entering repair cascade …",
                    tool_name, exc,
                )

            # ── 层 2a：截断法 JSON 修复 ──────────────────────────────────
            repaired = _repair_json_truncate(stripped)
            if repaired:
                try:
                    result = json.loads(repaired)
                    logger.info(
                        "Tool %s: truncation repair succeeded, action=%s",
                        tool_name, result.get("action"),
                    )
                    return self._convert_parameter_types(tool_name, result)
                except json.JSONDecodeError:
                    pass

            # ── 层 2b：单/双引号规范化 ───────────────────────────────────
            normalized = _normalize_quotes(stripped)
            if normalized != stripped:
                try:
                    result = json.loads(normalized)
                    logger.info(
                        "Tool %s: quote normalization succeeded, action=%s",
                        tool_name, result.get("action"),
                    )
                    return self._convert_parameter_types(tool_name, result)
                except json.JSONDecodeError:
                    # 对规范化后的字符串再做截断修复
                    repaired2 = _repair_json_truncate(normalized)
                    if repaired2:
                        try:
                            result = json.loads(repaired2)
                            logger.info(
                                "Tool %s: quote-norm + truncation succeeded, action=%s",
                                tool_name, result.get("action"),
                            )
                            return self._convert_parameter_types(tool_name, result)
                        except json.JSONDecodeError:
                            pass

            # ── 层 3：正则逐字段提取 ─────────────────────────────────────
            extracted = _extract_fields_regex(stripped)
            if extracted:
                logger.info(
                    "Tool %s: regex extraction produced %d fields: %s",
                    tool_name, len(extracted), list(extracted.keys()),
                )
                return self._convert_parameter_types(tool_name, extracted)

            logger.error(
                "Tool %s: ALL parse strategies failed. "
                "Raw params (truncated): %.300s",
                tool_name, stripped,
            )

        # ── 降级：父类 key=value 解析（非 JSON 格式）────────────────────
        return super()._parse_tool_parameters(tool_name, parameters)

    # ------------------------------------------------------------------
    # 覆盖 2：工具执行 + 自我纠错闭环
    # ------------------------------------------------------------------

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:  # type: ignore[override]
        """执行工具调用；失败时触发 LLM 自我纠错重试。

        自我纠错流程
        -----------
        1. 首次执行工具调用。
        2. 若结果包含 ``❌`` 或 ``参数验证失败``：
           a. 构建自然语言错误反馈消息。
           b. 向 LLM 发送修正请求（独立对话，不污染主历史）。
           c. 解析 LLM 返回的新工具调用，替换原始参数，重新执行。
           d. 最多重试 ``_self_correction_max_retries`` 次。
        3. 记录每次成功/失败的详细日志。
        """
        attempt = 0
        current_params = parameters
        last_result = ""

        while True:
            try:
                result = super()._execute_tool_call(tool_name, current_params)
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(
                    "Tool %s: exception during execution (attempt %d): %s\n%s",
                    tool_name, attempt, exc, tb,
                )
                result = f"❌ 工具调用异常: {exc}"

            failed = "❌" in result or "参数验证失败" in result

            if not failed:
                if attempt > 0:
                    logger.info(
                        "Tool %s: succeeded after %d self-correction attempt(s).",
                        tool_name, attempt,
                    )
                else:
                    logger.debug("Tool call OK — tool=%s result_prefix=%.80s", tool_name, result)
                return result

            last_result = result
            logger.error(
                "Tool call FAILED — tool=%s attempt=%d result=%r raw_params(truncated)=%.300s",
                tool_name, attempt, result, current_params,
            )

            # ── 超过最大重试次数，放弃 ───────────────────────────────────
            if attempt >= self._self_correction_max_retries:
                logger.error(
                    "Tool %s: giving up after %d self-correction attempt(s). Last result: %r",
                    tool_name, attempt, last_result,
                )
                return last_result

            # ── 构建自然语言纠错反馈并请求 LLM 修正 ─────────────────────
            attempt += 1
            feedback = self._build_correction_feedback(
                tool_name=tool_name,
                parameters=current_params,
                tool_result=result,
                attempt=attempt,
            )
            logger.info(
                "Tool %s: sending self-correction request (attempt %d/%d) …",
                tool_name, attempt, self._self_correction_max_retries,
            )

            try:
                corrected_params = self._request_self_correction(
                    tool_name=tool_name,
                    original_params=current_params,
                    feedback=feedback,
                )
                if corrected_params is not None:
                    current_params = corrected_params
                    logger.info(
                        "Tool %s: LLM provided corrected params (attempt %d).",
                        tool_name, attempt,
                    )
                else:
                    logger.warning(
                        "Tool %s: LLM did not produce a correctable tool call "
                        "in self-correction attempt %d.",
                        tool_name, attempt,
                    )
            except Exception as exc:
                logger.error(
                    "Tool %s: self-correction LLM call raised exception: %s",
                    tool_name, exc,
                )

    # ------------------------------------------------------------------
    # 内部辅助：自我纠错
    # ------------------------------------------------------------------

    def _build_correction_feedback(
        self,
        *,
        tool_name: str,
        parameters: str,
        tool_result: str,
        attempt: int,
    ) -> str:
        """将工具调用失败信息转化为自然语言纠错反馈。

        策略
        ----
        - 若参数本身无法被解析（JSON 层面失败），给出 JSON 格式提示。
        - 若参数能被解析但缺少必填字段，列出缺失字段。
        - 否则直接呈现工具返回的错误消息。
        """
        parsed: dict[str, Any] | None = None
        parse_error: str = ""
        stripped = parameters.strip()
        if stripped.startswith("{"):
            for candidate in [
                stripped,
                _repair_json_truncate(stripped) or "",
                _normalize_quotes(stripped),
            ]:
                if not candidate:
                    continue
                try:
                    parsed = json.loads(candidate)
                    break
                except json.JSONDecodeError as exc:
                    parse_error = str(exc)

        header = _SELF_CORRECTION_HEADER + f"（第 {attempt} 次纠错）\n\n"

        if parsed is None:
            body = _CORRECTION_HINT_INVALID_JSON.format(error=parse_error or "未知解析错误")
        else:
            missing: list[str] = []
            if tool_name == "note" and "action" not in parsed:
                missing.append("action")
            if missing:
                body = _CORRECTION_HINT_MISSING_PARAMS.format(
                    missing_fields=", ".join(missing),
                    tool_name=tool_name,
                )
            else:
                body = _CORRECTION_HINT_TOOL_FAILED.format(result=tool_result)

        return header + body

    def _request_self_correction(
        self,
        *,
        tool_name: str,
        original_params: str,
        feedback: str,
    ) -> str | None:
        """向 LLM 发送纠错请求，提取并返回修正后的工具参数字符串。

        实现方式
        --------
        通过调用 ``self.llm.chat`` 发起一次独立的纠错对话，
        不污染主对话历史，保持 Agent 状态干净。

        Returns
        -------
        str | None
            修正后的 JSON 参数字符串（不含外层 ``[TOOL_CALL:...]`` 包装），
            若 LLM 未输出有效工具调用则返回 ``None``。
        """
        correction_messages = [
            {
                "role": "user",
                "content": (
                    f"{feedback}\n\n"
                    f"原始工具名：{tool_name}\n"
                    f"原始参数（供参考）：{original_params[:500]}"
                ),
            }
        ]

        try:
            llm = getattr(self, "llm", None)
            if llm is None:
                logger.warning("RobustToolAwareAgent: no llm attribute, skipping LLM correction.")
                return None

            response = llm.chat(messages=correction_messages)
            if not response:
                return None

            content = response if isinstance(response, str) else str(response)
            return _extract_corrected_params(tool_name, content)

        except Exception as exc:
            logger.error("Self-correction LLM call failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------

def _repair_json_truncate(text: str) -> str | None:
    """截断法 JSON 修复：找到最外层合法闭合 ``}``，截断并返回候选字符串。

    使用状态机正确处理字符串内的转义序列与引号嵌套，避免误判。
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
                        break

        if last_closed_pos > 0:
            candidate = text[: last_closed_pos + 1]
            json.loads(candidate)  # 验证候选串合法性
            return candidate
    except Exception:
        pass
    return None


def _normalize_quotes(text: str) -> str:
    """将作为 JSON 定界符的单引号替换为双引号（处理常见 LLM 输出错误）。

    使用状态机扫描，区分定界符单引号与字符串内容中的单引号，
    仅替换起到分隔作用的单引号。不含单引号时原样返回（快速路径）。
    """
    if "'" not in text:
        return text

    result: list[str] = []
    i = 0
    n = len(text)
    in_double = False
    in_single = False

    while i < n:
        ch = text[i]

        if ch == "\\" and (in_double or in_single):
            # 转义序列：原样保留
            result.append(ch)
            i += 1
            if i < n:
                result.append(text[i])
                i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            result.append(ch)
        elif ch == "'" and not in_double:
            # 定界符单引号 → 替换为双引号
            in_single = not in_single
            result.append('"')
        else:
            result.append(ch)

        i += 1

    return "".join(result)


# 已知字段的正则提取模式（同时支持单/双引号键名）
_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "action":    re.compile(r"""["']action["']\s*:\s*["']([^"']+)["']"""),
    "note_id":   re.compile(r"""["']note_id["']\s*:\s*["']([^"']+)["']"""),
    "note_type": re.compile(r"""["']note_type["']\s*:\s*["']([^"']+)["']"""),
    "title":     re.compile(r"""["']title["']\s*:\s*["']([^"']{1,200})["']"""),
    "tags":      re.compile(r"""["']tags["']\s*:\s*(\[[^\]]*\])"""),
    "content":   re.compile(r"""["']content["']\s*:\s*["']([^"']{1,500})["']"""),
    "query":     re.compile(r"""["']query["']\s*:\s*["']([^"']{1,300})["']"""),
}
_TASK_ID_PATTERN = re.compile(r"""["']task_id["']\s*:\s*(\d+)""")


def _extract_fields_regex(text: str) -> dict[str, Any]:
    """从原始字符串中用正则逐字段提取已知字段（最终兜底解析）。"""
    result: dict[str, Any] = {}

    for field, pattern in _FIELD_PATTERNS.items():
        m = pattern.search(text)
        if m:
            if field == "tags":
                try:
                    result[field] = json.loads(m.group(1))
                except json.JSONDecodeError:
                    raw = m.group(1).strip("[] ")
                    result[field] = [
                        t.strip().strip("\"'")
                        for t in raw.split(",")
                        if t.strip()
                    ]
            else:
                result[field] = m.group(1)

    m2 = _TASK_ID_PATTERN.search(text)
    if m2:
        try:
            result["task_id"] = int(m2.group(1))
        except ValueError:
            pass

    return result


# 从 LLM 自我纠错响应中提取工具调用的正则
_TOOL_CALL_RE = re.compile(
    r"\[TOOL_CALL\s*:\s*(?P<name>[^\s:\]]+)\s*:\s*(?P<params>\{.*?\})\s*\]",
    re.DOTALL,
)


def _extract_corrected_params(tool_name: str, llm_response: str) -> str | None:
    """从 LLM 自我纠错响应中提取目标工具名的 JSON 参数字符串。

    Parameters
    ----------
    tool_name:
        期望匹配的工具名称（大小写不敏感）。
    llm_response:
        LLM 返回的完整响应文本。

    Returns
    -------
    str | None
        提取到的 JSON 参数字符串；未找到则返回 ``None``。
    """
    for m in _TOOL_CALL_RE.finditer(llm_response):
        name = m.group("name").strip()
        params = m.group("params").strip()
        if name.lower() == tool_name.lower():
            logger.debug(
                "Self-correction: extracted params for tool '%s': %.200s",
                tool_name, params,
            )
            return params

    # 宽松兜底：返回第一个找到的工具调用参数
    m = _TOOL_CALL_RE.search(llm_response)
    if m:
        params = m.group("params").strip()
        logger.debug(
            "Self-correction: name mismatch, using first found params: %.200s", params
        )
        return params

    return None
