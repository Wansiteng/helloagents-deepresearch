"""State models used by the deep research workflow."""

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from typing_extensions import Annotated


class SSEEventType(str, Enum):
    """SSE 事件类型枚举。

    后端通过 ``/research/stream`` 推送的所有事件均使用此枚举中的 ``type`` 字段，
    前端可据此分发处理逻辑，避免魔法字符串散落在各处。

    事件流顺序（典型）
    ------------------
    1. :attr:`STATUS` — 初始化提示
    2. :attr:`TODO_LIST` — 规划 Agent 生成的任务列表
    3. :attr:`TASK_STATUS` (``in_progress``) — 单任务开始执行
    4. :attr:`TOOL_CALL` — Agent 调用工具（笔记读写等）
    5. :attr:`SOURCES` — 搜索结果汇总
    6. :attr:`TASK_SUMMARY_CHUNK` — 摘要 Agent 逐 token 流式输出
    7. :attr:`TASK_STATUS` (``completed`` / ``skipped`` / ``failed``) — 单任务结束
    8. :attr:`REPORT_NOTE` — 报告写入笔记
    9. :attr:`FINAL_REPORT` — Writer Agent 生成最终报告
    10. :attr:`DONE` — 流结束信号
    11. :attr:`ERROR` — 异常（任意阶段）
    """

    STATUS = "status"
    """全局状态提示，如"初始化研究流程"。"""

    TODO_LIST = "todo_list"
    """Planner Agent 输出的任务列表（含 id / title / intent / query）。"""

    TASK_STATUS = "task_status"
    """单任务状态变更：``in_progress`` / ``completed`` / ``skipped`` / ``failed``。"""

    SOURCES = "sources"
    """搜索引擎返回的来源列表摘要，附带原始上下文。"""

    TASK_SUMMARY_CHUNK = "task_summary_chunk"
    """Summarizer Agent 流式输出的摘要文本片段（逐 token）。"""

    TOOL_CALL = "tool_call"
    """Agent 调用工具的详情（工具名、参数、返回值）。"""

    REPORT_NOTE = "report_note"
    """最终报告写入笔记工具的确认事件。"""

    FINAL_REPORT = "final_report"
    """Writer Agent 生成的完整 Markdown 研究报告。"""

    DONE = "done"
    """流式传输结束信号，前端收到后可关闭 SSE 连接。"""

    ERROR = "error"
    """任意阶段发生异常时推送的错误事件。"""

    SECTION_DRAFT = "section_draft"
    """单任务完成后立即推送的渐进式章节草稿（无额外 LLM 调用，直接格式化摘要）。"""

    DYNAMIC_TASKS = "dynamic_tasks"
    """动态规划 Agent 发现研究空白后补充的新任务列表。"""

    REFLECTION = "reflection"
    """反思 Agent 对最终报告的质量评审结果（评分 + 空白列表）。"""


@dataclass(kw_only=True)
class TodoItem:
    """单个待办任务项。"""

    id: int
    title: str
    intent: str
    query: str
    status: str = field(default="pending")
    summary: Optional[str] = field(default=None)
    sources_summary: Optional[str] = field(default=None)
    notices: list[str] = field(default_factory=list)
    note_id: Optional[str] = field(default=None)
    note_path: Optional[str] = field(default=None)
    stream_token: Optional[str] = field(default=None)
    section_draft: Optional[str] = field(default=None)
    """渐进式报告：任务完成后格式化的章节草稿（无 LLM 调用）。"""


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)  # Report topic
    search_query: str = field(default=None)  # Deprecated placeholder
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)  # Research loop count
    running_summary: str = field(default=None)  # Legacy summary field
    todo_items: Annotated[list, operator.add] = field(default_factory=list)
    structured_report: Optional[str] = field(default=None)
    report_note_id: Optional[str] = field(default=None)
    report_note_path: Optional[str] = field(default=None)


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)  # Backward-compatible文本
    report_markdown: Optional[str] = field(default=None)
    todo_items: List[TodoItem] = field(default_factory=list)

