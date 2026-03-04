"""Writer Agent — 专精逻辑组织与长文生成的单一职责 Agent。

该 Agent 汇总所有子任务的摘要与来源，生成完整的结构化 Markdown 研究报告。
是研究流水线的最后一环，消费 PlannerAgent 和 SummarizerAgent 的输出。

**单一职责**：仅负责报告的逻辑组织与长文写作，不执行搜索或摘要。
"""

from __future__ import annotations

import json

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import SummaryState
from services.text_processing import strip_tool_calls
from utils import strip_thinking_tokens


class WriterAgent:
    """Writer Agent：汇总任务结果，生成结构化 Markdown 研究报告。

    工具依赖
    --------
    - ``note``（可选）：通过 ``read`` 操作加载各子任务笔记，获取最新摘要；
      通过 ``create`` 操作将最终报告持久化为 ``conclusion`` 类型笔记。

    报告结构
    --------
    按 ``report_writer_instructions`` 模板生成，包含：

    1. 背景概览
    2. 核心洞见（3−5 条，附任务编号）
    3. 证据与数据
    4. 风险与挑战
    5. 参考来源
    """

    def __init__(self, report_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = report_agent
        self._config = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def generate_report(self, state: SummaryState) -> str:
        """根据已完成的任务状态生成完整研究报告。

        参数
        ----
        state:
            研究状态，``todo_items`` 中各任务的 ``summary`` 与
            ``sources_summary`` 应已填充完毕。

        返回
        ----
        Markdown 格式的完整研究报告字符串。
        """
        prompt = self._build_prompt(state)

        response = self._agent.run(prompt)
        self._agent.clear_history()

        report_text = response.strip()
        if self._config.strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        report_text = strip_tool_calls(report_text).strip()

        return report_text or "报告生成失败，请检查输入。"

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_prompt(self, state: SummaryState) -> str:
        """构造 Writer Agent 的完整用户提示。

        将所有子任务的标题、意图、状态、摘要和来源汇总为结构化提示，
        并附上笔记读写指引，使 Agent 可通过 NoteTool 获取更详细的任务信息。
        """
        tasks_block = []
        for task in state.todo_items:
            summary_block = task.summary or "暂无可用信息"
            sources_block = task.sources_summary or "暂无来源"
            tasks_block.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 检索查询：{task.query}\n"
                f"- 执行状态：{task.status}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
            )

        note_references = []
        for task in state.todo_items:
            if task.note_id:
                note_references.append(
                    f"- 任务 {task.id}《{task.title}》：note_id={task.note_id}"
                )

        notes_section = "\n".join(note_references) if note_references else "- 暂无可用任务笔记"

        read_template = json.dumps({"action": "read", "note_id": "<note_id>"}, ensure_ascii=False)
        create_conclusion_template = json.dumps(
            {
                "action": "create",
                "title": f"研究报告：{state.research_topic}",
                "note_type": "conclusion",
                "tags": ["deep_research", "report"],
                "content": "请在此沉淀最终报告要点",
            },
            ensure_ascii=False,
        )

        return (
            f"研究主题：{state.research_topic}\n"
            f"任务概览：\n{''.join(tasks_block)}\n"
            f"可用任务笔记：\n{notes_section}\n"
            f"请针对每条任务笔记使用格式：[TOOL_CALL:note:{read_template}] 读取内容，"
            f"整合所有信息后撰写报告。\n"
            f"如需输出汇总结论，可追加调用：[TOOL_CALL:note:{create_conclusion_template}] 保存报告要点。"
        )


# ── 向后兼容别名 ───────────────────────────────────────────────────────
#: ``ReportingService`` 已重命名为 ``WriterAgent``；此别名保持导入向后兼容。
ReportingService = WriterAgent

