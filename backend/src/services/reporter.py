"""Writer Agent — 专精逻辑组织与长文生成的单一职责 Agent。

该 Agent 汇总所有子任务的摘要与来源，生成完整的结构化 Markdown 研究报告。
是研究流水线的最后一环，消费 PlannerAgent 和 SummarizerAgent 的输出。

**单一职责**：仅负责报告的逻辑组织与长文写作，不执行搜索或摘要。

动态按需检索（Query-based Retrieval）
--------------------------------------
当 ``vector_store`` 可用时，`generate_report` 会自动切换为"分章节生成"策略：

1. **细粒度上下文组装**：逐章节生成，每次只将与当前章节最相关的片段送入 LLM。
2. **动态 Top-K 召回**：以"研究主题 + 章节标题"为 Query，从 ChromaDB 中语义检索
   Top-K（由 ``vector_top_k`` 配置）最相关文本块，实现按需供给。
3. **缓解注意力丢失**：每次生成仅关注当前章节所需的纯净知识，从根本上缓解
   长文生成时的"中间注意力丢失（Lost in the Middle）"现象，提高事实准确率。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from hello_agents import ToolAwareSimpleAgent

from config import Configuration
from models import SummaryState
from prompts import REPORT_SECTIONS, get_current_date, report_section_writer_instructions
from services.text_processing import strip_tool_calls
from utils import strip_thinking_tokens

if TYPE_CHECKING:
    from services.vector_store import VectorStore

logger = logging.getLogger(__name__)


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

    动态按需检索（Query-based Retrieval）
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    当传入 ``vector_store`` 时，自动启用分章节生成策略。每个章节独立召回
    Top-K 语义相关片段，显著减少无关上下文对 LLM 注意力的干扰。
    """

    def __init__(
        self,
        report_agent: ToolAwareSimpleAgent,
        config: Configuration,
        vector_store: Optional["VectorStore"] = None,
    ) -> None:
        self._agent = report_agent
        self._config = config
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def generate_report(self, state: SummaryState) -> str:
        """根据已完成的任务状态生成完整研究报告。

        当 vector_store 可用且已有存储内容时，自动切换为分章节生成策略
        （动态 Top-K 召回），否则回退到全量提示的传统策略。

        参数
        ----
        state:
            研究状态，``todo_items`` 中各任务的 ``summary`` 与
            ``sources_summary`` 应已填充完毕。

        返回
        ----
        Markdown 格式的完整研究报告字符串。
        """
        # ── 优先使用分章节动态检索策略 ───────────────────────────────
        if self._vector_store is not None and self._vector_store.count() > 0:
            logger.info(
                "WriterAgent: 启用分章节 Query-based Retrieval 策略 "
                "(vector_store chunks=%d, top_k=%d)",
                self._vector_store.count(),
                self._config.vector_top_k,
            )
            return self._generate_report_section_by_section(state)

        # ── 回退：传统全量提示策略 ───────────────────────────────────
        logger.info("WriterAgent: 使用传统全量提示策略（vector_store 未启用或为空）")
        return self._generate_report_fullcontext(state)

    # ------------------------------------------------------------------
    # 分章节生成（动态 Top-K 召回）
    # ------------------------------------------------------------------

    def _generate_report_section_by_section(self, state: SummaryState) -> str:
        """分章节生成报告，每节独立进行语义检索。

        策略
        ~~~~
        1. 遍历 REPORT_SECTIONS 中定义的每个章节。
        2. 以"研究主题 + 章节标题 + 章节目标"构建 Query。
        3. 从 ChromaDB 中语义检索 Top-K 最相关片段（动态 Top-K 召回）。
        4. 将检索到的片段 + 任务来源概览拼装为单章节提示，驱动 LLM 生成该节内容。
        5. 最终将各章节内容拼接为完整报告。
        """
        assert self._vector_store is not None  # 调用前已检查

        # 汇总所有任务的来源概览，供"参考来源"等章节使用
        sources_summary = self._build_sources_summary(state)
        topic = state.research_topic
        top_k = self._config.vector_top_k
        current_date = get_current_date()

        report_title = f"# 研究报告：{topic}\n"
        sections_output: list[str] = [report_title]

        for section_title, section_goal, section_heading in REPORT_SECTIONS:
            # ── 1. 构建语义检索 Query ────────────────────────────────
            query = f"{topic} {section_title} {section_goal}"

            # ── 2. Top-K 动态召回 ────────────────────────────────────
            try:
                hits = self._vector_store.query(query, n_results=top_k)
            except Exception as exc:
                logger.warning(
                    "WriterAgent: 章节'%s'向量检索失败，跳过补充上下文: %s",
                    section_title,
                    exc,
                )
                hits = []

            if hits:
                retrieved_context = "\n\n---\n\n".join(
                    f"**[知识片段 {i + 1}]** (相关度分数: {1 - hit['distance']:.3f})\n{hit['text']}"
                    for i, hit in enumerate(hits)
                )
                logger.info(
                    "WriterAgent: 章节'%s' 召回 %d 条片段 (top_k=%d)",
                    section_title,
                    len(hits),
                    top_k,
                )
            else:
                retrieved_context = "（未检索到相关知识片段，请根据研究主题进行合理推断。）"
                logger.info("WriterAgent: 章节'%s' 无命中片段", section_title)

            # ── 3. 组装单章节提示 ────────────────────────────────────
            section_prompt = report_section_writer_instructions.format(
                section_title=section_title,
                section_goal=section_goal,
                section_heading=section_heading,
                research_topic=topic,
                current_date=current_date,
                retrieved_context=retrieved_context,
                sources_summary=sources_summary,
            )

            # ── 4. LLM 生成该章节 ────────────────────────────────────
            section_text = self._run_agent_once(section_prompt)
            sections_output.append(section_text)

        # ── 5. 拼接完整报告 ──────────────────────────────────────────
        full_report = "\n\n".join(sections_output)
        return full_report or "报告生成失败，请检查输入。"

    def _run_agent_once(self, prompt: str) -> str:
        """调用 LLM 生成一次文本，并清理思考 token 与工具调用残留。"""
        response = self._agent.run(prompt)
        self._agent.clear_history()

        text = response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        return strip_tool_calls(text).strip()

    # ------------------------------------------------------------------
    # 传统全量提示策略（向后兼容）
    # ------------------------------------------------------------------

    def _generate_report_fullcontext(self, state: SummaryState) -> str:
        """传统策略：将全部任务摘要一次性拼入提示，适用于 vector_store 未启用时。"""
        prompt = self._build_fullcontext_prompt(state)

        response = self._agent.run(prompt)
        self._agent.clear_history()

        report_text = response.strip()
        if self._config.strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        report_text = strip_tool_calls(report_text).strip()
        return report_text or "报告生成失败，请检查输入。"

    def _build_fullcontext_prompt(self, state: SummaryState) -> str:
        """构造传统 Writer Agent 的完整用户提示（全量上下文）。

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

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _build_sources_summary(self, state: SummaryState) -> str:
        """将所有任务的来源概览整合为字符串，用于"参考来源"章节。"""
        parts: list[str] = []
        for task in state.todo_items:
            if task.sources_summary:
                parts.append(f"**任务 {task.id}《{task.title}》**\n{task.sources_summary}")
        return "\n\n".join(parts) if parts else "暂无来源信息"


# ── 向后兼容别名 ───────────────────────────────────────────────────────
#: ``ReportingService`` 已重命名为 ``WriterAgent``；此别名保持导入向后兼容。
ReportingService = WriterAgent

