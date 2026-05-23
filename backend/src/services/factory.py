"""Construction factory for the research service layer.

Both the legacy ``DeepResearchAgent`` and the new ``ResearchSession`` need the
same bundle of objects — LLM, tool registry, vector store, the four agent
services. This module owns that construction so neither orchestrator duplicates
it.

This factory builds ``hello_agents``-based objects and is therefore transitional
glue (it lives in ``services/`` rather than ``core/``). PR-4 removes it together
with the legacy agent layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent

from agents.robust_agent import RobustToolAwareAgent
from config import Configuration
from prompts import (
    open_source_model_constraint_prompt,
    report_writer_instructions,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from services.planner import PlannerAgent
from services.reflection import CriticAgent
from services.reporter import WriterAgent
from services.summarizer import SummarizerAgent
from services.tool_events import ToolCallTracker
from services.vector_store import VectorStore
from tool_registry import AgentToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ResearchServices:
    """The constructed service bundle shared by every orchestrator."""

    config: Configuration
    llm: HelloAgentsLLM
    tool_registry: AgentToolRegistry
    vector_store: VectorStore | None
    tool_tracker: ToolCallTracker
    planner: PlannerAgent
    summarizer: SummarizerAgent
    writer: WriterAgent
    critic: CriticAgent | None
    llm_concurrency: int
    knowledge_sources: list[Any]
    """Enabled :class:`~core.knowledge.KnowledgeSource` instances (web, vault, ...).
    Consumed by the new ``ResearchSession`` orchestrator; ignored by the legacy
    ``DeepResearchAgent``."""


def build_llm(config: Configuration) -> HelloAgentsLLM:
    """Instantiate ``HelloAgentsLLM`` following configuration preferences."""
    llm_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "timeout": config.llm_timeout,
    }

    model_id = config.llm_model_id or config.local_llm
    if model_id:
        llm_kwargs["model"] = model_id

    provider = (config.llm_provider or "").strip()
    if provider:
        llm_kwargs["provider"] = provider

    if provider == "ollama":
        llm_kwargs["base_url"] = config.sanitized_ollama_url()
        llm_kwargs["api_key"] = config.llm_api_key or "ollama"
    elif provider == "lmstudio":
        llm_kwargs["base_url"] = config.lmstudio_base_url
        llm_kwargs["api_key"] = config.llm_api_key or "lm-studio"
    else:
        if config.llm_base_url:
            llm_kwargs["base_url"] = config.llm_base_url
        if config.llm_api_key:
            llm_kwargs["api_key"] = config.llm_api_key

    return HelloAgentsLLM(**llm_kwargs)


def _build_vector_store(config: Configuration) -> VectorStore | None:
    """Construct the optional RAG vector store, or ``None`` if disabled/failed."""
    if not config.use_vector_store:
        return None
    try:
        store = VectorStore(
            workspace=config.vector_store_path,
            embedding_model=config.embedding_model,
            ollama_base_url=config.ollama_base_url,
            chunk_size=config.vector_chunk_size,
            chunk_overlap=config.vector_chunk_overlap,
        )
        logger.info(
            "VectorStore 已启用: path=%s model=%s",
            config.vector_store_path,
            config.embedding_model,
        )
        return store
    except Exception as exc:
        logger.warning("VectorStore 初始化失败，将禁用向量记忆: %s", exc)
        return None


def create_tool_aware_agent(
    *,
    config: Configuration,
    llm: HelloAgentsLLM,
    tool_registry: AgentToolRegistry,
    tool_tracker: ToolCallTracker,
    name: str,
    system_prompt: str,
) -> ToolAwareSimpleAgent:
    """Instantiate a ``RobustToolAwareAgent`` sharing the unified tool registry."""
    ha_registry = tool_registry.hello_agents_registry

    effective_prompt = system_prompt
    if config.use_open_source_mode:
        effective_prompt = system_prompt + "\n" + open_source_model_constraint_prompt
        logger.debug("open_source_mode: injecting constraint prompt for agent '%s'", name)

    if config.no_think_mode:
        effective_prompt = "/no_think\n" + effective_prompt
        logger.debug("no_think_mode: injecting /no_think directive for agent '%s'", name)

    max_retries = (
        config.open_source_model_max_retries if config.use_open_source_mode else 0
    )

    return RobustToolAwareAgent(
        name=name,
        llm=llm,
        system_prompt=effective_prompt,
        enable_tool_calling=ha_registry is not None,
        tool_registry=ha_registry,
        tool_call_listener=tool_tracker.record,
        self_correction_max_retries=max_retries,
    )


def build_research_services(config: Configuration) -> ResearchServices:
    """Construct the full service bundle for a research run."""
    llm = build_llm(config)
    tool_registry = AgentToolRegistry(config)
    vector_store = _build_vector_store(config)
    tool_tracker = ToolCallTracker(
        config.notes_workspace if config.enable_notes else None
    )

    def make_agent(name: str, system_prompt: str) -> ToolAwareSimpleAgent:
        return create_tool_aware_agent(
            config=config,
            llm=llm,
            tool_registry=tool_registry,
            tool_tracker=tool_tracker,
            name=name,
            system_prompt=system_prompt,
        )

    planner_agent = make_agent("研究规划专家", todo_planner_system_prompt.strip())
    writer_agent = make_agent("报告撰写专家", report_writer_instructions.strip())

    def summarizer_factory() -> ToolAwareSimpleAgent:
        return make_agent("任务总结专家", task_summarizer_instructions.strip())

    planner = PlannerAgent(planner_agent, config)
    summarizer = SummarizerAgent(summarizer_factory, config)
    writer = WriterAgent(writer_agent, config, vector_store=vector_store)

    critic: CriticAgent | None = None
    if config.enable_reflection:
        from prompts import reflection_instructions  # noqa: F401 – 验证提示词可导入

        def critic_factory() -> ToolAwareSimpleAgent:
            return make_agent(
                "报告质量评审专家",
                (
                    "你是一名专业的研究质量评审专家，擅长识别研究报告中的不足与空白。"
                    "请对给定的研究报告进行客观评审，输出结构化 JSON 格式的评估结果。"
                ),
            )

        critic = CriticAgent(critic_factory, config)

    knowledge_sources = _build_knowledge_sources(config)

    return ResearchServices(
        config=config,
        llm=llm,
        tool_registry=tool_registry,
        vector_store=vector_store,
        tool_tracker=tool_tracker,
        planner=planner,
        summarizer=summarizer,
        writer=writer,
        critic=critic,
        llm_concurrency=max(1, config.llm_concurrency),
        knowledge_sources=knowledge_sources,
    )


def _build_knowledge_sources(config: Configuration) -> list[Any]:
    """Construct the enabled knowledge sources for the new orchestrator.

    Web search is always enabled. The Obsidian vault source is added only when
    ``OBSIDIAN_VAULT_PATH`` is configured.
    """
    from core.sources.web import WebSearchSource

    sources: list[Any] = [WebSearchSource(config)]

    vault_path = (config.obsidian_vault_path or "").strip()
    if vault_path:
        from core.sources.obsidian import ObsidianVaultSource

        try:
            sources.append(ObsidianVaultSource.from_config(config))
            logger.info("Obsidian 知识源已启用: vault=%s", vault_path)
        except Exception as exc:
            logger.warning("Obsidian 知识源初始化失败，已跳过: %s", exc)

    if getattr(config, "enable_arxiv", False):
        from core.sources.arxiv import ArxivSource

        sources.append(ArxivSource(config))
        logger.info("arXiv 学术源已启用")

    if getattr(config, "enable_openalex", False):
        from core.sources.openalex import OpenAlexSource

        sources.append(OpenAlexSource(config))
        email = (config.openalex_email or "").strip()
        logger.info(
            "OpenAlex 学术源已启用%s",
            f"（polite pool: {email}）" if email else "（建议设置 OPENALEX_EMAIL）",
        )

    return sources
