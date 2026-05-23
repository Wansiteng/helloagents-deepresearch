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
    """Instantiate the *global* HelloAgentsLLM (legacy single-model path).

    Per-agent LLMs are built via :func:`build_llm_for_role`; this function is
    kept for backwards compatibility with code paths that still need one
    canonical LLM (e.g. some test fixtures and the legacy DeepResearchAgent).
    """
    return build_llm_for_role("global", config)


# ── Per-agent LLM resolution (roadmap item #3) ───────────────────────────────
# Three-layer precedence:
#   1. <role>_llm_provider / <role>_llm_model   (explicit override)
#   2. quality_mode preset                       (coarse routing)
#   3. global llm_* fields                       (fallback)
_AGENT_ROLES = {"planner", "summarizer", "reporter", "critic"}


def _resolve_llm_kwargs(role: str, config: Configuration) -> dict[str, Any]:
    """Resolve the effective LLM kwargs for ``role``.

    ``role`` is one of "planner" / "summarizer" / "reporter" / "critic", or
    "global" for the legacy single-LLM path.
    """
    # Defaults inherited from the legacy global settings.
    global_provider = (config.llm_provider or "").strip()
    global_model = config.llm_model_id or config.local_llm or ""
    global_base_url = config.llm_base_url
    global_api_key = config.llm_api_key

    if role == "global":
        provider, model = global_provider, global_model
        base_url, api_key = global_base_url, global_api_key
    else:
        # Step 1: explicit per-agent override.
        per_provider = getattr(config, f"{role}_llm_provider", None)
        per_model = getattr(config, f"{role}_llm_model", None)

        # Step 2: quality_mode preset.
        wants_frontier = (
            config.quality_mode == "frontier-only"
            or (config.quality_mode == "hybrid" and role != "planner")
        )

        if per_provider or per_model:
            # Honor explicit override; pull missing field from the same tier
            # as the override's provider (frontier if frontier provider,
            # else global).
            chose_frontier = (per_provider or "").lower() == (
                (config.frontier_provider or "").lower()
            ) and config.frontier_provider
            if chose_frontier:
                provider = per_provider or config.frontier_provider
                model = per_model or config.frontier_model or ""
                base_url = config.frontier_base_url
                api_key = config.frontier_api_key
            else:
                provider = per_provider or global_provider
                model = per_model or global_model
                base_url = global_base_url
                api_key = global_api_key
        elif wants_frontier and config.frontier_provider:
            provider = config.frontier_provider
            model = config.frontier_model or ""
            base_url = config.frontier_base_url
            api_key = config.frontier_api_key
        else:
            # Either local-only, or hybrid+planner, or no frontier configured.
            provider, model = global_provider, global_model
            base_url, api_key = global_base_url, global_api_key

    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "timeout": config.llm_timeout,
    }
    if model:
        kwargs["model"] = model
    if provider:
        kwargs["provider"] = provider

    # Provider-specific base_url / api_key conveniences (kept from the legacy
    # build_llm so Ollama / LMStudio still work the same way).
    if provider == "ollama":
        kwargs["base_url"] = base_url or config.sanitized_ollama_url()
        kwargs["api_key"] = api_key or "ollama"
    elif provider == "lmstudio":
        kwargs["base_url"] = base_url or config.lmstudio_base_url
        kwargs["api_key"] = api_key or "lm-studio"
    else:
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

    return kwargs


def build_llm_for_role(role: str, config: Configuration) -> HelloAgentsLLM:
    """Construct the HelloAgentsLLM that should drive ``role``'s agent."""
    kwargs = _resolve_llm_kwargs(role, config)
    return HelloAgentsLLM(**kwargs)


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
    """Construct the full service bundle for a research run.

    As of roadmap #3, each agent role (planner / summarizer / reporter /
    critic) gets its own LLM instance. By default they all resolve to the
    same global model (preserving previous behaviour), but with
    ``quality_mode`` or per-role overrides the heavier agents can be pushed
    to a frontier API while the planner stays local.
    """
    # Build one LLM per role. Cache identical resolved kwarg-dicts so we
    # don't create four parallel HTTP clients when everything is local.
    llm_cache: dict[tuple, HelloAgentsLLM] = {}

    def llm_for(role: str) -> HelloAgentsLLM:
        kwargs = _resolve_llm_kwargs(role, config)
        # Key by the resolved (provider, model, base_url) — temperature/timeout
        # are constant, so this identifies a unique upstream endpoint.
        cache_key = (
            kwargs.get("provider"),
            kwargs.get("model"),
            kwargs.get("base_url"),
        )
        if cache_key not in llm_cache:
            llm_cache[cache_key] = HelloAgentsLLM(**kwargs)
        logger.info(
            "Agent %-11s → provider=%s model=%s",
            role,
            kwargs.get("provider") or "(default)",
            kwargs.get("model") or "(default)",
        )
        return llm_cache[cache_key]

    planner_llm = llm_for("planner")
    summarizer_llm = llm_for("summarizer")
    reporter_llm = llm_for("reporter")
    # A single canonical LLM is still exposed on ResearchServices.llm for
    # legacy consumers (vector-store init, tool tracker introspection…).
    # Use the planner's LLM because it's the cheapest / fastest.
    llm = planner_llm

    tool_registry = AgentToolRegistry(config)
    vector_store = _build_vector_store(config)
    tool_tracker = ToolCallTracker(
        config.notes_workspace if config.enable_notes else None
    )

    def make_agent(
        agent_llm: HelloAgentsLLM, name: str, system_prompt: str
    ) -> ToolAwareSimpleAgent:
        return create_tool_aware_agent(
            config=config,
            llm=agent_llm,
            tool_registry=tool_registry,
            tool_tracker=tool_tracker,
            name=name,
            system_prompt=system_prompt,
        )

    planner_agent = make_agent(
        planner_llm, "研究规划专家", todo_planner_system_prompt.strip()
    )
    writer_agent = make_agent(
        reporter_llm, "报告撰写专家", report_writer_instructions.strip()
    )

    def summarizer_factory() -> ToolAwareSimpleAgent:
        return make_agent(
            summarizer_llm, "任务总结专家", task_summarizer_instructions.strip()
        )

    planner = PlannerAgent(planner_agent, config)
    summarizer = SummarizerAgent(summarizer_factory, config)
    writer = WriterAgent(writer_agent, config, vector_store=vector_store)

    critic: CriticAgent | None = None
    if config.enable_reflection:
        from prompts import reflection_instructions  # noqa: F401 – 验证提示词可导入

        critic_llm = llm_for("critic")

        def critic_factory() -> ToolAwareSimpleAgent:
            return make_agent(
                critic_llm,
                "报告质量评审专家",
                (
                    "你是一名专业的研究质量评审专家，擅长识别研究报告中的不足与空白。"
                    "请对给定的研究报告进行客观评审，输出结构化 JSON 格式的评估结果。"
                ),
            )

        critic = CriticAgent(critic_factory, config)

    logger.info(
        "build_research_services: quality_mode=%s, %d distinct LLM endpoint(s)",
        config.quality_mode,
        len(llm_cache),
    )

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
