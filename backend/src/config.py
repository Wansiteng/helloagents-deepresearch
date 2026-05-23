import os
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="Local Model Name",
        description="Name of the locally hosted LLM (Ollama/LMStudio)",
    )
    llm_provider: str = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider identifier (ollama, lmstudio, or custom)",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        title="Search API",
        description="Web search API to use",
    )
    enable_notes: bool = Field(
        default=True,
        title="Enable Notes",
        description="Whether to store task progress in NoteTool",
    )
    notes_workspace: str = Field(
        default="./notes",
        title="Notes Workspace",
        description="Directory for NoteTool to persist task notes",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        title="Ollama Base URL",
        description="Base URL for Ollama API (without /v1 suffix)",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        title="LLM API Key",
        description="Optional API key when using custom OpenAI-compatible services",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        title="LLM Base URL",
        description="Optional base URL when using custom OpenAI-compatible services",
    )
    llm_model_id: Optional[str] = Field(
        default=None,
        title="LLM Model ID",
        description="Optional model identifier for custom OpenAI-compatible services",
    )
    llm_timeout: int = Field(
        default=600,
        title="LLM Timeout",
        description="LLM request timeout in seconds (default 600). Local models like Ollama may need 5-10 min.",
    )
    use_vector_store: bool = Field(
        default=False,
        title="Use Vector Store",
        description="Enable RAG-based long-context memory via local ChromaDB vector store",
    )
    vector_store_path: str = Field(
        default="./vector_store",
        title="Vector Store Path",
        description="Directory for ChromaDB persistent storage",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        title="Embedding Model",
        description="Ollama embedding model name for vector store (e.g. nomic-embed-text)",
    )
    vector_chunk_size: int = Field(
        default=500,
        title="Vector Chunk Size",
        description="Max tokens per chunk for sliding-window text splitting",
    )
    vector_chunk_overlap: int = Field(
        default=50,
        title="Vector Chunk Overlap",
        description="Overlap tokens between adjacent chunks",
    )
    vector_top_k: int = Field(
        default=5,
        title="Vector Top K",
        description="Number of top similar chunks to retrieve from vector store",
    )
    search_results_per_task: int = Field(
        default=12,
        title="Search Results Per Task",
        description=(
            "Web pages (and per-source chunks) requested for each research "
            "sub-task. Higher = deeper coverage but slower per task and "
            "longer summarizer prompts (which can hit local-LLM context "
            "limits). Sensible range: 5-20. Default 12 is tuned for a "
            "'deep' feel without blowing up token budgets."
        ),
    )
    enable_arxiv: bool = Field(
        default=True,
        title="Enable arXiv Source",
        description=(
            "Query arXiv's open Atom API alongside web search. No key needed. "
            "Strongly recommended for CS / physics / math / quantitative "
            "biology topics — surfaces peer-reviewed and preprint papers that "
            "rarely rank on DDG."
        ),
    )
    enable_openalex: bool = Field(
        default=True,
        title="Enable OpenAlex Source",
        description=(
            "Query OpenAlex (~250M scholarly works, all disciplines) alongside "
            "web search. No key needed; set OPENALEX_EMAIL to enter the polite "
            "pool for higher rate limits."
        ),
    )
    openalex_email: Optional[str] = Field(
        default=None,
        title="OpenAlex Polite-Pool Email",
        description=(
            "Optional contact email passed to OpenAlex via mailto= and to "
            "arXiv via User-Agent. Pushes both APIs into their polite pools."
        ),
    )

    # ── Quality mode & per-agent LLM routing ─────────────────────────────────
    # The legacy llm_* fields above describe ONE global LLM. They still serve
    # as the fallback for every agent. On top of that we add a coarse-grained
    # "quality_mode" preset plus optional per-agent overrides so each agent
    # (planner / summarizer / reporter / critic) can run on a different model.
    #
    # Resolution precedence per agent role:
    #   1. <role>_llm_provider / <role>_llm_model env vars (most specific)
    #   2. quality_mode preset:
    #        local-only    → all roles use global llm_*
    #        hybrid        → planner uses global, others use frontier_*
    #        frontier-only → all roles use frontier_*
    #   3. global llm_* (fallback)
    quality_mode: Literal["local-only", "hybrid", "frontier-only"] = Field(
        default="local-only",
        title="Quality Mode",
        description=(
            "Coarse-grained agent routing preset. 'local-only' keeps every "
            "agent on the global LLM (typically Ollama). 'hybrid' runs the "
            "planner locally but pushes the heavier summarizer + reporter to "
            "the frontier API defined by frontier_* fields. 'frontier-only' "
            "runs every agent on the frontier API. Per-agent overrides "
            "(see <role>_llm_provider) take precedence over this preset."
        ),
    )

    frontier_provider: Optional[str] = Field(
        default=None,
        title="Frontier Provider",
        description=(
            "LLM provider used when quality_mode promotes an agent to the "
            "frontier path. Must be one of HelloAgentsLLM's supported "
            "providers: openai, deepseek, qwen, modelscope, kimi, zhipu, "
            "ollama, vllm, custom. Pair with frontier_model + frontier_api_key."
        ),
    )
    frontier_model: Optional[str] = Field(
        default=None,
        title="Frontier Model",
        description="Model id for the frontier provider (e.g., gpt-4o, deepseek-chat, qwen-plus).",
    )
    frontier_api_key: Optional[str] = Field(
        default=None,
        title="Frontier API Key",
        description="API key for the frontier provider. Read from FRONTIER_API_KEY env.",
    )
    frontier_base_url: Optional[str] = Field(
        default=None,
        title="Frontier Base URL",
        description="Optional base-URL override for the frontier provider (custom endpoints / proxies).",
    )

    # Per-agent overrides — set just the ones you want to deviate from the
    # quality_mode preset. Leaving them None means "follow the preset".
    planner_llm_provider: Optional[str] = Field(default=None)
    planner_llm_model: Optional[str] = Field(default=None)
    summarizer_llm_provider: Optional[str] = Field(default=None)
    summarizer_llm_model: Optional[str] = Field(default=None)
    reporter_llm_provider: Optional[str] = Field(default=None)
    reporter_llm_model: Optional[str] = Field(default=None)
    critic_llm_provider: Optional[str] = Field(default=None)
    critic_llm_model: Optional[str] = Field(default=None)
    use_open_source_mode: bool = Field(
        default=False,
        title="Use Open-Source Model Mode",
        description=(
            "Enable strict-constraint prompt engineering and self-correction retry "
            "loop optimized for local open-source models (e.g. Qwen, Llama) that "
            "have weaker instruction-following capabilities than frontier APIs."
        ),
    )
    open_source_model_max_retries: int = Field(
        default=2,
        title="Open-Source Model Self-Correction Max Retries",
        description=(
            "Maximum number of self-correction retry attempts when a tool call "
            "fails to produce valid JSON in open-source model mode."
        ),
    )
    no_think_mode: bool = Field(
        default=False,
        title="No Think Mode (Qwen3 series)",
        description=(
            "Qwen3 系列专用：在每个 Agent 的 System Prompt 开头注入 '/no_think' 指令，"
            "禁用思维链生成以加快响应速度并降低 token 消耗。"
            "适合对延迟敏感、无需深度推理的场景；"
            "深度研究任务建议保持默认关闭以发挥 Qwen3 思维链推理优势，"
            "配合 strip_thinking_tokens=True 自动清理思考输出。"
        ),
    )

    # ── 并发与性能 ────────────────────────────────────────────────────────
    llm_concurrency: int = Field(
        default=1,
        title="LLM Concurrency",
        description=(
            "并发 LLM 请求数上限（用于 Semaphore 控制）。"
            "默认 1 适合 Ollama 单实例；如启用 OLLAMA_NUM_PARALLEL=N，"
            "可将此值同步设置为 N 以充分利用并发能力。"
        ),
    )

    # ── 动态规划 ──────────────────────────────────────────────────────────
    enable_dynamic_planning: bool = Field(
        default=True,
        title="Enable Dynamic Planning",
        description=(
            "初始任务批次完成后，由 PlannerAgent 评估研究覆盖度，"
            "若发现明显空白则自动追加最多 max_dynamic_tasks 个补充任务。"
        ),
    )
    max_dynamic_tasks: int = Field(
        default=2,
        title="Max Dynamic Tasks",
        description="动态规划最多追加的额外任务数量。",
    )

    # ── 反思评审 ──────────────────────────────────────────────────────────
    enable_reflection: bool = Field(
        default=True,
        title="Enable Reflection",
        description=(
            "最终报告生成后，由 CriticAgent 进行质量评审。"
            "若评分低于 reflection_score_threshold，自动补充研究并重新生成报告。"
        ),
    )
    reflection_score_threshold: int = Field(
        default=6,
        title="Reflection Score Threshold",
        description="触发补充研究的质量评分下限（1-10），低于此分值将追加研究任务。",
    )

    # ── 搜索策略多样化 ────────────────────────────────────────────────────
    search_retry_on_empty: bool = Field(
        default=True,
        title="Search Retry on Empty Results",
        description=(
            "当搜索返回空结果时，自动尝试备用查询词（任务标题 + 研究主题 组合），"
            "提高搜索成功率，无需额外 LLM 调用。"
        ),
    )

    # ── 渐进式报告生成 ────────────────────────────────────────────────────
    enable_progressive_report: bool = Field(
        default=True,
        title="Enable Progressive Report",
        description=(
            "每个子任务完成后立即格式化该任务摘要为章节草稿并推送前端，"
            "用户可在等待最终报告期间实时阅读中间成果（无额外 LLM 调用）。"
        ),
    )

    # ── 知识源：Obsidian vault ────────────────────────────────────────────
    obsidian_vault_path: Optional[str] = Field(
        default=None,
        title="Obsidian Vault Path",
        description=(
            "本地 Obsidian vault 目录路径。设置后，新编排器会把 vault 笔记"
            "作为一个知识源参与研究检索（语义检索）。留空则不启用。"
        ),
    )
    obsidian_index_path: str = Field(
        default="./obsidian_index",
        title="Obsidian Index Path",
        description="Obsidian vault 语义索引（ChromaDB）与 mtime manifest 的存放目录。",
    )

    # ── 编排器选择 ────────────────────────────────────────────────────────
    use_new_orchestrator: bool = Field(
        default=False,
        title="Use New Orchestrator",
        description=(
            "实验性：`/research/stream` 改用新的 ResearchSession 异步状态机，"
            "而非旧的 DeepResearchAgent.run_stream。默认关闭，旧路径为默认。"
        ),
    )

    # ── 上下文窗口管理 ────────────────────────────────────────────────────
    context_max_chars: int = Field(
        default=15000,
        title="Context Max Chars",
        description=(
            "传入 WriterAgent 的任务摘要总字符上限。超出时自动按 summary_card_max_chars "
            "截断各任务摘要，防止提示词超出本地模型上下文窗口。0 表示不限制。"
        ),
    )
    summary_card_max_chars: int = Field(
        default=800,
        title="Summary Card Max Chars",
        description="上下文压缩时单个任务摘要的最大字符数。",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """Create a configuration object using environment variables and overrides."""

        raw_values: dict[str, Any] = {}

        # Load values from environment variables based on field names
        for field_name in cls.model_fields.keys():
            env_key = field_name.upper()
            if env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]

        # Additional mappings for explicit env names
        env_aliases = {
            "local_llm": os.getenv("LOCAL_LLM"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_api_key": os.getenv("LLM_API_KEY"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "use_tool_calling": os.getenv("USE_TOOL_CALLING"),
            "search_api": os.getenv("SEARCH_API"),
            "enable_notes": os.getenv("ENABLE_NOTES"),
            "notes_workspace": os.getenv("NOTES_WORKSPACE"),
            "llm_timeout": os.getenv("LLM_TIMEOUT"),
            "use_vector_store": os.getenv("USE_VECTOR_STORE"),
            "vector_store_path": os.getenv("VECTOR_STORE_PATH"),
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "vector_chunk_size": os.getenv("VECTOR_CHUNK_SIZE"),
            "vector_chunk_overlap": os.getenv("VECTOR_CHUNK_OVERLAP"),
            "vector_top_k": os.getenv("VECTOR_TOP_K"),
            "search_results_per_task": os.getenv("SEARCH_RESULTS_PER_TASK"),
            "enable_arxiv": os.getenv("ENABLE_ARXIV"),
            "enable_openalex": os.getenv("ENABLE_OPENALEX"),
            "openalex_email": os.getenv("OPENALEX_EMAIL"),
            "quality_mode": os.getenv("QUALITY_MODE"),
            "frontier_provider": os.getenv("FRONTIER_PROVIDER"),
            "frontier_model": os.getenv("FRONTIER_MODEL"),
            "frontier_api_key": os.getenv("FRONTIER_API_KEY"),
            "frontier_base_url": os.getenv("FRONTIER_BASE_URL"),
            "planner_llm_provider": os.getenv("PLANNER_LLM_PROVIDER"),
            "planner_llm_model": os.getenv("PLANNER_LLM_MODEL"),
            "summarizer_llm_provider": os.getenv("SUMMARIZER_LLM_PROVIDER"),
            "summarizer_llm_model": os.getenv("SUMMARIZER_LLM_MODEL"),
            "reporter_llm_provider": os.getenv("REPORTER_LLM_PROVIDER"),
            "reporter_llm_model": os.getenv("REPORTER_LLM_MODEL"),
            "critic_llm_provider": os.getenv("CRITIC_LLM_PROVIDER"),
            "critic_llm_model": os.getenv("CRITIC_LLM_MODEL"),
            "use_open_source_mode": os.getenv("USE_OPEN_SOURCE_MODE"),
            "open_source_model_max_retries": os.getenv("OPEN_SOURCE_MODEL_MAX_RETRIES"),
            "no_think_mode": os.getenv("NO_THINK_MODE"),
            "llm_concurrency": os.getenv("LLM_CONCURRENCY"),
            "enable_dynamic_planning": os.getenv("ENABLE_DYNAMIC_PLANNING"),
            "max_dynamic_tasks": os.getenv("MAX_DYNAMIC_TASKS"),
            "enable_reflection": os.getenv("ENABLE_REFLECTION"),
            "reflection_score_threshold": os.getenv("REFLECTION_SCORE_THRESHOLD"),
            "search_retry_on_empty": os.getenv("SEARCH_RETRY_ON_EMPTY"),
            "enable_progressive_report": os.getenv("ENABLE_PROGRESSIVE_REPORT"),
            "context_max_chars": os.getenv("CONTEXT_MAX_CHARS"),
            "summary_card_max_chars": os.getenv("SUMMARY_CARD_MAX_CHARS"),
            "use_new_orchestrator": os.getenv("USE_NEW_ORCHESTRATOR"),
            "obsidian_vault_path": os.getenv("OBSIDIAN_VAULT_PATH"),
            "obsidian_index_path": os.getenv("OBSIDIAN_INDEX_PATH"),
        }

        for key, value in env_aliases.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value
                elif key in raw_values:
                    # Explicit None in overrides clears a previously-set env value
                    del raw_values[key]

        return cls(**raw_values)

    def sanitized_ollama_url(self) -> str:
        """Ensure Ollama base URL includes the /v1 suffix required by OpenAI clients."""

        base = self.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def resolved_model(self) -> Optional[str]:
        """Best-effort resolution of the model identifier to use."""

        return self.llm_model_id or self.local_llm

