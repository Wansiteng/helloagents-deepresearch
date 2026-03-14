import os
from enum import Enum
from typing import Any, Optional

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
            "use_open_source_mode": os.getenv("USE_OPEN_SOURCE_MODE"),
            "open_source_model_max_retries": os.getenv("OPEN_SOURCE_MODEL_MAX_RETRIES"),
            "no_think_mode": os.getenv("NO_THINK_MODE"),
        }

        for key, value in env_aliases.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

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

