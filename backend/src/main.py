"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# 加载 .env 文件（从 src/ 向上查找到 backend/）
import os as _os
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# 从 .env 读取代理配置（load_dotenv 已加载），缓存后再清除 IDE 注入的随机端口。
_clash_proxy = _os.environ.get("HTTP_PROXY") or _os.environ.get("HTTPS_PROXY") or "http://127.0.0.1:7897"
for _proxy_key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                    "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"):
    _os.environ.pop(_proxy_key, None)
_os.environ["http_proxy"] = _clash_proxy
_os.environ["https_proxy"] = _clash_proxy
_os.environ["HTTP_PROXY"] = _clash_proxy
_os.environ["HTTPS_PROXY"] = _clash_proxy
_os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
_os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

# ddgs 内部使用 primp（Rust HTTP 客户端），不读取系统代理环境变量。
# 通过 monkey-patch DDGS.__init__ 强制注入 proxy 参数。
try:
    from ddgs import DDGS as _DDGS
    _orig_ddgs_init = _DDGS.__init__

    def _patched_ddgs_init(self, *args, **kwargs):
        if "proxy" not in kwargs:
            kwargs["proxy"] = _clash_proxy
        _orig_ddgs_init(self, *args, **kwargs)

    _DDGS.__init__ = _patched_ddgs_init
except Exception:
    pass

import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests as _requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI
from agent import DeepResearchAgent
from models import SSEEventType
from routers_history import router as history_router

# 添加控制台日志处理程序
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


# 添加错误日志文件处理程序
logger.add(
    sink=sys.stderr,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )
    llm_provider: str | None = Field(
        default=None,
        description="Override LLM provider: ollama | lmstudio | mlx | custom",
    )
    local_llm: str | None = Field(
        default=None,
        description="Override the model name/id to use for this request",
    )


class LocalLLMServiceInfo(BaseModel):
    running: bool
    models: list[str]


class ProbeLocalLLMsResponse(BaseModel):
    services: dict[str, LocalLLMServiceInfo]


class PreflightRequest(BaseModel):
    """Payload for LLM preflight check (same fields as ResearchRequest minus topic)."""

    llm_provider: str | None = Field(default=None)
    local_llm: str | None = Field(default=None)


class PreflightResponse(BaseModel):
    ok: bool
    error: str | None = None
    hint: str | None = None


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


_LOCAL_PROVIDERS = {"ollama", "lmstudio", "mlx"}


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api
    if payload.llm_provider is not None:
        overrides["llm_provider"] = payload.llm_provider
    if payload.local_llm is not None:
        overrides["local_llm"] = payload.local_llm
        # For local providers, the model is identified by local_llm.
        # If the env had LLM_MODEL_ID set for a *different* default provider,
        # clear it so resolved_model() correctly returns local_llm.
        effective_provider = payload.llm_provider or overrides.get("llm_provider")
        if effective_provider in _LOCAL_PROVIDERS:
            overrides["llm_model_id"] = None

    return Configuration.from_env(overrides=overrides)


_PROBE_TIMEOUT = 2.5  # seconds


def _probe_ollama(base_url: str = "http://localhost:11434") -> LocalLLMServiceInfo:
    """Check Ollama via GET /api/tags."""
    try:
        resp = _requests.get(
            f"{base_url.rstrip('/')}/api/tags",
            timeout=_PROBE_TIMEOUT,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        return LocalLLMServiceInfo(running=True, models=models)
    except Exception:
        return LocalLLMServiceInfo(running=False, models=[])


def _probe_openai_compatible(base_url: str) -> LocalLLMServiceInfo:
    """Check any OpenAI-compatible service via GET /v1/models."""
    try:
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        resp = _requests.get(
            f"{url}/models",
            timeout=_PROBE_TIMEOUT,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        return LocalLLMServiceInfo(running=True, models=models)
    except Exception:
        return LocalLLMServiceInfo(running=False, models=[])


def _preflight_hint(error_msg: str, provider: str) -> str:
    """Map a raw LLM error string to a user-friendly hint."""
    err_lower = error_msg.lower()

    if "compute error" in err_lower:
        base = (
            "LM Studio 推理引擎返回了 Compute error，常见原因：\n"
            "① 模型正在加载中，请等待 LM Studio 完全加载后重试；\n"
            "② 系统内存不足（尤其是带 mmproj 的多模态模型需要更多内存）；\n"
            "③ 尝试在 LM Studio 中卸载再重新加载该模型。"
        )
        return base

    if "connection refused" in err_lower or "connect" in err_lower:
        svc = {"ollama": "Ollama", "lmstudio": "LM Studio", "mlx": "mlx-lm"}.get(provider, "本地 LLM 服务")
        return f"无法连接到 {svc}，请确认服务已启动并监听默认端口。"

    if "not found" in err_lower or "404" in err_lower:
        return "模型未找到，请确认模型名称正确，并在本地服务中已完整加载该模型。"

    if "timeout" in err_lower:
        return "LLM 响应超时，模型可能仍在加载中，请稍等后重试。"

    if "401" in err_lower or "unauthorized" in err_lower:
        return "API 鉴权失败，请检查 .env 中的 LLM_API_KEY 配置。"

    return "LLM 调用失败，请检查本地服务状态和模型配置后重试。"


def create_app() -> FastAPI:
    app = FastAPI(title="HelloAgents Deep Researcher")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(history_router)

    @app.on_event("startup")
    def log_startup_configuration() -> None:
        from tool_registry import AgentToolRegistry
        config = Configuration.from_env()

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        registry = AgentToolRegistry(config)
        registered_tools = registry.list_tools() or ["（无工具）"]

        logger.info(
            "DeepResearch configuration loaded: provider={} model={} base_url={} search_api={} "
            "max_loops={} fetch_full_page={} tool_calling={} strip_thinking={} api_key={} "
            "llm_timeout={}s registered_tools={} vector_store={}",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
            _mask_secret(config.llm_api_key),
            config.llm_timeout,
            registered_tools,
            f"{config.vector_store_path} (model={config.embedding_model})" if config.use_vector_store else "禁用",
        )

    @app.get("/health")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/probe-local-llms", response_model=ProbeLocalLLMsResponse)
    def probe_local_llms() -> ProbeLocalLLMsResponse:
        """Probe locally-running LLM services and return available models.

        Supported services:
        - **ollama** — default port 11434
        - **lmstudio** — default port 1234 (OpenAI-compatible)
        - **mlx** — mlx-lm server, default port 8080 (OpenAI-compatible)
        """
        config = Configuration.from_env()

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_ollama = pool.submit(_probe_ollama, config.ollama_base_url)
            f_lmstudio = pool.submit(_probe_openai_compatible, config.lmstudio_base_url)
            f_mlx = pool.submit(_probe_openai_compatible, "http://localhost:8080")

            ollama_info = f_ollama.result()
            lmstudio_info = f_lmstudio.result()
            mlx_info = f_mlx.result()

        logger.info(
            "Local LLM probe: ollama={} lmstudio={} mlx={}",
            ollama_info.running,
            lmstudio_info.running,
            mlx_info.running,
        )

        return ProbeLocalLLMsResponse(
            services={
                "ollama": ollama_info,
                "lmstudio": lmstudio_info,
                "mlx": mlx_info,
            }
        )

    @app.post("/llm-preflight", response_model=PreflightResponse)
    def llm_preflight(payload: PreflightRequest) -> PreflightResponse:
        """Quick sanity-check: send a minimal message to the selected LLM.

        Returns ``{"ok": true}`` if the model responds, otherwise returns
        ``{"ok": false, "error": "...", "hint": "..."}`` with a human-readable
        hint so the frontend can display actionable guidance before wasting time
        on a full research run.
        """
        # Build a throw-away ResearchRequest so we can reuse _build_config
        dummy = ResearchRequest(
            topic="__preflight__",
            llm_provider=payload.llm_provider,
            local_llm=payload.local_llm,
        )
        try:
            config = _build_config(dummy)
            agent = DeepResearchAgent(config=config)
            llm = agent._init_llm()
            llm.invoke([{"role": "user", "content": "Reply with the single word: OK"}])
            return PreflightResponse(ok=True)
        except Exception as exc:
            raw = str(exc)
            hint = _preflight_hint(raw, payload.llm_provider or "")
            logger.warning("LLM preflight failed: {}", raw)
            return PreflightResponse(ok=False, error=raw, hint=hint)


    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
            result = agent.run(payload.topic)
        except ValueError as exc:  # Likely due to unsupported configuration
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Research failed") from exc

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "note_id": item.note_id,
                "note_path": item.note_path,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest) -> StreamingResponse:
        """SSE 流式研究接口。

        推送事件类型请参见 :class:`~models.SSEEventType`：

        - ``status`` — 全局状态提示
        - ``todo_list`` — Planner Agent 输出的任务列表
        - ``task_status`` — 单任务状态变更
        - ``sources`` — 搜索来源摘要
        - ``task_summary_chunk`` — Summarizer Agent 流式摘要片段
        - ``tool_call`` — Agent 工具调用详情
        - ``final_report`` — Writer Agent 最终报告
        - ``done`` — 流结束信号
        - ``error`` — 异常事件
        """
        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(payload.topic):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("type") in {SSEEventType.ERROR, SSEEventType.DONE}:
                        return
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Streaming research failed")
                error_payload = {"type": SSEEventType.ERROR, "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
