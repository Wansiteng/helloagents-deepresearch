"""LLM client abstraction for the rebuilt research engine.

Defines a provider-agnostic :class:`LLMClient` protocol plus a concrete
:class:`OpenAICompatibleClient` that talks to any OpenAI-compatible endpoint —
Ollama, LM Studio, vLLM, OpenAI, DeepSeek, Moonshot, OpenRouter, etc.

Anthropic's native API uses a different wire protocol and is intentionally out
of scope here; a separate client will cover it later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

from utils import strip_thinking_tokens

if TYPE_CHECKING:
    from config import Configuration

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(kw_only=True)
class ToolCall:
    """A single tool/function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(kw_only=True)
class LLMMessage:
    """A chat message in provider-agnostic form.

    ``role`` is one of ``system`` / ``user`` / ``assistant`` / ``tool``.
    ``tool_call_id`` is only populated on ``tool`` messages (the result of a
    prior tool call).
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat interface used by the research engine."""

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMMessage:
        """Run one non-streaming chat completion and return the reply."""
        ...

    def chat_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding content deltas.

        Streaming + tool calling is deliberately not supported here yet (see
        PR-2). This yields plain content fragments only.
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Message <-> OpenAI wire-format conversion
# ──────────────────────────────────────────────────────────────────────────────
def _to_openai_message(message: LLMMessage) -> dict[str, Any]:
    """Convert an :class:`LLMMessage` into an OpenAI chat message dict."""
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }

    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    """Parse OpenAI ``tool_calls`` into :class:`ToolCall` objects."""
    parsed: list[ToolCall] = []
    for raw in raw_tool_calls or []:
        function = getattr(raw, "function", None)
        if function is None:
            continue
        raw_args = getattr(function, "arguments", "") or "{}"
        try:
            arguments = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Tool call arguments not valid JSON: %r", raw_args)
            arguments = {}
        parsed.append(
            ToolCall(
                id=getattr(raw, "id", "") or "",
                name=getattr(function, "name", "") or "",
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible implementation
# ──────────────────────────────────────────────────────────────────────────────
class OpenAICompatibleClient:
    """:class:`LLMClient` backed by any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 600,
        strip_thinking: bool = True,
    ) -> None:
        """Build a client.

        Args:
            base_url: OpenAI-compatible base URL (should include ``/v1``).
            api_key: API key. Local servers accept any non-empty placeholder.
            model: Model identifier to request.
            timeout: Per-request timeout in seconds.
            strip_thinking: Whether to strip ``<think>`` chains from replies.
        """
        from openai import AsyncOpenAI

        self.model = model
        self.strip_thinking = strip_thinking
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    @classmethod
    def from_config(cls, config: Configuration) -> OpenAICompatibleClient:
        """Build a client from the legacy :class:`Configuration` object.

        This mirrors the provider routing in ``agent.py::_init_llm`` and exists
        as a transitional adapter so callers can keep using the env-driven
        config until the new orchestrator owns configuration end to end.
        """
        provider = (config.llm_provider or "").strip().lower()

        if provider == "ollama":
            base_url = config.sanitized_ollama_url()
            api_key = config.llm_api_key or "ollama"
        elif provider == "lmstudio":
            base_url = config.lmstudio_base_url
            api_key = config.llm_api_key or "lm-studio"
        else:
            base_url = config.llm_base_url or ""
            api_key = config.llm_api_key or ""

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=config.resolved_model() or config.local_llm,
            timeout=config.llm_timeout,
            strip_thinking=config.strip_thinking_tokens,
        )

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMMessage:
        """Run one non-streaming chat completion."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": temperature,
        }
        if tools:
            request["tools"] = tools

        response = await self._client.chat.completions.create(**request)
        choice = response.choices[0].message

        tool_calls = _parse_tool_calls(getattr(choice, "tool_calls", None))
        content = choice.content or ""
        # Only strip thinking tokens from plain replies; tool-call turns carry
        # no prose worth touching.
        if self.strip_thinking and not tool_calls and content:
            content = strip_thinking_tokens(content)

        return LLMMessage(role="assistant", content=content, tool_calls=tool_calls)

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding content deltas.

        Thinking-token stripping is not applied here: ``<think>`` spans cross
        chunk boundaries and need a streaming-aware stripper (deferred to PR-2).
        """
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=[_to_openai_message(m) for m in messages],
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece
