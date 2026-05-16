"""Unit tests for core.llm — no network, AsyncOpenAI is faked."""
from __future__ import annotations

from types import SimpleNamespace

from config import Configuration
from core.llm import (
    LLMClient,
    LLMMessage,
    OpenAICompatibleClient,
    ToolCall,
    _to_openai_message,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────
class _FakeCompletions:
    """Stands in for AsyncOpenAI().chat.completions."""

    def __init__(self, response, stream_chunks):
        self._response = response
        self._stream_chunks = stream_chunks
        self.last_request: dict = {}

    async def create(self, **kwargs):
        self.last_request = kwargs
        if kwargs.get("stream"):
            return _FakeAsyncStream(self._stream_chunks)
        return self._response


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:  # noqa: B904
            raise StopAsyncIteration


def _make_client(response=None, stream_chunks=()):
    """Build an OpenAICompatibleClient with its OpenAI client swapped for a fake."""
    client = OpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        api_key="placeholder",
        model="test-model",
    )
    fake_completions = _FakeCompletions(response, stream_chunks)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    return client, fake_completions


def _message(content=None, tool_calls=None):
    """Build a fake OpenAI choice.message object."""
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _stream_chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
def test_llm_message_defaults():
    msg = LLMMessage(role="user", content="hi")
    assert msg.role == "user"
    assert msg.content == "hi"
    assert msg.tool_calls == []
    assert msg.tool_call_id is None


def test_tool_call_construction():
    call = ToolCall(id="c1", name="search", arguments={"q": "python"})
    assert call.name == "search"
    assert call.arguments["q"] == "python"


def test_to_openai_message_tool_role():
    msg = LLMMessage(role="tool", content="result", tool_call_id="c1")
    assert _to_openai_message(msg) == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "result",
    }


def test_to_openai_message_assistant_with_tool_calls():
    msg = LLMMessage(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "x"})],
    )
    out = _to_openai_message(msg)
    assert out["role"] == "assistant"
    assert out["tool_calls"][0]["id"] == "c1"
    assert out["tool_calls"][0]["function"]["name"] == "search"
    assert '"q"' in out["tool_calls"][0]["function"]["arguments"]


# ──────────────────────────────────────────────────────────────────────────────
# from_config provider routing
# ──────────────────────────────────────────────────────────────────────────────
def test_from_config_ollama_routing():
    config = Configuration(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        local_llm="qwen3.5:9b",
    )
    client = OpenAICompatibleClient.from_config(config)
    assert client.model == "qwen3.5:9b"
    assert str(client._client.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_from_config_lmstudio_routing():
    config = Configuration(
        llm_provider="lmstudio",
        lmstudio_base_url="http://localhost:1234/v1",
        local_llm="qwen3.5-35b",
    )
    client = OpenAICompatibleClient.from_config(config)
    assert client.model == "qwen3.5-35b"
    assert str(client._client.base_url).rstrip("/") == "http://localhost:1234/v1"


def test_from_config_custom_routing():
    config = Configuration(
        llm_provider="custom",
        llm_base_url="https://api.deepseek.com/v1",
        llm_api_key="sk-secret",
        llm_model_id="deepseek-chat",
    )
    client = OpenAICompatibleClient.from_config(config)
    assert client.model == "deepseek-chat"
    assert str(client._client.base_url).rstrip("/") == "https://api.deepseek.com/v1"


# ──────────────────────────────────────────────────────────────────────────────
# chat()
# ──────────────────────────────────────────────────────────────────────────────
async def test_chat_plain_reply():
    client, fake = _make_client(_response(_message(content="hello world")))
    reply = await client.chat([LLMMessage(role="user", content="hi")])
    assert reply.role == "assistant"
    assert reply.content == "hello world"
    assert reply.tool_calls == []
    assert fake.last_request["model"] == "test-model"


async def test_chat_strips_thinking_tokens():
    client, _ = _make_client(
        _response(_message(content="<think>pondering</think>final answer"))
    )
    reply = await client.chat([LLMMessage(role="user", content="hi")])
    assert "<think>" not in reply.content
    assert reply.content == "final answer"


async def test_chat_keeps_thinking_when_disabled():
    client = OpenAICompatibleClient(
        base_url="http://x/v1", api_key="k", model="m", strip_thinking=False
    )
    fake = _FakeCompletions(
        _response(_message(content="<think>raw</think>answer")), ()
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    reply = await client.chat([LLMMessage(role="user", content="hi")])
    assert "<think>" in reply.content


async def test_chat_parses_tool_calls():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="web_search", arguments='{"query": "asyncio"}'),
    )
    client, fake = _make_client(
        _response(_message(content=None, tool_calls=[tool_call]))
    )
    reply = await client.chat(
        [LLMMessage(role="user", content="search")],
        tools=[{"type": "function", "function": {"name": "web_search"}}],
    )
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "web_search"
    assert reply.tool_calls[0].arguments == {"query": "asyncio"}
    assert "tools" in fake.last_request


async def test_chat_tolerates_malformed_tool_arguments():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="web_search", arguments="{not valid json"),
    )
    client, _ = _make_client(
        _response(_message(content=None, tool_calls=[tool_call]))
    )
    reply = await client.chat([LLMMessage(role="user", content="search")])
    assert reply.tool_calls[0].arguments == {}


# ──────────────────────────────────────────────────────────────────────────────
# chat_stream()
# ──────────────────────────────────────────────────────────────────────────────
async def test_chat_stream_yields_content_deltas():
    chunks = [_stream_chunk("hel"), _stream_chunk("lo"), _stream_chunk(None)]
    client, _ = _make_client(stream_chunks=chunks)
    pieces = [p async for p in client.chat_stream([LLMMessage(role="user", content="hi")])]
    assert pieces == ["hel", "lo"]


# ──────────────────────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────────────────────
def test_openai_client_satisfies_protocol():
    client = OpenAICompatibleClient(base_url="http://x/v1", api_key="k", model="m")
    assert isinstance(client, LLMClient)
