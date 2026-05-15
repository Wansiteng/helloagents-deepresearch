"""Process-level HTTP/HTTPS proxy configuration.

Some IDEs (PyCharm) inject random ``HTTP_PROXY`` ports into the subprocess
environment; if left untouched they break every outbound LLM/search call.
We always reset proxy-related env vars to a single canonical value derived
from ``LOCAL_PROXY_URL`` (or the existing ``HTTP_PROXY`` env var).

Call :func:`configure_proxy` from ``main.py`` *before* any HTTP client (LLM,
``requests``, ``httpx``, ``ddgs``) is instantiated.
"""
from __future__ import annotations

import os

_PROXY_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
)

_NO_PROXY_HOSTS = "localhost,127.0.0.1,0.0.0.0"


def resolve_proxy_url() -> str | None:
    """Return the proxy URL to use, or ``None`` for direct connections.

    Resolution order (first non-empty wins):
      1. ``LOCAL_PROXY_URL`` env var
      2. ``HTTP_PROXY`` env var (cached before we reset)
      3. ``HTTPS_PROXY`` env var (cached before we reset)
    """
    return (
        os.environ.get("LOCAL_PROXY_URL")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or None
    )


def configure_proxy(proxy_url: str | None = None) -> str | None:
    """Reset proxy env vars to a single, canonical value.

    Args:
        proxy_url: Override the resolved proxy URL. Pass ``None`` to use the
            value from :func:`resolve_proxy_url`. Pass an empty string to
            disable proxying entirely.

    Returns:
        The proxy URL that was set, or ``None`` if proxying is disabled.
    """
    effective = proxy_url if proxy_url is not None else resolve_proxy_url()

    for key in _PROXY_KEYS:
        os.environ.pop(key, None)

    if effective:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ[key] = effective

    os.environ["no_proxy"] = _NO_PROXY_HOSTS
    os.environ["NO_PROXY"] = _NO_PROXY_HOSTS

    return effective or None
