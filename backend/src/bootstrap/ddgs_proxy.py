"""Force ``ddgs`` (DuckDuckGo search) to honour our proxy configuration.

``ddgs`` delegates HTTP to ``primp``, a Rust client that does **not** read
``HTTP_PROXY``/``HTTPS_PROXY`` from the environment. The only way to route
its traffic through Clash/Mihomo/etc. is to pass ``proxy=<url>`` to the
``DDGS`` constructor on every call site.

Rather than thread that argument through every search caller, we monkey-patch
``DDGS.__init__`` once at process start to inject the proxy when none is
supplied explicitly.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_ddgs_proxy(proxy_url: str | None) -> bool:
    """Patch ``DDGS.__init__`` so it uses ``proxy_url`` by default.

    Args:
        proxy_url: Proxy URL to inject. If falsy, this is a no-op so callers
            can call unconditionally.

    Returns:
        ``True`` if the patch was applied, ``False`` otherwise.
    """
    if not proxy_url:
        return False

    try:
        from ddgs import DDGS
    except Exception as exc:  # pragma: no cover - dep guard
        logger.warning("ddgs not importable, skipping proxy patch: %s", exc)
        return False

    original_init = DDGS.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("proxy", proxy_url)
        original_init(self, *args, **kwargs)

    DDGS.__init__ = patched_init  # type: ignore[method-assign]
    return True
