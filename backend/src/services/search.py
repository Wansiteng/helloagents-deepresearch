"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

from hello_agents.tools import SearchTool

from config import Configuration
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000
_GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")


def _duckduckgo_search_direct(query: str, max_results: int = 5) -> dict[str, Any]:
    """直接使用 ddgs.DDGS 并注入代理，绕过 hello_agents 内部实现。"""
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError("ddgs 未安装")

    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    ddgs_kwargs: dict[str, Any] = {"timeout": 30}
    if proxy:
        ddgs_kwargs["proxy"] = proxy

    logger.debug("DuckDuckGo direct search: query=%s proxy=%s", query, proxy)

    with DDGS(**ddgs_kwargs) as client:
        raw = list(client.text(query, max_results=max_results))

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("href", ""),
            "content": item.get("body", ""),
        }
        for item in raw
        if item.get("href")
    ]

    return {"results": results, "backend": "duckduckgo", "answer": None, "notices": []}


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)

    # duckduckgo 直接走带代理的实现，其他 backend 走 hello_agents SearchTool
    if search_api == "duckduckgo":
        try:
            raw_response = _duckduckgo_search_direct(query)
        except Exception as exc:
            logger.exception("DuckDuckGo direct search failed: %s", exc)
            raise
    else:
        try:
            raw_response = _GLOBAL_SEARCH_TOOL.run(
                {
                    "input": query,
                    "backend": search_api,
                    "mode": "structured",
                    "fetch_full_page": config.fetch_full_page,
                    "max_results": 5,
                    "max_tokens_per_source": MAX_TOKENS_PER_SOURCE,
                    "loop_count": loop_count,
                }
            )
        except Exception as exc:
            logger.exception("Search backend %s failed: %s", search_api, exc)
            raise

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    return payload, notices, answer_text, backend_label


def dispatch_search_with_retry(
    query: str,
    config: Configuration,
    loop_count: int,
    fallback_queries: Optional[list[str]] = None,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """执行搜索，若结果为空则按顺序尝试备用查询词。

    参数
    ----
    query:
        主搜索查询词。
    config:
        运行时配置。
    loop_count:
        当前研究循环次数（传给底层 dispatch_search）。
    fallback_queries:
        备用查询词列表，主查询无结果时按序尝试。

    返回
    ----
    与 ``dispatch_search`` 相同的四元组。
    """
    result, notices, answer, backend = dispatch_search(query, config, loop_count)

    if result and result.get("results"):
        return result, notices, answer, backend

    if not fallback_queries:
        return result, notices, answer, backend

    for fallback in fallback_queries:
        if not fallback or fallback.strip() == query.strip():
            continue
        logger.info("主查询 '%s' 无结果，尝试备用查询 '%s'", query, fallback)
        try:
            result2, notices2, answer2, backend2 = dispatch_search(fallback, config, loop_count)
            if result2 and result2.get("results"):
                combined_notices = notices + notices2 + [f"使用备用查询词: {fallback}"]
                return result2, combined_notices, answer2, backend2
        except Exception as exc:
            logger.warning("备用查询 '%s' 失败: %s", fallback, exc)

    return result, notices, answer, backend


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context
