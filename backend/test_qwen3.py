#!/usr/bin/env python3
"""Qwen3.5 适配集成测试脚本。

运行方式（在 backend/ 目录下）:
    python test_qwen3.py

测试项目:
    1. Ollama 连通性 — 直接调用 qwen3.5:9b，确认可响应
    2. strip_thinking_tokens — 验证 <think> 各变体均被正确剥离
    3. DeepResearchAgent 初始化 — 确认配置加载无报错
    4. PlannerAgent 快速测试 — 发送短主题，断言返回 >= 1 个 TodoItem
"""

from __future__ import annotations

import sys
import os
import time

# ── 把 src/ 目录加入模块搜索路径 ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# 加载 .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    results.append((name, ok, detail))


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1: Ollama 连通性
# ──────────────────────────────────────────────────────────────────────────────

def test_ollama_connectivity() -> None:
    """两阶段连通性测试：
    1. 轻量级：通过 Ollama REST API 检查模型是否已加载（无推理）
    2. 推理验证：streaming 模式下接收到首批 token 即视为成功
       （Qwen3.5:9b 思维链可达 2000+ tokens，非 streaming 完整等待需 70-120s）
    """
    print("\n[1/4] Ollama 连通性测试 (qwen3.5:9b) ...")
    import urllib.request
    import json as _json

    # ── 阶段 1：轻量级检查——Ollama 是否运行且 qwen3.5:9b 可用 ─────────
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            tags_data = _json.loads(r.read())
        model_names = [m["name"] for m in tags_data.get("models", [])]
        model_available = any("qwen3.5" in n.lower() for n in model_names)
        print(f"      Ollama 已运行，可用模型: {model_names}")
        if not model_available:
            report("Ollama connectivity", False, f"qwen3.5:9b 未找到，已有: {model_names}")
            return
    except Exception as exc:
        report("Ollama connectivity", False, f"Ollama 服务不可达: {exc}")
        return

    # ── 阶段 2：streaming 推理验证（接收首批 token 即成功，无需完整等待） ─
    # Qwen3.5:9b 行为说明：
    # - 思维链内容经由 Ollama 专有的 reasoning_delta 字段流出
    # - 最终答案通过 content_delta 输出（可能在 thinking 完成后才到达）
    # - 收到任意 token（包括 reasoning）即证明模型推理正在进行
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=30,  # streaming 模式：首 token 通常 <10s
        )
        first_token = ""
        token_count = 0
        stream = client.chat.completions.create(
            model="qwen3.5:9b",
            messages=[{"role": "user", "content": "你好"}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            content_piece = delta.content or ""
            reasoning_piece = getattr(delta, "reasoning", "") or ""
            piece = content_piece or reasoning_piece
            if piece:
                if not first_token:
                    first_token = piece
                token_count += 1
                if token_count >= 3:  # 收到 3 个 token 即可，不必等全部输出
                    stream.close()
                    break

        ok = bool(first_token)
        report(
            "Ollama connectivity",
            ok,
            f"streaming OK, {token_count}+ tokens received" if ok else "no tokens received",
        )
        if first_token:
            print(f"      首个 token 预览: {repr(first_token[:40])}")
    except Exception as exc:
        report("Ollama connectivity", False, str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2: strip_thinking_tokens 多变体
# ──────────────────────────────────────────────────────────────────────────────

def test_strip_thinking_tokens() -> None:
    print("\n[2/4] strip_thinking_tokens 变体测试 ...")
    from utils import strip_thinking_tokens

    cases = [
        # (描述, 输入, 期望不含的字符串, 期望包含的字符串 or None)
        (
            "标准 <think>",
            "<think>这是思考过程</think>这是答案",
            "<think>",
            "这是答案",
        ),
        (
            "<|think|> 变体A（</|think|> 闭合）",
            "<|think|>内部思考</|think|>正式输出A",
            "<|think|>",
            "正式输出A",
        ),
        (
            "<|think|> 变体B（<|/think|> 闭合）",
            "<|think|>内部思考<|/think|>正式输出B",
            "<|think|>",
            "正式输出B",
        ),
        (
            "未闭合 <think> 兜底截断",
            "前缀内容<think>截断后的内容",
            "<think>",
            None,
        ),
        (
            "无 think 标签直通",
            "纯净答案文本",
            None,
            "纯净答案文本",
        ),
        (
            "混合：<think> 后跟正常文本",
            "<think>推理过程</think>\n最终结论",
            "<think>",
            "最终结论",
        ),
    ]

    all_ok = True
    for desc, inp, forbidden, expected in cases:
        try:
            out = strip_thinking_tokens(inp)
            ok = True

            if forbidden is not None and forbidden in out:
                ok = False
                print(f"      ❌ {desc}: 仍含 '{forbidden}', output='{out}'")
            elif expected is not None and expected not in out:
                ok = False
                print(f"      ❌ {desc}: 缺失期望内容 '{expected}', output='{out}'")
            else:
                print(f"      ✓ {desc}: output='{out[:60]}'")

            if not ok:
                all_ok = False
        except Exception as exc:
            all_ok = False
            print(f"      ❌ {desc}: raised {exc}")

    report("strip_thinking_tokens variants", all_ok)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 3: DeepResearchAgent 初始化
# ──────────────────────────────────────────────────────────────────────────────

def test_agent_init() -> None:
    print("\n[3/4] DeepResearchAgent 初始化测试 ...")
    try:
        from config import Configuration
        from agent import DeepResearchAgent

        config = Configuration.from_env()
        print(f"      provider={config.llm_provider}  model={config.resolved_model()}")
        print(f"      open_source_mode={config.use_open_source_mode}  max_retries={config.open_source_model_max_retries}")
        print(f"      strip_thinking_tokens={config.strip_thinking_tokens}  no_think_mode={config.no_think_mode}")

        agent = DeepResearchAgent(config=config)
        ok = agent.llm is not None
        report(
            "DeepResearchAgent init",
            ok,
            f"model={config.resolved_model()} open_source_mode={config.use_open_source_mode} no_think_mode={config.no_think_mode}",
        )
    except Exception as exc:
        report("DeepResearchAgent init", False, str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# 测试 4: PlannerAgent 快速任务拆解
# ──────────────────────────────────────────────────────────────────────────────

def test_planner_quick() -> None:
    print("\n[4/4] PlannerAgent 快速任务拆解测试 (最长 120s) ...")
    try:
        from config import Configuration
        from agent import DeepResearchAgent
        from models import SummaryState

        config = Configuration.from_env()
        # 把研究轮数设为 1，避免触发完整研究流程
        config_overrides = config.model_copy(update={"max_web_research_loops": 1})

        agent = DeepResearchAgent(config=config_overrides)

        start = time.time()
        state = SummaryState(research_topic="Python 异步编程 asyncio 最佳实践")
        todo_items = agent.planner.plan_todo_list(state)
        elapsed = time.time() - start

        ok = len(todo_items) >= 1
        detail = f"{len(todo_items)} tasks generated in {elapsed:.1f}s"
        report("PlannerAgent quick test", ok, detail)

        if todo_items:
            print("      生成的任务列表：")
            for t in todo_items:
                print(f"        [{t.id}] {t.title} — {t.query}")
    except Exception as exc:
        import traceback
        report("PlannerAgent quick test", False, str(exc))
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Qwen3.5 适配集成测试")
    print("=" * 60)

    test_ollama_connectivity()
    test_strip_thinking_tokens()
    test_agent_init()
    test_planner_quick()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  结果汇总: {passed}/{total} 通过")
    if passed == total:
        print("  All tests passed ✅")
        sys.exit(0)
    else:
        print("  部分测试失败 ❌")
        for name, ok, detail in results:
            if not ok:
                print(f"    - {name}: {detail}")
        sys.exit(1)
