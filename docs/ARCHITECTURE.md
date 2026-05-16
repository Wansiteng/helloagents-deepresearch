# 架构设计 (Architecture)

> 这份文档回答两个问题：
> 1. **当前架构的硬伤**——为什么需要重构
> 2. **目标架构与迁移路径**——重构的方向与如何分阶段安全推进

阅读前请先读 [`POSITIONING.md`](POSITIONING.md)，本文所有架构选型都服务于该定位。

---

## 一、当前架构现状

### 1.1 组件全景

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Vue 3 + Vite, src/App.vue)                   │
│  ──── HTTP / SSE ───────────────────────────────────────│
└──────────────────────────────────────┬──────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────┐
│  FastAPI app (backend/src/main.py)                      │
│  endpoints:                                             │
│    GET  /health                                         │
│    GET  /probe-local-llms                               │
│    POST /llm-preflight                                  │
│    POST /research        (sync)                         │
│    POST /research/stream (SSE)                          │
│    + routers_history.py: history CRUD                   │
└──────────────────────────────────────┬──────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────┐
│  DeepResearchAgent (backend/src/agent.py, ~600 lines)   │
│  - 同时承担 sync 路径 (run) 和 stream 路径 (run_stream) │
│  - 持有：LLM、tool_registry、4 个子 Agent、向量存储     │
│  - 控制流：Thread + Queue + Semaphore                   │
└──────────────────────────────────────┬──────────────────┘
                                       │
   ┌──────────────┬─────────────┬──────┴────────┬──────────────┐
   │              │             │               │              │
   ▼              ▼             ▼               ▼              ▼
PlannerAgent  Summarizer    WriterAgent    CriticAgent     SearchService
(planner.py)  (summarizer)  (reporter.py)  (reflection)    (search.py)
   │              │             │               │
   └──────────────┴──────┬──────┴───────────────┘
                         │
                         ▼
        ┌──────────────────────────────┐
        │ RobustToolAwareAgent         │
        │ (agents/robust_agent.py)     │
        │ ↓ 依赖 ↓                     │
        │ ToolAwareSimpleAgent         │
        │ (hello_agents 包)            │
        └──────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────┐
        │  AgentToolRegistry           │
        │  (tool_registry.py)          │
        │  └─ NoteTool (services/notes)│
        └──────────────────────────────┘
```

### 1.2 现状诊断

按严重程度排列：

#### 🔴 Critical: God class

[`backend/src/agent.py`](../backend/src/agent.py) 的 `DeepResearchAgent` 单类 600+ 行，承担：
- LLM 客户端初始化 (`_init_llm`)
- 子 Agent 工厂 (`_create_tool_aware_agent`)
- 任务执行（sync 路径 `_execute_task`，~150 行）
- 任务执行（stream 路径 `worker` 嵌套函数，~150 行）——**与 sync 路径几乎完全重复**
- SSE 事件队列编排 (`enqueue` / `run_batch`)
- 上下文压缩 (`_compress_state_for_writer`)
- 报告持久化 (`_persist_final_report`)
- NoteTool ID 解析 (`_find_existing_report_note_id`, `_extract_note_id_from_text`)
- 向量库读写

任何修改都要在两处同步——典型的 fragile design。

#### 🔴 Critical: 双路径重复

`run()` 调用 `_execute_task(emit_stream=False)`，`run_stream()` 内部定义嵌套 `worker()` 函数走 `enqueue()`。两者实现搜索 → RAG 检索 → LLM 摘要 → 状态更新的逻辑各自一遍。

证据（[`agent.py`](../backend/src/agent.py) 中的代码段）：
- L255~314 `worker` 内的 search → context 构造
- L590~644 `_execute_task` 内的 search → context 构造（near-identical）

#### 🟠 Major: 控制流脆弱

- 用 `Thread` + `Queue` + `Semaphore(1)` 模拟 actor 模型，没有显式 state machine
- `__task_done__` 哨兵事件混在数据流里，靠字符串区分
- worker 抛异常只能 catch 在最外层，状态恢复粗糙
- `state.research_loop_count += 1` 在 worker 之间靠 `_state_lock` 保护，仍有读改写竞态可能

#### 🟠 Major: 强耦合 hello-agents

- `RobustToolAwareAgent` 继承自 `hello_agents.ToolAwareSimpleAgent`
- 工具调用走字符串模板 `[TOOL_CALL:note:{json}]`，需要正则 + JSON 截断兜底解析
- 系统 prompt 必须遵守 hello-agents 的格式约定
- 想换底层 agent 框架（或不用 agent 框架）就得重写整个 prompt 体系

#### 🟡 Moderate: 隐式状态共享

- `SummaryState` 在多个 worker thread 间共享
- `state.web_research_results` / `state.sources_gathered` 用 `_state_lock` 保护
- `task` 对象本身（`status`, `summary`, `sources_summary`...）在 worker 内修改、在主线程读取，没有锁

#### 🟡 Moderate: 知识源是「写死的 web search」

- 当前架构没有 KnowledgeSource 抽象
- `dispatch_search_with_retry` 直接调用 web 搜索引擎
- 想接入 Obsidian / 本地 PDF / 代码仓必须改 `_execute_task`

#### 🟢 Minor: 兼容性遗留（已在 cleanup 分支处理）

- `self.reporting = self.writer` 别名
- `PlanningService = PlannerAgent` 等 service 别名
- `main.py` 里硬编码的 `127.0.0.1:7897` Clash 代理

---

## 二、目标架构

### 2.1 设计原则

#### 原则 1：状态显式 (Explicit State)

研究流程是一个**可序列化的状态机**。每一步状态可持久化、可恢复。
- 当前：状态散落在 `SummaryState`、`task.status`、`channel_map`、`event_queue`
- 目标：单一 `ResearchSession` 对象，可 `to_json()` / `from_json()`，崩溃可恢复

#### 原则 2：知识源多态 (Pluggable Knowledge Sources)

Web search 只是知识源之一。Obsidian、本地 PDF、代码仓库、Zotero、向量记忆——都实现同一个 `KnowledgeSource` 接口。
- 当前：知识源 = web search，硬写死
- 目标：`SearchSource`、`ObsidianSource`、`PdfSource`、`CodeRepoSource`、`VectorMemorySource` 全部实现同一个 protocol

#### 原则 3：轻量自建核心 + 成熟底层

- 自己写：orchestrator / state machine / event bus
- 用现成：LLM 调用走 OpenAI 兼容 SDK（兼容 Ollama / LM Studio / vLLM / 云端），向量库用 Chroma
- 不用：LangGraph（太重，API 不稳）、AutoGen（太多 magic）、CrewAI（过度抽象）、hello-agents（耦合教程示例代码风格）

#### 原则 4：Native function calling

- 当前：字符串模板 `[TOOL_CALL:note:{json}]` + 容错解析
- 目标：OpenAI-style function calling（Qwen3、Llama3.2、DeepSeek-V3 都已原生支持）
- 收益：不再需要为字符串解析失败做兜底；工具调用错误率显著下降

#### 原则 5：用户可干预

每个状态机节点都暴露 `pre_hook` / `post_hook`，前端可在 PlannerAgent 输出后让用户编辑 todo list、否决子任务、追加问题。

---

### 2.2 目标组件设计

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Vue 3)  ─── HTTP / SSE / WebSocket ──────────│
└──────────────────────────────────────┬──────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────┐
│  FastAPI app (backend/src/main.py)                      │
│  thin layer: validation + session create/resume         │
└──────────────────────────────────────┬──────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────┐
│  ResearchSession (state machine)                        │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │ Planner  │ Executor │ Critic   │ Reporter         │  │
│  │ Step     │ Step     │ Step     │ Step             │  │
│  └──────────┴──────────┴──────────┴──────────────────┘  │
│  - persistent state ──► sessions/<id>/state.json (or sqlite)
│  - event bus  ──► async iterator for SSE                │
│  - resumable: load → continue from last node            │
│  - interruptible: pre/post hooks per step               │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────▼─────────┐         ┌─────────────────┐
        │ KnowledgeSources │◄────────┤  ToolRegistry   │
        │  ├ WebSearch     │         │  ├ NoteTool     │
        │  ├ ObsidianVault │         │  ├ CitationTool │
        │  ├ LocalPDFs     │         │  └ ...          │
        │  ├ CodeRepo      │         └─────────────────┘
        │  └ VectorMemory  │
        └─────────┬────────┘
                  │
        ┌─────────▼────────┐
        │   LLMClient      │
        │ (OpenAI compat,  │
        │  native tools)   │
        └──────────────────┘
```

### 2.3 关键接口（先写 protocol，再写实现）

```python
# backend/src/core/llm.py
from typing import Protocol, AsyncIterator
from dataclasses import dataclass

@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMMessage: ...

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
    ) -> AsyncIterator[str]: ...   # 只 yield content 增量；流式+工具调用见 PR-2
```

> **PR-1 落地修正**：`chat_stream` 原设计返回 `AsyncIterator[LLMMessage]`，实现时
> 改为 `AsyncIterator[str]`（content 增量）——流式场景前端只消费文本片段，
> 流式 + 工具调用的复杂度推迟到 PR-2。

```python
# backend/src/core/knowledge.py
from typing import Protocol
from dataclasses import dataclass

@dataclass
class KnowledgeQuery:
    text: str
    intent: str           # 子任务的研究意图，给数据源做 query rewrite
    max_results: int = 5

@dataclass
class KnowledgeChunk:
    source: str           # "web:duckduckgo" | "obsidian:vault_name" | "pdf:filename" | ...
    title: str
    url_or_path: str
    content: str
    metadata: dict

class KnowledgeSource(Protocol):
    name: str  # "web", "obsidian", "pdf", "code", "vector_memory"
    async def query(self, q: KnowledgeQuery) -> list[KnowledgeChunk]: ...
    @property
    def is_local(self) -> bool: ...   # 用于"完全本地模式"的过滤
```

```python
# backend/src/core/session.py
from enum import Enum
from typing import AsyncIterator

class StepName(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"     # 对每个 todo 跑搜索 + 摘要
    CRITIC = "critic"
    REPORT = "report"
    DONE = "done"

class ResearchSession:
    """状态机。每个节点：
    - pre_hook: 可被前端拦截编辑（让用户改 plan、否决 todo 等）
    - run: 实际执行
    - post_hook: 持久化、推 SSE 事件
    """
    session_id: str
    topic: str
    state: dict              # 全部可序列化字段
    current_step: StepName

    async def run(self) -> AsyncIterator[dict]:  # yields SSE events
        ...

    def save(self) -> None: ...
    @classmethod
    def load(cls, session_id: str) -> "ResearchSession": ...
    async def resume(self) -> AsyncIterator[dict]: ...
```

### 2.4 与现有代码的对应关系

| 当前文件 | 命运 | 备注 |
|---|---|---|
| `agent.py` (`DeepResearchAgent`) | **删除** | 拆分到 `core/session.py` + `core/steps/*.py` |
| `services/planner.py` | **重写** | 实现为 `core/steps/plan.py` 的 step；prompt 单独抽到 `prompts/` |
| `services/summarizer.py` | **重写** | 实现为 `core/steps/execute.py` 内的 summarize 阶段 |
| `services/reporter.py` | **重写** | 实现为 `core/steps/report.py` |
| `services/reflection.py` | **重写** | 实现为 `core/steps/critic.py`（保留逻辑，重写接口） |
| `services/search.py` | **重构** | 改为 `core/sources/web.py`，实现 `KnowledgeSource` |
| `services/vector_store.py` | **保留 + 包装** | 包成 `core/sources/vector_memory.py`，实现 `KnowledgeSource` |
| `services/notes.py` | **保留** | 仍作为 NoteTool，对接新的 ToolRegistry |
| `tool_registry.py` | **重写** | 与 hello-agents 解耦，使用 OpenAI function spec |
| `agents/robust_agent.py` | **删除** | native function calling 后不再需要容错解析 |
| `models.py` (`SummaryState`, `TodoItem`, ...) | **保留 + 演进** | dataclass 大部分可复用 |
| `routers_history.py` | **保留 + 接入新 session 表** | 与 ResearchSession 持久化合并 |
| `bootstrap/` | **保留** | 进程级初始化已经清晰 |
| `prompts.py` | **拆分** | 按 step 拆到 `prompts/{plan,execute,critic,report}.py` |

依赖删除清单：
- 移除 `hello-agents==0.2.9`
- 新增（如需要）：`tenacity`（重试）、`structlog`（结构化日志）

---

## 三、关键技术决策

每个决策都列出 **trade-off** 与 **理由**。

### D-1: Orchestration 框架

| 选项 | 优点 | 缺点 |
|---|---|---|
| 自建 state machine | 可控、依赖少、贴合定位 | 自己维护事件总线、持久化 |
| **LangGraph** | state graph 思想成熟、有 Studio 调试 | 太重、API 频繁变、强绑定 LangChain 生态 |
| AutoGen | 多 agent 对话原语 | 不适合本场景（不需要 agent 间对话）|
| Pydantic-AI | 类型安全 | 还在快速变化，function calling 抽象不够灵活 |

**决策：自建**，但**借鉴 LangGraph 的 state graph 思想**（节点 / 边 / 持久化 checkpoint 的概念）。

### D-2: 工具调用协议

| 选项 | 优点 | 缺点 |
|---|---|---|
| 字符串模板 + 容错解析 (现状) | 兼容老模型 | 错误率高、难调试、维护负担重 |
| **OpenAI function calling** | 标准、错误率低、广泛支持 | 极少数老模型不支持 |

**决策：硬切到 native function calling**。
- 代价：旧的 `RobustToolAwareAgent` 容错解析废弃
- 收益：不再需要 100+ 行的兜底解析；工具调用错误率从 ~5% 降到 <1%
- 退路：保留 prompt 中"如果模型不支持 function calling，可以用 ```TOOL_CALL: ...``` 格式"的兜底说明，但默认走 native

### D-3: 状态持久化

| 选项 | 优点 | 缺点 |
|---|---|---|
| JSON files (sessions/<id>/state.json) | 简单、易调试、人类可读 | 并发写有竞态、查询慢 |
| **SQLite** | 事务、并发安全、可查询、与 routers_history 合并 | 二进制、调试需要工具 |
| Postgres | 多用户场景方便 | 单用户太重 |

**决策：SQLite**。理由：
- 跟 `routers_history.py` 的现有需求自然合并（一个 DB 文件）
- 单用户场景足够，未来要多用户再升级
- WAL 模式下并发表现完全够用

### D-4: 向量库

| 选项 | 优点 | 缺点 |
|---|---|---|
| **Chroma (现状)** | 已实现、API 简单 | 性能一般、社区在转向其他选项 |
| LanceDB | 性能好、嵌入式 | 需要重写 vector_store.py |
| Qdrant (local) | 生产级 | 多一个服务进程 |

**决策：暂留 Chroma**。理由：vector_store.py 已经能用，换库是 M5 之后再考虑的事，本轮重构不引入新依赖。

### D-5: 异步模型

| 选项 | 优点 | 缺点 |
|---|---|---|
| 当前的 Thread + Queue | 已实现 | 与 FastAPI 的 async 不匹配 |
| **asyncio + `asyncio.Queue`** | 原生 async、与 FastAPI 一致 | 需要重写 worker 逻辑 |

**决策：asyncio**。理由：FastAPI、httpx、ChromaDB、OpenAI SDK 全部支持 async；线程模型只是历史遗留。

---

## 四、迁移路径（非破坏性，分 4 个 PR）

不能 big-bang 重写。每个 PR 都要保证旧路径仍可运行。

### PR-1: 引入 core 抽象（不接入主流程）— ✅ 已完成

**目标**：把新接口骨架搭起来，跑通 unit test，但不接入 `/research/stream`。

新增：
- `backend/src/core/__init__.py`
- `backend/src/core/llm.py` —— `LLMClient` protocol + `OpenAICompatibleClient` 实现（基于 `openai.AsyncOpenAI`）
- `backend/src/core/knowledge.py` —— `KnowledgeSource` protocol + `KnowledgeQuery` / `KnowledgeChunk`
- `backend/src/core/sources/web.py` —— `WebSearchSource`（包装现有 `services/search.py`）
- `backend/src/core/sources/vector_memory.py` —— `VectorMemorySource`（包装现有 `services/vector_store.py`）
- `backend/tests/unit/test_llm_client.py`、`backend/tests/unit/test_knowledge_source.py`
- `backend/tests/integration/test_core_live.py`（可选 live smoke test）

不动：所有现有文件。

**验收**：unit test 全绿；`uv run python src/main.py` 行为不变。

> 落地说明：core 接口确定为 **async**；`OpenAICompatibleClient` 提供 `from_config()`
> 过渡期适配器；`chat_stream` 返回 `AsyncIterator[str]`（见 2.3 节修正注）。

---

### PR-2: ResearchSession + feature flag — ✅ 已完成

**目标**：状态机骨架完成，通过 feature flag 在 `/research/stream` 选择走旧路径或新路径。

新增：
- `backend/src/services/factory.py` —— `ResearchServices` + `build_research_services()` 工厂
- `backend/src/core/session.py` —— `ResearchSession` async 状态机
- `backend/src/core/steps/{plan,execute,report}.py` —— 三个 step（委托现有 service）
- `backend/tests/unit/test_research_session.py`、`tests/integration/test_new_orchestrator_smoke.py`
- `.env.example` 加 `USE_NEW_ORCHESTRATOR=False` flag

修改：
- `agent.py.__init__` 改用 `build_research_services`（保行为重构）
- `config.py` 加 `use_new_orchestrator` 字段
- `main.py` 在 `/research/stream` 根据 flag 分发到旧 `DeepResearchAgent.run_stream()` 或新 `ResearchSession.run()`

> 落地说明（与原设计的偏差）：
> - PR-2 走**最小可用流程**（plan→execute→report，串行、非流式摘要）。反思评审、动态规划、
>   渐进式草稿、`tool_call` 事件、并行执行、流式摘要——推迟到后续 PR。
> - `core/steps/critic.py` 与 `prompts/` 拆包未在 PR-2 落地（无 critic step、step 仍复用
>   `prompts.py` 经由现有 service）。
> - `ResearchSession` 委托现有 `services/*`，不重写 agent 内部逻辑；`core/llm`、`core/sources`
>   仍未接入（分别待 PR-4、PR-3）。

**验收**：
- 旧路径行为不变（flag 默认关）
- 新路径能跑通最小研究流程
- flag 切换无需重启

---

### PR-3: 接入 Obsidian + LocalPDF 知识源（实现差异化价值）

**目标**：第一次让本项目跟 ChatGPT/Claude Deep Research 出现真正差异。

新增：
- `backend/src/core/sources/obsidian.py` —— 读取 vault 路径，按 frontmatter / link / tag 检索
- `backend/src/core/sources/local_pdf.py` —— 索引一个目录下所有 PDF，向量化 + 全文检索
- `.env.example` 加 `OBSIDIAN_VAULT_PATH`、`LOCAL_PDF_PATH`
- 前端配置面板加「数据源」开关

修改：
- `core/steps/execute.py` 的搜索阶段并发查询所有启用的 KnowledgeSource，结果合并

**验收**：
- 切到新 orchestrator + 启用 Obsidian 源
- 输入研究主题，能看到引用来自本地 vault 的笔记
- README 截图更新

---

### PR-4: 删除旧实现

**目标**：把过渡期代码清理掉。

删除：
- `backend/src/agent.py`（`DeepResearchAgent` 整个类）
- `backend/src/agents/robust_agent.py`
- `backend/src/services/{planner,summarizer,reporter,reflection}.py`（逻辑已迁移到 `core/steps/`）
- 移除 `hello-agents==0.2.9` 依赖
- 移除 feature flag

修改：
- `main.py` 直接调用新 `ResearchSession`
- README 同步删除旧架构描述

**验收**：
- 整个 integration test suite 通过
- `git grep "hello_agents"` 无匹配
- 跑一次完整研究，报告质量不退化

---

## 五、不在本轮做的事（明确推迟）

| 事项 | 推迟原因 | 何时做 |
|---|---|---|
| 实际写 PR-1 ~ PR-4 的代码 | 本轮（M1）只产出文档蓝图 | M2 起 |
| 多模态（PDF 图片、扫描件 OCR） | 不影响主流程 | M3 之后 |
| 替换 Chroma → LanceDB | 现有方案能用，不改 | M5 之后 |
| 多用户支持（OAuth / 用户隔离）| 与「自己用」的定位冲突 | 永不？|
| 移动端 / Chrome 插件 | 偏离用户画像 | 不做 |
| Agent 间对话 / debate | 增加复杂度无产品价值 | 不做 |

---

## 六、参考决策来源

- `agent.py` 现状代码（god class 的具体行号见 1.2 节）
- 同类项目的架构选择：[Anthropic Claude Code](https://docs.claude.com/en/docs/claude-code)、[OpenHands](https://github.com/All-Hands-AI/OpenHands)、[gpt-researcher](https://github.com/assafelovic/gpt-researcher)
- LangGraph state graph 模式：[langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)（仅借鉴思想，不直接依赖）
- 用户立意：见 [`POSITIONING.md`](POSITIONING.md)

---

> 这份文档跟 POSITIONING.md 一样要定期复盘。M2 写代码时如果发现接口不对劲，**先回来改文档**，再改代码。
