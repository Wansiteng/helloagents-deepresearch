# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。日期格式 `YYYY-MM-DD`。

## [Unreleased]

### Changed
- 仓库整体结构清理：测试文件归位到 `backend/tests/`，pytest 化重写
- 项目立意从「通用深度研究」收敛到「**隐私 × 个人知识库** deep research」，详见 [`docs/POSITIONING.md`](docs/POSITIONING.md)
- 立意细化：「不依赖云端 LLM」改为「**默认本地 LLM；云端是显式 opt-in**（用户自带 API key）」。索引/笔记层始终本地不让步
- 架构演进蓝图沉淀为文档，详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

### Added
- `backend/src/bootstrap/` 模块：把 `main.py` 里的代理配置和 `ddgs` monkey-patch 抽出
- `backend/.env.example` 新增 `LOCAL_PROXY_URL` 配置项
- `backend/tests/integration/test_qwen3_smoke.py`：pytest 化的 Qwen3 集成 smoke test，标记 `@pytest.mark.integration` 默认跳过
- `pyproject.toml` dev 依赖加入 `pytest` 与 `pytest-asyncio`，配置 `integration` marker
- **M2 / PR-1：core 抽象层**（不接入主流程，纯增量）
  - `backend/src/core/llm.py`：provider 无关的 `LLMClient` 协议 + `OpenAICompatibleClient`（基于 `openai.AsyncOpenAI`，支持 native function calling，含 `from_config` 过渡适配器）
  - `backend/src/core/knowledge.py`：`KnowledgeSource` 协议 + `KnowledgeQuery` / `KnowledgeChunk`
  - `backend/src/core/sources/web.py`、`vector_memory.py`：包装现有 `services/search.py`、`services/vector_store.py` 的两个知识源
  - `backend/tests/unit/test_llm_client.py`、`test_knowledge_source.py`：纯 unit 测试（mock，无网络）
  - `backend/tests/integration/test_core_live.py`：真实 Ollama smoke test（默认跳过）
- **M2 / PR-2：ResearchSession 状态机 + feature flag**
  - `backend/src/services/factory.py`：`build_research_services()` 工厂，统一构造 service 层；`agent.py.__init__` 改用之（保行为重构）
  - `backend/src/core/session.py` + `core/steps/{plan,execute,report}.py`：新的 async 状态机编排器，跑通最小研究流程（plan→execute→report），委托现有 service
  - `config.py` 新增 `use_new_orchestrator` 开关；`/research/stream` 按 flag 分发新旧编排器（默认走旧路径）
  - `.env.example` 新增 `USE_NEW_ORCHESTRATOR`
  - `backend/tests/unit/test_research_session.py`、`tests/integration/test_new_orchestrator_smoke.py`

### Removed
- `backend/test_qwen3.py`（旧的 `__main__` 风格集成脚本，已迁移到 pytest）
- `main.py` 硬编码的 Clash 代理 fallback `http://127.0.0.1:7897`（改走 `LOCAL_PROXY_URL` 环境变量）
- 教程遗留的向后兼容别名：`PlanningService` / `SummarizationService` / `ReportingService` / `DeepResearchAgent.reporting`

---

## 0.0.1 — 演进历史（2026-03-04 初始仓库 ~ 2026-03-19）

> 本节归档自原 README 的「致谢与所做修改」段落，保留作为项目早期演进记录。

项目源自 [Wansiteng/hello-agents](https://github.com/Wansiteng/hello-agents) 教程仓库第十四章示例代码（`code/chapter14/helloagents-deepresearch`）。在原始教程基础之上，本仓库做了以下修改与重构：

### 2026-03-04 — 初始 fork + 三项 bug 修复
1. **修复 `.env` 未加载**：原代码缺少 `load_dotenv()`，模型始终回退到默认 `llama3.2`。在 `src/main.py` 启动时加入 dotenv 加载。
2. **修复并发 LLM 超时**：原代码对所有子任务并发调用 LLM，本地 Ollama 仅支持单请求处理，导致排队超时全部失败。引入 `threading.Semaphore(1)` 串行化 LLM 调用。
3. **修复 NoteTool 调用验证失败**：LLM 生成 `[TOOL_CALL:note:{...}]` 时把含未转义双引号/换行的摘要文本写入 `content` 字段，导致 `json.loads` 抛 `JSONDecodeError`。新增 `backend/src/agents/robust_agent.py` 的 `RobustToolAwareAgent`：JSON 截断修复 + 正则逐字段提取兜底。

### 2026-03-04 — 多 Agent 架构重构
- **模块化 Agent 设计**：拆解为 `PlannerAgent` / `SummarizerAgent` / `WriterAgent` 三个单一职责 Agent
- **统一工具注册表**：新增 `backend/src/tool_registry.py` 的 `AgentToolRegistry`，链式注册、单一实例、共享给三个 Agent
- **SSE 异步事件流**：在 `models.py` 新增 `SSEEventType(str, Enum)`，定义 10 种标准化事件类型；`/research/stream` 端点返回 `StreamingResponse(media_type="text/event-stream")`

### 2026-03-05 — 历史持久化
- 新增 `backend/src/routers_history.py` 与 `backend/src/services/vector_store.py`，研究历史可查询与向量检索

### 2026-03-14 — Qwen3 适配
- 新增 `STRIP_THINKING_TOKENS` 配置，自动剥离 `<think>` 多种变体
- 新增 `NO_THINK_MODE` 在 system prompt 注入 `/no_think`
- LM Studio + Qwen3.5-35b 作为可选基座

### 2026-03-19 — 动态规划 + 反思
- 新增 `services/reflection.py` 的 `CriticAgent`：报告生成后评分 + 触发补充研究
- 新增 `/probe-local-llms`、`/llm-preflight` 端点：本地 LLM 探测与预检
