# DeepResearch (Local-First, Privacy-First)

> 一个**完全本地运行**、能深度阅读你**私人知识库**的研究助手。
>
> 项目定位、与 ChatGPT/Claude Deep Research 的差异、目标用户与演进里程碑：详见 [`docs/POSITIONING.md`](docs/POSITIONING.md)。
>
> 当前架构现状、迁移到目标架构的路径：详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20LMStudio%20%7C%20Custom-blue)
![搜索](https://img.shields.io/badge/Search-DuckDuckGo%20%7C%20Tavily%20%7C%20Perplexity-green)
![前端](https://img.shields.io/badge/Frontend-Vue%203%20%2B%20Vite-brightgreen)
![后端](https://img.shields.io/badge/Backend-FastAPI-red)

---

## 它是什么

输入一个研究主题，系统自动：

1. **拆解**任务（PlannerAgent → 3~5 个互补子任务）
2. **检索**信息（DuckDuckGo / Tavily / Perplexity / SearXNG）
3. **摘要**每个子任务（SummarizerAgent，流式输出）
4. **撰写**完整 Markdown 研究报告（WriterAgent）
5. **反思**报告质量并按需补充研究（CriticAgent，可选）

整个流程通过 SSE 实时流式推送到 Vue 3 前端。所有 LLM 推理走本地（Ollama / LM Studio / mlx-lm），数据不出本机。

## 它的差异点

| 维度 | ChatGPT Deep Research | Claude Research | 本项目 |
|---|---|---|---|
| 数据出本机（默认本地 LLM）| 是 | 是 | **否** |
| 接入个人知识库（Obsidian / PDF / 代码仓） | 否 | 否 | **规划中** |
| 通用 web 检索质量 | ★★★★★ | ★★★★★ | ★★ |
| 可中途干预研究流程 | 否 | 否 | **规划中** |
| 云端 LLM 选项（用户自带 API key）| — | — | **支持（opt-in）** |
| 单次成本 | 订阅 | 订阅 | **0**（本地）/ 按量计费（云端）|

详见 [`docs/POSITIONING.md`](docs/POSITIONING.md)。

---

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/)（推荐，已锁定依赖于 `backend/uv.lock`）
- [Ollama](https://ollama.com) / [LM Studio](https://lmstudio.ai/) / mlx-lm 任选其一

### 启动后端

```bash
cd backend
cp .env.example .env       # 编辑 .env：选择 LLM_PROVIDER、模型、搜索引擎
uv sync                    # 安装锁定依赖
uv run python src/main.py  # 默认监听 http://localhost:8000
```

最小配置（Ollama + qwen3.5:9b）：

```dotenv
LLM_PROVIDER=ollama
LOCAL_LLM=qwen3.5:9b
OLLAMA_BASE_URL=http://localhost:11434
SEARCH_API=duckduckgo
```

### 启动前端

```bash
cd frontend
npm install
npm run dev                # 默认 http://localhost:5173
```

打开浏览器访问前端，输入研究主题即可。

---

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_PROVIDER` | LLM 提供者：`ollama` / `lmstudio` / `custom` | `ollama` |
| `LOCAL_LLM` | 本地模型名称 | `llama3.2` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `LMSTUDIO_BASE_URL` | LM Studio OpenAI 兼容地址 | `http://localhost:1234/v1` |
| `SEARCH_API` | 搜索引擎：`duckduckgo` / `tavily` / `perplexity` / `searxng` | `duckduckgo` |
| `TAVILY_API_KEY` | Tavily API Key（使用 Tavily 时必填）| — |
| `LLM_TIMEOUT` | LLM 请求超时秒数 | `120` |
| `MAX_WEB_RESEARCH_LOOPS` | 研究迭代轮数 | `3` |
| `STRIP_THINKING_TOKENS` | 是否剥离 `<think>` 标签 | `True` |
| `LOCAL_PROXY_URL` | 出站 HTTP 代理（如 Clash/Mihomo），留空直连 | — |

完整配置见 [`backend/.env.example`](backend/.env.example)。

---

## 测试

```bash
cd backend
uv run pytest                              # 跑单元测试（默认跳过 integration）
uv run pytest -m integration               # 跑需要本地 LLM 的集成 smoke test
```

集成测试需要 Ollama 已加载 Qwen3.5-family 模型；缺少时会自动 skip 而不是 fail。

---

## 项目结构

```
.
├── backend/
│   ├── src/
│   │   ├── agent.py              # DeepResearchAgent 主控（待重构，见 ARCHITECTURE.md）
│   │   ├── main.py               # FastAPI 入口、SSE 端点
│   │   ├── bootstrap/            # 进程级初始化（代理配置、ddgs monkey-patch）
│   │   ├── agents/               # 容错工具调用解析层
│   │   ├── services/             # PlannerAgent / SummarizerAgent / WriterAgent / CriticAgent
│   │   └── tool_registry.py      # 统一工具注册表
│   ├── tests/                    # pytest 测试
│   └── pyproject.toml
├── frontend/                     # Vue 3 + Vite + TypeScript
├── docs/
│   ├── POSITIONING.md            # 项目定位与差异化
│   └── ARCHITECTURE.md           # 当前架构 + 目标架构 + 迁移路径
└── CHANGELOG.md
```

---

## 演进历史

详见 [`CHANGELOG.md`](CHANGELOG.md)。项目源自 [Wansiteng/hello-agents](https://github.com/Wansiteng/hello-agents) 教程仓库第十四章示例，正在向独立的「隐私优先 + 个人知识库」架构演进。

## 许可证

源自 hello-agents 教程的部分遵循其原始许可；本仓库新增内容以 MIT 开放。

## 相关链接

- 立意文档：[`docs/POSITIONING.md`](docs/POSITIONING.md)
- 架构文档：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 演进历史：[`CHANGELOG.md`](CHANGELOG.md)
- 原始教程：[Wansiteng/hello-agents](https://github.com/Wansiteng/hello-agents)
- [Ollama 官网](https://ollama.com) ｜ [Tavily 搜索 API](https://tavily.com)
