# HelloAgents DeepResearch

一个基于多 Agent 协作的**本地深度研究助手**，输入研究主题后自动完成任务规划、网络检索、内容摘要与报告生成，结果实时流式呈现。

![研究流程](https://img.shields.io/badge/LLM-Ollama%20%7C%20LMStudio%20%7C%20Custom-blue)
![搜索引擎](https://img.shields.io/badge/Search-DuckDuckGo%20%7C%20Tavily%20%7C%20Perplexity-green)
![前端](https://img.shields.io/badge/Frontend-Vue%203%20%2B%20Vite-brightgreen)
![后端](https://img.shields.io/badge/Backend-FastAPI-red)

---

## 功能特性

- **自动任务分解**：规划 Agent 将研究主题拆解为 3~5 个互补子任务
- **多引擎检索**：支持 DuckDuckGo（免费）、Tavily、Perplexity、SearXNG
- **多 Agent 流水线**：规划 → 搜索 → 摘要 → 报告，四阶段专家 Agent 协作
- **实时流式输出**：SSE 推送，每个子任务进度在前端实时展示
- **笔记持久化**：NoteTool 将中间进度和最终报告保存为本地 Markdown 文件
- **完全本地运行**：支持 Ollama / LMStudio 等本地 LLM，数据不出本机

---

## 系统架构

```
用户输入研究主题
        ↓
  PlanningService（规划 Agent）
    → 拆解为 N 个 TodoItem（含搜索关键词）
        ↓
  SearchService（并发搜索）
    → DuckDuckGo / Tavily / Perplexity
        ↓
  SummarizationService（摘要 Agent）
    → 对每个子任务提炼关键信息
        ↓
  ReportingService（报告 Agent）
    → 汇总生成完整 Markdown 研究报告
        ↓
  前端实时流式展示（Vue 3 + SSE）
```

---

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com)（推荐）或其他 LLM 服务

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/helloagents-deepresearch.git
cd helloagents-deepresearch
```

### 2. 配置后端

```bash
cd backend
cp .env.example .env
```

编辑 `.env`，填写你的配置（以 Ollama + qwen3.5:9b 为例）：

```dotenv
LLM_PROVIDER=ollama
LOCAL_LLM=qwen3.5:9b
OLLAMA_BASE_URL=http://localhost:11434
SEARCH_API=duckduckgo
```

如需使用代理（如 Clash），追加：

```dotenv
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897
NO_PROXY=localhost,127.0.0.1
```

### 3. 启动后端

```bash
cd backend
pip install fastapi "hello-agents==0.2.9" tavily-python python-dotenv requests openai "uvicorn[standard]" ddgs loguru
python src/main.py
```

> 后端默认监听 `http://localhost:8000`

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

> 前端默认运行在 `http://localhost:5173`

### 5. 开始研究

打开浏览器访问 `http://localhost:5173`，在输入框中填写研究主题，点击提交即可。

---

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_PROVIDER` | LLM 提供者：`ollama` / `lmstudio` / `custom` | `ollama` |
| `LOCAL_LLM` | 本地模型名称 | `llama3.2` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `SEARCH_API` | 搜索引擎：`duckduckgo` / `tavily` / `perplexity` / `searxng` | `duckduckgo` |
| `TAVILY_API_KEY` | Tavily API Key（使用 Tavily 时必填）| — |
| `LLM_TIMEOUT` | LLM 请求超时秒数 | `120` |
| `MAX_WEB_RESEARCH_LOOPS` | 研究迭代轮数 | `3` |
| `STRIP_THINKING_TOKENS` | 是否剥离 `<think>` 标签（适用于思维链模型）| `True` |

---

## 致谢与声明

### 原始项目来源

本项目直接源自 **[Wansiteng/hello-agents](https://github.com/Wansiteng/hello-agents)** 教程仓库中第十四章的示例代码（`code/chapter14/helloagents-deepresearch`）。

该课程由 **万思腾（Wansiteng）** 创作，是 HelloAgents 智能体开发教程系列的一部分，系统介绍了如何使用 `hello-agents` 框架构建多 Agent 应用。

特此致谢原作者的开源贡献。

### 本仓库所做的修改

在原始教程代码基础之上，本仓库进行了以下修改与修复：

1. **修复 `.env` 未加载问题**：原代码缺少 `load_dotenv()` 调用，导致环境变量配置完全失效，模型始终使用默认的 `llama3.2`。已在 `src/main.py` 启动时加入正确的 dotenv 加载路径。

2. **修复并发 LLM 超时导致任务失败**：原代码对所有子任务同时开启独立线程并发调用 LLM，本地 Ollama 仅支持单请求处理，导致排队超时后任务全部标记为 failed。已引入 `threading.Semaphore(1)` 对 LLM 调用串行化。

3. **代理支持**：补充了 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` 环境变量的说明与配置，确保国内网络环境下搜索服务可正常使用。

### 许可证

原始代码遵循 hello-agents 仓库的许可协议。本仓库的修改部分同样以相同许可开放。

---

## 相关链接

- [HelloAgents 主仓库](https://github.com/Wansiteng/hello-agents)
- [hello-agents PyPI 包](https://pypi.org/project/hello-agents/)
- [Ollama 官网](https://ollama.com)
- [Tavily 搜索 API](https://tavily.com)
