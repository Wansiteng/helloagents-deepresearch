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

该课程是 HelloAgents 智能体开发教程系列的一部分，系统介绍了如何使用 `hello-agents` 框架构建多 Agent 应用。

特此致谢原作者的开源贡献。

### 本仓库所做的修改

在原始教程代码基础之上，本仓库进行了以下修改与修复：

1. **修复 `.env` 未加载问题**：原代码缺少 `load_dotenv()` 调用，导致环境变量配置完全失效，模型始终使用默认的 `llama3.2`。已在 `src/main.py` 启动时加入正确的 dotenv 加载路径。

2. **修复并发 LLM 超时导致任务失败**：原代码对所有子任务同时开启独立线程并发调用 LLM，本地 Ollama 仅支持单请求处理，导致排队超时后任务全部标记为 failed。已引入 `threading.Semaphore(1)` 对 LLM 调用串行化。

3. **修复 note 工具调用验证失败**：LLM 生成 `[TOOL_CALL:note:{...}]` 时，会把含有未转义双引号或换行的摘要文本写入 `content` 字段，导致 `json.loads` 抛出 `JSONDecodeError` → `action` 字段丢失 → `NoteTool.validate_parameters` 返回 `False` → 工具返回"参数验证失败"，LLM 随即输出"由于工具调用验证失败，本次总结基于上下文直接生成"的免责声明，并跳过笔记同步。

   **修复内容：**

   - 新增 `backend/src/agents/robust_agent.py`，实现 `RobustToolAwareAgent`（继承自 `ToolAwareSimpleAgent`）：
     - 覆盖 `_parse_tool_parameters`，在标准 `json.loads` 失败后依次执行：
       1. **JSON 截断修复**：找到最后一个合法闭合的 `}` 截断后重新解析；
       2. **正则逐字段提取**：确保 `action`、`note_id`、`task_id`、`title`、`note_type`、`tags` 等关键字段不因 `content` 破坏而全部丢失；
     - 覆盖 `_execute_tool_call`，工具调用失败时记录完整的原始参数日志，便于排查。
   - `DeepResearchAgent._create_tool_aware_agent` 改为实例化 `RobustToolAwareAgent`，规划、摘要、报告三个子 Agent 全部使用容错解析。
   - 优化 `backend/src/services/notes.py` 笔记工具调用引导语：明确要求 LLM 在 `TOOL_CALL` 的 `content` 字段只填写**一句话简短状态描述**，完整摘要须在工具调用成功后以普通 Markdown 输出，从源头降低 JSON 格式破坏概率。
   - 修正 `backend/src/prompts.py` 中 `task_summarizer_instructions` 的 update 示例：去掉未替换的 `{task_id}` 格式化占位符，并加入同样的 `content` 约束说明。

### 许可证

原始代码遵循 hello-agents 仓库的许可协议。本仓库的修改部分同样以相同许可开放。

---

## 相关链接

- [HelloAgents 主仓库](https://github.com/Wansiteng/hello-agents)
- [hello-agents PyPI 包](https://pypi.org/project/hello-agents/)
- [Ollama 官网](https://ollama.com)
- [Tavily 搜索 API](https://tavily.com)
