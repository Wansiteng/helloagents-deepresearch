# HelloAgents DeepResearch

一个基于多 Agent 协作的**本地深度研究助手**，输入研究主题后自动完成任务规划、网络检索、内容摘要与报告生成，结果实时流式呈现。

![研究流程](https://img.shields.io/badge/LLM-Ollama%20%7C%20LMStudio%20%7C%20Custom-blue)
![搜索引擎](https://img.shields.io/badge/Search-DuckDuckGo%20%7C%20Tavily%20%7C%20Perplexity-green)
![前端](https://img.shields.io/badge/Frontend-Vue%203%20%2B%20Vite-brightgreen)
![后端](https://img.shields.io/badge/Backend-FastAPI-red)

---

## 功能特性

- **自动任务分解**：`PlannerAgent` 将研究主题拆解为 3~5 个互补子任务，每个任务携带独立搜索关键词与研究意图
- **多引擎检索**：支持 DuckDuckGo（免费）、Tavily、Perplexity、SearXNG
- **三 Agent 协作流水线**：`PlannerAgent` → `SummarizerAgent` → `WriterAgent`，各司其职、单一职责
- **统一工具注册表**：`AgentToolRegistry` 集中管理所有工具，支持链式注册与按需扩展
- **10 种 SSE 事件推送**：每个子任务进度、摘要片段、最终报告均通过 Server-Sent Events 实时播报
- **笔记持久化**：`NoteTool` 将中间进度和最终报告保存为本地 Markdown 文件
- **完全本地运行**：支持 Ollama / LMStudio 等本地 LLM，数据不出本机

---

## 系统架构

```
用户输入研究主题
        ↓
  PlannerAgent（规划 Agent）
    → 拆解为 N 个 TodoItem（含搜索关键词）
        ↓
  SearchService（并发搜索）          AgentToolRegistry（工具注册表）
    → DuckDuckGo / Tavily / Perplexity  ← NoteTool / 可插拔扩展工具
        ↓
  SummarizerAgent（摘要 Agent）
    → 对每个子任务流式提炼关键信息（SSE 推送 task_summary_chunk）
        ↓
  WriterAgent（报告 Agent）
    → 汇总生成完整 Markdown 研究报告
        ↓
  前端实时流式展示（Vue 3 + SSE，10 种事件类型）
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

4. **多智能体协同与通信解决方案设计** <sub>（更新于 2026-03-05）</sub>

   在前三项修复的基础上，对系统架构进行了更深层的重构，完整实现以下三个子功能：

   ---

   **4-1. 模块化 Agent 设计**

   将原来依赖单体大模型堆叠 Prompt 的实现，拆解为三个**单一职责 Agent**，彻底解耦研究流水线：

   | Agent 类 | 文件 | 单一职责 |
   |---|---|---|
   | `PlannerAgent` | `services/planner.py` | 意图理解与任务拆解，输出 `List[TodoItem]` |
   | `SummarizerAgent` | `services/summarizer.py` | 网页阅读与信息浓缩，支持同步与流式两种模式 |
   | `WriterAgent` | `services/reporter.py` | 逻辑组织与长文生成，消费前两个 Agent 的输出 |

   各 Agent 均保留向后兼容别名（`PlanningService = PlannerAgent` 等），不破坏旧引用。

   在 `agent.py` 中，三个 Agent 通过依赖注入组合成有序流水线：

   ```
   planner.plan_todo_list(state)
       → [并发] summarizer.stream_task_summary(state, task, context)
           → writer.generate_report(state)
   ```

   每个 Agent 独立接收 `ToolAwareSimpleAgent`（或 `RobustToolAwareAgent`）实例，互不干扰，可单独替换或测试。

   ---

   **4-2. 统一工具注册表（AgentToolRegistry）**

   新增 `backend/src/tool_registry.py`，实现可插拔的工具管理模块：

   ```python
   class AgentToolRegistry:
       def register(self, name: str, tool) -> "AgentToolRegistry": ...
       def get(self, name: str): ...
       def list_tools(self) -> list[str]: ...

       @property
       def hello_agents_registry(self) -> ToolRegistry | None: ...

       @property
       def note_tool(self) -> NoteTool | None: ...
   ```

   核心特性：
   - **链式注册**：`registry.register("wiki", WikiTool()).register("kb", KBTool())`
   - **自动内置**：`enable_notes=True` 时自动注册 `NoteTool`，无需手动配置
   - **单一实例**：`agent.py` 中 `self.tool_registry = AgentToolRegistry(self.config)` 替代原来分散的 `self.note_tool + self.tools_registry`，三个 Agent 共享同一注册表
   - **启动可观测**：FastAPI 启动时通过 `registry.list_tools()` 打印所有已注册工具名称

   ---

   **4-3. SSE 异步事件流机制**

   在 `models.py` 新增 `SSEEventType(str, Enum)`，定义 10 种标准化事件类型：

   | 事件类型 | 触发时机 |
   |---|---|
   | `status` | 全局阶段切换（初始化、生成报告等）|
   | `todo_list` | `PlannerAgent` 输出任务列表后 |
   | `task_status` | 单任务状态变更（`in_progress` / `completed` / `failed`）|
   | `sources` | 搜索完成，来源 URL 汇总 |
   | `task_summary_chunk` | `SummarizerAgent` 流式输出摘要片段 |
   | `tool_call` | Agent 调用工具时的详情 |
   | `report_note` | 最终报告笔记已持久化 |
   | `final_report` | `WriterAgent` 完整报告生成完毕 |
   | `done` | 流正常结束 |
   | `error` | 异常事件 |

   `/research/stream` 端点（`main.py`）返回 `StreamingResponse(media_type="text/event-stream")`，事件格式为标准 SSE：

   ```
   data: {"type": "task_summary_chunk", "task_id": 1, "chunk": "量子纠缠是...", "step": 1}
   ```

   并发实现：`run_stream()` 为每个子任务启动独立 `Thread`，通过线程安全的 `Queue` 汇聚事件；`Semaphore(1)` 保证本地 Ollama 串行调用，避免并发超时。

### 许可证

原始代码遵循 hello-agents 仓库的许可协议。本仓库的修改部分同样以相同许可开放。

---

## 相关链接

- [HelloAgents 主仓库](https://github.com/Wansiteng/hello-agents)
- [hello-agents PyPI 包](https://pypi.org/project/hello-agents/)
- [Ollama 官网](https://ollama.com)
- [Tavily 搜索 API](https://tavily.com)
