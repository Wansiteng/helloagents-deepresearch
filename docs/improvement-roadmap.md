# DeepResearch — 走向硕士级科研助手的改进路线

> 本文记录把当前 DeepResearch 工具从「快速摸底助手」升级到「能产出硕士级研究草稿」所需的全部工程改造，按 **ROI** 排序成 4 个 Tier，方便跨设备 / 跨阶段开发时随时回查。
>
> 编写日期：2026-05-21 · 维护者：Edison

---

## 0. TL;DR

| 维度 | 现状 | 改造目标 |
|---|---|---|
| 输入源 | DuckDuckGo / Tavily 等通用 web 搜 | 接 arXiv / OpenAlex / Semantic Scholar 等学术源 |
| 单源理解 | 只看 search snippet（200-500 字）| 抓 PDF / HTML 全文，按 abstract/methods/results 切片 |
| 任务协同 | N 个独立任务并行，互不通气 | 共享 running insights，后续任务能用前面任务的发现 |
| 引用 | 末尾列 URL，正文无引用 | 每句 inline `[^s3]` 引用 + 文末 IEEE/APA 参考文献 |
| 事实可信度 | 直接信 LLM 输出 | Verifier agent 交叉验证每条 claim |
| 迭代 | 一次性流水线 + 1 轮 gap fill | 真正的 spiral：跑完后让用户 push 某 task 再深挖 |
| 模型 | 默认 gemma 4B 本地 | 升 Qwen 30B+ 本地 或 / 加 Claude/GPT-4 API 路径 |

完整 16 条改进按 Tier 拆分，Tier 1 是**决定能不能用**的硬伤，Tier 2 是**让产出能进 thesis**的关键，Tier 3-4 是从「能用」到「好用」。

---

## 1. 现状评估（基线）

### 1.1 完整流程

```
用户输入主题
  ↓
PlannerAgent  ─→  4-7 个 TodoItem（title/intent/query）
  ↓
ExecuteStep（串行）
  for task in tasks:
    fan-out 到所有 KnowledgeSource（web / obsidian / vector）
      └─ web 默认拿 12 条 search snippet（提过的 5 → 12）
    build_context(chunks) → SummarizerAgent → task.summary
  ↓
（可选）GapAssessment：planner 再补 ≤2 个空白任务，跑一轮
  ↓
ReporterAgent  →  全任务摘要 → 最终 Markdown 报告
  ↓
（可选）ReflectionAgent：质量评审
```

### 1.2 维度评分

| 维度 | 评分 | 痛点 |
|---|---|---|
| 搜索源质量 | ★★☆☆☆ | 通用 web 搜，学术内容稀有 |
| 单源理解深度 | ★★☆☆☆ | search snippet 200-500 字，读不了 PDF |
| 任务拆解广度 | ★★★★☆ | 已重构，6 角度组合 + 多角度兜底 |
| 任务间协同 | ★☆☆☆☆ | 完全孤立 |
| 迭代深化 | ★★☆☆☆ | 一轮 gap 不够 |
| 事实可信度 | ★☆☆☆☆ | LLM 幻觉直接进报告 |
| 引用规范 | ★☆☆☆☆ | 末尾 URL 列表，正文无引用 |
| 模型天花板 | ★★☆☆☆ | gemma 4B 抓不到跨论文 synthesis |
| 报告结构 | ★★★☆☆ | 看似齐全，但缺方法对比 / taxonomy / limitations |

### 1.3 核心瓶颈判断

**最大杠杆**：学术源 + frontier-tier 模型 + inline 引用 + 交叉验证。这 4 条任一缺失，硕士级产出都做不到。

---

## 2. 整体策略

1. **保 SSE / Vue / Aether 架构不动**——重构都在 backend agent 层和知识源层
2. **保持 `KnowledgeSource` 抽象**——新源都实现这个 Protocol（见 `backend/src/core/knowledge.py`）
3. **保持本地优先 + 远程可选混合模式**——隐私敏感任务（Obsidian / 内部资料）走本地 LLM，公开综述用远程 frontier API
4. **每条改进项独立 mergeable**——可单独验收、单独回滚

---

## 3. Tier 1 — 决定「能不能用」的硬伤

### #1 接入学术检索源 · 4-6D · ★★★★★

**问题**：当前唯一外部源是 `WebSearchSource`（DDG / Tavily / Perplexity / SearXNG），返回的全是博客 / Wikipedia / 新闻 / 营销页。peer-reviewed 论文几乎搜不到，搜到也排在第 8 页。

**目标**：让 ExecuteStep 在每个 task 的 search 阶段同时命中：
- **arXiv**：CS/Physics/Math 预印本，开放 API，无 key
- **OpenAlex**：2.5 亿论文，含 citation graph，开放 API，无 key，推荐邮箱头
- **Semantic Scholar**（Tier 1.5）：补充覆盖，要 API key

**实施要点**：
- 文件：
  - `backend/src/core/sources/arxiv.py` — `ArxivSource` 实现 `KnowledgeSource`
  - `backend/src/core/sources/openalex.py` — `OpenAlexSource`
  - `backend/src/core/sources/semantic_scholar.py`（可选）
- 配置：
  - `Configuration.enable_arxiv: bool = True`
  - `Configuration.enable_openalex: bool = True`
  - `Configuration.openalex_email: str | None`（polite pool）
  - `Configuration.semantic_scholar_api_key: str | None`
  - env: `ENABLE_ARXIV`, `ENABLE_OPENALEX`, `OPENALEX_EMAIL`, `SEMANTIC_SCHOLAR_API_KEY`
- 工厂注册：
  - 在 `_build_knowledge_sources()` 里按 flag 追加
- chunk 内容：标题 + 作者 + 年份 + 摘要 + DOI/arxiv ID
- 用 `httpx` async（已在 lockfile）

**API 速查**：
```
arXiv:    GET http://export.arxiv.org/api/query?search_query=all:{q}&max_results=N&sortBy=relevance
          → Atom XML，stdlib xml.etree 即可

OpenAlex: GET https://api.openalex.org/works?search={q}&per-page=N&sort=relevance_score:desc
          → JSON。abstract_inverted_index 需要重建为正常文本
          Header: User-Agent: "ResearchAssistant ({email})" 进入 polite pool

Semantic Scholar: GET https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=N
                  → JSON。free tier 100 req/5min；有 key 1000 req/sec
                  Header: x-api-key: {key}
```

**验收**：
- 主题「Transformer 架构演进」跑一次，每个 task 的 sources 里至少 30% 是 arXiv/OpenAlex 链接
- 报告里能引到具体 paper title + 作者 + 年份
- 单元测试：mock httpx 返回固定 fixture，断言 chunks 数和字段映射

---

### #2 PDF 解析管线 · 3-5D · ★★★★☆

**问题**：search 拿到的 academic 链接很多是 PDF。当前 web source 只拿 search snippet（200-500 字），完整论文进不来，方法 / 数据 / 局限性这些 thesis 必需的细节全丢。

**目标**：当 chunk URL 是 PDF（或可下载 PDF），自动下载、解析、按 abstract / introduction / methods / results / discussion / refs 切片，喂给后续 vector store / summarizer。

**实施要点**：
- 新依赖：`pymupdf`（PyMuPDF，最快最准）或 `pypdf`
- 新模块：`backend/src/services/pdf_ingest.py`
  ```python
  async def fetch_and_parse_pdf(url: str) -> list[KnowledgeChunk]:
      """download → save to cache → extract text → section-split → return chunks"""
  ```
- section 切分启发式：正则匹配大写或粗体的 `Abstract` / `Introduction` / `Method[s]?` / `Result[s]?` / `Discussion` / `Conclusion` / `References`
- cache：`{cache_dir}/{sha256(url)}.pdf` + `.chunks.json`
- 在 academic source 里：如果 paper 有开放 PDF URL，调用 pdf_ingest，把每个 section 当成单独 KnowledgeChunk

**验收**：
- 任意 arXiv 论文 URL 进来，能产出 ≥5 个 chunk，每个标记 section
- summarizer prompt 里能看到 abstract + methods 这种学术结构
- cache 命中时 < 100ms 返回

---

### #3 Frontier-tier 模型路径 · 1-2D · ★★★★★

**问题**：默认本地 4B 模型推理能力不够。Qwen 30B+ 是可行的本地替代但慢；Claude / GPT-4 是质量上限。

**目标**：让用户在 UI 或 env 里选三档：
- `local-only`：planner + summarizer + reporter 全本地
- `hybrid`：planner 本地，summarizer + reporter 远程（性价比最高）
- `frontier-only`：全部远程

**实施要点**：
- `Configuration.quality_mode: Literal["local-only", "hybrid", "frontier-only"]`
- `services/factory.py` 按 mode 注入不同 agent
- 已有的 `llm_provider` / `llm_model_id` 体系够用，只需要给每个 agent 单独的 provider 配置：
  - `PLANNER_LLM_PROVIDER`, `PLANNER_LLM_MODEL`
  - `SUMMARIZER_LLM_PROVIDER`, `SUMMARIZER_LLM_MODEL`
  - `REPORTER_LLM_PROVIDER`, `REPORTER_LLM_MODEL`
- 前端搜索栏「模型」chip 弹出新增"质量模式"选项
- 在 sidebar 展示当前每个 agent 用的模型

**验收**：
- 切到 `frontier-only` 跑一次，比 local 速度快 3-5×，产出明显更长更深
- 切到 `hybrid` 时，planner 走本地，summarizer 调 Claude API 成功
- 任一 agent 调用失败有清晰错误提示

---

## 4. Tier 2 — 让产出能进 thesis

### #4 任务间共享发现 · 2-3D · ★★★★☆

**问题**：当前每个 task 在独立 prompt 里跑，task N 看不到 task 1..N-1 的发现。研究的本质是 *连接*，目前是 *并联*。

**目标**：每个 task 完成后，summarizer 之外再调一次 LLM（或复用），抽取 3-5 个 `KeyFinding`（{entity, claim, source_ids, confidence}），加入 `state.running_insights`。下一个 task 的 summarizer prompt 把 running_insights 作为「已知信息」前置。

**实施要点**：
- 数据：`SummaryState.running_insights: list[KeyFinding]`
- 新 agent / 新方法：`SummarizerAgent.extract_key_findings(task, summary) → list[KeyFinding]`
- prompt 改：
  ```
  <CONTEXT>
  以下是之前 N 个任务已经得到的关键发现，请在本任务的分析中显式建立关联或区分：
  [F1] {finding 1 + sources}
  [F2] ...
  </CONTEXT>
  ```
- 新 KeyFinding 也参与 #5 引用系统

**验收**：
- 跑「Transformer 演进」主题，第 3 个 task（说"关键技术"）的摘要里能 cite 第 1 个 task（"概念"）的发现
- running_insights 不会让 prompt 爆 ctx（cap 总条数 / 总字符）

---

### #5 Inline 引用 + 参考文献 · 3-4D · ★★★★★

**问题**：报告里"宁德时代 BMS 用了 X 算法"这种 claim 完全不知道出处。学术写作首要规范：**每个事实可追溯**。

**目标**：
1. 每个 KnowledgeChunk 在 build_context 时分配 stable `cite_id`（s1, s2, ..., a1, a2 for arxiv...）
2. summarizer / reporter prompt 强制 `每句陈述末尾以 [^s3] 标注来源`
3. 报告末自动生成 `## 参考文献` 节，按 IEEE 或 APA 格式列
4. frontend marked 渲染器把 `[^s3]` 渲染为可点击的上标，点击跳到参考列表

**实施要点**：
- `core/knowledge.py`: KnowledgeChunk 加 `cite_id: str | None`，由 `build_context` 分配
- `services/citation.py`（新）：
  - `assign_cite_ids(chunks) → chunks_with_ids`
  - `format_bibliography(chunks, style="IEEE")` → markdown 字符串
- prompt 改：
  - summarizer/reporter 系统提示加入"**每个事实陈述都必须以 [^cite_id] 结尾**，多个来源用 [^s1][^s2] 并列"
- frontend：
  - 已用 marked，加 `marked-footnote` 插件或自己实现 `[^x]` → `<sup><a href="#fn-x">` 渲染
  - 报告 footer 已有 export 区，加 "参考文献" 章节

**验收**：
- 报告里每段至少有 1 个 inline cite
- 点击 cite 上标跳到对应参考条目
- 导出 MD 时 cite 保留为标准 markdown footnote 语法

---

### #6 Claim verification · 2-3D · ★★★★☆

**问题**：LLM 幻觉照写不误。30B+ 模型幻觉率低但非零，4B 则非常严重。

**目标**：Reporter 之后新增 `VerifierAgent`：
1. 拆报告为「事实 claim」列表
2. 每个 claim 回去检索 `state.all_chunks`，看是否在 ≥1 个 chunk 里有支持
3. 标注：
   - `✓ 多源一致`：≥2 个独立 source
   - `△ 单源`：仅 1 个 source 支持
   - `⚠ 来源分歧`：source 之间矛盾
   - `❌ 未找到来源`：可能幻觉
4. 报告里给标注，或自动改写删除 `❌` 类 claim

**实施要点**：
- 新 agent: `backend/src/services/verifier.py`
- 第一版可用简单 string-match + embedding 相似度（vector store 已有）
- 进阶版：让 LLM 判断"chunk 是否支持该 claim"

**验收**：
- 报告每段 claim 都有可视化置信度标签
- 故意输入有歧义的主题（"GPT-5 的训练成本"），看到的多是 `⚠` 或 `❌`

---

### #7 二轮深挖（spiral iteration）· 3-4D · ★★★★☆

**问题**：一次性流水线，研究 = "拆 5 任务 → 全跑完 → 出报告"。真研究是 spiral：读完一轮，发现关键问题，回去深挖。

**目标**：报告生成后，UI 显示 5 个 task 的 "深度评分"，user 点击某 task → 该 task 的发现 + 剩余疑问喂给 planner，再拆出 2-3 个**子-子任务**，跑一轮，merge 进原 task。

**实施要点**：
- backend 新 endpoint：`POST /research/{id}/deepen-task` body=`{task_id, focus_question?}`
- 复用 PlannerAgent，prompt 改为「针对已有发现 + 用户聚焦问题，再拆 2-3 个子任务」
- 复用 ExecuteStep + SummarizerAgent
- frontend：task-detail 加 "再深挖" 按钮

**验收**：
- 跑完一轮后点 "再深挖" 某 task，新增的 subtasks 在 UI 里以缩进显示
- 报告 regenerate 后包含深挖内容

---

## 5. Tier 3 — 从「能用」到「好用」

### #8 Source 质量加权 · 1D
- 自动给 chunk 打标签：`academic` / `authoritative` / `wiki` / `news` / `blog`
- 简单规则：URL 域名匹配（`arxiv.org` / `*.edu` / `*.gov` / `nature.com` → academic；`wikipedia.org` → wiki；其它默认 web）
- summarizer prompt：「academic / authoritative 来源优先采信」

### #9 时间过滤 · 0.5D
- planner 可输出 `time_window: "recent_3y" | "recent_5y" | "any"` per task
- "现状 / 最新进展" 类任务默认 recent_3y；"历史 / 概念" 类不限
- 学术源调用时传 date filter（OpenAlex 有 `filter=from_publication_date:2023-01-01`）

### #10 报告模板 · 2D
- 三种模板：
  - **Literature Review**：abstract / taxonomy / methods comparison / open problems / refs
  - **Concept Primer**：definitions / history / how it works / examples / FAQ
  - **Decision Brief**：context / options / pros-cons / recommendation / risks
- 前端搜索栏旁加 chip "模板"，下拉选

### #11 对比表自动生成 · 1-2D
- "对比 / 比较" 类任务的 summarizer prompt 加 `必须输出 Markdown 表格对比`
- 后处理：识别 `| ... |` 表格，在 frontend markdown 渲染为美观表

### #12 BibTeX 导出 · 0.5D
- 已有 MD / PDF 两个导出按钮，加 ".bib"
- 用 `citation.py` 的 chunk metadata 渲染 BibTeX 条目

### #13 方法论摘要 · 2D
- 对每篇 academic chunk，额外 prompt 抽五元组：`(methodology, N/dataset, main_finding, limitation, year)`
- 存 chunk.metadata，在报告 Literature Review 模板里以表格呈现

---

## 6. Tier 4 — 研究方法论纵深

### #14 Citation graph 雪球抽样 · 4-5D
- 找到 seminal paper（高引用数）后，自动追：
  - 它引的论文（cited_by → backward citations）
  - 引它的论文（cites → forward citations）
- OpenAlex 的 `referenced_works` + `cited_by_count` 直接用
- 限制：每个 seminal paper 各方向最多 5 跳，避免爆炸

### #15 Confidence labeling · 1-2D
- 每条 reporter 输出的 claim 标 high / mid / low confidence
- 依据：source 数 + source 类型权重 + #6 verifier 输出

### #16 自承不足 · 1D
- 报告末尾必带 "本报告的局限"：
  - 哪些角度覆盖薄弱（task summary 字数低于阈值）
  - 哪些 claim 单源
  - 哪些子领域没找到 academic source（可能需要付费数据库）

---

## 7. 模型层补充

### 7.1 Qwen 30B+ 本地：可行的"中档"路径
- **够用范围**：summarizer + reporter，前提是配套 #5 #6 #4
- **不够的地方**：跨论文深层 synthesis 仍弱于 Claude/GPT-4，约慢 3-5× per token
- **硬件**：单卡 24GB 跑 Q4_K_M 量化，约 10-20 tok/s
- **建议**：
  - planner 用本地小模型（gemma 4B 够）
  - summarizer / reporter 用 Qwen 30B+
  - 一次研究总耗时 25-40 分钟（含 search latency）

### 7.2 远程 API 路径
- **首选**：Claude 3.5 Sonnet 或 GPT-4o（性价比 + 质量平衡）
- **顶配**：Claude 3.5 Opus 或 GPT-4 Turbo
- **配置**：env 加 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`，agent 工厂按 quality_mode 路由

### 7.3 混合策略表

| 任务 | 推荐模型档位 | 备注 |
|---|---|---|
| Planner | 4B 本地 | 拆任务对模型要求低 |
| Summarizer | 30B+ 本地 / Claude Sonnet | per-task synthesis |
| KeyFinding 抽取（#4） | 同 Summarizer | 复用 agent |
| Reporter | Claude Sonnet+ / GPT-4o+ | 跨任务 final synthesis |
| Verifier（#6） | 30B+ 本地 / Sonnet | 二元判断够 |
| Reflection（既有）| 30B+ 本地 | 质量评审 |

---

## 8. 推荐执行顺序

### 阶段 1：MVP for grad-level（2-3 周专注）
1. **#1 学术源** ← *正在做*
2. **#3 frontier 模型路径**（含 Qwen 30B 本地集成）
3. **#5 inline 引用**
4. **#4 任务间共享发现**
5. **#6 claim verification**

完成这 5 条，工具达到「**草稿可直接用，需人工核验**」水位。

### 阶段 2：研究方法论补全（再 2-3 周）
6. **#2 PDF 解析**
7. **#7 二轮深挖**
8. **#10 报告模板**
9. **#13 方法论摘要**

完成后达到「**接近文献综述质量**」水位。

### 阶段 3：研究产品化（按需）
- #8 #9 #11 #12 #14 #15 #16

---

## 9. 跨设备开发须知

- **主分支**：`main`，所有改动直接推（与项目当前流程一致）
- **测试**：`backend/`：`uv run pytest tests/unit -x -q`；`frontend/`：`npm run build`
- **本地服务**：backend `uv run uvicorn src.main:app --reload --port 8000`；frontend `npm run dev`
- **当前模型**：默认 `ollama:gemma4:e4b`（4B），ENV 切其它
- **配置入口**：所有运行时配置走 `backend/src/config.py::Configuration`，env 映射在该文件底部 `_env_map`
- **新增 KnowledgeSource**：照 `backend/src/core/sources/{web,obsidian,vector_memory}.py` 任一模板写
- **commit message 风格**：`feat(area): xxx` / `fix(area): xxx`，正文用主动语态描述「为什么」而非「做了什么」

### 跨设备同步检查清单
- [ ] `git pull --rebase`
- [ ] `cd backend && uv sync`（同步依赖）
- [ ] `cd frontend && npm install`
- [ ] 检查 `.env` 是否需要补字段（参照 `Configuration._env_map`）
- [ ] 跑一遍单元测试和构建
- [ ] 在 README / 这个文档里更新「正在做」标记

---

## 10. 进度跟踪

| # | 状态 | 备注 |
|---|---|---|
| 1 学术源 | 🚧 in progress | arXiv + OpenAlex 第一版 |
| 2 PDF 解析 | ⬜ planned | |
| 3 Frontier 模型路径 | ⬜ planned | |
| 4 任务间共享 | ⬜ planned | |
| 5 Inline 引用 | ⬜ planned | |
| 6 Claim verification | ⬜ planned | |
| 7 二轮深挖 | ⬜ planned | |
| 8 Source 质量加权 | ⬜ planned | |
| 9 时间过滤 | ⬜ planned | |
| 10 报告模板 | ⬜ planned | |
| 11 对比表自动 | ⬜ planned | |
| 12 BibTeX 导出 | ⬜ planned | |
| 13 方法论摘要 | ⬜ planned | |
| 14 Citation graph | ⬜ planned | |
| 15 Confidence labeling | ⬜ planned | |
| 16 自承不足 | ⬜ planned | |

> 状态标记：⬜ planned / 🚧 in progress / ✅ done / ❌ blocked

完成后请在表里更新状态。

---

*文档持续更新——欢迎在 main 分支直接改。*
