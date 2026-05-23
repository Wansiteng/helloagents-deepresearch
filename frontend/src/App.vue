<template>
  <main class="app-shell" :class="{ expanded: isExpanded }">
    <button
      v-if="!isExpanded"
      class="history-toggle-btn"
      @click="isHistoryOpen = true"
    >
      <span class="material-symbols-outlined" aria-hidden="true">history</span>
      历史记录
    </button>

    <HistoryModal :isOpen="isHistoryOpen" @close="isHistoryOpen = false" />

    <!-- 初始状态：居中输入卡片 -->
    <div v-if="!isExpanded" class="layout layout-centered">
      <section class="landing">
        <header class="landing-head">
          <div class="logo" aria-hidden="true">
            <div class="logo-grid">
              <i style="opacity: 0.95"></i>
              <i style="opacity: 0.55"></i>
              <i class="accent"></i>
              <i style="opacity: 0.55"></i>
              <i style="opacity: 0.95"></i>
              <i style="opacity: 0.35"></i>
              <i style="opacity: 0.35"></i>
              <i style="opacity: 0.55"></i>
              <i style="opacity: 0.95"></i>
            </div>
          </div>
          <div>
            <h1>深度研究助手</h1>
            <p>结合多轮智能检索与总结，实时呈现洞见与引用。</p>
          </div>
        </header>

        <form class="search-form" @submit.prevent="handleSubmit">
          <div class="search-bar" :class="{ 'is-error': !!error }">
            <span class="material-symbols-outlined search-bar-icon" aria-hidden="true">search</span>

            <textarea
              v-model="form.topic"
              ref="topicEl"
              class="search-bar-input"
              rows="1"
              placeholder="输入研究主题"
              required
              @input="autosizeTopic"
            ></textarea>

            <div class="search-bar-tools">
              <!-- 搜索引擎 chip + 弹层 -->
              <div class="search-pop-wrap" ref="engineWrapEl">
                <button
                  type="button"
                  class="search-chip"
                  :class="{ active: openMenu === 'engine' }"
                  @click="toggleMenu('engine')"
                  :aria-expanded="openMenu === 'engine'"
                  aria-haspopup="menu"
                >
                  <span class="search-chip-label">搜索</span>
                  <span class="search-chip-value">{{ engineLabel }}</span>
                </button>
                <div v-if="openMenu === 'engine'" class="popover popover-left" role="menu">
                  <div class="popover-head">搜索引擎</div>
                  <button
                    v-for="opt in engineOptions"
                    :key="opt.id"
                    type="button"
                    class="popover-item"
                    :class="{ selected: form.searchApi === opt.id }"
                    role="menuitemradio"
                    :aria-checked="form.searchApi === opt.id"
                    @click="selectEngine(opt.id)"
                  >
                    <span>{{ opt.label }}</span>
                    <span
                      v-if="form.searchApi === opt.id"
                      class="material-symbols-outlined"
                      aria-hidden="true"
                    >check</span>
                  </button>
                </div>
              </div>

              <!-- 本地 LLM chip + 弹层 -->
              <div class="search-pop-wrap" ref="llmWrapEl">
                <button
                  type="button"
                  class="search-chip"
                  :class="{ active: openMenu === 'llm' }"
                  @click="toggleMenu('llm')"
                  :aria-expanded="openMenu === 'llm'"
                  aria-haspopup="menu"
                >
                  <span class="search-chip-label">模型</span>
                  <span class="search-chip-value">{{ llmLabel }}</span>
                </button>
                <div v-if="openMenu === 'llm'" class="popover popover-right popover-llm" role="menu">
                  <header class="popover-head popover-head-row">
                    <span>本地 LLM</span>
                    <button
                      type="button"
                      class="popover-refresh"
                      :disabled="probeLoading"
                      @click="refreshProbe"
                    >
                      <span
                        class="material-symbols-outlined"
                        :class="{ spinning: probeLoading }"
                        aria-hidden="true"
                      >refresh</span>
                      {{ probeLoading ? '探测中' : '刷新' }}
                    </button>
                  </header>

                  <p v-if="probeError" class="popover-error">{{ probeError }}</p>

                  <template v-else-if="!probeLoading">
                    <div v-if="runningProviders.length === 0" class="popover-empty">
                      未检测到本地 LLM 服务（Ollama / LM Studio / mlx-lm），将沿用后端配置。
                    </div>
                    <template v-else>
                      <div class="popover-section-eyebrow">服务</div>
                      <button
                        type="button"
                        class="popover-item"
                        :class="{ selected: form.llmProvider === '' }"
                        @click="selectProvider('')"
                      >
                        <span>沿用后端配置</span>
                        <span
                          v-if="form.llmProvider === ''"
                          class="material-symbols-outlined"
                          aria-hidden="true"
                        >check</span>
                      </button>
                      <button
                        v-for="key in runningProviders"
                        :key="key"
                        type="button"
                        class="popover-item"
                        :class="{ selected: form.llmProvider === key }"
                        @click="selectProvider(key)"
                      >
                        <span>{{ PROVIDER_LABELS[key] ?? key }}</span>
                        <span
                          v-if="form.llmProvider === key"
                          class="material-symbols-outlined"
                          aria-hidden="true"
                        >check</span>
                      </button>

                      <template v-if="form.llmProvider && modelsForProvider.length > 0">
                        <div class="popover-divider" aria-hidden="true"></div>
                        <div class="popover-section-eyebrow">模型</div>
                        <button
                          v-for="m in modelsForProvider"
                          :key="m"
                          type="button"
                          class="popover-item"
                          :class="{ selected: form.llmModel === m }"
                          @click="selectModel(m)"
                        >
                          <span>{{ m }}</span>
                          <span
                            v-if="form.llmModel === m"
                            class="material-symbols-outlined"
                            aria-hidden="true"
                          >check</span>
                        </button>
                      </template>
                      <p
                        v-else-if="form.llmProvider"
                        class="popover-empty"
                      >该服务下当前无可列出的模型</p>
                    </template>
                  </template>

                  <div v-else class="popover-loading">
                    <span class="material-symbols-outlined spinner-sm" aria-hidden="true">progress_activity</span>
                    正在探测本地 LLM 服务…
                  </div>
                </div>
              </div>

              <!-- AI 提交 -->
              <span class="ai-halo" :class="{ active: loading }">
                <button
                  class="search-submit"
                  type="submit"
                  :disabled="loading || preflightChecking"
                >
                  <span class="material-symbols-outlined" aria-hidden="true">
                    {{ (loading || preflightChecking) ? 'auto_awesome' : 'arrow_forward' }}
                  </span>
                  <span class="search-submit-label">
                    {{ preflightChecking ? '检测中' : loading ? '研究中' : '开始研究' }}
                  </span>
                </button>
              </span>
            </div>
          </div>

          <div v-if="error" class="alert alert-error search-error" role="alert">
            <span class="material-symbols-outlined filled" aria-hidden="true">error</span>
            <div>
              <p class="search-error-msg">{{ error }}</p>
              <p
                v-if="preflightHint"
                class="search-error-hint"
                v-html="preflightHint.replace(/\n/g, '<br/>')"
              ></p>
            </div>
          </div>
          <p v-else-if="loading" class="search-status hint muted">
            正在收集线索与证据，实时进展见右侧区域。
          </p>

          <div v-if="loading" class="search-cancel-row">
            <button
              type="button"
              class="secondary-btn"
              @click="cancelResearch"
            >取消研究</button>
          </div>
        </form>
      </section>
    </div>

    <!-- 全屏状态：左右分栏布局 -->
    <div v-else class="layout layout-fullscreen" :class="{ 'sidebar-collapsed': sidebarCollapsed }">
      <!-- 折叠后顶部左侧的展开按钮 -->
      <button
        v-if="sidebarCollapsed"
        type="button"
        class="sidebar-expand-btn"
        @click="toggleSidebar"
        title="展开侧边栏"
        aria-label="展开侧边栏"
      >
        <span class="material-symbols-outlined" aria-hidden="true">menu_open</span>
      </button>

      <!-- 左侧：研究信息 -->
      <aside class="sidebar" :aria-hidden="sidebarCollapsed">
        <div class="sidebar-header">
          <div class="sidebar-brand">
            <div class="logo logo-sm" aria-hidden="true">
              <div class="logo-grid">
                <i style="opacity: 0.95"></i>
                <i style="opacity: 0.55"></i>
                <i class="accent"></i>
                <i style="opacity: 0.55"></i>
                <i style="opacity: 0.95"></i>
                <i style="opacity: 0.35"></i>
                <i style="opacity: 0.35"></i>
                <i style="opacity: 0.55"></i>
                <i style="opacity: 0.95"></i>
              </div>
            </div>
            <h2>深度研究助手</h2>
          </div>
          <button
            type="button"
            class="sidebar-collapse-btn"
            @click="toggleSidebar"
            title="收起侧边栏"
            aria-label="收起侧边栏"
            :tabindex="sidebarCollapsed ? -1 : 0"
          >
            <span class="material-symbols-outlined" aria-hidden="true">menu_open</span>
          </button>
        </div>

        <div class="research-info">
          <div class="info-item">
            <label>研究主题</label>
            <p class="topic-display">{{ form.topic }}</p>
          </div>

          <div class="info-item" v-if="form.searchApi">
            <label>搜索引擎</label>
            <p>{{ form.searchApi }}</p>
          </div>

          <div class="info-item" v-if="form.llmProvider">
            <label>本地模型</label>
            <p>{{ PROVIDER_LABELS[form.llmProvider] ?? form.llmProvider }}{{ form.llmModel ? ` · ${form.llmModel}` : "" }}</p>
          </div>

          <div class="info-item error-info" v-if="error && !loading">
            <label>错误详情</label>
            <p class="error-detail-text">{{ error }}</p>
          </div>

          <div class="info-item" v-if="totalTasks > 0">
            <label>研究进度</label>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: `${(completedTasks / totalTasks) * 100}%` }"></div>
            </div>
            <p class="progress-text">{{ completedTasks }} / {{ totalTasks }} 任务完成</p>
          </div>
        </div>

        <div class="sidebar-actions">
          <button class="new-research-btn" @click="startNewResearch">
            <span class="material-symbols-outlined" aria-hidden="true">add</span>
            开始新研究
          </button>
        </div>
      </aside>

      <!-- 右侧：研究结果 -->
      <section
        class="panel-result"
        v-if="todoTasks.length || reportMarkdown || progressLogs.length"
      >
        <header class="status-bar">
          <div class="status-main">
            <div class="status-chip" :class="researchPhase">
              <span
                v-if="researchPhase === 'complete'"
                class="material-symbols-outlined status-check"
                aria-hidden="true"
              >check_circle</span>
              <span v-else class="dot"></span>
              {{ researchPhaseLabel }}
            </div>
            <span class="status-meta">
              任务进度：{{ completedTasks }} / {{ totalTasks || todoTasks.length || 1 }}
              · 阶段记录 {{ progressLogs.length }} 条
            </span>
          </div>
          <div class="status-controls">
            <button
              v-if="loading"
              type="button"
              class="secondary-btn status-stop-btn"
              @click="cancelResearch"
              title="强制停止当前研究"
            >
              <span class="material-symbols-outlined" aria-hidden="true">stop_circle</span>
              停止研究
            </button>
            <button class="secondary-btn" @click="logsCollapsed = !logsCollapsed">
              {{ logsCollapsed ? "展开流程" : "收起流程" }}
            </button>
            <button
              type="button"
              class="secondary-btn status-history-btn"
              @click="isHistoryOpen = true"
              title="查看历史记录"
            >
              <span class="material-symbols-outlined" aria-hidden="true">history</span>
              历史记录
            </button>
          </div>
        </header>

        <div class="timeline-wrapper" v-show="!logsCollapsed && progressLogs.length">
          <transition-group name="timeline" tag="ul" class="timeline">
            <li v-for="(log, index) in progressLogs" :key="`${log}-${index}`">
              <span class="timeline-node"></span>
              <p>{{ log }}</p>
            </li>
          </transition-group>
        </div>

        <div class="tasks-section" v-if="todoTasks.length">
          <aside class="tasks-list">
            <h3>任务清单</h3>
            <ul>
              <li
                v-for="task in todoTasks"
                :key="task.id"
                :class="['task-item', { active: task.id === activeTaskId, completed: task.status === 'completed' }]"
              >
                <button
                  type="button"
                  class="task-button"
                  @click="activeTaskId = task.id"
                >
                  <span class="task-title">{{ task.title }}</span>
                  <span class="task-status" :class="task.status">
                    <span
                      v-if="task.status === 'in_progress'"
                      class="material-symbols-outlined"
                      aria-hidden="true"
                    >progress_activity</span>
                    {{ formatTaskStatus(task.status) }}
                  </span>
                </button>
                <p class="task-intent">{{ task.intent }}</p>
              </li>
            </ul>
          </aside>

          <article class="task-detail" v-if="currentTask">
            <header class="task-header">
              <div>
                <h3>{{ currentTaskTitle || "当前任务" }}</h3>
                <p class="muted" v-if="currentTaskIntent">
                  {{ currentTaskIntent }}
                </p>
              </div>
              <div class="task-chip-group">
                <span class="task-label">查询：{{ currentTaskQuery || "" }}</span>
                <span
                  v-if="currentTaskNoteId"
                  class="task-label note-chip"
                  :title="currentTaskNoteId"
                >
                  笔记：{{ currentTaskNoteId }}
                </span>
                <span
                  v-if="currentTaskNotePath"
                  class="task-label note-chip path-chip"
                  :title="currentTaskNotePath"
                >
                  <span class="path-label">路径：</span>
                  <span class="path-text">{{ currentTaskNotePath }}</span>
                  <button
                    class="chip-action"
                    type="button"
                    @click="copyNotePath(currentTaskNotePath)"
                  >
                    复制
                  </button>
                </span>
              </div>
            </header>

            <section v-if="currentTask && currentTask.notices.length" class="task-notices">
              <h4>系统提示</h4>
              <ul>
                <li v-for="(notice, idx) in currentTask.notices" :key="`${notice}-${idx}`">
                  {{ notice }}
                </li>
              </ul>
            </section>

            <section
              class="sources-block"
              :class="{ 'block-highlight': sourcesHighlight }"
            >
              <h3>最新来源</h3>
              <template v-if="currentTaskSources.length">
                <ul class="sources-list">
                  <li
                    v-for="(item, index) in currentTaskSources"
                    :key="`${item.title}-${index}`"
                    class="source-item"
                  >
                    <a
                      class="source-link"
                      :href="item.url || '#'"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {{ item.title || item.url || `来源 ${index + 1}` }}
                    </a>
                    <div v-if="item.snippet || item.raw" class="source-tooltip">
                      <p v-if="item.snippet">{{ item.snippet }}</p>
                      <p v-if="item.raw" class="muted-text">{{ item.raw }}</p>
                    </div>
                  </li>
                </ul>
              </template>
              <p v-else class="muted">暂无可用来源</p>
            </section>

            <section
              class="summary-block"
              :class="{ 'block-highlight': summaryHighlight }"
            >
              <h3>任务总结</h3>
              <div
                v-if="currentTaskSummary"
                class="markdown-body"
                v-html="renderedTaskSummary"
              ></div>
              <p v-else class="muted">暂无可用信息</p>
            </section>

            <section
              class="tools-block"
              :class="{ 'block-highlight': toolHighlight }"
              v-if="currentTaskToolCalls.length"
            >
              <h3>工具调用记录</h3>
              <ul class="tool-list">
                <li
                  v-for="entry in currentTaskToolCalls"
                  :key="`${entry.eventId}-${entry.timestamp}`"
                  class="tool-entry"
                >
                  <div class="tool-entry-header">
                    <span class="tool-entry-title">
                      #{{ entry.eventId }} {{ entry.agent }} → {{ entry.tool }}
                    </span>
                    <span
                      v-if="entry.noteId"
                      class="tool-entry-note"
                    >
                      笔记：{{ entry.noteId }}
                    </span>
                  </div>
                  <p v-if="entry.notePath" class="tool-entry-path">
                    笔记路径：
                    <button
                      class="link-btn"
                      type="button"
                      @click="copyNotePath(entry.notePath)"
                    >
                      复制
                    </button>
                    <span class="path-text">{{ entry.notePath }}</span>
                  </p>
                  <p class="tool-subtitle">参数</p>
                  <pre class="tool-pre">{{ formatToolParameters(entry.parameters) }}</pre>
                  <template v-if="entry.result">
                    <p class="tool-subtitle">执行结果</p>
                    <pre class="tool-pre">{{ formatToolResult(entry.result) }}</pre>
                  </template>
                </li>
              </ul>
            </section>
          </article>

          <article class="task-detail" v-else>
            <p class="muted">等待任务规划或执行结果。</p>
          </article>
        </div>

        <div
          v-if="reportMarkdown"
          class="report-block"
          :class="{ 'block-highlight': reportHighlight }"
        >
          <header class="report-head">
            <h3>最终报告</h3>
            <div class="report-actions">
              <button
                type="button"
                class="secondary-btn"
                @click="exportMarkdown"
                :disabled="!reportMarkdown"
                title="下载原始 Markdown"
              >
                <span class="material-symbols-outlined" aria-hidden="true">description</span>
                导出 MD
              </button>
              <button
                type="button"
                class="secondary-btn"
                @click="exportPdf"
                :disabled="!reportMarkdown || exportingPdf"
                title="导出为 PDF（含目录与排版）"
              >
                <span class="material-symbols-outlined" aria-hidden="true">picture_as_pdf</span>
                {{ exportingPdf ? "生成中…" : "导出 PDF" }}
              </button>
            </div>
          </header>
          <div class="report-layout" :class="{ 'no-toc': reportToc.length === 0 }">
            <nav class="report-toc" v-if="reportToc.length" aria-label="报告目录">
              <p class="toc-title">目录</p>
              <ul>
                <li
                  v-for="item in reportToc"
                  :key="item.id"
                  :class="['toc-level-' + item.level, { active: activeTocId === item.id }]"
                >
                  <a :href="`#${item.id}`" @click.prevent="scrollToHeading(item.id)">{{ item.text }}</a>
                </li>
              </ul>
            </nav>
            <article
              class="report-main markdown-body"
              ref="reportMainEl"
              v-html="renderedReport"
            ></article>
          </div>
        </div>
      </section>

    </div>
  </main>
</template>

<script lang="ts" setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { marked } from "marked";
// Renders `[^N]` inline citations + auto-anchors the "[^N]: …" footnote
// definitions emitted by the backend ReporterAgent's bibliography pass.
import markedFootnote from "marked-footnote";
import HistoryModal from "./components/HistoryModal.vue";

interface TocItem {
  id: string;
  level: number;
  text: string;
}

import {
  runResearchStream,
  probeLocalLLMs,
  llmPreflight,
  type ResearchStreamEvent,
  type LocalLLMServiceInfo
} from "./services/api";

const isHistoryOpen = ref(false);

interface SourceItem {
  title: string;
  url: string;
  snippet: string;
  raw: string;
}

interface ToolCallLog {
  eventId: number;
  agent: string;
  tool: string;
  parameters: Record<string, unknown>;
  result: string;
  noteId: string | null;
  notePath: string | null;
  timestamp: number;
}

interface TodoTaskView {
  id: number;
  title: string;
  intent: string;
  query: string;
  status: string;
  summary: string;
  sourcesSummary: string;
  sourceItems: SourceItem[];
  notices: string[];
  noteId: string | null;
  notePath: string | null;
  toolCalls: ToolCallLog[];
}

const form = reactive({
  topic: "",
  searchApi: "",
  llmProvider: "",
  llmModel: ""
});

// ── Local LLM probe ──────────────────────────────────────────────────────────
const probeServices = ref<Record<string, LocalLLMServiceInfo>>({});
const probeLoading = ref(false);
const probeError = ref("");

const PROVIDER_LABELS: Record<string, string> = {
  ollama: "Ollama",
  lmstudio: "LM Studio",
  mlx: "MLX (mlx-lm)",
};

const runningProviders = computed(() =>
  Object.entries(probeServices.value)
    .filter(([, info]) => info.running)
    .map(([key]) => key)
);

const modelsForProvider = computed<string[]>(() => {
  if (!form.llmProvider) return [];
  const info = probeServices.value[form.llmProvider];
  return info?.models ?? [];
});

async function refreshProbe() {
  probeLoading.value = true;
  probeError.value = "";
  form.llmProvider = "";
  form.llmModel = "";
  try {
    const result = await probeLocalLLMs();
    probeServices.value = result.services;
    const first = Object.entries(result.services).find(([, v]) => v.running);
    if (first) {
      form.llmProvider = first[0];
      form.llmModel = first[1].models[0] ?? "";
    }
  } catch (e) {
    probeError.value = "无法连接后端探测接口，请确认后端已启动";
  } finally {
    probeLoading.value = false;
  }
}

onMounted(() => {
  refreshProbe();
  document.addEventListener("click", onDocClick);
  document.addEventListener("keydown", onDocKeydown);
});
// ─────────────────────────────────────────────────────────────────────────────

const loading = ref(false);
const preflightChecking = ref(false);
const preflightHint = ref("");
const error = ref("");
const progressLogs = ref<string[]>([]);
const logsCollapsed = ref(false);
const isExpanded = ref(false);
const sidebarCollapsed = ref(false);

function toggleSidebar(): void {
  sidebarCollapsed.value = !sidebarCollapsed.value;
}

/**
 * Status chip phase:
 *   - "running": SSE still streaming AND no final report yet
 *   - "complete": final report arrived (even if stream is still finishing)
 *   - "idle":    nothing has run yet on this screen
 *
 * We key on reportMarkdown (not on loading alone) so the chip flips to
 * "完成" the moment the report event lands, instead of waiting for the
 * stream's tail to close.
 */
const researchPhase = computed<"running" | "complete" | "idle">(() => {
  const hasReport = reportMarkdown.value && reportMarkdown.value !== "暂无生成的报告";
  if (hasReport) return "complete";
  if (loading.value) return "running";
  return "idle";
});

const researchPhaseLabel = computed(() => {
  switch (researchPhase.value) {
    case "running":  return "研究进行中";
    case "complete": return "研究完成";
    default:         return "等待开始";
  }
});

const todoTasks = ref<TodoTaskView[]>([]);
const activeTaskId = ref<number | null>(null);
const reportMarkdown = ref("");

const summaryHighlight = ref(false);
const sourcesHighlight = ref(false);
const reportHighlight = ref(false);
const toolHighlight = ref(false);

let currentController: AbortController | null = null;

const searchOptions = [
  "advanced",
  "duckduckgo",
  "tavily",
  "perplexity",
  "searxng"
];

// ── Landing search bar — chip popovers + keyboard handling ─────────────
const openMenu = ref<null | "engine" | "llm">(null);
const topicEl = ref<HTMLTextAreaElement | null>(null);
const engineWrapEl = ref<HTMLElement | null>(null);
const llmWrapEl = ref<HTMLElement | null>(null);

const engineOptions = computed(() => [
  { id: "", label: "默认" },
  ...searchOptions.map((id) => ({ id, label: id })),
]);

const engineLabel = computed(() => {
  const found = engineOptions.value.find((o) => o.id === form.searchApi);
  return found ? found.label : "默认";
});

const llmLabel = computed(() => {
  if (!form.llmProvider) return "默认";
  const provider = PROVIDER_LABELS[form.llmProvider] ?? form.llmProvider;
  return form.llmModel ? `${provider} · ${form.llmModel}` : provider;
});

function toggleMenu(k: "engine" | "llm"): void {
  openMenu.value = openMenu.value === k ? null : k;
}
function closeMenu(): void {
  openMenu.value = null;
}

function selectEngine(id: string): void {
  form.searchApi = id;
  closeMenu();
}
function selectProvider(id: string): void {
  form.llmProvider = id;
  // Reset model when provider changes (matches the original select onChange)
  form.llmModel = id ? (modelsForProvider.value[0] ?? "") : "";
  if (!id) closeMenu();
}
function selectModel(m: string): void {
  form.llmModel = m;
  closeMenu();
}


function autosizeTopic(): void {
  const el = topicEl.value;
  if (!el) return;
  el.style.height = "auto";
  el.style.height = Math.min(96, el.scrollHeight) + "px";
}

function onDocClick(e: MouseEvent): void {
  if (!openMenu.value) return;
  const target = e.target as Node;
  if (engineWrapEl.value?.contains(target)) return;
  if (llmWrapEl.value?.contains(target)) return;
  closeMenu();
}
function onDocKeydown(e: KeyboardEvent): void {
  if (e.key === "Escape" && openMenu.value) closeMenu();
}

const TASK_STATUS_LABEL: Record<string, string> = {
  pending: "待执行",
  in_progress: "进行中",
  completed: "已完成",
  skipped: "已跳过"
};

function formatTaskStatus(status: string): string {
  return TASK_STATUS_LABEL[status] ?? status;
}

const totalTasks = computed(() => todoTasks.value.length);
const completedTasks = computed(() =>
  todoTasks.value.filter((task) => task.status === "completed").length
);

const currentTask = computed(() => {
  if (activeTaskId.value !== null) {
    return todoTasks.value.find((task) => task.id === activeTaskId.value) ?? null;
  }
  return todoTasks.value[0] ?? null;
});

const currentTaskSources = computed(() => currentTask.value?.sourceItems ?? []);
const currentTaskSummary = computed(() => currentTask.value?.summary ?? "");

// ── Markdown rendering ─────────────────────────────────────────────────
// marked is configured for GFM (tables, strikethrough) + line-break = enter.
marked.setOptions({ gfm: true, breaks: true });
marked.use(markedFootnote({ description: "参考文献" }));

function renderMarkdown(src: string): string {
  if (!src) return "";
  try {
    return marked.parse(src) as string;
  } catch (err) {
    console.error("markdown render failed", err);
    return src;
  }
}

const renderedTaskSummary = computed(() => renderMarkdown(currentTaskSummary.value));

const renderedReport = ref("");
const reportToc = ref<TocItem[]>([]);
const activeTocId = ref<string>("");
const exportingPdf = ref(false);
const reportMainEl = ref<HTMLElement | null>(null);

let tocObserver: IntersectionObserver | null = null;

/** Slugify text → DOM id. Keeps Latin alphanumerics + CJK; everything else becomes "-". */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w一-龥]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Walk rendered HTML, inject ids on h1/h2/h3, and collect a TOC. */
function injectHeadingIdsAndCollectToc(html: string): { html: string; toc: TocItem[] } {
  const toc: TocItem[] = [];
  const counts = new Map<string, number>();
  const next = html.replace(/<h([1-3])([^>]*)>([\s\S]*?)<\/h\1>/g, (_m, lvl, attrs, inner) => {
    const text = String(inner).replace(/<[^>]+>/g, "").trim();
    // Skip the auto-injected footnote-section label (marked-footnote emits
    // <h2 id="footnote-label" class="sr-only">参考文献</h2>) — it's a screen-
    // reader anchor, not a real outline entry.
    const isFootnoteLabel =
      /\bid=["']footnote-label["']/i.test(attrs) ||
      /\bclass=["'][^"']*\bsr-only\b/i.test(attrs);

    let slug = slugify(text);
    if (!slug) slug = `h-${toc.length + 1}`;
    const c = counts.get(slug) ?? 0;
    counts.set(slug, c + 1);
    const id = c === 0 ? slug : `${slug}-${c}`;
    if (!isFootnoteLabel) {
      toc.push({ id, level: Number(lvl), text });
    }
    return `<h${lvl}${attrs} id="${id}">${inner}</h${lvl}>`;
  });
  return { html: next, toc };
}

function rebuildReport(md: string): void {
  if (!md) {
    renderedReport.value = "";
    reportToc.value = [];
    activeTocId.value = "";
    return;
  }
  const rendered = renderMarkdown(md);
  const { html, toc } = injectHeadingIdsAndCollectToc(rendered);
  renderedReport.value = html;
  reportToc.value = toc;
  activeTocId.value = toc[0]?.id ?? "";
  // Re-arm the IntersectionObserver after the new DOM is in place.
  nextTick(() => setupTocObserver());
}

function setupTocObserver(): void {
  if (tocObserver) {
    tocObserver.disconnect();
    tocObserver = null;
  }
  const root = reportMainEl.value;
  if (!root || reportToc.value.length === 0) return;
  const headings = Array.from(root.querySelectorAll<HTMLElement>("h1[id], h2[id], h3[id]"));
  if (headings.length === 0) return;
  tocObserver = new IntersectionObserver(
    (entries) => {
      // Pick the topmost heading currently intersecting near the top of viewport.
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length > 0) {
        const id = (visible[0].target as HTMLElement).id;
        if (id) activeTocId.value = id;
      }
    },
    { rootMargin: "-80px 0px -70% 0px", threshold: 0 },
  );
  headings.forEach((h) => tocObserver!.observe(h));
}

function scrollToHeading(id: string): void {
  const el = reportMainEl.value?.querySelector<HTMLElement>(`#${CSS.escape(id)}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  activeTocId.value = id;
}

watch(reportMarkdown, (md) => rebuildReport(md), { immediate: true });

// ── Exports ────────────────────────────────────────────────────────────
function reportFilename(ext: string): string {
  const topic = (form.topic || "deep-research").trim().slice(0, 40);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  const safe = topic.replace(/[\\/:*?"<>|\s]+/g, "_");
  return `${safe}-${stamp}.${ext}`;
}

function exportMarkdown(): void {
  if (!reportMarkdown.value) return;
  const blob = new Blob([reportMarkdown.value], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = reportFilename("md");
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Free the blob URL on the next tick so the download has time to start.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportPdf(): Promise<void> {
  if (!reportMarkdown.value || exportingPdf.value) return;
  const target = reportMainEl.value;
  if (!target) return;
  exportingPdf.value = true;
  try {
    // Dynamic-import so the libraries are only pulled in when the user actually exports.
    const [{ default: html2canvas }, jspdfModule] = await Promise.all([
      import("html2canvas"),
      import("jspdf"),
    ]);
    const JsPdfCtor = (jspdfModule as any).jsPDF ?? (jspdfModule as any).default;

    // Render against the current theme's page background so dark mode prints correctly.
    const bg =
      getComputedStyle(document.documentElement).getPropertyValue("--bg-page").trim() ||
      "#ffffff";

    const canvas = await html2canvas(target, {
      scale: 2,
      useCORS: true,
      backgroundColor: bg,
      windowWidth: target.scrollWidth,
    });

    const pdf = new JsPdfCtor({ unit: "pt", format: "a4", orientation: "portrait" });
    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    const margin = 32;
    const usableW = pageW - margin * 2;
    const imgH = (canvas.height * usableW) / canvas.width;

    // Paginate by drawing the same tall image with a shifted Y offset on each page.
    const imgData = canvas.toDataURL("image/png");
    let heightLeft = imgH;
    let position = margin;
    pdf.addImage(imgData, "PNG", margin, position, usableW, imgH);
    heightLeft -= pageH - margin * 2;
    while (heightLeft > 0) {
      pdf.addPage();
      position = margin - (imgH - heightLeft);
      pdf.addImage(imgData, "PNG", margin, position, usableW, imgH);
      heightLeft -= pageH - margin * 2;
    }
    pdf.save(reportFilename("pdf"));
  } catch (err) {
    console.error("PDF export failed", err);
    alert(`PDF 导出失败：${(err as Error).message || err}`);
  } finally {
    exportingPdf.value = false;
  }
}
const currentTaskTitle = computed(() => currentTask.value?.title ?? "");
const currentTaskIntent = computed(() => currentTask.value?.intent ?? "");
const currentTaskQuery = computed(() => currentTask.value?.query ?? "");
const currentTaskNoteId = computed(() => currentTask.value?.noteId ?? "");
const currentTaskNotePath = computed(() => currentTask.value?.notePath ?? "");
const currentTaskToolCalls = computed(
  () => currentTask.value?.toolCalls ?? []
);

const pulse = (flag: typeof summaryHighlight) => {
  flag.value = false;
  requestAnimationFrame(() => {
    flag.value = true;
    window.setTimeout(() => {
      flag.value = false;
    }, 1200);
  });
};

function parseSources(raw: string): SourceItem[] {
  if (!raw) {
    return [];
  }

  const items: SourceItem[] = [];
  const lines = raw.split("\n");

  let current: SourceItem | null = null;
  const truncate = (value: string, max = 360) => {
    const trimmed = value.trim();
    return trimmed.length > max ? `${trimmed.slice(0, max)}…` : trimmed;
  };

  const flush = () => {
    if (!current) {
      return;
    }
    const normalized: SourceItem = {
      title: current.title?.trim() || "",
      url: current.url?.trim() || "",
      snippet: current.snippet ? truncate(current.snippet) : "",
      raw: current.raw ? truncate(current.raw, 420) : ""
    };

    if (
      normalized.title ||
      normalized.url ||
      normalized.snippet ||
      normalized.raw
    ) {
      if (!normalized.title && normalized.url) {
        normalized.title = normalized.url;
      }
      items.push(normalized);
    }
    current = null;
  };

  const ensureCurrent = () => {
    if (!current) {
      current = { title: "", url: "", snippet: "", raw: "" };
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    if (/^\*/.test(trimmed) && trimmed.includes(" : ")) {
      flush();
      const withoutBullet = trimmed.replace(/^\*\s*/, "");
      const [titlePart, urlPart] = withoutBullet.split(" : ");
      current = {
        title: titlePart?.trim() || "",
        url: urlPart?.trim() || "",
        snippet: "",
        raw: ""
      };
      continue;
    }

    if (/^(Source|信息来源)\s*:/.test(trimmed)) {
      flush();
      const [, titlePart = ""] = trimmed.split(/:\s*(.+)/);
      current = {
        title: titlePart.trim(),
        url: "",
        snippet: "",
        raw: ""
      };
      continue;
    }

    if (/^URL\s*:/.test(trimmed)) {
      ensureCurrent();
      const [, urlPart = ""] = trimmed.split(/:\s*(.+)/);
      current!.url = urlPart.trim();
      continue;
    }

    if (
      /^(Most relevant content from source|信息内容)\s*:/.test(trimmed)
    ) {
      ensureCurrent();
      const [, contentPart = ""] = trimmed.split(/:\s*(.+)/);
      current!.snippet = contentPart.trim();
      continue;
    }

    if (
      /^(Full source content limited to|信息内容限制为)\s*:/.test(trimmed)
    ) {
      ensureCurrent();
      const [, rawPart = ""] = trimmed.split(/:\s*(.+)/);
      current!.raw = rawPart.trim();
      continue;
    }

    if (/^https?:\/\//.test(trimmed)) {
      ensureCurrent();
      if (!current!.url) {
        current!.url = trimmed;
        continue;
      }
    }

    ensureCurrent();
    current!.raw = current!.raw ? `${current!.raw}\n${trimmed}` : trimmed;
  }

  flush();
  return items;
}

function extractOptionalString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function ensureRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function applyNoteMetadata(
  task: TodoTaskView,
  payload: Record<string, unknown>
): void {
  const noteId = extractOptionalString(payload.note_id);
  if (noteId) {
    task.noteId = noteId;
  }
  const notePath = extractOptionalString(payload.note_path);
  if (notePath) {
    task.notePath = notePath;
  }
}

function formatToolParameters(parameters: Record<string, unknown>): string {
  try {
    return JSON.stringify(parameters, null, 2);
  } catch (error) {
    console.warn("无法格式化工具参数", error, parameters);
    return Object.entries(parameters)
      .map(([key, value]) => `${key}: ${String(value)}`)
      .join("\n");
  }
}

function formatToolResult(result: string): string {
  const trimmed = result.trim();
  const limit = 900;
  if (trimmed.length > limit) {
    return `${trimmed.slice(0, limit)}…`;
  }
  return trimmed;
}

async function copyNotePath(path: string | null | undefined) {
  if (!path) {
    return;
  }

  try {
    await navigator.clipboard.writeText(path);
    progressLogs.value.push(`已复制笔记路径：${path}`);
  } catch (error) {
    console.warn("无法直接复制到剪贴板", error);
    window.prompt("复制以下笔记路径", path);
    progressLogs.value.push("请手动复制笔记路径");
  }
}

function resetWorkflowState() {
  todoTasks.value = [];
  activeTaskId.value = null;
  reportMarkdown.value = "";
  progressLogs.value = [];
  summaryHighlight.value = false;
  sourcesHighlight.value = false;
  reportHighlight.value = false;
  toolHighlight.value = false;
  logsCollapsed.value = false;
}

function findTask(taskId: unknown): TodoTaskView | undefined {
  const numeric =
    typeof taskId === "number"
      ? taskId
      : typeof taskId === "string"
      ? Number(taskId)
      : NaN;
  if (Number.isNaN(numeric)) {
    return undefined;
  }
  return todoTasks.value.find((task) => task.id === numeric);
}

function upsertTaskMetadata(task: TodoTaskView, payload: Record<string, unknown>) {
  if (typeof payload.title === "string" && payload.title.trim()) {
    task.title = payload.title.trim();
  }
  if (typeof payload.intent === "string" && payload.intent.trim()) {
    task.intent = payload.intent.trim();
  }
  if (typeof payload.query === "string" && payload.query.trim()) {
    task.query = payload.query.trim();
  }
}

const handleSubmit = async () => {
  if (!form.topic.trim()) {
    error.value = "请输入研究主题";
    return;
  }

  // ── LLM 预检：仅在用户选择了本地 provider 时执行 ───────────────────────
  if (form.llmProvider) {
    preflightChecking.value = true;
    preflightHint.value = "";
    error.value = "";
    try {
      const result = await llmPreflight(form.llmProvider, form.llmModel || undefined);
      if (!result.ok) {
        error.value = result.error ?? "LLM 预检失败";
        preflightHint.value = result.hint ?? "";
        preflightChecking.value = false;
        return;
      }
    } catch {
      // 预检接口本身不可用时不阻断，让正式流程去报错
    }
    preflightChecking.value = false;
  }
  // ──────────────────────────────────────────────────────────────────────────

  if (currentController) {
    currentController.abort();
    currentController = null;
  }

  loading.value = true;
  error.value = "";
  preflightHint.value = "";
  isExpanded.value = true;
  resetWorkflowState();

  const controller = new AbortController();
  currentController = controller;

  const payload = {
    topic: form.topic.trim(),
    search_api: form.searchApi || undefined,
    llm_provider: form.llmProvider || undefined,
    local_llm: form.llmModel || undefined,
  };

  try {
    await runResearchStream(
      payload,
      (event: ResearchStreamEvent) => {
        if (event.type === "status") {
          const message =
            typeof event.message === "string" && event.message.trim()
              ? event.message
              : "流程状态更新";
          progressLogs.value.push(message);

          const payload = event as Record<string, unknown>;
          const task = findTask(payload.task_id);
          if (task && message) {
            task.notices.push(message);
            applyNoteMetadata(task, payload);
          }
          return;
        }

        if (event.type === "todo_list") {
          const tasks = Array.isArray(event.tasks)
            ? (event.tasks as Record<string, unknown>[])
            : [];

          todoTasks.value = tasks.map((item, index) => {
            const rawId =
              typeof item.id === "number"
                ? item.id
                : typeof item.id === "string"
                ? Number(item.id)
                : index + 1;
            const id = Number.isFinite(rawId) ? Number(rawId) : index + 1;
            const noteId =
              typeof item.note_id === "string" && item.note_id.trim()
                ? item.note_id.trim()
                : null;
            const notePath =
              typeof item.note_path === "string" && item.note_path.trim()
                ? item.note_path.trim()
                : null;

            return {
              id,
              title:
                typeof item.title === "string" && item.title.trim()
                  ? item.title.trim()
                  : `任务${id}`,
              intent:
                typeof item.intent === "string" && item.intent.trim()
                  ? item.intent.trim()
                  : "探索与主题相关的关键信息",
              query:
                typeof item.query === "string" && item.query.trim()
                  ? item.query.trim()
                  : form.topic.trim(),
              status:
                typeof item.status === "string" && item.status.trim()
                  ? item.status.trim()
                  : "pending",
              summary: "",
              sourcesSummary: "",
              sourceItems: [],
              notices: [],
              noteId,
              notePath,
              toolCalls: []
            } as TodoTaskView;
          });

          if (todoTasks.value.length) {
            activeTaskId.value = todoTasks.value[0].id;
            progressLogs.value.push("已生成任务清单");
          } else {
            progressLogs.value.push("未生成任务清单，使用默认任务继续");
          }
          return;
        }

        if (event.type === "task_status") {
          const payload = event as Record<string, unknown>;
          const task = findTask(event.task_id);
          if (!task) {
            return;
          }

          upsertTaskMetadata(task, payload);
          applyNoteMetadata(task, payload);
          const status =
            typeof event.status === "string" && event.status.trim()
              ? event.status.trim()
              : task.status;
          task.status = status;

          if (status === "in_progress") {
            task.summary = "";
            task.sourcesSummary = "";
            task.sourceItems = [];
            task.notices = [];
            activeTaskId.value = task.id;
            progressLogs.value.push(`开始执行任务：${task.title}`);
          } else if (status === "completed") {
            if (typeof event.summary === "string" && event.summary.trim()) {
              task.summary = event.summary.trim();
            }
            if (
              typeof event.sources_summary === "string" &&
              event.sources_summary.trim()
            ) {
              task.sourcesSummary = event.sources_summary.trim();
              task.sourceItems = parseSources(task.sourcesSummary);
            }
            progressLogs.value.push(`完成任务：${task.title}`);
            if (activeTaskId.value === task.id) {
              pulse(summaryHighlight);
              pulse(sourcesHighlight);
            }
          } else if (status === "skipped") {
            progressLogs.value.push(`任务跳过：${task.title}`);
          }
          return;
        }

        if (event.type === "sources") {
          const payload = event as Record<string, unknown>;
          const task = findTask(event.task_id);
          if (!task) {
            return;
          }

          const textCandidates = [
            payload.latest_sources,
            payload.sources_summary,
            payload.raw_context
          ];
          const latestText = textCandidates
            .map((value) => (typeof value === "string" ? value.trim() : ""))
            .find((value) => value);

          if (latestText) {
            task.sourcesSummary = latestText;
            task.sourceItems = parseSources(latestText);
            if (activeTaskId.value === task.id) {
              pulse(sourcesHighlight);
            }
            progressLogs.value.push(`已更新任务来源：${task.title}`);
          }

          if (typeof payload.backend === "string") {
            progressLogs.value.push(
              `当前使用搜索后端：${payload.backend}`
            );
          }

          applyNoteMetadata(task, payload);

          return;
        }

        if (event.type === "task_summary_chunk") {
          const payload = event as Record<string, unknown>;
          const task = findTask(event.task_id);
          if (!task) {
            return;
          }
          const chunk =
            typeof event.content === "string" ? event.content : "";
          task.summary += chunk;
          applyNoteMetadata(task, payload);
          if (activeTaskId.value === task.id) {
            pulse(summaryHighlight);
          }
          return;
        }

        if (event.type === "tool_call") {
          const payload = event as Record<string, unknown>;
          const eventId =
            typeof payload.event_id === "number"
              ? payload.event_id
              : Date.now();
          const agent =
            typeof payload.agent === "string" && payload.agent.trim()
              ? payload.agent.trim()
              : "Agent";
          const tool =
            typeof payload.tool === "string" && payload.tool.trim()
              ? payload.tool.trim()
              : "tool";
          const parameters = ensureRecord(payload.parameters);
          const result =
            typeof payload.result === "string" ? payload.result : "";
          const noteId = extractOptionalString(payload.note_id);
          const notePath = extractOptionalString(payload.note_path);

          const task = findTask(payload.task_id);
          if (task) {
            task.toolCalls.push({
              eventId,
              agent,
              tool,
              parameters,
              result,
              noteId,
              notePath,
              timestamp: Date.now()
            });
            if (noteId) {
              task.noteId = noteId;
            }
            if (notePath) {
              task.notePath = notePath;
            }
            const logSummary = noteId
              ? `${agent} 调用了 ${tool}（任务 ${task.id}，笔记 ${noteId}）`
              : `${agent} 调用了 ${tool}（任务 ${task.id}）`;
            progressLogs.value.push(logSummary);
            if (activeTaskId.value === task.id) {
              pulse(toolHighlight);
            }
          } else {
            progressLogs.value.push(`${agent} 调用了 ${tool}`);
          }
          return;
        }

        if (event.type === "final_report") {
          const report =
            typeof event.report === "string" && event.report.trim()
              ? event.report.trim()
              : "";
          reportMarkdown.value = report || "报告生成失败，未获得有效内容";
          pulse(reportHighlight);
          progressLogs.value.push("最终报告已生成");
          return;
        }

        if (event.type === "error") {
          const detail =
            typeof event.detail === "string" && event.detail.trim()
              ? event.detail
              : "研究过程中发生错误";
          error.value = detail;
          progressLogs.value.push("研究失败，已停止流程");
        }
      },
      { signal: controller.signal }
    );

    if (!reportMarkdown.value) {
      reportMarkdown.value = "暂无生成的报告";
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      progressLogs.value.push("已取消当前研究任务");
    } else {
      error.value = err instanceof Error ? err.message : "请求失败";
    }
  } finally {
    loading.value = false;
    if (currentController === controller) {
      currentController = null;
    }
  }
};

const cancelResearch = () => {
  if (!loading.value || !currentController) {
    return;
  }
  progressLogs.value.push("正在尝试取消当前研究任务…");
  currentController.abort();
};

const goBack = () => {
  if (loading.value) {
    return; // 研究进行中不允许返回
  }
  isExpanded.value = false;
};

const startNewResearch = () => {
  // 研究进行中时，开始新研究会中断当前研究并清空其进度。
  // 必须先确认，避免误点导致正在进行的研究从界面消失且无法找回。
  if (loading.value) {
    const confirmed = window.confirm(
      "当前研究仍在进行中。开始新研究会中断当前研究、清空其进度且无法恢复，确定要继续吗？"
    );
    if (!confirmed) {
      return;
    }
    cancelResearch();
  }
  resetWorkflowState();
  isExpanded.value = false;
  form.topic = "";
  form.searchApi = "";
};

onBeforeUnmount(() => {
  if (currentController) {
    currentController.abort();
    currentController = null;
  }
  if (tocObserver) {
    tocObserver.disconnect();
    tocObserver = null;
  }
  document.removeEventListener("click", onDocClick);
  document.removeEventListener("keydown", onDocKeydown);
});
</script>


<style scoped>
/* ============================================================
   DeepResearch v4 "Aether" — App.vue scoped styles
   Frosted-glass surfaces over the body aurora.
   All easing uses --aether-ease (decelerated, no spring).
============================================================ */

.app-shell {
  position: relative;
  z-index: 1;
  min-height: 100vh;
  padding: 96px 32px 64px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: var(--aether-ink-2);
  overflow: hidden;
  box-sizing: border-box;
  transition: padding 400ms var(--aether-ease);
}

.app-shell.expanded {
  padding: 0;
  align-items: stretch;
}

/* ── Layout ──────────────────────────────────────────────── */
.layout {
  position: relative;
  width: 100%;
  display: flex;
  z-index: 1;
}

.layout-centered {
  max-width: 720px;
  justify-content: center;
  align-items: center;
}

.layout-fullscreen {
  display: grid;
  grid-template-columns: 380px 1fr;
  height: 100vh;
  max-width: 100%;
  align-items: stretch;
  /* Modern browsers (Chrome 119+, Firefox 115+, Safari 17+) animate
     grid-template-columns; on older browsers the change is instant. */
  transition: grid-template-columns 420ms var(--aether-ease);
}

.layout-fullscreen.sidebar-collapsed {
  grid-template-columns: 0px 1fr;
}

/* ── Landing composition ────────────────────────────────── */
.landing {
  width: 100%;
  max-width: 720px;
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  box-shadow: none;
  animation: aetherRise 800ms var(--aether-ease) both;
  display: flex;
  flex-direction: column;
  gap: 0;
}

.landing-head {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 28px;
  margin-bottom: 56px;
  text-align: center;
}

.landing-head > div {
  display: flex;
  flex-direction: column;
  gap: 14px;
  align-items: center;
}

.landing-head h1 {
  margin: 0;
  font-size: 52px;
  font-weight: 600;
  letter-spacing: -0.025em;
  line-height: 1.04;
  color: var(--aether-ink);
  background: linear-gradient(180deg, var(--aether-ink) 0%, #2b3243 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.landing-head p {
  margin: 0;
  color: var(--aether-ink-3);
  font-size: 19px;
  line-height: 1.5;
  letter-spacing: -0.01em;
  max-width: 520px;
}

/* ── Logo (3×3 grid tile) ───────────────────────────────── */
.logo {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  background: linear-gradient(135deg, var(--primary-500), var(--primary-700));
  padding: 10px;
  box-shadow: var(--soft-2);
  flex-shrink: 0;
  display: grid;
  place-items: center;
  transition: transform 500ms var(--aether-ease),
    box-shadow 500ms var(--aether-ease);
}

.logo:hover {
  transform: translateY(-2px) scale(1.03);
  box-shadow: var(--soft-3);
}

.logo:active {
  transform: scale(0.97);
}

.logo-sm {
  width: 40px;
  height: 40px;
  padding: 7px;
  border-radius: 12px;
}

.logo-sm .logo-grid {
  grid-template-columns: repeat(3, 7px);
  grid-template-rows: repeat(3, 7px);
}

.logo-sm .logo-grid > i {
  width: 7px;
  height: 7px;
}

.logo-grid {
  display: grid;
  grid-template-columns: repeat(3, 8px);
  grid-template-rows: repeat(3, 8px);
  gap: 2px;
}

.logo-grid > i {
  width: 8px;
  height: 8px;
  border-radius: 1.6px;
  background: #fff;
  display: block;
}

.logo-grid > i.accent {
  background: var(--secondary-500);
  opacity: 1;
}

/* ── Form panel — the single frosted glass on landing ──── */
.form {
  display: flex;
  flex-direction: column;
  gap: 22px;
  padding: 32px;
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur-lg);
  -webkit-backdrop-filter: var(--glass-blur-lg);
  border: var(--glass-border);
  border-radius: 28px;
  box-shadow: var(--soft-3);
}

.field {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.field > span {
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--aether-ink-4);
}

textarea,
input,
select {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid var(--aether-line);
  background: var(--surface);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  color: var(--aether-ink);
  font-size: 15px;
  font-family: inherit;
  letter-spacing: -0.005em;
  transition: border-color 200ms var(--aether-ease),
    background 200ms var(--aether-ease),
    box-shadow 200ms var(--aether-ease);
}

textarea {
  resize: vertical;
  min-height: 96px;
  line-height: 1.55;
}

textarea::placeholder,
input::placeholder {
  color: var(--aether-ink-4);
}

select {
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%23828a9c' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 16px center;
  padding-right: 40px;
  cursor: pointer;
}

textarea:hover,
input:hover,
select:hover {
  background: var(--surface-bright);
  border-color: var(--aether-line-strong);
}

textarea:focus,
input:focus,
select:focus {
  outline: none;
  border-color: var(--primary-400);
  background: var(--surface-container-highest);
  box-shadow: var(--focus-ring);
}

.options {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}

.option {
  flex: 1;
  min-width: 140px;
}

/* ── Local LLM probe — nested inset frosted card ───────── */
.llm-probe-section {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 20px;
  background: var(--surface-2);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--aether-line);
  border-radius: 18px;
}

.llm-probe-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.llm-probe-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--aether-ink-2);
}

.probe-refresh-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 14px;
  font-size: 12px;
  font-weight: 500;
  color: var(--aether-ink-2);
  background: var(--surface);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid var(--aether-line);
  border-radius: 999px;
  cursor: pointer;
  font-family: inherit;
  transition: background 220ms var(--aether-ease),
    border-color 220ms var(--aether-ease),
    transform 220ms var(--aether-ease);
}

.probe-refresh-btn:hover:not(:disabled) {
  background: var(--surface-container-highest);
  border-color: var(--aether-line-strong);
  transform: translateY(-1px);
}

.probe-refresh-btn:active:not(:disabled) {
  transform: translateY(0) scale(0.98);
  transition-duration: 120ms;
}

.probe-refresh-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.probe-refresh-icon {
  font-size: 16px;
}

.probe-refresh-icon.spinning {
  animation: aetherSpin 0.9s linear infinite;
}

.probe-empty {
  font-size: 12px;
  color: var(--aether-ink-4);
  line-height: 1.5;
}

.probe-error {
  font-size: 12px;
  color: var(--fg-danger);
  margin: 0;
}

.probe-loading {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--aether-ink-3);
}

.spinner-sm {
  font-size: 16px;
  color: var(--primary-500);
  animation: aetherSpin 1.2s linear infinite;
}

.llm-selects {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ── Submit + AI halo ──────────────────────────────────── */
.form-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.form-actions .ai-halo {
  align-self: flex-start;
}

.submit {
  padding: 16px 32px;
  border-radius: 999px;
  border: none;
  background: var(--primary-500);
  color: #fff;
  font-size: 16px;
  font-weight: 500;
  letter-spacing: -0.01em;
  font-family: inherit;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.4) inset,
    0 2px 6px rgba(37, 99, 235, 0.32),
    0 16px 36px -12px rgba(37, 99, 235, 0.55);
  transition: background 220ms var(--aether-ease),
    box-shadow 220ms var(--aether-ease),
    transform 220ms var(--aether-ease);
}

.submit-label {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.submit .spinner {
  font-size: 18px;
  animation: aetherSpin 1.2s linear infinite;
}

.submit:not(:disabled):hover {
  background: var(--primary-600);
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.4) inset,
    0 4px 10px rgba(37, 99, 235, 0.35),
    0 24px 48px -14px rgba(37, 99, 235, 0.6);
  transform: translateY(-1px);
}

.submit:not(:disabled):active {
  transform: translateY(0) scale(0.98);
  transition-duration: 120ms;
}

.submit:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.secondary-btn {
  padding: 9px 18px;
  border-radius: 999px;
  background: var(--surface-thin);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid var(--aether-line-strong);
  color: var(--aether-ink-2);
  font-size: 13px;
  font-weight: 500;
  font-family: inherit;
  cursor: pointer;
  transition: background 220ms var(--aether-ease),
    border-color 220ms var(--aether-ease),
    transform 220ms var(--aether-ease);
}

.secondary-btn:hover:not(:disabled) {
  background: var(--surface-bright);
  border-color: rgba(10, 14, 26, 0.2);
  transform: translateY(-1px);
}

.secondary-btn:active:not(:disabled) {
  transform: translateY(0) scale(0.98);
}

.secondary-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

/* ── Error / hint ──────────────────────────────────────── */
.error-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 16px;
}

.error-chip {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  background: var(--bg-danger);
  border: 1px solid var(--border-danger);
  border-radius: 14px;
  color: var(--fg-danger);
  font-size: 14px;
  letter-spacing: -0.005em;
  animation: aetherFadeUp 500ms var(--aether-ease) both;
}

.error-chip .material-symbols-outlined {
  font-size: 20px;
  flex-shrink: 0;
}

.preflight-hint {
  margin: 0;
  padding: 12px 16px;
  font-size: 13px;
  line-height: 1.55;
  color: var(--fg-warning);
  background: var(--bg-warning);
  border: 1px solid var(--border-warning);
  border-radius: 14px;
}

.hint.muted {
  margin-top: 16px;
  color: var(--aether-ink-4);
  font-size: 13px;
}

/* ── Workflow result panel ─────────────────────────────── */
.panel-result {
  height: 100vh;
  overflow-y: auto;
  padding: 40px 48px 64px;
  display: flex;
  flex-direction: column;
  gap: 28px;
  position: relative;
  transition: padding 420ms var(--aether-ease);
}

/* When the sidebar is collapsed the floating expand FAB lives at
   top: 24, left: 24 (40px wide). Push the result panel's left padding
   in so its first content row (the status-bar) doesn't sit underneath
   the FAB. Animates in sync with the sidebar's own collapse. */
.layout-fullscreen.sidebar-collapsed .panel-result {
  padding-left: 88px;
}

.status-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.status-main {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.status-controls {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}

.status-history-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.status-history-btn .material-symbols-outlined {
  font-size: 16px;
}

/* Force-stop the running SSE — danger-tinted so it's unmistakable */
.status-stop-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-danger);
  border-color: var(--border-danger);
  color: var(--fg-danger);
}
.status-stop-btn .material-symbols-outlined { font-size: 16px; }
.status-stop-btn:hover:not(:disabled) {
  background: rgba(239, 68, 68, 0.18);
  border-color: rgba(239, 68, 68, 0.35);
  color: var(--fg-danger);
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 16px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: -0.005em;
  color: var(--aether-ink-2);
  background: var(--glass-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--aether-line);
  animation: aetherFadeIn 500ms var(--aether-ease) both;
}

.status-chip .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--aether-ink-4);
  transition: background var(--dur-medium) var(--aether-ease);
}

.status-chip.running .dot {
  background: var(--primary-500);
  animation: aetherPulse 2s var(--aether-ease) infinite;
}

/* Completed: emerald success styling — clearly distinct from "running" */
.status-chip.complete {
  background: var(--bg-success);
  color: var(--fg-success);
  border-color: var(--border-success);
}
.status-chip .status-check {
  font-size: 16px;
  color: currentColor;
}

.status-meta {
  color: var(--aether-ink-4);
  font-size: 13px;
}

@keyframes aetherPulse {
  0%   { transform: scale(1);    box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.45); }
  60%  { transform: scale(1.15); box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); }
  100% { transform: scale(1);    box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}

/* ── Timeline ──────────────────────────────────────────── */
.timeline-wrapper {
  max-height: 280px;
  overflow-y: auto;
  padding: 18px 22px;
  background: var(--glass-bg-thin);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--aether-line);
  border-radius: 20px;
  scrollbar-width: thin;
  scrollbar-color: var(--aether-line-strong) transparent;
}

.timeline-wrapper::-webkit-scrollbar { width: 6px; }
.timeline-wrapper::-webkit-scrollbar-thumb {
  background: var(--aether-line-strong);
  border-radius: 999px;
}

.timeline {
  list-style: none;
  padding: 0 0 0 18px;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  position: relative;
}

.timeline::before {
  content: "";
  position: absolute;
  top: 8px;
  bottom: 8px;
  left: 4px;
  width: 1px;
  background: linear-gradient(180deg,
    rgba(10, 14, 26, 0) 0%,
    rgba(10, 14, 26, 0.15) 15%,
    rgba(10, 14, 26, 0.15) 85%,
    rgba(10, 14, 26, 0) 100%);
}

.timeline li {
  position: relative;
  padding-left: 18px;
  color: var(--aether-ink-2);
  font-size: 14px;
  line-height: 1.55;
  letter-spacing: -0.005em;
  animation: aetherFadeUp 600ms var(--aether-ease) both;
}

.timeline li:nth-child(1) { animation-delay: 40ms; }
.timeline li:nth-child(2) { animation-delay: 90ms; }
.timeline li:nth-child(3) { animation-delay: 140ms; }
.timeline li:nth-child(4) { animation-delay: 190ms; }
.timeline li:nth-child(5) { animation-delay: 240ms; }
.timeline li:nth-child(6) { animation-delay: 290ms; }

.timeline-node {
  position: absolute;
  left: -3px;
  top: 8px;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--aether-ink-4);
  box-sizing: border-box;
}

.timeline li:last-child .timeline-node {
  background: var(--primary-500);
  animation: aetherDotPulse 1.8s var(--aether-ease) infinite;
}

@keyframes aetherDotPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); transform: scale(1); }
  50%      { box-shadow: 0 0 0 8px rgba(37, 99, 235, 0); transform: scale(1.25); }
}

.timeline-enter-active,
.timeline-leave-active {
  transition: opacity 280ms var(--aether-ease),
    transform 280ms var(--aether-ease);
}

.timeline-enter-from,
.timeline-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

/* ── Tasks ─────────────────────────────────────────────── */
.tasks-section {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 24px;
  align-items: start;
}

.tasks-list {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: var(--glass-border);
  border-radius: 22px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  box-shadow: var(--soft-1);
}

.tasks-list h3 {
  margin: 0 0 4px 4px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--aether-ink-4);
}

.tasks-list ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.task-item {
  border-radius: 14px;
  transition: background 220ms var(--aether-ease),
    transform 280ms var(--aether-ease);
  animation: aetherFadeUp 600ms var(--aether-ease) both;
}

.task-item:hover {
  background: rgba(10, 14, 26, 0.04);
}

.task-item.active {
  background: rgba(37, 99, 235, 0.08);
}

.task-item.active:hover {
  background: rgba(37, 99, 235, 0.1);
}

.task-item:nth-child(1) { animation-delay: 30ms; }
.task-item:nth-child(2) { animation-delay: 70ms; }
.task-item:nth-child(3) { animation-delay: 110ms; }
.task-item:nth-child(4) { animation-delay: 150ms; }
.task-item:nth-child(5) { animation-delay: 190ms; }
.task-item:nth-child(6) { animation-delay: 230ms; }
.task-item:nth-child(7) { animation-delay: 270ms; }
.task-item:nth-child(8) { animation-delay: 310ms; }

.task-button {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px 4px;
  background: transparent;
  border: none;
  color: inherit;
  cursor: pointer;
  text-align: left;
  font-family: inherit;
}

.task-title {
  font-weight: 500;
  font-size: 14px;
  color: var(--aether-ink);
  letter-spacing: -0.005em;
}

.task-status {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 500;
  border: 1px solid transparent;
  color: var(--aether-ink-4);
  background: rgba(10, 14, 26, 0.05);
}

.task-status .material-symbols-outlined {
  font-size: 13px;
}

.task-status.pending {
  background: rgba(10, 14, 26, 0.05);
  color: var(--aether-ink-4);
}

.task-status.in_progress {
  background: rgba(37, 99, 235, 0.1);
  color: var(--primary-600);
}

.task-status.in_progress .material-symbols-outlined {
  animation: aetherSpin 1.6s linear infinite;
}

.task-status.completed {
  background: rgba(16, 185, 129, 0.1);
  color: var(--success-700);
}

.task-status.skipped {
  background: rgba(239, 68, 68, 0.08);
  color: var(--danger-700);
}

.task-intent {
  margin: 0;
  padding: 0 14px 12px;
  font-size: 12px;
  color: var(--aether-ink-4);
  line-height: 1.5;
}

/* ── Task detail — large glass card ────────────────────── */
.task-detail {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: var(--glass-border);
  border-radius: 24px;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 22px;
  box-shadow: var(--soft-2);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 16px;
}

.task-header h3 {
  margin: 0;
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: var(--aether-ink);
}

.task-header .muted {
  margin: 6px 0 0;
  color: var(--aether-ink-4);
  font-size: 13px;
}

.task-chip-group {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.task-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  background: var(--surface);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  color: var(--aether-ink-2);
  border: 1px solid var(--aether-line);
}

.task-label.note-chip {
  background: rgba(16, 185, 129, 0.1);
  color: var(--success-700);
  border-color: rgba(16, 185, 129, 0.2);
}

.task-label.path-chip {
  max-width: 360px;
  background: rgba(99, 102, 241, 0.08);
  color: var(--tertiary-700);
  border-color: rgba(99, 102, 241, 0.18);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-label .material-symbols-outlined {
  font-size: 14px;
}

.path-label {
  font-weight: 500;
}

.path-text {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chip-action {
  border: none;
  background: rgba(99, 102, 241, 0.16);
  color: var(--tertiary-700);
  padding: 3px 8px;
  border-radius: 8px;
  font-size: 11px;
  font-family: inherit;
  cursor: pointer;
  transition: background 220ms var(--aether-ease);
}

.chip-action:hover {
  background: rgba(99, 102, 241, 0.28);
}

.task-notices {
  background: rgba(37, 99, 235, 0.06);
  border: 1px solid rgba(37, 99, 235, 0.18);
  border-radius: 14px;
  padding: 14px 18px;
  color: var(--aether-ink-2);
}

.task-notices h4 {
  margin: 0 0 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--aether-ink);
}

.task-notices ul {
  list-style: disc;
  margin: 0 0 0 18px;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.task-notices li {
  font-size: 13px;
  letter-spacing: -0.005em;
}

/* ── Sources / summary nested blocks ───────────────────── */
.sources-block,
.summary-block {
  position: relative;
  padding: 22px;
  border-radius: 18px;
  background: var(--surface-2);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--aether-line);
}

.sources-block h3,
.summary-block h3 {
  margin: 0 0 14px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--aether-ink-4);
}

.sources-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.source-item {
  position: relative;
  display: inline-flex;
  flex-direction: column;
  gap: 6px;
}

.source-link {
  color: var(--aether-ink);
  text-decoration: none;
  font-weight: 500;
  font-size: 14.5px;
  line-height: 1.45;
  letter-spacing: -0.005em;
  transition: color 200ms var(--aether-ease);
}

.source-link::after {
  content: " ↗";
  font-size: 12px;
  opacity: 0.5;
}

.source-link:hover {
  color: var(--primary-600);
}

.source-tooltip {
  display: none;
  position: absolute;
  bottom: calc(100% + 12px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--glass-bg-strong);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  color: var(--aether-ink-2);
  padding: 14px 16px;
  border-radius: 14px;
  box-shadow: var(--soft-3);
  width: min(420px, 90vw);
  z-index: 20;
  border: 1px solid var(--aether-line);
  letter-spacing: -0.005em;
}

.source-tooltip p {
  margin: 0 0 8px;
  font-size: 13px;
  line-height: 1.55;
}

.source-tooltip p:last-child { margin-bottom: 0; }

.muted-text {
  color: var(--aether-ink-4);
}

.source-item:hover .source-tooltip,
.source-item:focus-within .source-tooltip {
  display: block;
}

.block-pre {
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--aether-ink-2);
  background: var(--surface-2);
  padding: 16px 18px;
  border-radius: 14px;
  border: 1px solid var(--aether-line);
  overflow: auto;
  max-height: 420px;
  margin: 0;
}

.summary-block .block-pre,
.sources-block .block-pre {
  max-height: 360px;
}

.block-highlight {
  animation: aetherGlow 1.4s var(--aether-ease);
}

@keyframes aetherGlow {
  0%   { box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.24); }
  100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}

/* ── Tool calls ────────────────────────────────────────── */
.tools-block {
  padding: 22px;
  border-radius: 18px;
  background: var(--surface-2);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--aether-line);
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.tools-block h3 {
  margin: 0;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--aether-ink-4);
}

.tool-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tool-entry {
  background: var(--surface);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--aether-line);
  border-radius: 14px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.tool-entry-header {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
}

.tool-entry-title {
  font-weight: 500;
  color: var(--aether-ink);
  font-size: 14px;
  letter-spacing: -0.005em;
}

.tool-entry-note {
  font-size: 12px;
  color: var(--success-700);
}

.tool-entry-path {
  margin: 0;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--primary-600);
}

.tool-subtitle {
  margin: 0;
  font-size: 12px;
  color: var(--aether-ink-4);
  font-weight: 500;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.tool-pre {
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--aether-ink-2);
  background: var(--surface-2);
  padding: 12px;
  border-radius: 12px;
  border: 1px solid var(--aether-line);
  overflow: auto;
  max-height: 260px;
}

.link-btn {
  background: none;
  border: none;
  color: var(--primary-600);
  cursor: pointer;
  padding: 0 4px;
  font-size: 12px;
  font-family: inherit;
  border-radius: 4px;
  transition: color 200ms var(--aether-ease);
}

.link-btn:hover {
  color: var(--primary-700);
}

/* ── Sidebar — floating frosted glass panel ────────────── */
.sidebar {
  margin: 24px 0 24px 24px;
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur-lg);
  -webkit-backdrop-filter: var(--glass-blur-lg);
  border: var(--glass-border);
  border-radius: 24px;
  box-shadow: var(--soft-2);
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  height: calc(100vh - 48px);
  overflow: hidden auto;
  position: sticky;
  top: 24px;
  transition:
    opacity var(--dur-medium) var(--aether-ease),
    transform var(--dur-long) var(--aether-ease),
    margin var(--dur-long) var(--aether-ease),
    padding var(--dur-long) var(--aether-ease),
    box-shadow var(--dur-long) var(--aether-ease);
}

/* Collapsed: fade the panel, slide it slightly left, and zero out the
   padding/margin so the grid 0-column transition reads as a real
   "drawer closing" instead of a sudden cut. */
.layout-fullscreen.sidebar-collapsed .sidebar {
  opacity: 0;
  transform: translateX(-12px);
  margin-left: 0;
  margin-right: 0;
  padding-left: 0;
  padding-right: 0;
  pointer-events: none;
  box-shadow: none;
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.sidebar-collapse-btn {
  width: 36px;
  height: 36px;
  display: inline-grid;
  place-items: center;
  flex-shrink: 0;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 999px;
  color: var(--aether-ink-3);
  font-family: inherit;
  cursor: pointer;
  transition: background var(--dur-medium) var(--aether-ease),
    color var(--dur-medium) var(--aether-ease),
    border-color var(--dur-medium) var(--aether-ease),
    transform var(--dur-medium) var(--aether-ease);
}
.sidebar-collapse-btn .material-symbols-outlined { font-size: 20px; }
.sidebar-collapse-btn:hover {
  background: rgba(10, 14, 26, 0.05);
  color: var(--aether-ink);
}
.sidebar-collapse-btn:active { transform: scale(0.94); }

/* Floating "expand sidebar" FAB — only shown when sidebar is collapsed */
.sidebar-expand-btn {
  position: absolute;
  top: 24px;
  left: 24px;
  width: 40px;
  height: 40px;
  display: inline-grid;
  place-items: center;
  z-index: 100;
  background: var(--glass-bg-strong);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: var(--glass-border);
  border-radius: 12px;
  color: var(--aether-ink-2);
  font-family: inherit;
  cursor: pointer;
  box-shadow: var(--soft-1);
  transition: background var(--dur-medium) var(--aether-ease),
    box-shadow var(--dur-medium) var(--aether-ease),
    transform var(--dur-medium) var(--aether-ease);
  animation: aetherFadeIn var(--dur-long) var(--aether-ease) both;
}
.sidebar-expand-btn .material-symbols-outlined {
  font-size: 20px;
  /* Mirror the menu_open icon so the arrow points right (= "open") */
  transform: scaleX(-1);
}
.sidebar-expand-btn:hover {
  background: var(--surface-container-highest);
  box-shadow: var(--soft-2);
  transform: translateY(-1px);
}
.sidebar-expand-btn:active {
  transform: translateY(0) scale(0.96);
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.sidebar-header h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: var(--aether-ink);
}

.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  background: var(--surface-thin);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid var(--aether-line);
  border-radius: 999px;
  color: var(--aether-ink-2);
  font-size: 13px;
  font-weight: 500;
  font-family: inherit;
  cursor: pointer;
  width: fit-content;
  transition: background 220ms var(--aether-ease),
    transform 220ms var(--aether-ease);
}

.back-btn .material-symbols-outlined { font-size: 16px; }

.back-btn:hover:not(:disabled) {
  background: var(--surface-bright);
  transform: translateY(-1px);
}

.back-btn:active:not(:disabled) {
  transform: translateY(0) scale(0.98);
}

.back-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.research-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.info-item label {
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--aether-ink-4);
}

.info-item p {
  margin: 0;
  font-size: 14px;
  color: var(--aether-ink-2);
  letter-spacing: -0.005em;
  line-height: 1.55;
}

.topic-display {
  font-size: 16px !important;
  font-weight: 500;
  line-height: 1.45;
  color: var(--aether-ink) !important;
  padding: 0;
  background: transparent;
  border-left: none;
  letter-spacing: -0.015em;
}

.progress-bar {
  width: 100%;
  height: 4px;
  background: rgba(10, 14, 26, 0.06);
  border-radius: 999px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary-500), var(--primary-400));
  border-radius: 999px;
  transition: width 0.8s var(--aether-ease);
}

.progress-text {
  font-size: 12px !important;
  color: var(--aether-ink-4) !important;
  letter-spacing: 0;
}

.sidebar-actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-top: 4px;
  /* (border-top removed — the divider felt heavy against the glass) */
}

.new-research-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 14px 24px;
  background: var(--primary-500);
  border: none;
  border-radius: 999px;
  color: #fff;
  font-size: 14px;
  font-weight: 500;
  letter-spacing: -0.005em;
  font-family: inherit;
  cursor: pointer;
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.4) inset,
    0 2px 6px rgba(37, 99, 235, 0.28),
    0 12px 28px -10px rgba(37, 99, 235, 0.5);
  transition: background 220ms var(--aether-ease),
    box-shadow 220ms var(--aether-ease),
    transform 220ms var(--aether-ease);
}

.new-research-btn .material-symbols-outlined { font-size: 18px; }

.new-research-btn:hover {
  background: var(--primary-600);
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.4) inset,
    0 4px 10px rgba(37, 99, 235, 0.32),
    0 20px 40px -12px rgba(37, 99, 235, 0.55);
  transform: translateY(-1px);
}

.new-research-btn:active {
  transform: translateY(0) scale(0.98);
}

.error-info label {
  color: var(--fg-danger) !important;
}

.error-detail-text {
  font-size: 12px !important;
  color: var(--fg-danger) !important;
  word-break: break-all;
  background: var(--bg-danger);
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid var(--border-danger);
  margin: 0;
}

/* ── History toggle (FAB) ──────────────────────────────── */
.history-toggle-btn {
  position: absolute;
  top: 24px;
  right: 24px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 16px;
  background: var(--surface);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--aether-line);
  border-radius: 999px;
  color: var(--aether-ink-2);
  font-size: 13px;
  font-weight: 500;
  font-family: inherit;
  cursor: pointer;
  z-index: 1000;
  box-shadow: var(--soft-1);
  transition: background 220ms var(--aether-ease),
    box-shadow 220ms var(--aether-ease),
    transform 220ms var(--aether-ease);
}

.history-toggle-btn .material-symbols-outlined { font-size: 18px; }

.history-toggle-btn:hover {
  background: var(--surface-container-highest);
  box-shadow: var(--soft-2);
  transform: translateY(-1px);
}

.history-toggle-btn:active {
  transform: translateY(0) scale(0.96);
}

/* In v4.2 the history toggle stays top-right in every state — the
   back-button slot is now occupied by the sidebar collapse control,
   and the floating expand button (when sidebar is collapsed) lives
   on the LEFT, so the two FABs no longer collide. */

/* ── Report header with export actions ─────────────────── */
.report-block {
  padding: 32px;
  border-radius: 24px;
  background: var(--glass-bg-strong);
  backdrop-filter: var(--glass-blur-lg);
  -webkit-backdrop-filter: var(--glass-blur-lg);
  border: var(--glass-border);
  box-shadow: var(--soft-2);
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.report-block h3 {
  margin: 0;
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: var(--aether-ink);
}

.report-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.report-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.report-actions .secondary-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 16px;
  font-size: 13px;
}

.report-actions .material-symbols-outlined {
  font-size: 18px;
}

/* ── Report layout: sticky TOC + main column ──────────── */
.report-layout {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 28px;
  align-items: start;
}

.report-layout.no-toc {
  grid-template-columns: 1fr;
}

.report-toc {
  position: sticky;
  top: 16px;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
  padding: 4px 4px 4px 0;
  border-right: 1px solid var(--aether-line);
  scrollbar-width: thin;
  scrollbar-color: var(--aether-line-strong) transparent;
}

.report-toc .toc-title {
  margin: 0 0 12px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--aether-ink-4);
}

.report-toc ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.report-toc li {
  font-size: 13px;
  line-height: 1.4;
  letter-spacing: -0.005em;
}

.report-toc li a {
  display: block;
  padding: 6px 12px;
  border-radius: 999px;
  color: var(--aether-ink-3);
  text-decoration: none;
  border: 1px solid transparent;
  transition: background 220ms var(--aether-ease),
    color 220ms var(--aether-ease),
    border-color 220ms var(--aether-ease),
    box-shadow 220ms var(--aether-ease);
}

.report-toc li a:hover {
  background: rgba(10, 14, 26, 0.04);
  color: var(--aether-ink);
}

/* Active item: sapphire-tinted glass pill — matches the search-chip
   active recipe (rgba(37,99,235,0.08) + border + primary-600 text).
   No more left rail. */
.report-toc li.active > a {
  background: rgba(37, 99, 235, 0.10);
  color: var(--primary-600);
  border-color: rgba(37, 99, 235, 0.18);
  font-weight: 500;
}

.report-toc li.toc-level-2 a { padding-left: 24px; }
.report-toc li.toc-level-3 a { padding-left: 36px; font-size: 12px; }

.report-main {
  min-width: 0;
}

/* ── Markdown body — task summary + final report ───────── */
.markdown-body {
  color: var(--aether-ink-2);
  font-size: 15px;
  line-height: 1.65;
  letter-spacing: -0.005em;
  word-wrap: break-word;
}

.markdown-body :where(h1, h2, h3, h4, h5, h6) {
  color: var(--aether-ink);
  font-weight: 600;
  letter-spacing: -0.015em;
  line-height: 1.3;
  margin: 1.6em 0 0.5em;
  scroll-margin-top: 16px;
}

.markdown-body > :first-child { margin-top: 0; }
.markdown-body h1 { font-size: 28px; }
.markdown-body h2 { font-size: 22px; }
.markdown-body h3 { font-size: 18px; }
.markdown-body h4 { font-size: 16px; font-weight: 500; }
.markdown-body h5,
.markdown-body h6 { font-size: 14px; font-weight: 500; }

.markdown-body p {
  margin: 0 0 0.9em;
}

.markdown-body a {
  color: var(--primary-600);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 200ms var(--aether-ease);
}

.markdown-body a:hover {
  border-bottom-color: currentColor;
}

.markdown-body ul,
.markdown-body ol {
  margin: 0 0 0.9em;
  padding-left: 1.6em;
}

.markdown-body li {
  margin: 0.3em 0;
}

.markdown-body li > p {
  margin-bottom: 0.4em;
}

.markdown-body strong {
  font-weight: 600;
  color: var(--aether-ink);
}

.markdown-body em { font-style: italic; }
.markdown-body del { color: var(--aether-ink-4); }

.markdown-body code {
  font-family: var(--font-mono);
  font-size: 0.88em;
  background: rgba(37, 99, 235, 0.08);
  color: var(--primary-700);
  padding: 2px 6px;
  border-radius: 6px;
}

.markdown-body pre {
  margin: 0 0 1em;
  padding: 16px 18px;
  background: var(--surface-2);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid var(--aether-line);
  border-radius: 14px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.7;
}

.markdown-body pre code {
  background: transparent;
  color: var(--aether-ink-2);
  padding: 0;
}

.markdown-body blockquote {
  margin: 0 0 1em;
  padding: 10px 16px;
  border-left: 2px solid var(--primary-500);
  background: rgba(37, 99, 235, 0.04);
  color: var(--aether-ink-3);
  border-radius: 0 12px 12px 0;
}

.markdown-body blockquote > :last-child { margin-bottom: 0; }

.markdown-body hr {
  margin: 1.8em 0;
  border: 0;
  border-top: 1px solid var(--aether-line);
}

.markdown-body table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 1em;
  font-size: 13.5px;
}

.markdown-body th,
.markdown-body td {
  padding: 10px 14px;
  border: 1px solid var(--aether-line);
  text-align: left;
  vertical-align: top;
}

.markdown-body th {
  background: var(--surface-2);
  font-weight: 600;
  color: var(--aether-ink);
  letter-spacing: 0.02em;
}

.markdown-body img {
  max-width: 100%;
  height: auto;
  border-radius: 12px;
}

/* ── Footnotes (marked-footnote extension) ────────────────── */
.markdown-body section.footnotes,
.markdown-body section[data-footnotes] {
  margin-top: 2em;
  padding-top: 1.4em;
  border-top: 1px solid var(--aether-line);
}

/* marked-footnote ships <h2 class="sr-only" id="footnote-label">参考文献</h2>;
   un-hide it (the extension assumes a Tailwind .sr-only rule that we don't
   ship) and style it like a normal H2 — gives the section a real heading. */
.markdown-body section.footnotes h2.sr-only,
.markdown-body section[data-footnotes] h2.sr-only,
.markdown-body h2#footnote-label {
  position: static !important;
  width: auto !important;
  height: auto !important;
  margin: 0 0 0.8em !important;
  clip: auto !important;
  overflow: visible !important;
  white-space: normal !important;
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: var(--aether-ink);
}

.markdown-body section.footnotes ol,
.markdown-body section[data-footnotes] ol {
  padding-left: 1.4em;
  margin: 0;
  font-size: 13.5px;
  color: var(--aether-ink-2);
  counter-reset: footnote;
}

.markdown-body section.footnotes ol li,
.markdown-body section[data-footnotes] ol li {
  margin: 0.45em 0;
  line-height: 1.6;
}

/* Marker styling: in-text superscript and the back-ref arrow at the end of
   each footnote definition. */
.markdown-body sup a,
.markdown-body a[data-footnote-ref] {
  display: inline-block;
  padding: 0 4px;
  margin: 0 2px;
  border-radius: 6px;
  background: rgba(37, 99, 235, 0.08);
  color: var(--primary-600);
  font-size: 0.75em;
  font-weight: 500;
  text-decoration: none;
  border-bottom: none !important;
  vertical-align: super;
  transition: background var(--dur-short) var(--aether-ease);
}

.markdown-body sup a:hover,
.markdown-body a[data-footnote-ref]:hover {
  background: rgba(37, 99, 235, 0.16);
}

.markdown-body a[data-footnote-backref] {
  color: var(--aether-ink-4);
  text-decoration: none;
  margin-left: 0.4em;
  border-bottom: none !important;
  font-size: 0.9em;
}

.markdown-body a[data-footnote-backref]:hover {
  color: var(--primary-600);
}

/* ── Responsive ───────────────────────────────────────── */
@media (max-width: 1024px) {
  .layout-fullscreen { grid-template-columns: 320px 1fr; }
  .panel-result { padding: 32px 32px 48px; }
  .landing-head h1 { font-size: 44px; }
}

@media (max-width: 900px) {
  .report-layout { grid-template-columns: 1fr; }
  .report-toc {
    position: static;
    max-height: none;
    border-right: none;
    border-bottom: 1px solid var(--aether-line);
    padding-bottom: 12px;
  }
}

@media (max-width: 768px) {
  .layout-fullscreen { grid-template-columns: 1fr; }
  .sidebar {
    position: static;
    height: auto;
    margin: 16px;
  }
  .panel-result {
    height: auto;
    padding: 24px 20px 40px;
  }
  .tasks-section { grid-template-columns: 1fr; }
  .landing-head h1 { font-size: 36px; }
  .landing-head p { font-size: 17px; }
  .form { padding: 24px; }
}

@media (max-width: 600px) {
  .options { flex-direction: column; }
  .status-meta { font-size: 12px; }
}

/* ============================================================
   v4.1 — Landing hero scale-up + Google-style search bar
============================================================ */

/* Landing composition widens to 820px in v4.1 */
.landing { max-width: 820px; }

/* Hero logo bumps to 56px when it sits inside the landing head */
.landing-head .logo {
  width: 56px;
  height: 56px;
  border-radius: 16px;
  padding: 12px;
}
.landing-head .logo .logo-grid {
  grid-template-columns: repeat(3, 9px);
  grid-template-rows: repeat(3, 9px);
  gap: 2.5px;
}
.landing-head .logo .logo-grid > i {
  width: 9px;
  height: 9px;
  border-radius: 2px;
}

/* ── Search bar — the pill ───────────────────────────────── */
.search-form {
  display: flex;
  flex-direction: column;
  gap: 14px;
  width: 100%;
}

.search-bar {
  position: relative;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 14px 14px 24px;
  background: var(--glass-bg-strong);
  backdrop-filter: var(--glass-blur-lg);
  -webkit-backdrop-filter: var(--glass-blur-lg);
  border: var(--glass-border);
  border-radius: 999px;
  box-shadow: var(--soft-3);
  transition:
    box-shadow 260ms var(--aether-ease),
    border-color 260ms var(--aether-ease),
    background 260ms var(--aether-ease);
}
.search-bar:hover {
  box-shadow: var(--soft-3), 0 0 0 4px rgba(37, 99, 235, 0.06);
}
.search-bar:focus-within {
  border-color: rgba(37, 99, 235, 0.35);
  box-shadow: var(--soft-3), 0 0 0 5px rgba(37, 99, 235, 0.14);
  background: var(--surface-container-highest);
}
.search-bar.is-error {
  border-color: rgba(239, 68, 68, 0.4);
  box-shadow: var(--soft-3), 0 0 0 5px rgba(239, 68, 68, 0.12);
}

.search-bar-icon {
  color: var(--aether-ink-4);
  flex-shrink: 0;
  font-size: 22px;
}

/* The textarea is styled like a single-line field but grows to 96px */
.search-bar-input {
  flex: 1;
  border: none;
  background: transparent;
  padding: 8px 0;
  font-family: inherit;
  font-size: 17px;
  letter-spacing: -0.01em;
  color: var(--aether-ink);
  line-height: 1.4;
  resize: none;
  min-height: 28px;
  max-height: 96px;
  outline: none;
  /* Override the global textarea recipe in scoped CSS above */
  border-radius: 0;
  box-shadow: none;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
  letter-spacing: -0.01em;
  transition: none;
}
.search-bar-input::placeholder {
  color: var(--aether-ink-4);
}
.search-bar-input:focus,
.search-bar-input:hover {
  outline: none;
  box-shadow: none;
  background: transparent;
  border-color: transparent;
}

.search-bar-tools {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

/* ── Chip buttons inside the bar ────────────────────────── */
.search-pop-wrap {
  position: relative;
}

.search-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  height: 40px;
  border-radius: 999px;
  background: transparent;
  border: 1px solid transparent;
  color: var(--aether-ink-3);
  font-family: inherit;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: -0.005em;
  cursor: pointer;
  transition:
    background 200ms var(--aether-ease),
    color 200ms var(--aether-ease),
    border-color 200ms var(--aether-ease);
}
.search-chip:hover {
  background: rgba(10, 14, 26, 0.05);
  color: var(--aether-ink);
}
.search-chip.active {
  background: rgba(37, 99, 235, 0.08);
  color: var(--primary-600);
  border-color: rgba(37, 99, 235, 0.15);
}
.search-chip .material-symbols-outlined { font-size: 16px; }
.search-chip-label { font-weight: 500; }
.search-chip-value {
  color: var(--aether-ink-4);
  font-weight: 400;
  max-width: 110px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.search-chip.active .search-chip-value {
  color: var(--primary-500);
  opacity: 0.85;
}

/* ── Popovers — small frosted menus anchored under the chip */
.popover {
  position: absolute;
  top: calc(100% + 10px);
  z-index: 60;
  min-width: 200px;
  padding: 6px;
  border-radius: 16px;
  background: var(--glass-bg-strong);
  backdrop-filter: var(--glass-blur-lg);
  -webkit-backdrop-filter: var(--glass-blur-lg);
  border: var(--glass-border);
  box-shadow: var(--soft-3);
  display: flex;
  flex-direction: column;
  animation: popoverIn 220ms var(--aether-ease);
}
.popover-left { left: 0; }
.popover-right { right: 0; }
.popover-llm { width: 280px; max-height: 70vh; overflow-y: auto; }

@keyframes popoverIn {
  from { opacity: 0; transform: translateY(-4px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

.popover-head {
  padding: 10px 12px 6px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--aether-ink-4);
}

.popover-head-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.popover-head-row > span:first-child {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--aether-ink-4);
}

.popover-refresh {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 999px;
  background: transparent;
  border: 1px solid var(--aether-line-strong);
  color: var(--aether-ink-3);
  font-family: inherit;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: background 200ms var(--aether-ease), color 200ms var(--aether-ease);
}
.popover-refresh:hover:not(:disabled) {
  background: rgba(10, 14, 26, 0.04);
  color: var(--aether-ink);
}
.popover-refresh:disabled { opacity: 0.5; cursor: not-allowed; }
.popover-refresh .material-symbols-outlined { font-size: 14px; }
.popover-refresh .spinning { animation: aetherSpin 0.9s linear infinite; }

.popover-section-eyebrow {
  padding: 6px 12px 4px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--aether-ink-4);
}

.popover-divider {
  height: 1px;
  background: var(--aether-line);
  margin: 6px 8px;
}

.popover-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 9px 12px;
  border-radius: 12px;
  background: transparent;
  border: none;
  color: var(--aether-ink-2);
  font-family: inherit;
  font-size: 13.5px;
  letter-spacing: -0.005em;
  text-align: left;
  cursor: pointer;
  transition: background 180ms var(--aether-ease), color 180ms var(--aether-ease);
}
.popover-item:hover { background: rgba(10, 14, 26, 0.05); color: var(--aether-ink); }
.popover-item.selected {
  background: rgba(37, 99, 235, 0.08);
  color: var(--primary-700);
  font-weight: 500;
}
.popover-item .material-symbols-outlined {
  font-size: 16px;
  color: var(--primary-500);
}

.popover-empty,
.popover-error,
.popover-loading {
  padding: 12px;
  font-size: 12.5px;
  color: var(--aether-ink-3);
  line-height: 1.5;
}
.popover-error { color: var(--fg-danger); }
.popover-loading {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── AI submit button inside the bar ────────────────────── */
.search-submit {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 44px;
  padding: 0 22px;
  border-radius: 999px;
  border: none;
  background: linear-gradient(180deg,
    var(--primary-400) 0%,
    var(--primary-500) 55%,
    var(--primary-600) 100%);
  color: #fff;
  font-family: inherit;
  font-size: 14px;
  font-weight: 500;
  letter-spacing: -0.005em;
  cursor: pointer;
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.4) inset,
    0 1px 2px rgba(37, 99, 235, 0.3),
    0 12px 28px -10px rgba(37, 99, 235, 0.55);
  transition:
    background 220ms var(--aether-ease),
    box-shadow 220ms var(--aether-ease),
    transform 220ms var(--aether-ease);
}

.search-submit .material-symbols-outlined { font-size: 18px; }

.search-submit:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.4) inset,
    0 2px 4px rgba(37, 99, 235, 0.32),
    0 18px 36px -12px rgba(37, 99, 235, 0.6);
}
.search-submit:active:not(:disabled) {
  transform: translateY(0) scale(0.98);
  transition-duration: 120ms;
}
.search-submit:disabled { opacity: 0.6; cursor: not-allowed; }

/* ── Status / error rows below the bar ──────────────────── */
.search-status {
  text-align: center;
}

.alert {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px 16px;
  border-radius: 14px;
  font-size: 14px;
  letter-spacing: -0.005em;
}
.alert .material-symbols-outlined { font-size: 20px; flex-shrink: 0; }
.alert-error {
  background: var(--bg-danger);
  color: var(--fg-danger);
  border: 1px solid var(--border-danger);
}
.alert-warning {
  background: var(--bg-warning);
  color: var(--fg-warning);
  border: 1px solid var(--border-warning);
}
.search-error { animation: aetherFadeUp 400ms var(--aether-ease) both; }
.search-error-msg {
  margin: 0;
  font-weight: 500;
}
.search-error-hint {
  margin: 6px 0 0;
  font-size: 13px;
  color: var(--fg-warning);
  font-weight: 400;
}

.search-cancel-row {
  display: flex;
  justify-content: center;
}

/* ── Responsive collapse of the bar — README spec ───────── */
@media (max-width: 880px) {
  .search-chip-value { display: none; }
}
@media (max-width: 640px) {
  .search-bar { padding: 12px 12px 12px 18px; gap: 10px; }
  .search-chip-label { display: none; }
  .search-submit { padding: 0 0; width: 44px; justify-content: center; }
  .search-submit-label { display: none; }
}
</style>
