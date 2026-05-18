<template>
  <main class="app-shell" :class="{ expanded: isExpanded }">
    <button class="history-toggle-btn" @click="isHistoryOpen = true">
      <span class="material-symbols-outlined" aria-hidden="true">history</span>
      历史记录
    </button>

    <HistoryModal :isOpen="isHistoryOpen" @close="isHistoryOpen = false" />

    <!-- 初始状态：居中输入卡片 -->
    <div v-if="!isExpanded" class="layout layout-centered">
      <section class="panel panel-form panel-centered">
        <header class="panel-head">
          <div class="logo dr-pop" aria-hidden="true">
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

        <form class="form" @submit.prevent="handleSubmit">
          <label class="field">
            <span>研究主题</span>
            <textarea
              v-model="form.topic"
              placeholder="例如：探索多模态模型在 2025 年的关键突破"
              rows="4"
              required
            ></textarea>
          </label>

          <section class="options">
            <label class="field option">
              <span>搜索引擎</span>
              <select v-model="form.searchApi">
                <option value="">沿用后端配置</option>
                <option
                  v-for="option in searchOptions"
                  :key="option"
                  :value="option"
                >
                  {{ option }}
                </option>
              </select>
            </label>
          </section>

          <!-- 本地 LLM 选择 -->
          <section class="llm-probe-section">
            <div class="llm-probe-header">
              <span class="llm-probe-title">本地 LLM</span>
              <button
                type="button"
                class="probe-refresh-btn"
                :disabled="probeLoading"
                @click="refreshProbe"
                :title="probeLoading ? '探测中…' : '重新探测本地服务'"
              >
                <span
                  class="material-symbols-outlined probe-refresh-icon"
                  :class="{ spinning: probeLoading }"
                  aria-hidden="true"
                >refresh</span>
                {{ probeLoading ? "探测中…" : "刷新" }}
              </button>
            </div>

            <p v-if="probeError" class="probe-error">{{ probeError }}</p>

            <template v-else-if="!probeLoading">
              <div v-if="runningProviders.length === 0" class="probe-empty">
                未检测到本地 LLM 服务（Ollama / LM Studio / mlx-lm），将沿用后端配置。
              </div>

              <div v-else class="llm-selects">
                <label class="field option">
                  <span>服务</span>
                  <select v-model="form.llmProvider" @change="form.llmModel = modelsForProvider[0] ?? ''">
                    <option value="">沿用后端配置</option>
                    <option
                      v-for="key in runningProviders"
                      :key="key"
                      :value="key"
                    >
                      {{ PROVIDER_LABELS[key] ?? key }}
                    </option>
                  </select>
                </label>

                <label class="field option" v-if="form.llmProvider">
                  <span>模型</span>
                  <select v-model="form.llmModel">
                    <template v-if="modelsForProvider.length > 0">
                      <option
                        v-for="m in modelsForProvider"
                        :key="m"
                        :value="m"
                      >{{ m }}</option>
                    </template>
                    <template v-else>
                      <option value="">（服务已运行，当前无可列出的模型）</option>
                    </template>
                  </select>
                </label>
              </div>
            </template>

            <div v-else class="probe-loading">
              <span class="material-symbols-outlined spinner-sm" aria-hidden="true">progress_activity</span>
              正在探测本地 LLM 服务…
            </div>
          </section>

          <div class="form-actions">
            <div class="ai-halo" :class="{ active: loading }">
              <button class="submit" type="submit" :disabled="loading || preflightChecking">
                <span class="submit-label">
                  <span
                    v-if="loading || preflightChecking"
                    class="material-symbols-outlined spinner"
                    aria-hidden="true"
                  >progress_activity</span>
                  {{ preflightChecking ? "检测模型中…" : loading ? "研究进行中..." : "开始研究" }}
                </span>
              </button>
            </div>
            <button
              v-if="loading"
              type="button"
              class="secondary-btn"
              @click="cancelResearch"
            >
              取消研究
            </button>
          </div>
        </form>

        <div v-if="error" class="error-block">
          <p class="error-chip">
            <span class="material-symbols-outlined" aria-hidden="true">error</span>
            {{ error }}
          </p>
          <p v-if="preflightHint" class="preflight-hint" v-html="preflightHint.replace(/\n/g, '<br/>')"></p>
        </div>
        <p v-else-if="loading" class="hint muted">
          正在收集线索与证据，实时进展见右侧区域。
        </p>
      </section>
    </div>

    <!-- 全屏状态：左右分栏布局 -->
    <div v-else class="layout layout-fullscreen">
      <!-- 左侧：研究信息 -->
      <aside class="sidebar">
        <div class="sidebar-header">
          <button class="back-btn" @click="goBack" :disabled="loading">
            <span class="material-symbols-outlined" aria-hidden="true">arrow_back</span>
            返回
          </button>
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
        class="panel panel-result"
        v-if="todoTasks.length || reportMarkdown || progressLogs.length"
      >
        <header class="status-bar">
          <div class="status-main">
            <div class="status-chip" :class="{ active: loading }">
              <span class="dot"></span>
              {{ loading ? "研究进行中" : "研究流程完成" }}
            </div>
            <span class="status-meta">
              任务进度：{{ completedTasks }} / {{ totalTasks || todoTasks.length || 1 }}
              · 阶段记录 {{ progressLogs.length }} 条
            </span>
          </div>
          <div class="status-controls">
            <button class="secondary-btn" @click="logsCollapsed = !logsCollapsed">
              {{ logsCollapsed ? "展开流程" : "收起流程" }}
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
              <pre class="block-pre">{{ currentTaskSummary || "暂无可用信息" }}</pre>
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
          <h3>最终报告</h3>
          <pre class="block-pre">{{ reportMarkdown }}</pre>
        </div>
      </section>

    </div>
  </main>
</template>

<script lang="ts" setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import HistoryModal from "./components/HistoryModal.vue";

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
});
// ─────────────────────────────────────────────────────────────────────────────

const loading = ref(false);
const preflightChecking = ref(false);
const preflightHint = ref("");
const error = ref("");
const progressLogs = ref<string[]>([]);
const logsCollapsed = ref(false);
const isExpanded = ref(false);

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
});
</script>


<style scoped>
/* ============================================================
   DeepResearch v3 — Sapphire Ink + Spring Motion
   Tokens from style.css (global). Scoped component styles.
============================================================ */

.app-shell {
  position: relative;
  min-height: 100vh;
  padding: 72px 24px;
  display: flex;
  justify-content: center;
  align-items: center;
  background: var(--bg-page);
  color: var(--on-surface);
  overflow: hidden;
  box-sizing: border-box;
  transition: padding var(--dur-long) var(--ease-standard);
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
  gap: 24px;
  z-index: 1;
}

.layout-centered {
  max-width: 640px;
  justify-content: center;
  align-items: center;
}

.layout-fullscreen {
  height: 100vh;
  max-width: 100%;
  gap: 0;
  align-items: stretch;
}

/* ── Panels ──────────────────────────────────────────────── */
.panel {
  position: relative;
  flex: 1 1 360px;
  padding: 24px;
  border-radius: var(--radius-xl);
  background: var(--surface);
  border: 1px solid var(--outline-variant);
  box-shadow: var(--elev-2);
  overflow: hidden;
}

.panel-form {
  max-width: 640px;
}

.panel-centered {
  width: 100%;
  max-width: 640px;
  padding: 40px;
  border-radius: var(--radius-2xl);
  box-shadow: var(--elev-2);
  animation: drBounceIn var(--dur-xlong) var(--ease-spring) both;
}

.panel-result {
  min-width: 360px;
  flex: 2 1 420px;
}

.panel-form h1 {
  margin: 0;
  font-size: 24px;
  font-weight: var(--weight-semibold);
  letter-spacing: var(--tracking-tight);
}

.panel-form p {
  margin: 4px 0 0;
  color: var(--on-surface-muted);
  font-size: 13px;
}

.panel-head {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 28px;
}

/* ── Logo (3×3 grid mark) ────────────────────────────────── */
.logo {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-md);
  background: linear-gradient(135deg, var(--primary-500), var(--primary-700));
  padding: 10px;
  box-shadow: var(--elev-2);
  flex-shrink: 0;
  display: grid;
  place-items: center;
  transition: transform var(--dur-medium) var(--ease-spring),
    box-shadow var(--dur-medium) var(--ease-spring);
}

.logo:hover {
  transform: translateY(-2px) rotate(-3deg) scale(1.04);
  box-shadow: var(--elev-3);
}

.logo-sm {
  width: 40px;
  height: 40px;
  padding: 7px;
  border-radius: var(--radius-sm);
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

/* ── Form ────────────────────────────────────────────────── */
.form {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field span {
  font-size: 13px;
  font-weight: var(--weight-medium);
  color: var(--on-surface-muted);
}

textarea,
input,
select {
  padding: 14px 16px;
  border-radius: var(--radius-md);
  border: 1px solid var(--outline);
  background: var(--surface-bright);
  color: var(--on-surface);
  font-size: 14px;
  font-family: inherit;
  transition: border-color var(--dur-short) var(--ease-standard),
    box-shadow var(--dur-short) var(--ease-standard);
}

textarea {
  resize: vertical;
  min-height: 88px;
  line-height: var(--leading-base);
}

select {
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%238696a0' stroke-width='2' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 16px center;
  padding-right: 40px;
  cursor: pointer;
}

textarea:focus,
input:focus,
select:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: var(--focus-ring);
}

.options {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.option {
  flex: 1;
  min-width: 140px;
}

/* ── Local LLM probe section ─────────────────────────────── */
.llm-probe-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 18px;
  background: var(--surface-2);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-lg);
}

.llm-probe-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.llm-probe-title {
  font-size: 14px;
  font-weight: var(--weight-semibold);
  color: var(--on-surface);
}

.probe-refresh-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 6px 12px;
  font-size: 12px;
  font-weight: var(--weight-medium);
  color: var(--primary-700);
  background: transparent;
  border: 1px solid var(--outline);
  border-radius: var(--radius-pill);
  cursor: pointer;
  transition: background var(--dur-short) var(--ease-standard),
    border-color var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
}

.probe-refresh-btn:hover:not(:disabled) {
  background: var(--primary-100);
  transform: translateY(-1px);
}

.probe-refresh-btn:active:not(:disabled) {
  transform: scale(0.96);
}

.probe-refresh-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.probe-refresh-icon {
  font-size: 16px;
  flex-shrink: 0;
}

.probe-refresh-icon.spinning {
  animation: spin 0.8s linear infinite;
}

.probe-empty {
  font-size: 12px;
  color: var(--on-surface-faint);
  line-height: var(--leading-snug);
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
  color: var(--on-surface-muted);
}

.spinner-sm {
  font-size: 16px;
  color: var(--primary);
  animation: spin 1s linear infinite;
}

.llm-selects {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* ── Submit + AI halo ────────────────────────────────────── */
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
  padding: 14px 28px;
  border-radius: var(--radius-pill);
  border: none;
  background: var(--primary);
  color: var(--on-primary);
  font-size: 15px;
  font-weight: var(--weight-medium);
  font-family: inherit;
  cursor: pointer;
  box-shadow: var(--elev-2);
  display: inline-flex;
  align-items: center;
  gap: 10px;
  transition: background var(--dur-short) var(--ease-standard),
    box-shadow var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
}

.submit-label {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.submit .spinner {
  font-size: 18px;
  animation: spin 1s linear infinite;
}

.submit:not(:disabled):hover {
  background: var(--primary-hover);
  box-shadow: var(--elev-3);
  transform: translateY(-1px);
}

.submit:not(:disabled):active {
  transform: scale(0.96);
  transition-duration: var(--dur-instant);
}

.submit:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.secondary-btn {
  padding: 9px 21px;
  border-radius: var(--radius-pill);
  background: transparent;
  border: 1px solid var(--outline);
  color: var(--primary-700);
  font-size: 14px;
  font-weight: var(--weight-medium);
  font-family: inherit;
  cursor: pointer;
  transition: background var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
}

.secondary-btn:hover {
  background: var(--primary-100);
  transform: translateY(-1px);
}

.secondary-btn:active {
  transform: scale(0.96);
}

/* ── Error / hint ────────────────────────────────────────── */
.error-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.error-chip {
  margin-top: 16px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: var(--bg-danger);
  border: 1px solid var(--border-danger);
  border-radius: var(--radius-md);
  color: var(--fg-danger);
  font-size: 14px;
  animation: drFadeUp var(--dur-long) var(--ease-spring-soft) both;
}

.error-chip .material-symbols-outlined {
  font-size: 20px;
  flex-shrink: 0;
}

.preflight-hint {
  margin: 0;
  padding: 12px 16px;
  font-size: 13px;
  line-height: var(--leading-base);
  color: var(--fg-warning);
  background: var(--bg-warning);
  border: 1px solid var(--border-warning);
  border-radius: var(--radius-md);
}

.hint.muted {
  color: var(--on-surface-muted);
}

/* ── Result panel ────────────────────────────────────────── */
.panel-result {
  display: flex;
  flex-direction: column;
  gap: 20px;
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
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--surface-2);
  color: var(--on-surface-muted);
  padding: 6px 14px;
  border-radius: var(--radius-pill);
  font-size: 13px;
  font-weight: var(--weight-medium);
  border: 1px solid var(--outline-variant);
  animation: drPopIn var(--dur-long) var(--ease-spring) both;
}

.status-chip.active {
  background: var(--primary-100);
  color: var(--primary-700);
  border-color: var(--primary-200);
}

.status-chip .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--neutral-400);
}

.status-chip.active .dot {
  background: var(--primary);
  animation: drPulse 1.4s var(--ease-spring) infinite;
}

.status-meta {
  color: var(--on-surface-muted);
  font-size: 13px;
}

@keyframes drPulse {
  0%   { transform: scale(1);    box-shadow: 0 0 0 0 rgba(0, 128, 105, 0.55); }
  50%  { transform: scale(1.25); box-shadow: 0 0 0 8px rgba(0, 128, 105, 0); }
  100% { transform: scale(1);    box-shadow: 0 0 0 0 rgba(0, 128, 105, 0); }
}

/* ── Timeline ────────────────────────────────────────────── */
.timeline-wrapper {
  max-height: 240px;
  overflow-y: auto;
  padding-right: 8px;
  scrollbar-width: thin;
  scrollbar-color: var(--primary-300) var(--surface-2);
}

.timeline-wrapper::-webkit-scrollbar {
  width: 6px;
}

.timeline-wrapper::-webkit-scrollbar-track {
  background: var(--surface-2);
  border-radius: var(--radius-pill);
}

.timeline-wrapper::-webkit-scrollbar-thumb {
  background: var(--primary-300);
  border-radius: var(--radius-pill);
}

.timeline-wrapper::-webkit-scrollbar-thumb:hover {
  background: var(--primary);
}

.timeline {
  list-style: none;
  padding: 0 0 0 16px;
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
  background: var(--outline);
}

.timeline li {
  position: relative;
  padding-left: 20px;
  color: var(--on-surface);
  font-size: 14px;
  line-height: var(--leading-snug);
  animation: drFadeUp var(--dur-long) var(--ease-spring-soft) both;
}

.timeline li:nth-child(1) { animation-delay: 40ms; }
.timeline li:nth-child(2) { animation-delay: 90ms; }
.timeline li:nth-child(3) { animation-delay: 140ms; }
.timeline li:nth-child(4) { animation-delay: 190ms; }
.timeline li:nth-child(5) { animation-delay: 240ms; }
.timeline li:nth-child(6) { animation-delay: 290ms; }

.timeline-node {
  position: absolute;
  left: -1px;
  top: 6px;
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: var(--primary);
  border: 2px solid var(--primary);
  box-sizing: border-box;
  animation: drDotPop var(--dur-long) var(--ease-spring) both;
}

.timeline li:last-child .timeline-node {
  background: var(--secondary-500);
  border-color: var(--secondary-500);
  animation: drDotPulse 1.6s var(--ease-spring) infinite;
}

@keyframes drDotPop {
  0%   { transform: scale(0); }
  60%  { transform: scale(1.25); }
  100% { transform: scale(1); }
}

@keyframes drDotPulse {
  0%       { box-shadow: 0 0 0 0 rgba(37, 211, 102, 0.55); transform: scale(1); }
  50%      { box-shadow: 0 0 0 6px rgba(37, 211, 102, 0); transform: scale(1.18); }
  100%     { box-shadow: 0 0 0 0 rgba(37, 211, 102, 0); transform: scale(1); }
}

.timeline-enter-active,
.timeline-leave-active {
  transition: opacity var(--dur-medium) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
}

.timeline-enter-from,
.timeline-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

/* ── Tasks ───────────────────────────────────────────────── */
.tasks-section {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 20px;
  align-items: start;
}

.tasks-list {
  background: var(--surface);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-lg);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tasks-list h3 {
  margin: 0;
  font-size: 15px;
  font-weight: var(--weight-semibold);
  color: var(--on-surface-strong);
}

.tasks-list ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.task-item {
  border-radius: var(--radius-md);
  border: 1px solid transparent;
  transition: background var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
  animation: drFadeUp var(--dur-long) var(--ease-spring-soft) both;
}

.task-item:nth-child(1) { animation-delay: 30ms; }
.task-item:nth-child(2) { animation-delay: 70ms; }
.task-item:nth-child(3) { animation-delay: 110ms; }
.task-item:nth-child(4) { animation-delay: 150ms; }
.task-item:nth-child(5) { animation-delay: 190ms; }
.task-item:nth-child(6) { animation-delay: 230ms; }
.task-item:nth-child(7) { animation-delay: 270ms; }
.task-item:nth-child(8) { animation-delay: 310ms; }

.task-item:hover {
  background: var(--surface-2);
  transform: translateX(2px);
}

.task-item.completed {
  background: var(--surface-2);
}

.task-item.active,
.task-item.active:hover {
  background: var(--primary-100);
  border-color: var(--primary-200);
}

.task-button {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px 4px;
  background: transparent;
  border: none;
  color: inherit;
  cursor: pointer;
  text-align: left;
  font-family: inherit;
}

.task-title {
  font-weight: var(--weight-medium);
  font-size: 14px;
  color: var(--on-surface-strong);
}

.task-status {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: var(--radius-pill);
  font-size: 11px;
  font-weight: var(--weight-medium);
  color: var(--on-surface-muted);
  background: var(--surface-2);
}

.task-status .material-symbols-outlined {
  font-size: 13px;
}

.task-status.pending {
  background: var(--surface-2);
  color: var(--on-surface-muted);
}

.task-status.in_progress {
  background: var(--primary-100);
  color: var(--primary-700);
  animation: drStatusBob 1.6s var(--ease-spring) infinite;
}

.task-status.in_progress .material-symbols-outlined {
  animation: spin 1.6s linear infinite;
}

@keyframes drStatusBob {
  0%, 100% { transform: translateY(0); }
  50%      { transform: translateY(-2px); }
}

.task-status.completed {
  background: var(--bg-success);
  color: var(--fg-success);
}

.task-status.skipped {
  background: var(--bg-danger);
  color: var(--fg-danger);
}

.task-intent {
  margin: 0;
  padding: 0 14px 10px;
  font-size: 12px;
  color: var(--on-surface-muted);
}

/* ── Task detail ─────────────────────────────────────────── */
.task-detail {
  background: var(--surface);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-lg);
  padding: 22px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 12px;
}

.task-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: var(--weight-semibold);
  color: var(--on-surface-strong);
}

.task-header .muted {
  margin: 6px 0 0;
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
  border-radius: var(--radius-pill);
  font-size: 12px;
  font-weight: var(--weight-medium);
  background: var(--primary-container);
  color: var(--on-primary-container);
  border: 1px solid var(--primary-200);
}

.task-label.note-chip {
  background: var(--bg-success);
  color: var(--fg-success);
  border-color: var(--border-success);
}

.task-label.path-chip {
  max-width: 360px;
  background: var(--tertiary-container);
  color: var(--on-tertiary-container);
  border-color: var(--tertiary-outline);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.path-label {
  font-weight: var(--weight-medium);
}

.path-text {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chip-action {
  border: none;
  background: var(--tertiary-container);
  color: var(--on-tertiary-container);
  padding: 3px 8px;
  border-radius: var(--radius-sm);
  font-size: 11px;
  font-family: inherit;
  cursor: pointer;
  transition: background var(--dur-short) var(--ease-standard);
}

.chip-action:hover {
  background: var(--tertiary-outline);
  color: var(--on-tertiary-container);
}

.task-notices {
  background: var(--primary-50);
  border: 1px solid var(--primary-200);
  border-radius: var(--radius-md);
  padding: 14px 18px;
  color: var(--on-surface);
}

.task-notices h4 {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: var(--weight-semibold);
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
}

/* ── Sources / summary / report blocks ───────────────────── */
.sources-block,
.summary-block {
  position: relative;
  padding: 18px;
  border-radius: var(--radius-lg);
  background: var(--surface-bright);
  border: 1px solid var(--outline-variant);
}

.sources-block h3,
.summary-block h3 {
  margin: 0 0 14px;
  font-size: 16px;
  font-weight: var(--weight-semibold);
  color: var(--on-surface-strong);
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
  color: var(--primary-700);
  text-decoration: none;
  font-weight: var(--weight-medium);
  font-size: 14px;
  transition: color var(--dur-short) var(--ease-standard);
}

.source-link::after {
  content: " ↗";
  font-size: 12px;
  opacity: 0.6;
}

.source-link:hover {
  color: var(--on-surface-strong);
}

.source-tooltip {
  display: none;
  position: absolute;
  bottom: calc(100% + 12px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--surface-bright);
  color: var(--on-surface);
  padding: 14px 16px;
  border-radius: var(--radius-lg);
  box-shadow: var(--elev-4);
  width: min(420px, 90vw);
  z-index: 20;
  border: 1px solid var(--outline-variant);
}

.source-tooltip::before {
  content: "";
  position: absolute;
  bottom: -12px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 12px 10px 0 10px;
  border-style: solid;
  border-color: var(--surface-bright) transparent transparent transparent;
  filter: drop-shadow(0 2px 3px rgba(10, 17, 36, 0.12));
}

.source-tooltip p {
  margin: 0 0 8px;
  font-size: 13px;
  line-height: var(--leading-base);
}

.source-tooltip p:last-child {
  margin-bottom: 0;
}

.muted-text {
  color: var(--on-surface-muted);
}

.source-item:hover .source-tooltip,
.source-item:focus-within .source-tooltip {
  display: block;
}

.block-pre {
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: var(--leading-relaxed);
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--on-surface);
  background: var(--surface);
  padding: 14px 16px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--outline-variant);
  overflow: auto;
  max-height: 420px;
  margin: 0;
  scrollbar-width: thin;
  scrollbar-color: var(--primary-300) var(--surface-2);
}

.block-pre::-webkit-scrollbar {
  width: 6px;
}

.block-pre::-webkit-scrollbar-track {
  background: var(--surface-2);
  border-radius: var(--radius-pill);
}

.block-pre::-webkit-scrollbar-thumb {
  background: var(--primary-300);
  border-radius: var(--radius-pill);
}

.block-pre::-webkit-scrollbar-thumb:hover {
  background: var(--primary);
}

.summary-block .block-pre,
.sources-block .block-pre {
  max-height: 360px;
}

.report-block {
  background: var(--surface);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-lg);
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  box-shadow: var(--elev-1);
}

.report-block h3 {
  margin: 0;
  font-size: 18px;
  font-weight: var(--weight-semibold);
  color: var(--on-surface-strong);
}

.report-block .block-pre {
  background: var(--surface-bright);
  max-height: 440px;
}

.block-highlight {
  animation: glow 1.2s var(--ease-standard);
}

@keyframes glow {
  0%   { box-shadow: 0 0 0 3px rgba(0, 128, 105, 0.28); }
  100% { box-shadow: 0 0 0 0 rgba(0, 128, 105, 0); }
}

/* ── Tool calls ──────────────────────────────────────────── */
.tools-block {
  position: relative;
  padding: 20px;
  border-radius: var(--radius-lg);
  background: var(--surface);
  border: 1px solid var(--outline-variant);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.tools-block h3 {
  margin: 0;
  font-size: 16px;
  font-weight: var(--weight-semibold);
  color: var(--on-surface-strong);
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
  background: var(--surface-bright);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-md);
  padding: 14px;
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
  font-weight: var(--weight-semibold);
  color: var(--on-surface-strong);
}

.tool-entry-note {
  font-size: 12px;
  color: var(--secondary-700);
}

.tool-entry-path {
  margin: 0;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--primary-700);
}

.tool-subtitle {
  margin: 0;
  font-size: 13px;
  color: var(--on-surface-muted);
  font-weight: var(--weight-medium);
}

.tool-pre {
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: var(--leading-base);
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--on-surface);
  background: var(--surface);
  padding: 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--outline-variant);
  overflow: auto;
  max-height: 260px;
  scrollbar-width: thin;
  scrollbar-color: var(--primary-300) var(--surface-2);
}

.tool-pre::-webkit-scrollbar {
  width: 6px;
}

.tool-pre::-webkit-scrollbar-track {
  background: var(--surface-2);
}

.tool-pre::-webkit-scrollbar-thumb {
  background: var(--primary-300);
  border-radius: var(--radius-pill);
}

.link-btn {
  background: none;
  border: none;
  color: var(--primary-700);
  cursor: pointer;
  padding: 0 4px;
  font-size: 12px;
  font-family: inherit;
  border-radius: var(--radius-xs);
  transition: color var(--dur-short) var(--ease-standard);
}

.link-btn:hover {
  color: var(--on-surface-strong);
}

/* ── Sidebar ─────────────────────────────────────────────── */
.sidebar {
  width: 400px;
  min-width: 400px;
  height: 100vh;
  background: var(--surface);
  border-right: 1px solid var(--outline-variant);
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 22px;
  overflow-y: auto;
}

.sidebar-header {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.sidebar-header h2 {
  font-size: 20px;
  font-weight: var(--weight-semibold);
  margin: 0;
  color: var(--on-surface-strong);
}

.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 16px;
  background: transparent;
  border: 1px solid var(--outline);
  border-radius: var(--radius-pill);
  color: var(--primary-700);
  font-size: 14px;
  font-weight: var(--weight-medium);
  font-family: inherit;
  cursor: pointer;
  width: fit-content;
  transition: background var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
}

.back-btn .material-symbols-outlined {
  font-size: 18px;
}

.back-btn:hover:not(:disabled) {
  background: var(--primary-100);
  transform: translateY(-1px);
}

.back-btn:active:not(:disabled) {
  transform: scale(0.96);
}

.back-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.research-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.info-item label {
  font-size: 12px;
  font-weight: var(--weight-medium);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wide);
  color: var(--on-surface-muted);
}

.info-item p {
  margin: 0;
  font-size: 14px;
  color: var(--on-surface);
  line-height: var(--leading-base);
}

.topic-display {
  font-size: 15px !important;
  font-weight: var(--weight-medium);
  color: var(--on-surface-strong) !important;
  padding: 14px 16px;
  background: var(--surface-bright);
  border-radius: var(--radius-md);
  border-left: 3px solid var(--primary);
}

.progress-bar {
  width: 100%;
  height: 6px;
  background: var(--surface-2);
  border-radius: var(--radius-pill);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--primary);
  border-radius: var(--radius-pill);
  transition: width var(--dur-xlong) var(--ease-standard);
}

.progress-text {
  font-size: 13px !important;
  color: var(--on-surface-muted) !important;
  font-weight: var(--weight-medium);
}

.sidebar-actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-top: 16px;
  border-top: 1px solid var(--outline-variant);
}

.new-research-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 14px 28px;
  background: var(--primary);
  border: none;
  border-radius: var(--radius-pill);
  color: var(--on-primary);
  font-size: 15px;
  font-weight: var(--weight-medium);
  font-family: inherit;
  cursor: pointer;
  box-shadow: var(--elev-2);
  transition: background var(--dur-short) var(--ease-standard),
    box-shadow var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
}

.new-research-btn .material-symbols-outlined {
  font-size: 18px;
}

.new-research-btn:hover {
  background: var(--primary-hover);
  box-shadow: var(--elev-3);
  transform: translateY(-1px);
}

.new-research-btn:active {
  transform: scale(0.96);
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
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-danger);
  margin: 0;
}

/* ── Fullscreen result panel ─────────────────────────────── */
.layout-fullscreen .panel-result {
  flex: 1;
  height: 100vh;
  border-radius: 0;
  border: none;
  box-shadow: none;
  overflow-y: auto;
  max-width: none;
  padding: 28px 32px;
}

/* ── History toggle (FAB) ────────────────────────────────── */
.history-toggle-btn {
  position: absolute;
  top: 24px;
  right: 24px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: var(--surface-container-high);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-pill);
  color: var(--on-surface);
  font-size: 13px;
  font-weight: var(--weight-medium);
  font-family: inherit;
  cursor: pointer;
  z-index: 1000;
  box-shadow: var(--elev-1);
  transition: background var(--dur-short) var(--ease-standard),
    box-shadow var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
}

.history-toggle-btn .material-symbols-outlined {
  font-size: 18px;
}

.history-toggle-btn:hover {
  background: var(--surface-container-highest);
  box-shadow: var(--elev-2);
  transform: translateY(-1px) scale(1.04);
}

.history-toggle-btn:active {
  transform: scale(0.92);
}

.expanded .history-toggle-btn {
  top: 16px;
  right: auto;
  left: 24px;
}

/* ── Shared spin keyframe ────────────────────────────────── */
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 1024px) {
  .sidebar {
    width: 320px;
    min-width: 320px;
  }
}

@media (max-width: 960px) {
  .app-shell {
    padding: 56px 16px;
  }

  .layout {
    flex-direction: column;
    align-items: stretch;
  }

  .panel {
    padding: 22px;
  }

  .panel-form,
  .panel-result {
    max-width: none;
  }

  .tasks-section {
    grid-template-columns: 1fr;
  }

  .status-bar {
    flex-direction: column;
    align-items: flex-start;
  }

  .status-main,
  .status-controls {
    width: 100%;
  }
}

@media (max-width: 768px) {
  .layout-fullscreen {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
    min-width: 100%;
    height: auto;
    max-height: 40vh;
  }

  .layout-fullscreen .panel-result {
    height: 60vh;
  }
}

@media (max-width: 600px) {
  .options {
    flex-direction: column;
  }

  .status-meta {
    font-size: 12px;
  }

  .panel-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .panel-form h1 {
    font-size: 22px;
  }
}
</style>
