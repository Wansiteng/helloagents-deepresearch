<template>
  <div class="modal-overlay" v-if="isOpen" @click.self="close">
    <div class="modal-content">
      <div class="modal-header">
        <h2>研究历史记录</h2>
        <button class="close-btn" @click="close">&times;</button>
      </div>
      <div class="modal-body">
        <div v-if="loading" class="loading-state">加载中...</div>
        <div v-else-if="error" class="error-state">{{ error }}</div>
        <div v-else-if="notes.length === 0" class="empty-state">暂无历史记录</div>
        <ul v-else class="history-list">
          <li v-for="note in notes" :key="note.id" class="history-item" @click="viewNote(note)">
            <div class="item-title">{{ note.title || '无标题' }}</div>
            <div class="item-meta">
              <span class="item-time">{{ formatDate(note.created_at) }}</span>
              <span class="item-type">{{ note.type === 'conclusion' ? '研究报告' : '任务笔记' }}</span>
            </div>
          </li>
        </ul>
      </div>
    </div>
    
    <!-- 内容查看侧滑面板/内部模态框 -->
    <div class="detail-modal" v-if="selectedNoteId" @click.self="closeDetail">
      <div class="detail-content">
         <div class="modal-header">
           <h3>{{ selectedNoteTitle }}</h3>
           <button class="close-btn" @click="closeDetail">&times;</button>
         </div>
         <div class="modal-body markdown-body" v-html="parsedContent"></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, computed } from 'vue';
import { getHistory, getHistoryDetail, HistoryNote } from '../services/api';
// @ts-ignore
import { marked } from 'marked';

const props = defineProps<{ isOpen: boolean }>();
const emit = defineEmits(['close']);

const notes = ref<HistoryNote[]>([]);
const loading = ref(false);
const error = ref('');

const selectedNoteId = ref('');
const selectedNoteTitle = ref('');
const noteContent = ref('');

const parsedContent = computed(() => {
  return marked(noteContent.value || '');
});

watch(() => props.isOpen, async (newVal) => {
  if (newVal) {
    await fetchHistory();
  } else {
    selectedNoteId.value = '';
    noteContent.value = '';
  }
});

async function fetchHistory() {
  loading.value = true;
  error.value = '';
  try {
    const data = await getHistory();
    // 优先展示 conclusion
    notes.value = data.notes.filter(n => n.type === 'conclusion').sort((a,b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  } catch (err: any) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
}

function formatDate(dateStr: string) {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  return d.toLocaleString();
}

async function viewNote(note: HistoryNote) {
  selectedNoteId.value = note.id;
  selectedNoteTitle.value = note.title;
  noteContent.value = '加载中...';
  try {
    const data = await getHistoryDetail(note.id);
    noteContent.value = data.content;
  } catch (err: any) {
    noteContent.value = `加载失败：${err.message}`;
  }
}

function close() {
  emit('close');
}

function closeDetail() {
  selectedNoteId.value = '';
  noteContent.value = '';
}
</script>

<style scoped>
/* ============================================================
   HistoryModal v3 — Sapphire Ink light dialog
============================================================ */

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(10, 17, 36, 0.36);
  z-index: 1000;
  display: grid;
  place-items: center;
  backdrop-filter: blur(3px);
  animation: scrimIn var(--dur-medium) var(--ease-standard);
}

@keyframes scrimIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}

.modal-content,
.detail-content {
  background: var(--surface);
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-2xl);
  width: min(620px, 90vw);
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: var(--elev-5);
  color: var(--on-surface);
  animation: dialogIn var(--dur-long) var(--ease-spring);
}

@keyframes dialogIn {
  0%   { transform: translateY(20px) scale(0.94); opacity: 0; }
  60%  { transform: translateY(-4px) scale(1.01); opacity: 1; }
  100% { transform: translateY(0)    scale(1);    opacity: 1; }
}

.detail-content {
  width: min(800px, 95vw);
  height: 90vh;
  max-height: 90vh;
}

.detail-modal {
  position: absolute;
  inset: 0;
  background: rgba(10, 17, 36, 0.36);
  display: grid;
  place-items: center;
  z-index: 1010;
  backdrop-filter: blur(3px);
  animation: scrimIn var(--dur-medium) var(--ease-standard);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px 14px;
  border-bottom: 1px solid var(--outline-variant);
}

.modal-header h2,
.modal-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: var(--weight-semibold);
  color: var(--on-surface-strong);
}

.close-btn {
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  font-size: 1.5rem;
  line-height: 1;
  cursor: pointer;
  color: var(--on-surface-muted);
  transition: background var(--dur-short) var(--ease-standard),
    color var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
}

.close-btn:hover {
  background: var(--surface-2);
  color: var(--on-surface-strong);
}

.close-btn:active {
  transform: scale(0.92);
}

.modal-body {
  padding: 16px 20px 20px;
  overflow-y: auto;
  flex: 1;
}

.history-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.history-item {
  padding: 14px 16px;
  border: 1px solid var(--outline-variant);
  border-radius: var(--radius-md);
  cursor: pointer;
  background: var(--surface);
  transition: background var(--dur-short) var(--ease-standard),
    border-color var(--dur-short) var(--ease-standard),
    transform var(--dur-medium) var(--ease-spring);
  animation: drFadeUp var(--dur-medium) var(--ease-spring-soft) both;
}

.history-item:nth-child(1) { animation-delay: 30ms; }
.history-item:nth-child(2) { animation-delay: 70ms; }
.history-item:nth-child(3) { animation-delay: 110ms; }
.history-item:nth-child(4) { animation-delay: 150ms; }
.history-item:nth-child(5) { animation-delay: 190ms; }
.history-item:nth-child(6) { animation-delay: 230ms; }
.history-item:nth-child(7) { animation-delay: 270ms; }
.history-item:nth-child(8) { animation-delay: 310ms; }

.history-item:hover {
  background: var(--surface-2);
  border-color: var(--primary-200);
  transform: translateY(-2px);
}

.item-title {
  font-weight: var(--weight-medium);
  font-size: 14px;
  color: var(--on-surface-strong);
  margin-bottom: 8px;
}

.item-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: var(--on-surface-muted);
}

.item-type {
  background: var(--secondary-container);
  color: var(--on-secondary-container);
  padding: 3px 10px;
  border-radius: var(--radius-pill);
  font-size: 11px;
  font-weight: var(--weight-medium);
}

.loading-state,
.error-state,
.empty-state {
  text-align: center;
  padding: 2rem;
  color: var(--on-surface-muted);
}

.error-state {
  color: var(--fg-danger);
}

.markdown-body {
  line-height: var(--leading-base);
  color: var(--on-surface);
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3 {
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  color: var(--on-surface-strong);
}

.markdown-body p {
  margin-bottom: 1em;
}
</style>
