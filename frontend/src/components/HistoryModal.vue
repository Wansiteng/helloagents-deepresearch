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
   HistoryModal v4 "Aether" — glass dialog over blurred scrim
============================================================ */

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(10, 14, 26, 0.32);
  backdrop-filter: saturate(160%) blur(12px);
  -webkit-backdrop-filter: saturate(160%) blur(12px);
  z-index: 1000;
  display: grid;
  place-items: center;
  padding: 32px;
  animation: aetherScrimIn 300ms var(--aether-ease);
}

@keyframes aetherScrimIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}

.modal-content,
.detail-content {
  width: min(640px, 100%);
  max-height: 80vh;
  background: var(--glass-bg-strong);
  backdrop-filter: var(--glass-blur-lg);
  -webkit-backdrop-filter: var(--glass-blur-lg);
  border: 1px solid rgba(255, 255, 255, 0.70);
  border-radius: 28px;
  box-shadow: var(--soft-5);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  color: var(--aether-ink-2);
  animation: aetherDialogIn 500ms var(--aether-ease);
}

@keyframes aetherDialogIn {
  from { transform: translateY(20px) scale(0.96); opacity: 0; }
  to   { transform: translateY(0) scale(1); opacity: 1; }
}

.detail-content {
  width: min(880px, 100%);
  height: 90vh;
  max-height: 90vh;
}

.detail-modal {
  position: absolute;
  inset: 0;
  background: rgba(10, 14, 26, 0.32);
  backdrop-filter: saturate(160%) blur(12px);
  -webkit-backdrop-filter: saturate(160%) blur(12px);
  display: grid;
  place-items: center;
  z-index: 1010;
  padding: 32px;
  animation: aetherScrimIn 300ms var(--aether-ease);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px 24px 12px;
  border-bottom: 1px solid var(--aether-line);
}

.modal-header h2,
.modal-header h3 {
  margin: 0;
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: var(--aether-ink);
}

.close-btn {
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--aether-line);
  border-radius: 12px;
  font-size: 1.4rem;
  line-height: 1;
  color: var(--aether-ink-3);
  cursor: pointer;
  font-family: inherit;
  box-shadow: var(--soft-1);
  transition: background 220ms var(--aether-ease),
    color 220ms var(--aether-ease),
    transform 220ms var(--aether-ease);
}

.close-btn:hover {
  background: rgba(255, 255, 255, 0.85);
  color: var(--aether-ink);
  transform: translateY(-1px);
}

.close-btn:active {
  transform: scale(0.96);
}

.modal-body {
  padding: 12px 18px 22px;
  overflow-y: auto;
  flex: 1;
}

.history-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.history-item {
  padding: 14px;
  border-radius: 14px;
  cursor: pointer;
  background: transparent;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: background 220ms var(--aether-ease);
  animation: aetherFadeUp 500ms var(--aether-ease) both;
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
  background: rgba(10, 14, 26, 0.04);
}

.item-title {
  font-weight: 500;
  font-size: 14.5px;
  color: var(--aether-ink);
  letter-spacing: -0.005em;
  line-height: 1.4;
}

.item-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: var(--aether-ink-4);
}

.item-type {
  padding: 3px 10px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.08);
  color: var(--primary-600);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.02em;
}

.loading-state,
.error-state,
.empty-state {
  text-align: center;
  padding: 2.5rem 1rem;
  color: var(--aether-ink-4);
  font-size: 14px;
}

.error-state {
  color: var(--fg-danger);
}

.markdown-body {
  line-height: 1.65;
  letter-spacing: -0.005em;
  color: var(--aether-ink-2);
  font-size: 15px;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3 {
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  color: var(--aether-ink);
  letter-spacing: -0.015em;
}

.markdown-body p {
  margin-bottom: 0.9em;
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

.markdown-body code {
  font-family: var(--font-mono);
  font-size: 0.88em;
  background: rgba(37, 99, 235, 0.08);
  color: var(--primary-700);
  padding: 2px 6px;
  border-radius: 6px;
}

.markdown-body pre {
  background: rgba(255, 255, 255, 0.5);
  border: 1px solid var(--aether-line);
  border-radius: 14px;
  padding: 14px 16px;
  overflow-x: auto;
}

.markdown-body pre code {
  background: transparent;
  color: var(--aether-ink-2);
  padding: 0;
}
</style>
