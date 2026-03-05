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
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  z-index: 1000;
  display: flex;
  justify-content: center;
  align-items: center;
  backdrop-filter: blur(4px);
}

.modal-content, .detail-content {
  background: var(--bg-card, #1e1e24);
  border: 1px solid var(--border-color, #333);
  border-radius: 12px;
  width: 90%;
  max-width: 600px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  color: var(--text-normal, #e0e0e0);
}

.detail-content {
  max-width: 800px;
  width: 95%;
  height: 90vh;
  max-height: 90vh;
}

.detail-modal {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1010;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border-color, #333);
}

.modal-header h2, .modal-header h3 {
  margin: 0;
}

.close-btn {
  background: transparent;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text-muted, #999);
}
.close-btn:hover {
  color: var(--text-normal, #fff);
}

.modal-body {
  padding: 24px;
  overflow-y: auto;
  flex: 1;
}

.history-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.history-item {
  padding: 16px;
  border: 1px solid var(--border-color, #333);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  background: var(--bg-card, #1e1e24);
}

.history-item:hover {
  border-color: var(--primary-color, #10a37f);
  transform: translateY(-2px);
}

.item-title {
  font-weight: 500;
  margin-bottom: 8px;
}

.item-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: var(--text-muted, #999);
}

.item-type {
  background: var(--primary-color, #10a37f);
  color: #fff;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
}

.loading-state, .error-state, .empty-state {
  text-align: center;
  padding: 2rem;
  color: var(--text-muted, #999);
}
.error-state {
  color: #ef4444;
}

.markdown-body {
  line-height: 1.6;
}
.markdown-body h1, .markdown-body h2, .markdown-body h3 {
  margin-top: 1.5em;
  margin-bottom: 0.5em;
}
.markdown-body p {
  margin-bottom: 1em;
}
</style>
