"""向量存储服务 — 基于 ChromaDB 的 RAG 长上下文记忆层。

架构概览
--------
本模块提供两个核心能力：

1. **语义分块（Chunking）**：
   :func:`sliding_window_chunk` 按段落语义优先切分，再做滑动窗口截断，
   保证每个 Chunk 在 *chunk_size* Token 以内、相邻 Chunk 有 *overlap* Token 重叠，
   避免关键信息在切割点丢失。

2. **向量化存储与检索（VectorStore）**：
   :class:`VectorStore` 封装 ChromaDB 持久化客户端，使用 Ollama 本地 Embedding
   模型将 Chunk 向量化后写入，并提供余弦相似度语义检索接口。

使用方式::

    from services.vector_store import VectorStore

    vs = VectorStore(
        workspace="./vector_store",
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
    )

    # 存入摘要
    vs.add_document(
        text="任务摘要文本 ...",
        metadata={"task_id": 1, "topic": "深度研究主题"},
        doc_id="task_1",
    )

    # 语义检索
    results = vs.query("相关查询词", n_results=5)
    for r in results:
        print(r["text"], r["distance"])
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Token 估算常量
# 中文字符约 1.5 chars/token，英文词约 1 token，此处用字符数粗估
# ─────────────────────────────────────────────────────────────────────────────
_AVG_CHARS_PER_TOKEN: float = 2.0  # 保守估计，中英混合场景


def _estimate_tokens(text: str) -> int:
    """粗估文本 token 数（字符数 / 均值字符密度）。"""
    return max(1, int(len(text) / _AVG_CHARS_PER_TOKEN))


def _token_budget_chars(tokens: int) -> int:
    """将 token 预算转换为字符数上界。"""
    return int(tokens * _AVG_CHARS_PER_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# 分块逻辑
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """对文本进行语义感知的滑动窗口分块。

    策略
    ~~~~
    1. 先按**空行**切分段落（保留语义完整性）。
    2. 逐段累积，Token 数超过 *chunk_size* 时提交当前 Chunk 并保留
       最后 *overlap* Token 的内容作为下一 Chunk 的前缀（滑动窗口）。
    3. 若单段落本身超过 *chunk_size*，则在段落内部按句子边界（。！？. ! ?）
       做二次切割，实在无句边界时按字符数硬切。

    参数
    ----
    text:
        待分块的原始文本。
    chunk_size:
        每个 Chunk 的 Token 上限（默认 500）。
    overlap:
        相邻 Chunk 之间的重叠 Token 数（默认 50）。

    返回
    ----
    非空字符串列表，每个元素为一个 Chunk。
    """
    if not text or not text.strip():
        return []

    chunk_chars = _token_budget_chars(chunk_size)
    overlap_chars = _token_budget_chars(overlap)

    # ── Step 1: 先按空行切段落 ────────────────────────────────────────────
    raw_paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # ── Step 2: 将段落拆解为不超过 chunk_chars 的原子单元 ─────────────────
    atoms: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_chars:
            atoms.append(para)
        else:
            # 段落过长：按句子边界再切
            sentences = re.split(r"(?<=[。！？.!?])\s*", para)
            buf = ""
            for sent in sentences:
                if not sent:
                    continue
                if len(buf) + len(sent) > chunk_chars:
                    if buf:
                        atoms.append(buf.strip())
                        buf = sent
                    else:
                        # 单句超长：硬切
                        start = 0
                        while start < len(sent):
                            atoms.append(sent[start: start + chunk_chars])
                            start += chunk_chars - overlap_chars
                else:
                    buf += ("" if not buf else " ") + sent
            if buf.strip():
                atoms.append(buf.strip())

    # ── Step 3: 滑动窗口拼装 Chunks ───────────────────────────────────────
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0

    for atom in atoms:
        atom_len = len(atom)
        if current_len + atom_len > chunk_chars and current_parts:
            # 提交当前 Chunk
            chunk_text = "\n\n".join(current_parts)
            chunks.append(chunk_text)

            # 保留 overlap 内容作为下一 Chunk 的前缀
            overlap_text = chunk_text[-overlap_chars:] if overlap_chars > 0 else ""
            current_parts = [overlap_text, atom] if overlap_text else [atom]
            current_len = len(overlap_text) + atom_len
        else:
            current_parts.append(atom)
            current_len += atom_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# 向量存储
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """轻量级本地向量数据库（ChromaDB + Ollama Embedding）。

    特性
    ~~~~
    - **持久化**：数据写入磁盘，进程重启后自动恢复。
    - **余弦相似度**：集合使用 ``hnsw:space=cosine``，语义相关度排序更准确。
    - **幂等 upsert**：相同 ``doc_id`` 重复写入时自动覆盖，避免重复 Chunk。
    - **懒加载 chromadb**：import 推迟到首次构造，不影响未启用时的启动速度。

    参数
    ----
    workspace:
        ChromaDB 持久化目录，不存在时自动创建。
    embedding_model:
        Ollama 向量化模型名称（需提前 ``ollama pull <model>``）。
    ollama_base_url:
        Ollama 服务地址（无需 ``/v1`` 后缀）。
    chunk_size:
        分块 Token 上限。
    chunk_overlap:
        相邻 Chunk 重叠 Token 数。
    collection_name:
        ChromaDB 集合名称。
    """

    def __init__(
        self,
        workspace: str = "./vector_store",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        collection_name: str = "deep_research",
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chromadb 未安装，请执行: pip install chromadb>=0.6.0"
            ) from exc

        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._embedding_model = embedding_model
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        self._client = chromadb.PersistentClient(
            path=str(self._workspace),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore 已初始化: workspace=%s model=%s chunk_size=%d overlap=%d",
            workspace,
            embedding_model,
            chunk_size,
            chunk_overlap,
        )

    # ── Embedding ────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """调用 Ollama /api/embed 批量向量化文本列表。

        注意
        ~~~~
        - 使用 ``/api/embed``（多文本批量接口），比 ``/api/embeddings`` 更高效。
        - 超时设 60 秒；本地小模型通常 < 2 秒。
        """
        import requests  # noqa: PLC0415

        if not texts:
            return []

        url = f"{self._ollama_base_url}/api/embed"
        payload: dict[str, Any] = {
            "model": self._embedding_model,
            "input": texts,
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Ollama /api/embed 返回格式:
            # {"model": "...", "embeddings": [[...], [...]]}
            embeddings: list[list[float]] = data["embeddings"]
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding 返回数量不匹配: 预期 {len(texts)}, 实际 {len(embeddings)}"
                )
            return embeddings
        except Exception as exc:
            logger.error("Embedding 失败 (model=%s): %s", self._embedding_model, exc)
            raise

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def add_document(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> int:
        """将文本分块、向量化后写入 ChromaDB，返回实际写入的 Chunk 数量。

        参数
        ----
        text:
            待入库的原始文本（研究摘要、网页内容等）。
        metadata:
            附加在每个 Chunk 上的元信息字典（task_id、topic 等）。
        doc_id:
            文档唯一标识符；省略时自动按内容生成；重复写入时执行 upsert。

        返回
        ----
        写入的 Chunk 数量（0 表示文本为空或分块失败）。
        """
        if not text or not text.strip():
            logger.warning("add_document: 空文本，跳过。")
            return 0

        chunks = sliding_window_chunk(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            logger.warning("add_document: 分块结果为空，跳过。")
            return 0

        base_id = doc_id or hashlib.sha256(text.encode()).hexdigest()[:16]
        ids = [f"{base_id}_c{i}" for i in range(len(chunks))]

        base_meta: dict[str, Any] = metadata or {}
        metadatas: list[dict[str, Any]] = [
            {**base_meta, "chunk_index": i, "total_chunks": len(chunks), "doc_id": base_id}
            for i in range(len(chunks))
        ]

        try:
            embeddings = self._embed(chunks)
        except Exception:
            logger.error("add_document: embedding 失败，文档未写入。")
            return 0

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(
            "add_document: 写入 %d chunks (doc_id=%s, collection_size=%d)",
            len(chunks),
            base_id,
            self._collection.count(),
        )
        return len(chunks)

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """语义相似度检索，返回最相关的 Chunk 列表。

        参数
        ----
        query_text:
            查询文本（任务标题、关键词等）。
        n_results:
            最多返回 Chunk 数量。
        filter_metadata:
            ChromaDB ``where`` 过滤器（如 ``{"task_id": {"$eq": 1}}``）。

        返回
        ----
        字典列表，每项包含::

            {
                "text": str,          # Chunk 文本
                "metadata": dict,      # 元信息
                "distance": float,     # 余弦距离（越小越相关）
            }
        """
        if not query_text or not query_text.strip():
            return []

        total = self._collection.count()
        if total == 0:
            return []

        safe_n = min(n_results, total)

        try:
            query_embedding = self._embed([query_text])[0]
        except Exception:
            logger.error("query: embedding 失败，返回空结果。")
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": safe_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self._collection.query(**kwargs)

        items: list[dict[str, Any]] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]

        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            items.append({"text": doc, "metadata": meta or {}, "distance": dist})

        logger.debug("query: 返回 %d 条结果 (query=%r)", len(items), query_text[:60])
        return items

    # ── 辅助 ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """返回已存储的 Chunk 总数。"""
        return self._collection.count()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VectorStore(workspace={self._workspace!r}, "
            f"model={self._embedding_model!r}, "
            f"chunks={self.count()})"
        )
