# core/retriever.py
import faiss
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
import logging
import hashlib
from typing import List, Tuple
from config import HYBRID_SEARCH_ALPHA


# --- BM25 Keyword Search ---
class BM25IndexManager:
    # ... (BM25IndexManager class code from the original file, no changes needed)
    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}  # 映射BM25索引位置到文档ID
        self.tokenized_corpus = []
        self.raw_corpus = []

    def build_index(self, documents, doc_ids):
        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        return True

    def search(self, query, top_k=5):
        if not self.bm25_index: return []
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        return [{
            'id': self.doc_mapping[idx],
            'score': float(bm25_scores[idx]),
            'content': self.raw_corpus[idx]
        } for idx in top_indices if bm25_scores[idx] > 0]

    def clear(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []


# --- FAISS Vector Search ---
class FaissIndexManager:
    def __init__(self):
        self.faiss_index = None  # IndexIDMap 包裹的底层索引（IndexFlatL2）
        self.faiss_contents_map = {}  # original_id -> content
        self.faiss_metadatas_map = {}  # original_id -> metadata
        # 双向ID映射：original_id<->int64 faiss_id
        self.original_to_faiss_id = {}
        self.faiss_id_to_original_id = {}
        # 新增：保存插入顺序，供 UI/统计使用
        self.faiss_id_order_for_index = []  # List[str] of original_ids

    # --- 内部工具 ---
    @staticmethod
    def _to_faiss_id(original_id: str) -> np.int64:
        """稳定地将字符串ID映射为 int64。使用 md5 截断为 63bit 避免符号问题。"""
        h = hashlib.md5(original_id.encode('utf-8')).digest()
        val = int.from_bytes(h[:8], byteorder='big', signed=False) & ((1 << 63) - 1)
        return np.int64(val)

    def _ensure_index(self, dim: int):
        if self.faiss_index is None:
            base = faiss.IndexFlatL2(dim)
            self.faiss_index = faiss.IndexIDMap(base)
            logging.info("已创建 IndexIDMap(IndexFlatL2) 索引。")

    # --- 构建与批量重建 ---
    def build_index(self, embeddings_np, chunks, metadatas, original_ids):
        # TODO: (改进方向) 高级FAISS索引与管理
        # 思路:
        # 1. 替换 IndexFlatL2: 对于大规模数据，IndexFlatL2（暴力搜索）会很慢。
        #    - 尝试 `faiss.IndexIVFPQ`: 这是一种更高效的索引，结合了倒排文件、产品量化。
        #    - 它需要一个 `quantizer` (通常是 IndexFlatL2) 和训练步骤 `index.train(embeddings)`.
        #    - `index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)`
        # 2. 支持删除/更新: IndexFlatL2 不支持删除。
        #    - 使用 `faiss.IndexIDMap`: 这是一个包装器，它允许你为向量关联一个自定义的64位ID。
        #    - `index = faiss.IndexFlatL2(dimension)`
        #    - `id_map_index = faiss.IndexIDMap(index)`
        #    - 添加时使用: `id_map_index.add_with_ids(embeddings, ids_array)`
        #    - 删除时使用: `id_map_index.remove_ids(ids_to_remove_array)`
        #    这对于动态更新知识库（比如移除过时文档的chunks）至关重要。
        if embeddings_np.shape[0] == 0:
            logging.warning("没有可供索引的嵌入向量。")
            return
        # 重置索引与映射
        self.clear()
        dimension = int(embeddings_np.shape[1])
        embeddings_np = np.asarray(embeddings_np, dtype='float32')
        self._ensure_index(dimension)
        # 生成/记录 ID 映射
        ids = [self._to_faiss_id(oid) for oid in original_ids]
        for oid, fid in zip(original_ids, ids):
            self.original_to_faiss_id[oid] = fid
            self.faiss_id_to_original_id[fid] = oid
        # 添加向量
        ids_np = np.asarray(ids, dtype='int64')
        self.faiss_index.add_with_ids(embeddings_np, ids_np)
        # 维护负载与顺序
        for i, oid in enumerate(original_ids):
            self.faiss_contents_map[oid] = chunks[i]
            self.faiss_metadatas_map[oid] = metadatas[i]
        self.faiss_id_order_for_index = list(original_ids)
        logging.info(f"FAISS索引构建完成，共索引 {self.faiss_index.ntotal} 个文本块。")

    # --- 增删改 API（供 UI 与流程调用） ---
    def add_items(self, embeddings_np, chunks: List[str], metadatas: List[dict], original_ids: List[str]):
        if embeddings_np is None or len(original_ids) == 0:
            return 0
        embeddings_np = np.asarray(embeddings_np, dtype='float32')
        dim = int(embeddings_np.shape[1])
        self._ensure_index(dim)
        # 处理重复：若已存在相同 original_id，先移除
        ids = [self._to_faiss_id(oid) for oid in original_ids]
        existing_ids = [fid for oid, fid in zip(original_ids, ids) if oid in self.original_to_faiss_id]
        if existing_ids:
            try:
                existing_np = np.asarray(existing_ids, dtype='int64')
                selector = faiss.IDSelectorArray(len(existing_np), existing_np)
                removed = self.faiss_index.remove_ids(selector)
                logging.info(f"添加前移除 {removed} 个已存在ID，避免重复。")
            except Exception as e:
                logging.warning(f"移除已存在ID失败，将继续添加（可能引发重复）: {e}")
            # 从顺序与映射中清理旧项
            for oid in original_ids:
                if oid in self.original_to_faiss_id:
                    old_fid = self.original_to_faiss_id.pop(oid, None)
                    if old_fid is not None:
                        self.faiss_id_to_original_id.pop(old_fid, None)
                    if oid in self.faiss_id_order_for_index:
                        try:
                            self.faiss_id_order_for_index.remove(oid)
                        except ValueError:
                            pass
        # 记录映射与负载
        for oid, fid, chunk, meta in zip(original_ids, ids, chunks, metadatas):
            self.original_to_faiss_id[oid] = fid
            self.faiss_id_to_original_id[fid] = oid
            self.faiss_contents_map[oid] = chunk
            self.faiss_metadatas_map[oid] = meta
        # 批量添加
        ids_np = np.asarray(ids, dtype='int64')
        self.faiss_index.add_with_ids(embeddings_np, ids_np)
        # 维护顺序：新项追加到末尾
        self.faiss_id_order_for_index.extend(original_ids)
        return len(original_ids)

    def remove_items(self, original_ids: List[str]) -> int:
        if not original_ids or self.faiss_index is None:
            return 0
        ids = [self.original_to_faiss_id.get(oid) for oid in original_ids if oid in self.original_to_faiss_id]
        if not ids:
            return 0
        try:
            ids_np = np.asarray(ids, dtype='int64')
            selector = faiss.IDSelectorArray(len(ids_np), ids_np)
            removed = int(self.faiss_index.remove_ids(selector))
        except Exception as e:
            logging.error(f"从FAISS移除ID失败: {e}")
            removed = 0
        # 同步移除映射与负载
        for oid in original_ids:
            fid = self.original_to_faiss_id.pop(oid, None)
            if fid is not None:
                self.faiss_id_to_original_id.pop(fid, None)
            self.faiss_contents_map.pop(oid, None)
            self.faiss_metadatas_map.pop(oid, None)
            if oid in self.faiss_id_order_for_index:
                try:
                    self.faiss_id_order_for_index.remove(oid)
                except ValueError:
                    pass
        return removed

    def update_items(self, embeddings_np, chunks: List[str], metadatas: List[dict], original_ids: List[str]) -> int:
        # 简单策略：remove → add（保持相同 ID）
        self.remove_items(original_ids)
        return self.add_items(embeddings_np, chunks, metadatas, original_ids)

    # --- 检索 ---
    def search(self, query_embedding_np, top_k):
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
        query_embedding_np = np.asarray(query_embedding_np, dtype='float32')
        distances, labels = self.faiss_index.search(query_embedding_np, k=top_k)
        # 组装返回
        docs, metadatas, ids, dists = [], [], [], []
        for dist, fid in zip(distances[0], labels[0]):
            if fid == -1:  # 无效项
                continue
            oid = self.faiss_id_to_original_id.get(np.int64(fid))
            if oid is None:
                continue
            ids.append(oid)
            docs.append(self.faiss_contents_map.get(oid, ""))
            metadatas.append(self.faiss_metadatas_map.get(oid, {}))
            dists.append(float(dist))
        # 返回与ChromaDB相似的格式，并附带距离
        return {"documents": [docs], "metadatas": [metadatas], "ids": [ids], "distances": [dists]}

    def clear(self):
        self.faiss_index = None
        self.faiss_contents_map = {}
        self.faiss_metadatas_map = {}
        self.original_to_faiss_id = {}
        self.faiss_id_to_original_id = {}
        self.faiss_id_order_for_index = []  # 清空顺序

    def get_all_docs_and_ids(self) -> Tuple[List[str], List[str]]:
        """获取所有文档和ID，用于构建BM25索引"""
        # 使用维护的插入顺序，过滤已删除项
        doc_ids = [oid for oid in self.faiss_id_order_for_index if oid in self.faiss_contents_map]
        documents = [self.faiss_contents_map.get(doc_id, "") for doc_id in doc_ids]
        return documents, doc_ids


# --- Hybrid Search ---
def hybrid_merge(semantic_results, bm25_results, faiss_meta_map):
    # ... (hybrid_merge function code from the original file)
    # 融合改为使用向量距离→相似度（或回退排名分），并做归一化。
    merged_dict = {}

    # Semantic results processing
    if isinstance(semantic_results, dict) and semantic_results.get('documents') and semantic_results['documents'][0]:
        docs = semantic_results['documents'][0]
        metas = semantic_results['metadatas'][0]
        ids = semantic_results['ids'][0]
        dists = semantic_results.get('distances', [[]])[0] if semantic_results.get('distances') else None
        if dists and len(dists) == len(docs):
            # 距离转相似度：s = 1 / (1 + d)
            sims = [1.0 / (1.0 + max(0.0, float(d))) for d in dists]
            # 归一化到 [0,1]
            s_min, s_max = (min(sims), max(sims)) if sims else (0.0, 1.0)
            denom = (s_max - s_min) if (s_max - s_min) > 1e-12 else 1.0
            sims_norm = [(s - s_min) / denom for s in sims]
            for doc_id, doc, meta, s in zip(ids, docs, metas, sims_norm):
                merged_dict[doc_id] = {'score': HYBRID_SEARCH_ALPHA * float(s), 'content': doc, 'metadata': meta}
        else:
            # 回退：按排名线性赋分
            num_results = len(docs)
            for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
                score = 1.0 - (i / max(1, num_results))
                merged_dict[doc_id] = {'score': HYBRID_SEARCH_ALPHA * score, 'content': doc, 'metadata': meta}

    # BM25 results processing（max 归一化）
    if bm25_results:
        valid_scores = [r['score'] for r in bm25_results if 'score' in r]
        max_bm25_score = max(valid_scores) if valid_scores else 1.0
        for result in bm25_results:
            doc_id = result['id']
            normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0.0
            if doc_id in merged_dict:
                merged_dict[doc_id]['score'] += (1 - HYBRID_SEARCH_ALPHA) * normalized_score
            else:
                merged_dict[doc_id] = {
                    'score': (1 - HYBRID_SEARCH_ALPHA) * normalized_score,
                    'content': result['content'],
                    'metadata': faiss_meta_map.get(doc_id, {})
                }

    return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)