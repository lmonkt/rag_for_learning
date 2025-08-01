# core/retriever.py
import faiss
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
import logging
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

    def add_documents(self, documents, doc_ids):
        """增量添加文档到BM25索引。"""
        if not documents:
            return
            
        # 扩展现有语料库
        start_idx = len(self.raw_corpus)
        self.raw_corpus.extend([doc['content'] for doc in documents])
        
        # 更新映射
        for i, doc_id in enumerate(doc_ids):
            self.doc_mapping[start_idx + i] = doc_id
            
        # 重新分词并更新语料库
        new_tokenized = [list(jieba.cut(doc['content'])) for doc in documents]
        self.tokenized_corpus.extend(new_tokenized)
        
        # 重建BM25索引（注意：BM25Okapi不支持增量更新）
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        
        logging.info(f"成功添加 {len(documents)} 个文档到BM25索引。索引总数: {len(self.raw_corpus)}")


# --- FAISS Vector Search ---
class FaissIndexManager:
    def __init__(self):
        self.faiss_index = None
        self.faiss_contents_map = {}
        self.faiss_metadatas_map = {}
        self.faiss_id_order_for_index = []

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

        dimension = embeddings_np.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_np)

        for i, original_id in enumerate(original_ids):
            self.faiss_contents_map[original_id] = chunks[i]
            self.faiss_metadatas_map[original_id] = metadatas[i]
        self.faiss_id_order_for_index.extend(original_ids)
        logging.info(f"FAISS索引构建完成，共索引 {self.faiss_index.ntotal} 个文本块。")

    def search(self, query_embedding_np, top_k):
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return [], [], []

        distances, indices = self.faiss_index.search(query_embedding_np, k=top_k)

        docs, metadatas, ids = [], [], []
        for faiss_idx in indices[0]:
            if faiss_idx != -1 and faiss_idx < len(self.faiss_id_order_for_index):
                original_id = self.faiss_id_order_for_index[faiss_idx]
                docs.append(self.faiss_contents_map.get(original_id, ""))
                metadatas.append(self.faiss_metadatas_map.get(original_id, {}))
                ids.append(original_id)

        # 返回与ChromaDB相似的格式，便于下游处理
        return {"documents": [docs], "metadatas": [metadatas], "ids": [ids]}

    def clear(self):
        self.faiss_index = None
        self.faiss_contents_map = {}
        self.faiss_metadatas_map = {}
        self.faiss_id_order_for_index = []

    def add_documents(self, chunks, embeddings_np, metadatas, original_ids):
        """增量添加文档到现有索引。"""
        if embeddings_np.shape[0] == 0:
            logging.warning("没有可供添加的嵌入向量。")
            return

        # 如果还没有索引，则创建新的
        if self.faiss_index is None:
            dimension = embeddings_np.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)

        # 添加新的嵌入向量
        self.faiss_index.add(embeddings_np)

        # 更新映射
        for i, original_id in enumerate(original_ids):
            self.faiss_contents_map[original_id] = chunks[i]
            self.faiss_metadatas_map[original_id] = metadatas[i]
        self.faiss_id_order_for_index.extend(original_ids)
        
        logging.info(f"成功添加 {len(chunks)} 个文本块到FAISS索引。索引总数: {self.faiss_index.ntotal}")

    def get_all_docs_and_ids(self):
        """获取所有文档和ID，用于构建BM25索引"""
        doc_ids = self.faiss_id_order_for_index
        documents = [self.faiss_contents_map.get(doc_id, "") for doc_id in doc_ids]
        return documents, doc_ids


# --- Hybrid Search ---
def hybrid_merge(semantic_results, bm25_results, faiss_meta_map):
    # ... (hybrid_merge function code from the original file)
    # 略作修改以适应新的FaissIndexManager
    merged_dict = {}

    # Semantic results processing
    if semantic_results and semantic_results.get('documents') and semantic_results['documents'][0]:
        docs = semantic_results['documents'][0]
        metas = semantic_results['metadatas'][0]
        ids = semantic_results['ids'][0]
        num_results = len(docs)
        for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
            score = 1.0 - (i / max(1, num_results))
            merged_dict[doc_id] = {'score': HYBRID_SEARCH_ALPHA * score, 'content': doc, 'metadata': meta}

    # BM25 results processing
    if bm25_results:
        valid_scores = [r['score'] for r in bm25_results if 'score' in r]
        max_bm25_score = max(valid_scores) if valid_scores else 1.0
        for result in bm25_results:
            doc_id = result['id']
            normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
            if doc_id in merged_dict:
                merged_dict[doc_id]['score'] += (1 - HYBRID_SEARCH_ALPHA) * normalized_score
            else:
                merged_dict[doc_id] = {
                    'score': (1 - HYBRID_SEARCH_ALPHA) * normalized_score,
                    'content': result['content'],
                    'metadata': faiss_meta_map.get(doc_id, {})
                }

    return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)