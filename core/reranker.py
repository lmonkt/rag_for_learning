# core/reranker.py
import logging
import threading
from functools import lru_cache
import re
from sentence_transformers import CrossEncoder
import requests

from config import (
    CROSS_ENCODER_MODEL_NAME, RERANK_METHOD, OLLAMA_API_URL,
    GENERATOR_MODEL_OLLAMA_LIGHT, RERANKER_TIMEOUT, HTTP_RETRIES
)

# --- Cross Encoder Reranking ---

cross_encoder = None
cross_encoder_lock = threading.Lock()

# 创建一个会话以复用连接
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=HTTP_RETRIES))


def get_cross_encoder():
    """线程安全地延迟加载交叉编码器模型。"""
    global cross_encoder
    if cross_encoder is None:
        with cross_encoder_lock:
            if cross_encoder is None:
                try:
                    logging.info(f"正在加载交叉编码器模型: {CROSS_ENCODER_MODEL_NAME}")
                    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
                    logging.info("交叉编码器模型加载成功。")
                except Exception as e:
                    logging.error(f"加载交叉编码器失败: {e}")
                    logging.warning("将禁用交叉编码器重排序功能，使用简单排序作为后备方案。")
                    # 设置为None以便后续检查
                    cross_encoder = None
    return cross_encoder


def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k):
    """使用交叉编码器对结果进行重排序。"""
    # TODO: (改进方向) 更复杂的重排序模型/策略
    # 思路:
    # 1. 尝试更先进的交叉编码器模型:
    #    - bge-reranker-large: 一个当前非常流行的、效果很好的重排模型。
    #    - Cohere Rerank: 如果使用Cohere的API，他们的重排模型效果业界领先。
    # 2. 这里的修改很简单，只需要在 config.py 中更改 CROSS_ENCODER_MODEL_NAME 即可。
    encoder = get_cross_encoder()
    if not docs or encoder is None:
        # 如果没有文档或模型加载失败，返回原始顺序
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx / len(docs)})
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]

    cross_inputs = [[query, doc] for doc in docs]
    try:
        scores = encoder.predict(cross_inputs)
        results = sorted(
            [
                (doc_id, {'content': doc, 'metadata': meta, 'score': float(score)})
                for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
            ],
            key=lambda x: x[1]['score'],
            reverse=True
        )
        return results[:top_k]
    except Exception as e:
        logging.error(f"交叉编码器重排序失败: {e}")
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0})
                for doc_id, doc, meta in zip(doc_ids, docs, metadata_list)]


# --- LLM-based Reranking ---

@lru_cache(maxsize=128)  # 增加缓存大小以应对多次调用
def get_llm_relevance_score(query, doc):
    """使用LLM对查询和文档的相关性进行评分（带缓存）。"""
    prompt = f"""评估查询和文档的相关性，仅返回0-10的整数分数。
查询: {query}
文档: {doc}
相关性分数(0-10):"""
    try:
        response = session.post(
            OLLAMA_API_URL,
            json={"model": GENERATOR_MODEL_OLLAMA_LIGHT, "prompt": prompt, "stream": False},
            timeout=RERANKER_TIMEOUT
        )
        result = response.json().get("response", "0").strip()
        match = re.search(r'\b([0-9]|10)\b', result)
        return float(match.group(1)) if match else 0.0
    except Exception as e:
        logging.error(f"LLM评分失败: {e}")
        return 0.0  # 评分失败应给予最低分


def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k):
    """使用LLM对结果进行重排序。"""
    # TODO: (改进方向) 更复杂的重排序策略 - Listwise Reranking
    # 思路:
    # 当前是 "pointwise" 排序，即独立地给每个文档打分，这忽略了文档间的相对关系。
    # "Listwise" 方式效果更好：
    # 1. 构建一个prompt，将所有文档片段传递给LLM。
    # 2. 要求LLM直接输出一个重排后的文档ID列表。
    # 3. Prompt示例:
    #    "Query: {query}\n
    #    Documents:\n
    #    [1] {doc1}\n
    #    [2] {doc2}\n
    #    ...
    #    [10] {doc10}\n
    #    请按相关性从高到低排序，仅输出文档编号序列，如: 3,7,1,9,2,..."

    if not docs:
        return []

    results = []
    for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
        score = get_llm_relevance_score(query, doc)
        results.append((doc_id, {'content': doc, 'metadata': meta, 'score': score}))

    # 按分数降序排序
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    return results[:top_k]


def rerank_documents(query, docs, doc_ids, metadata_list, top_k, method=None):
    """
    重排序文档的主要接口。

    Args:
        query: 查询字符串
        docs: 文档内容列表
        doc_ids: 文档ID列表
        metadata_list: 文档元数据列表
        top_k: 返回的文档数量
        method: 重排序方法，如果为None则使用配置中的默认方法

    Returns:
        排序后的文档列表，格式为 [(doc_id, {'content': doc, 'metadata': meta, 'score': score}), ...]
    """
    if method is None:
        method = RERANK_METHOD

    if method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    elif method == "llm":
        return rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
    else:
        # 如果方法不识别，返回原始顺序
        logging.warning(f"未知的重排序方法: {method}，使用原始顺序")
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx / len(docs)})
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]