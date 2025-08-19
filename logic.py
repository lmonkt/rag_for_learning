# logic.py
import logging
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
import gradio as gr
# 导入核心组件
from core.document_processor import process_files_to_chunks, FileProcessor
from core.retriever import FaissIndexManager, BM25IndexManager, hybrid_merge
from core.reranker import rerank_documents
from core.llm_interface import call_ollama_api_stream, call_siliconflow_api, generate_query_variations
from core.web_search import serpapi_search
import core.reranker as reranker  # 导入模块以访问get_cross_encoder

# 导入配置
from config import (
    EMBED_MODEL_NAME, RETRIEVER_TOP_K, RERANKER_TOP_K,
    RECURSIVE_RETRIEVAL_MAX_ITERATIONS, RERANK_METHOD, OLLAMA_EMBED_MODEL
)
from utils.helpers import is_embedding_model_available

# --- 全局状态/资源管理器 ---
# 这些对象代表了应用的核心状态，由logic.py统一管理
# 在生产环境中，这些可能会被持久化存储（如数据库、文件系统）所取代
file_processor = FileProcessor()
faiss_manager = FaissIndexManager()
bm25_manager = BM25IndexManager()
embed_model = None
get_embedding: None = None


def initialize_models():
    """初始化所有需要预加载的模型。"""
    global embed_model, get_embedding  # <<< 重要：现在需要声明两个全局变量

    if embed_model is None:
        logging.info("正在加载嵌入模型...")

        # 1. 首先检查 Ollama 模型是否可用
        if is_embedding_model_available(OLLAMA_EMBED_MODEL):
            print(f"✅ 检测到 Ollama 已存在嵌入模型: {OLLAMA_EMBED_MODEL}")
            print("=> 将使用 Ollama 提供的嵌入服务。")

            # 定义使用 Ollama 的 get_embedding 函数
            def ollama_get_embedding(texts):
                # 确保输入是字符串列表，处理各种可能的输入格式
                if isinstance(texts, str):
                    texts = [texts]
                elif isinstance(texts, list):
                    # 如果是嵌套列表，展平它
                    flattened = []
                    for item in texts:
                        if isinstance(item, list):
                            # 如果item本身也是列表，递归展平
                            for subitem in item:
                                if isinstance(subitem, str):
                                    flattened.append(subitem)
                                else:
                                    flattened.append(str(subitem))
                        elif isinstance(item, str):
                            flattened.append(item)
                        else:
                            flattened.append(str(item))
                    texts = flattened
                else:
                    # 如果不是字符串也不是列表，转换为字符串列表
                    texts = [str(texts)]

                # 确保列表不为空
                if not texts:
                    texts = [""]

                # 记录调试信息
                logging.debug(f"ollama_get_embedding接收到的输入类型: {type(texts)}, 内容: {texts[:2] if len(texts) > 1 else texts}")

                response = ollama.embed(model=OLLAMA_EMBED_MODEL, input=texts)
                embeddings = response['embeddings']
                return np.array(embeddings, dtype='float32')

            # 将全局的 get_embedding 指向这个函数
            get_embedding = ollama_get_embedding

        else:
            print(f"❌ Ollama 中未找到嵌入模型: {OLLAMA_EMBED_MODEL}，推荐下载，因为SentenceTransformer模型加载速度较慢。")
            print(f"=> 将回退到 SentenceTransformer 模型: {EMBED_MODEL_NAME}")

            # 加载 SentenceTransformer 模型
            embed_model = SentenceTransformer(EMBED_MODEL_NAME)  # <<< 注意：这里才真正赋值给 embed_model

            # 定义使用 SentenceTransformer 的 get_embedding 函数
            def st_get_embedding(texts):
                embeddings = embed_model.encode(texts, show_progress_bar=True)
                return np.array(embeddings, dtype='float32')

            # 将全局的 get_embedding 指向这个函数
            get_embedding = st_get_embedding

        logging.info("嵌入模型加载完成。")
    # 预热交叉编码器
    if RERANK_METHOD == 'cross_encoder':
        logging.info("正在预加载交叉编码器...")
        reranker.get_cross_encoder()
        logging.info("交叉编码器预加载完成。")


def process_uploaded_files(files, progress=None):  # 增加默认值，更规范
    """处理上传的PDF文件，构建或重建知识库。"""
    if not files:
        return "请选择要上传的PDF文件", []

    # 1. 清理旧数据
    if progress is not None:
        progress(0.1, desc="清理历史数据...")

    faiss_manager.clear()
    bm25_manager.clear()
    file_processor.clear_files()

    # 2. 处理文件，切分文本块
    # 注意：我们将progress对象传递给下一层函数
    chunks, metadatas, original_ids = process_files_to_chunks(files, file_processor, progress)

    if not chunks:
        return "未能从文件中提取任何文本块。", file_processor.get_file_list()

    # 3. 生成嵌入
    # --- FIX START: 检查progress对象是否存在 ---
    if progress:
        progress(0.8, desc="生成文本嵌入...")
    # --- FIX END ---
    embeddings_np = get_embedding(chunks)

    # 4. 构建FAISS索引
    # --- FIX START: 检查progress对象是否存在 ---
    if progress is not None:
        progress(0.9, desc="构建FAISS索引...")
    # --- FIX END ---
    faiss_manager.build_index(embeddings_np, chunks, metadatas, original_ids)

    # 5. 构建BM25索引
    docs_for_bm25, ids_for_bm25 = faiss_manager.get_all_docs_and_ids()
    bm25_manager.build_index(docs_for_bm25, ids_for_bm25)

    summary = f"成功处理 {len(files)} 个文件，生成 {len(chunks)} 个文本块。"
    return summary, file_processor.get_file_list()


def recursive_retrieval(initial_query, enable_web_search, model_choice):
    """递归检索，整合本地与网络信息。"""
    logging.info("--- 开始执行递归检索 ---")
    query = initial_query
    all_contexts, all_doc_ids, all_metadata = [], [], []

    for i in range(RECURSIVE_RETRIEVAL_MAX_ITERATIONS):
        logging.info(f"递归检索迭代 {i + 1}，查询: {query}")

        # 1. 语义检索 (FAISS)
        logging.info("步骤 1/5: 执行向量编码和FAISS搜索...")
        query_embedding_np=get_embedding([query])
        semantic_results = faiss_manager.search(query_embedding_np, top_k=RETRIEVER_TOP_K)
        logging.info("FAISS搜索完成。")

        # 2. 关键字检索 (BM25)
        logging.info("步骤 2/5: 执行BM25关键字搜索...")
        bm25_results = bm25_manager.search(query, top_k=RETRIEVER_TOP_K)
        logging.info("BM25搜索完成。")

        # 3. 混合与排序
        logging.info("步骤 3/5: 合并检索结果...")
        merged_results = hybrid_merge(semantic_results, bm25_results, faiss_manager.faiss_metadatas_map)

        # 提取用于重排序的材料
        docs_to_rerank = [res['content'] for _, res in merged_results]
        ids_to_rerank = [doc_id for doc_id, _ in merged_results]
        metas_to_rerank = [res['metadata'] for _, res in merged_results]
        logging.info("结果合并完成。")

        # 4. 重排序
        logging.info("步骤 4/5: 执行重排序...")
        reranked_results = rerank_documents(query, docs_to_rerank, ids_to_rerank, metas_to_rerank, top_k=RERANKER_TOP_K)
        logging.info("重排序完成。")

        # TODO: (改进方向) 上下文管理与压缩
        # 思路:
        # 在将 reranked_results 添加到 all_contexts 之前，检查总长度。
        # 1. 计算当前 `all_contexts` 的总token数。
        # 2. 如果加上新的 `reranked_results` 会超过LLM的上下文窗口限制（例如4096个token）。
        # 3. 触发压缩策略:
        #    - `summarize_context(long_context, query)`: 使用LLM将一个或多个chunks总结成更短的文本。
        #    - `filter_context(chunks, query)`: 使用LLM或更简单的启发式方法（如关键字密度）筛选出最不相关的chunks并丢弃。
        # 4. 将压缩或筛选后的上下文添加到`all_contexts`。

        current_iter_docs = []
        for doc_id, result_data in reranked_results:
            if doc_id not in all_doc_ids:  # 避免重复添加
                all_contexts.append(result_data['content'])
                all_doc_ids.append(doc_id)
                all_metadata.append(result_data['metadata'])
                current_iter_docs.append(result_data['content'])

        # 5. 网络搜索 (如果启用)
        web_texts = []
        if enable_web_search:
            try:
                web_results = serpapi_search(query)
                for res in web_results:
                    # 将网络结果也视为一种上下文
                    text = f"标题：{res.get('title', '')}\n摘要：{res.get('snippet', '')}"
                    web_texts.append(text)
                    # 为了简单起见，我们不将网络结果加入向量库，只作为当轮的临时上下文
            except Exception as e:
                logging.error(f"网络搜索错误: {e}")

        # 6. 判断是否需要继续
        if i == RECURSIVE_RETRIEVAL_MAX_ITERATIONS - 1:
            break

        context_for_next_query = "\n".join(web_texts + current_iter_docs)
        if not context_for_next_query.strip():
            break  # 如果本轮没有任何新信息，则停止

        new_query = generate_query_variations(initial_query, context_for_next_query[:1000], model_choice)
        if new_query:
            # generate_query_variations返回的是列表，取第一个作为新的查询
            query = new_query[0] if isinstance(new_query, list) and new_query else new_query
        else:
            break  # LLM认为不需要继续
    logging.info(f"--- 递归检索结束，共找到 {len(all_contexts)} 条上下文 ---")
    return all_contexts, all_doc_ids, all_metadata


def answer_question_stream(question, enable_web_search, model_choice):
    """处理问答请求并以流式方式返回答案。"""
    # 检查知识库状态
    if not faiss_manager.faiss_index or faiss_manager.faiss_index.ntotal == 0:
        if not enable_web_search:
            yield "⚠️ 知识库为空，请先上传文档。", "完成"
            return
        logging.warning("知识库为空，将仅使用网络搜索结果。")

    # 1. 递归检索获取上下文
    contexts, doc_ids, metadatas = recursive_retrieval(question, enable_web_search, model_choice)

    # 2. 构建Prompt
    context_str = "\n\n---\n\n".join(
        f"[来源: {meta.get('source', '未知')}]\n{ctx}" for ctx, meta in zip(contexts, metadatas)
    )
    if not context_str.strip():
        context_str = "未在本地文档和网络中找到相关信息。"

    prompt = f"""你是一个专业的问答助手。请基于以下提供的参考内容来回答用户的问题。
请遵循以下原则：
1. 仅根据参考内容回答，不要使用自己的知识。
2. 如果内容不足，请坦诚告知。
3. 回答要清晰、有条理。
4. 在回答的末尾，以列表形式明确标注出你参考的所有来源文档名称，例如：`来源: [doc1.pdf, doc2.pdf]`。

--- 参考内容 ---
{context_str}
---
用户问题: {question}

你的回答:
"""
    # TODO: (改进方向) 答案生成效果评估与追溯
    # 思路:
    # 1. 更精细的追溯（Citations）:
    #    - 在生成最终答案后，可以增加一个步骤：
    #    - `citation_prompt = f"Context: {context_str}\nAnswer: {final_answer}\n请为答案中的每一句话找到最相关的来源，并以 '答案 [来源: doc.pdf]' 的格式重写答案。"`
    #    - 这比在末尾列出所有来源更具可追溯性。
    # 2. 答案评估（Faithfulness）:
    #    - `eval_prompt = f"Context: {context_str}\nAnswer: {final_answer}\n请判断答案中的所有信息是否都完全由上下文支持。回答'是'或'否'。"`
    #    - 如果LLM回答“否”，可以标记这个答案为“可能包含外部知识”，提醒用户注意。

    # 3. 生成答案
    if model_choice == "siliconflow":
        full_answer = call_siliconflow_api(prompt)
        yield full_answer, "完成"
    else:
        full_answer = ""
        for chunk in call_ollama_api_stream(prompt):
            full_answer += chunk
            yield full_answer, "生成中..."
        yield full_answer, "完成"
