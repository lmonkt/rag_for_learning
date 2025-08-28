# logic.py
import math
import logging
import requests
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
import gradio as gr
# 导入核心组件
from core.document_processor import process_files_to_chunks, FileProcessor
from core.retriever import FaissIndexManager, BM25IndexManager, hybrid_merge
from core.reranker import rerank_documents
from core.llm_interface import call_ollama_api_stream, call_siliconflow_api, call_siliconflow_api_stream, generate_query_variations
from core.web_search import serpapi_search
import core.reranker as reranker  # 导入模块以访问get_cross_encoder

# 导入配置
from config import (
    EMBED_MODEL_NAME, RETRIEVER_TOP_K, RERANKER_TOP_K,
    RECURSIVE_RETRIEVAL_MAX_ITERATIONS, RERANK_METHOD, OLLAMA_EMBED_MODEL,
    CONTEXT_MAX_TOKENS, TOKENS_PER_CHAR, SILICONFLOW_API_KEY, GENERATOR_MODEL_OLLAMA_LIGHT, OLLAMA_API_URL,
    OLLAMA_TIMEOUT,
)
from utils.helpers import is_embedding_model_available
from core.llm_interface import call_siliconflow_api  # 复用现有 SF 客户端

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
                    logging.debug(
                        f"ollama_get_embedding接收到的输入类型: {type(texts)}, "
                        f"元素数量: {len(texts)}, "
                        f"元素类型: {[type(t).__name__ for t in texts[:2]]}{'...' if len(texts) > 2 else ''}"
                    )

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

def _estimate_tokens(text: str) -> int:
    """基于字符数的粗略 token 估算。"""
    if not text:
        return 0
    return int(math.ceil(len(text) * TOKENS_PER_CHAR))

def _total_tokens(texts: list[str]) -> int:
    return sum(_estimate_tokens(t) for t in texts)

def _ollama_complete(prompt: str, timeout: int | None = None) -> str:
    """非流式本地总结调用（轻量模型），作为 SF 的后备。"""
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={
                "model": GENERATOR_MODEL_OLLAMA_LIGHT,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2
            },
            timeout=timeout or OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except Exception as e:
        logging.error(f"Ollama 总结失败: {e}")
        return ""

def _summarize_context(text: str, query: str, target_tokens: int) -> str:
    """将单个 chunk 压缩到目标 token 限额内（尽量保留关键事实）。"""
    # 控制输入长度，避免提示过长
    max_chars = max(512, int(target_tokens / max(TOKENS_PER_CHAR, 0.1) * 2))
    if len(text) > max_chars:
        text = text[:max_chars]

    prompt = (
        f"请用中文在不丢失关键信息的前提下，将下述与查询相关的内容压缩成精炼摘要，"
        f"限制在约{target_tokens}个 token 内，保留数字、时间、实体、结论：\n\n"
        f"查询：{query}\n\n"
        f"内容：\n{text}\n\n"
        f"摘要："
    )

    if SILICONFLOW_API_KEY:
        summary = call_siliconflow_api(prompt, temperature=0.2, max_tokens=max(128, min(1024, target_tokens)))
    else:
        summary = _ollama_complete(prompt)

    return (summary or "").strip()

def _fit_into_budget(
    reranked_results: list[tuple[str, dict]],
    all_contexts: list[str],
    all_doc_ids: list[str],
    all_metadata: list[dict],
    query: str
) -> list[str]:
    """
    将当前重排结果装入上下文预算：
    - 贪心加入最高分内容
    - 若遇到超限，对第一个超限内容做摘要以适配剩余预算
    - 返回本轮加入的文本列表（供后续上下文统计或日志）
    """
    current_tokens = _total_tokens(all_contexts)
    budget = CONTEXT_MAX_TOKENS
    joined_this_round: list[str] = []

    for doc_id, result in reranked_results:
        if doc_id in all_doc_ids:
            continue

        content = result.get("content", "")
        meta = result.get("metadata", {}) or {}
        need = _estimate_tokens(content)

        if current_tokens + need <= budget:
            all_contexts.append(content)
            all_doc_ids.append(doc_id)
            all_metadata.append(meta)
            joined_this_round.append(content)
            current_tokens += need
            continue

        # 尝试在剩余预算内压缩后加入
        remaining = max(0, budget - current_tokens)
        # 阈值：剩余不足 10% 预算则直接停止，避免无意义过短摘要
        if remaining >= int(0.1 * budget):
            summary = _summarize_context(content, query, target_tokens=remaining)
            if summary:
                all_contexts.append(summary)
                all_doc_ids.append(doc_id)
                meta = {**meta, "compressed": True}
                all_metadata.append(meta)
                joined_this_round.append(summary)
                current_tokens += _estimate_tokens(summary)
        # 无论摘要是否成功，预算已近极限，结束本轮装配
        break

    return joined_this_round

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

        current_iter_docs = _fit_into_budget(
            reranked_results=reranked_results,
            all_contexts=all_contexts,
            all_doc_ids=all_doc_ids,
            all_metadata=all_metadata,
            query=query  # 确保此处能拿到当前用户查询
        )

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
    yield "🔍 正在搜索相关文档...", "检索中"
    contexts, doc_ids, metadatas = recursive_retrieval(question, enable_web_search, model_choice)

    # 2. 构建Prompt
    yield "📝 正在构建提示词...", "准备中"
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

    # 3. 生成答案（真正的流式输出）
    yield "🤖 正在生成回答...", "生成中"
    raw_answer = ""

    if model_choice == "siliconflow":
        # 对于SiliconFlow，我们需要实现真正的流式输出
        try:
            # 首先尝试流式调用，如果不支持则回退到非流式
            response = call_siliconflow_api_stream(prompt)
            if response:  # 如果支持流式
                for chunk in response:
                    raw_answer += chunk
                    yield raw_answer, "生成中"
            else:  # 回退到非流式
                raw_answer = call_siliconflow_api(prompt)
                # 模拟流式输出，分段显示
                words = raw_answer.split()
                current_text = ""
                for i, word in enumerate(words):
                    current_text += word + " "
                    if i % 3 == 0 or i == len(words) - 1:  # 每3个词更新一次
                        yield current_text.strip(), "生成中"
        except Exception as e:
            logging.error(f"SiliconFlow生成失败: {e}")
            raw_answer = f"抱歉，生成回答时出现错误：{str(e)}"
            yield raw_answer, "错误"
            return
    else:
        # Ollama流式输出
        try:
            for chunk in call_ollama_api_stream(prompt):
                raw_answer += chunk
                yield raw_answer, "生成中"
        except Exception as e:
            logging.error(f"Ollama生成失败: {e}")
            raw_answer = f"抱歉，生成回答时出现错误：{str(e)}"
            yield raw_answer, "错误"
            return

    # 4. 简化后续处理，避免阻塞流式输出
    try:
        # 直接添加来源信息，不进行复杂的评估和重写
        sources = []
        for meta in metadatas:
            source_name = meta.get('source', '未知来源')
            if source_name not in sources:
                sources.append(source_name)

        if sources:
            sources_text = f"\n\n📚 **参考来源**: {', '.join(sources)}"
            final_answer = raw_answer + sources_text
        else:
            final_answer = raw_answer

        yield final_answer, "完成"

    except Exception as e:
        logging.error(f"后处理过程出错: {e}")
        # 如果后处理失败，返回原始答案
        yield raw_answer, "完成"


def _evaluate_answer_faithfulness(answer, context, model_choice):
    """评估答案的忠实性（是否完全基于上下文）"""
    eval_prompt = f"""请评估以下答案是否完全基于提供的上下文内容。

上下文内容:
{context}

答案:
{answer}

请按以下格式回答：
1. 忠实性评分（0-10分，10分表示完全忠实）：
2. 是否忠实（是/否）：
3. 简要说明：

格式示例：
评分：8
忠实：是
说明：答案大部分内容都有上下文支持，仅有少量合理推理。
"""

    try:
        if model_choice == "siliconflow":
            eval_result = call_siliconflow_api(eval_prompt)
        else:
            eval_result = ""
            for chunk in call_ollama_api_stream(eval_prompt):
                eval_result += chunk

        # 解析评估结果
        lines = eval_result.strip().split('\n')
        score = 7.0  # 默认分数
        is_faithful = True  # 默认忠实

        for line in lines:
            if '评分' in line or '分数' in line:
                try:
                    score = float([s for s in line.split() if s.replace('.', '').isdigit()][0])
                except:
                    pass
            if '忠实' in line:
                is_faithful = '是' in line or 'True' in line.upper()

        # 如果评分低于6分，认为不够忠实
        if score < 6:
            is_faithful = False

        return score, is_faithful

    except Exception as e:
        logging.error(f"忠实性评估失败: {e}")
        return 7.0, True  # 默认认为忠实


def _add_detailed_citations(answer, context, metadatas, model_choice):
    """为答案添加详细的引用标注"""
    # 构建来源映射
    sources = {}
    for i, meta in enumerate(metadatas):
        source_name = meta.get('source', f'来源{i+1}')
        sources[f'来源{i+1}'] = source_name

    source_list = '\n'.join([f"{key}: {value}" for key, value in sources.items()])

    citation_prompt = f"""请为以下答案中的每个重要陈述添加具体的来源标注。

可用来源：
{source_list}

上下文内容：
{context}

原始答案：
{answer}

请重写答案，在每个重要陈述后面添加引用标注，格式为 [来源X]。
注意：
1. 只在确实有对应来源支持的陈述后添加引用
2. 保持答案的自然流畅性
3. 最后仍需保留完整的来源列表

重写后的答案：
"""

    try:
        if model_choice == "siliconflow":
            enhanced_answer = call_siliconflow_api(citation_prompt)
        else:
            enhanced_answer = ""
            for chunk in call_ollama_api_stream(citation_prompt):
                enhanced_answer += chunk

        # 如果增强失败，返回原答案
        if not enhanced_answer.strip():
            return answer

        return enhanced_answer

    except Exception as e:
        logging.error(f"添加引用标注失败: {e}")
        return answer
