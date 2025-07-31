# core/llm_interface.py
import logging
import json
import requests
from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_API_URL, GENERATOR_MODEL_SILICONFLOW,
    OLLAMA_API_URL, GENERATOR_MODEL_OLLAMA, GENERATOR_MODEL_OLLAMA_LIGHT
)

# 创建一个会话以复用连接
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))


def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536):
    """调用SiliconFlow API。"""
    if not SILICONFLOW_API_KEY:
        raise ValueError("未设置 SILICONFLOW_API_KEY 环境变量。")

    payload = {
        "model": GENERATOR_MODEL_SILICONFLOW,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }

    try:
        response = session.post(
            SILICONFLOW_API_URL,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"].get("content", "")
        reasoning = result["choices"][0]["message"].get("reasoning_content", "")
        return f"{content}<think>{reasoning}</think>" if reasoning else content
    except Exception as e:
        logging.error(f"调用SiliconFlow API出错: {e}")
        return f"调用API时出错: {e}"


def call_ollama_api_stream(prompt):
    """以流式方式调用本地Ollama API。"""
    payload = {
        "model": GENERATOR_MODEL_OLLAMA,
        "prompt": prompt,
        "stream": True
    }
    try:
        with session.post(OLLAMA_API_URL, json=payload, stream=True, timeout=180) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8')).get("response", "")
                    yield chunk
    except Exception as e:
        logging.error(f"调用Ollama流式API出错: {e}")
        yield f"调用Ollama模型时出错: {e}"


def generate_new_query(initial_query, context_summary, model_choice):
    """使用LLM分析并生成新的查询。"""
    # TODO: (改进方向) 查询改写与意图识别
    # 思路: 这是实现查询扩展的核心位置。
    # 1. HyDE (Hypothetical Document Embeddings):
    #    - 在检索前，让LLM基于`initial_query`生成一个假设性的答案。
    #    - `hyde_prompt = f"请为以下问题写一个简短的、假设性的答案: {initial_query}"`
    #    - 然后用这个假设性答案的向量去检索，这比用问题的向量效果可能更好。
    # 2. Step-Back Prompting:
    #    - `step_back_prompt = f"你是一个善于思考的助手。对于用户的问题 '{initial_query}'，它的核心、更一般性的概念是什么？"`
    #    - 用LLM生成一个更宽泛的问题，然后同时检索原始问题和宽泛问题，合并结果。
    # 3. 意图识别:
    #    - `intent_prompt = f"用户问题是'{initial_query}'。它的意图是'事实查询'，'比较'，还是'摘要'？"`
    #    - 根据识别的意图，可以调整后续的RAG流程（例如，'摘要'意图可能需要更大的top_k）。
    prompt = f"""基于原始问题: {initial_query} 和已检索信息: \n{context_summary}\n
分析是否需要用不同角度或更具体的关键词进行追问。如果需要，请直接提供新的查询问题。如果信息已充分，请回复'不需要进一步查询'。
新查询:"""

    try:
        if model_choice == "siliconflow":
            new_query = call_siliconflow_api(prompt, temperature=0.5, max_tokens=100)
        else:
            response = session.post(
                OLLAMA_API_URL,
                json={"model": GENERATOR_MODEL_OLLAMA_LIGHT, "prompt": prompt, "stream": False},
                timeout=30
            )
            new_query = response.json().get("response", "").strip()

        if "不需要" in new_query or len(new_query) < 5:
            return None
        return new_query

    except Exception as e:
        logging.error(f"生成新查询时出错: {e}")
        return None