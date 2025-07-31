# rag_project/services/llm_services.py

import json
import re
import logging
from functools import lru_cache

from config import (
    session, SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_MODEL,
    OLLAMA_API_URL, OLLAMA_MODEL_SMALL
)


def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536):
    """调用SiliconFlow云端LLM API。"""
    if not SILICONFLOW_API_KEY:
        return "错误：未配置 SiliconFlow API 密钥。"

    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, "max_tokens": max_tokens, "temperature": temperature
    }
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}

    try:
        response = session.post(SILICONFLOW_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()

        content = result["choices"][0]["message"].get("content", "")
        # SiliconFlow API可能返回带思维链的内容，这里统一处理
        if reasoning := result["choices"][0]["message"].get("reasoning_content", ""):
            return f"{content}<think>{reasoning}</think>"
        return content

    except Exception as e:
        logging.error(f"调用SiliconFlow API时出错: {e}")
        return f"调用API时出错: {e}"


def call_ollama_api(prompt, model, stream=False, timeout=120):
    """调用本地Ollama LLM API。"""
    try:
        response = session.post(
            OLLAMA_API_URL,
            json={"model": model, "prompt": prompt, "stream": stream},
            timeout=timeout,
            stream=stream
        )
        response.raise_for_status()
        return response
    except Exception as e:
        logging.error(f"调用Ollama API时出错: {e}")
        raise


@lru_cache(maxsize=32)  # 使用缓存避免对相同(查询,文档)对重复调用LLM
def get_llm_relevance_score(query, doc):
    """
    使用一个小的本地LLM来为(查询, 文档)对的相关性打分。
    这是一个将LLM用于重排序的例子。
    """
    prompt = f"""评估以下查询和文档片段的相关性，仅返回0-10的整数分数。
    查询: {query}
    文档片段: {doc}
    相关性分数(0-10):"""

    try:
        response = call_ollama_api(prompt, model=OLLAMA_MODEL_SMALL, stream=False, timeout=30)
        result_text = response.json().get("response", "").strip()

        # 从LLM的回答中提取数字分数
        if match := re.search(r'\b([0-9]|10)\b', result_text):
            return float(match.group(1))
        return 5.0  # 如果无法解析，返回一个中性分数
    except Exception as e:
        logging.error(f"LLM评分失败: {e}")
        return 5.0  # 出错时返回中性分数