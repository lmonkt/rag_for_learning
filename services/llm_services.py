# rag_project/services/llm_services.py

import json
import re
import logging
from functools import lru_cache

from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_API_URL, GENERATOR_MODEL_SILICONFLOW,
    OLLAMA_API_URL, GENERATOR_MODEL_OLLAMA_LIGHT, HTTP_RETRIES,
    SILICONFLOW_TEMPERATURE, SILICONFLOW_MAX_TOKENS, SILICONFLOW_TIMEOUT,
    OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT
)

# 创建一个会话以复用连接
import requests
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=HTTP_RETRIES))


def call_siliconflow_api(prompt, temperature=None, max_tokens=None):
    """调用SiliconFlow云端LLM API。"""
    if not SILICONFLOW_API_KEY:
        return "错误：未配置 SiliconFlow API 密钥。"

    # 使用配置中的默认值，除非明确指定
    if temperature is None:
        temperature = SILICONFLOW_TEMPERATURE
    if max_tokens is None:
        max_tokens = SILICONFLOW_MAX_TOKENS

    payload = {
        "model": GENERATOR_MODEL_SILICONFLOW,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, "max_tokens": max_tokens, "temperature": temperature
    }
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}

    try:
        response = session.post(SILICONFLOW_API_URL, json=payload, headers=headers, timeout=SILICONFLOW_TIMEOUT)
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


def call_ollama_api(prompt, model, stream=False, timeout=None):
    """调用本地Ollama LLM API。"""
    if timeout is None:
        timeout = OLLAMA_TIMEOUT

    try:
        response = session.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "temperature": OLLAMA_TEMPERATURE
            },
            timeout=timeout,
            stream=stream
        )
        response.raise_for_status()

        if stream:
            # 流式处理
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            full_response += data['response']
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
        else:
            # 非流式处理
            result = response.json()
            return result.get("response", "")

    except Exception as e:
        logging.error(f"调用Ollama API时出错: {e}")
        if stream:
            yield f"调用API时出错: {e}"
        else:
            return f"调用API时出错: {e}"


def call_ollama_small_model(prompt):
    """调用小型Ollama模型进行快速推理。"""
    try:
        response = call_ollama_api(prompt, model=GENERATOR_MODEL_OLLAMA_LIGHT, stream=False, timeout=30)
        return response
    except Exception as e:
        logging.error(f"调用小型模型失败: {e}")
        return f"调用模型时出错: {e}"


@lru_cache(maxsize=32)
def remove_thinking_tags(text: str) -> str:
    """移除文本中的<think>...</think>标签。"""
    if not text:
        return text
    # 使用正则表达式替换所有<think>...</think>片段
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()
