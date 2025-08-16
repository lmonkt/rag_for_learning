# core/llm_interface.py
import logging
import json
import requests
from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_API_URL, GENERATOR_MODEL_SILICONFLOW,
    OLLAMA_API_URL, GENERATOR_MODEL_OLLAMA, GENERATOR_MODEL_OLLAMA_LIGHT,
    SILICONFLOW_TEMPERATURE, SILICONFLOW_MAX_TOKENS, SILICONFLOW_TIMEOUT,
    OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT, QUERY_GENERATION_TEMPERATURE,
    QUERY_GENERATION_MAX_TOKENS, QUERY_GENERATION_TIMEOUT, HTTP_RETRIES
)

# 创建一个会话以复用连接
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=HTTP_RETRIES))


def call_siliconflow_api(prompt, temperature=None, max_tokens=None):
    """调用SiliconFlow API。"""
    if not SILICONFLOW_API_KEY:
        raise ValueError("未设置 SILICONFLOW_API_KEY 环境变量。")

    # 使用配置中的默认值，除非明确指定
    if temperature is None:
        temperature = SILICONFLOW_TEMPERATURE
    if max_tokens is None:
        max_tokens = SILICONFLOW_MAX_TOKENS

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
            timeout=SILICONFLOW_TIMEOUT
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
        "stream": True,
        "temperature": OLLAMA_TEMPERATURE,
    }

    try:
        with session.post(OLLAMA_API_URL, json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as response:
            response.raise_for_status()
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
    except Exception as e:
        logging.error(f"调用Ollama API出错: {e}")
        yield f"调用API时出错: {e}"


def generate_query_variations(original_query: str, context_summary=None, model_choice=None) -> list:
    """生成查询变体以提高检索效果。

    Args:
        original_query (str): 原始查询
        context_summary: 上下文摘要（可选）
        model_choice: 模型选择（可选）
    """
    prompt = f"""
请为以下查询生成2-3个语义相似但表达不同的变体查询，用于提高信息检索的召回率：

原查询：{original_query}

要求：
1. 保持核心语义不变
2. 使用不同的词汇和表达方式
3. 每行一个变体查询
4. 不要添加额外解释
"""

    try:
        # 使用配置的查询生成参数
        if GENERATOR_MODEL_SILICONFLOW and SILICONFLOW_API_KEY:
            new_query = call_siliconflow_api(
                prompt,
                temperature=QUERY_GENERATION_TEMPERATURE,
                max_tokens=QUERY_GENERATION_MAX_TOKENS
            )
        else:
            response = session.post(
                OLLAMA_API_URL,
                json={
                    "model": GENERATOR_MODEL_OLLAMA_LIGHT,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": QUERY_GENERATION_TEMPERATURE,
                },
                timeout=QUERY_GENERATION_TIMEOUT
            )
            response.raise_for_status()
            new_query = response.json().get("response", "")

        # 解析变体查询
        variations = [line.strip() for line in new_query.split('\n') if line.strip() and not line.startswith('```')]
        return [original_query] + variations[:3]  # 包含原查询，最多4个查询

    except Exception as e:
        logging.error(f"生成查询变体失败: {e}")
        return [original_query]  # 失败时返回原查询
