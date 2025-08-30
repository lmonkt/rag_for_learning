# core/llm_interface.py
import logging
import json
import requests
from requests.adapters import HTTPAdapter
from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_API_URL, GENERATOR_MODEL_SILICONFLOW,
    OLLAMA_API_URL, GENERATOR_MODEL_OLLAMA, GENERATOR_MODEL_OLLAMA_LIGHT,
    SILICONFLOW_TEMPERATURE, SILICONFLOW_MAX_TOKENS, SILICONFLOW_TIMEOUT,
    OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT, QUERY_GENERATION_TEMPERATURE,
    QUERY_GENERATION_MAX_TOKENS, QUERY_GENERATION_TIMEOUT, HTTP_RETRIES,
    # 新增：DeepSeek & 阿里云
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL, GENERATOR_MODEL_DEEPSEEK,
    DEEPSEEK_TEMPERATURE, DEEPSEEK_MAX_TOKENS, DEEPSEEK_TIMEOUT,
    DASHSCOPE_API_KEY, ALIYUN_API_URL, GENERATOR_MODEL_ALIYUN,
    ALIYUN_TEMPERATURE, ALIYUN_MAX_TOKENS, ALIYUN_TIMEOUT
)

# 创建一个会话以复用连接
session = requests.Session()
session.mount('http://', HTTPAdapter(max_retries=HTTP_RETRIES))
session.mount('https://', HTTPAdapter(max_retries=HTTP_RETRIES))


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


def call_siliconflow_api_stream(prompt, temperature=None, max_tokens=None):
    """以流式方式调用SiliconFlow API。"""
    if not SILICONFLOW_API_KEY:
        return None  # 返回None表示不支持流式

    # 使用配置中的默认值，除非明确指定
    if temperature is None:
        temperature = SILICONFLOW_TEMPERATURE
    if max_tokens is None:
        max_tokens = SILICONFLOW_MAX_TOKENS

    payload = {
        "model": GENERATOR_MODEL_SILICONFLOW,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,  # 启用流式输出
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
            timeout=SILICONFLOW_TIMEOUT,
            stream=True
        )
        response.raise_for_status()

        # 处理流式响应
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    line_str = line_str[6:]  # 移除'data: '前缀

                if line_str.strip() == '[DONE]':
                    break

                try:
                    data = json.loads(line_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        logging.error(f"SiliconFlow流式调用失败: {e}")
        return None  # 返回None表示流式调用失败


# ----------------- 新增：DeepSeek 官方 API -----------------

def call_deepseek_api(prompt, temperature=None, max_tokens=None):
    """调用 DeepSeek Chat Completions（非流）。"""
    if not DEEPSEEK_API_KEY:
        return "错误：未配置 DeepSeek API 密钥。"

    if temperature is None:
        temperature = DEEPSEEK_TEMPERATURE
    if max_tokens is None:
        max_tokens = DEEPSEEK_MAX_TOKENS

    payload = {
        "model": GENERATOR_MODEL_DEEPSEEK or "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }

    try:
        resp = session.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=DEEPSEEK_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        msg = result.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        return f"{content}<think>{reasoning}</think>" if reasoning else content
    except Exception as e:
        logging.error(f"调用DeepSeek API失败: {e}")
        return f"调用API时出错: {e}"


def call_deepseek_api_stream(prompt, temperature=None, max_tokens=None):
    """流式调用 DeepSeek Chat Completions，返回逐段内容生成器；若不可用返回None。"""
    if not DEEPSEEK_API_KEY:
        return None

    if temperature is None:
        temperature = DEEPSEEK_TEMPERATURE
    if max_tokens is None:
        max_tokens = DEEPSEEK_MAX_TOKENS

    payload = {
        "model": GENERATOR_MODEL_DEEPSEEK or "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }

    try:
        with session.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=DEEPSEEK_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                if line.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'choices' in data and data['choices']:
                    delta = data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
    except Exception as e:
        logging.error(f"DeepSeek 流式调用失败: {e}")
        return None


# ----------------- 新增：阿里云 DashScope 兼容 API -----------------

def call_aliyun_api(prompt, temperature=None, max_tokens=None):
    """调用阿里云 OpenAI 兼容 Chat Completions（非流）。"""
    if not DASHSCOPE_API_KEY:
        return "错误：未配置 DASHSCOPE_API_KEY。"

    if temperature is None:
        temperature = ALIYUN_TEMPERATURE
    if max_tokens is None:
        max_tokens = ALIYUN_MAX_TOKENS

    payload = {
        "model": GENERATOR_MODEL_ALIYUN or "qwen-plus",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }

    try:
        resp = session.post(ALIYUN_API_URL, json=payload, headers=headers, timeout=ALIYUN_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        msg = result.get("choices", [{}])[0].get("message", {})
        return msg.get("content", "")
    except Exception as e:
        logging.error(f"调用阿里云 API 失败: {e}")
        return f"调用API时出错: {e}"


def call_aliyun_api_stream(prompt, temperature=None, max_tokens=None):
    """流式调用阿里云 OpenAI 兼容 Chat Completions，返回逐段内容生成器；若不可用返回None。"""
    if not DASHSCOPE_API_KEY:
        return None

    if temperature is None:
        temperature = ALIYUN_TEMPERATURE
    if max_tokens is None:
        max_tokens = ALIYUN_MAX_TOKENS

    payload = {
        "model": GENERATOR_MODEL_ALIYUN or "qwen-plus",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # 可选："stream_options": {"include_usage": True}
    }
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }

    try:
        with session.post(ALIYUN_API_URL, json=payload, headers=headers, timeout=ALIYUN_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                if line.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'choices' in data and data['choices']:
                    delta = data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
    except Exception as e:
        logging.error(f"阿里云流式调用失败: {e}")
        return None


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
        # 使用配置的查询生成参数；优先使用当前选择的云端模型
        new_query = None
        if model_choice == "deepseek" and DEEPSEEK_API_KEY:
            new_query = call_deepseek_api(
                prompt,
                temperature=QUERY_GENERATION_TEMPERATURE,
                max_tokens=QUERY_GENERATION_MAX_TOKENS
            )
        elif model_choice == "aliyun" and DASHSCOPE_API_KEY:
            # Qwen3 部分模型仅支持流式，使用流式并拼接
            text = ""
            stream_gen = call_aliyun_api_stream(
                prompt,
                temperature=QUERY_GENERATION_TEMPERATURE,
                max_tokens=QUERY_GENERATION_MAX_TOKENS
            )
            if stream_gen:
                for chunk in stream_gen:
                    text += chunk or ""
                new_query = text
            else:
                # 回退（若不可用）
                new_query = call_aliyun_api(
                    prompt,
                    temperature=QUERY_GENERATION_TEMPERATURE,
                    max_tokens=QUERY_GENERATION_MAX_TOKENS
                )
        elif GENERATOR_MODEL_SILICONFLOW and SILICONFLOW_API_KEY:
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
        variations = [line.strip() for line in (new_query or "").split('\n') if line.strip() and not line.startswith('```')]
        return [original_query] + variations[:3]  # 包含原查询，最多4个查询

    except Exception as e:
        logging.error(f"生成查询变体失败: {e}")
        return [original_query]  # 失败时返回原查询
