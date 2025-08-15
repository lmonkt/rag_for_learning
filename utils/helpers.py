# utils/helpers.py
import socket
import requests
import re
import json
import logging

# 导入配置以访问API URL和模型信息
from config import (
    OLLAMA_CHECK_URL, OLLAMA_TAGS_URL, GENERATOR_MODEL_OLLAMA,
    EMBED_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, HYBRID_SEARCH_ALPHA,
    CROSS_ENCODER_MODEL_NAME, CONNECTION_TIMEOUT, REQUEST_TIMEOUT,
    APP_HOST
)


def is_port_available(port: int) -> bool:
    """
    检查指定的本地端口是否可用。

    返回:
        bool: 如果端口可用，返回 True，否则返回 False。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)  # 设置一个短暂的超时
        # connect_ex 返回错误指示器，0表示成功连接（即端口被占用）
        return s.connect_ex((APP_HOST, port)) != 0


def check_environment():
    """
    检查运行环境是否满足基本要求，主要是Ollama服务及其模型。
    """
    try:
        # 1. 检查Ollama服务是否在运行
        response = requests.get(
            OLLAMA_TAGS_URL.replace("/api/tags", ""),  # 访问根路径
            proxies={"http": None, "https": None},
            timeout=CONNECTION_TIMEOUT
        )
        if response.status_code != 200:
            print(f"❌ Ollama服务连接异常，返回状态码: {response.status_code}")
            print("请确保Ollama服务正在运行。")
            return False
        print("✅ Ollama服务连接正常。")

        # 2. 检查所需模型是否存在
        model_check_response = requests.post(
            OLLAMA_CHECK_URL,
            json={"name": GENERATOR_MODEL_OLLAMA},
            timeout=REQUEST_TIMEOUT
        )
        if model_check_response.status_code != 200:
            print(f"❌ 未找到所需模型: {GENERATOR_MODEL_OLLAMA}")
            print(f"请先执行 'ollama pull {GENERATOR_MODEL_OLLAMA}' 来下载模型。")
            return False
        print(f"✅ 所需模型 '{GENERATOR_MODEL_OLLAMA}' 已存在。")

        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ 连接Ollama服务失败: {e}")
        print(f"请确认Ollama服务已在 {OLLAMA_CHECK_URL.replace('/api/show', '')} 启动。")
        return False


def is_embedding_model_available(model_name: str = None) -> bool:
    """
    检查指定的嵌入模型是否已通过 Ollama 下载并可用。
    借鉴了 check_environment 函数的风格，使用 /api/show API。

    Args:
        model_name (str): 要检查的嵌入模型名称，如果不指定则使用配置中的默认模型。

    Returns:
        bool: 如果模型存在则返回 True，否则返回 False。
    """
    # 如果没有指定模型名称，使用配置中的默认嵌入模型
    if model_name is None:
        from config import OLLAMA_EMBED_MODEL
        model_name = OLLAMA_EMBED_MODEL

    try:
        response = requests.post(
            OLLAMA_CHECK_URL,
            json={"name": model_name},
            timeout=REQUEST_TIMEOUT,
            proxies={"http": None, "https": None}
        )
        return response.status_code == 200

    except requests.exceptions.RequestException as e:
        print(f"⚠️  检查模型 {model_name} 时发生网络错误: {e}，考虑下载或确定ollama服务是否正常运行。")
        return False


def get_system_models_info() -> dict:
    """
    返回一个包含系统所用模型和技术信息的字典，用于在UI中展示。
    从配置中动态获取信息，确保与实际配置保持一致。
    """
    return {
        "嵌入模型": EMBED_MODEL_NAME,
        "分块方法": f"RecursiveCharacterTextSplitter (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})",
        "检索方法": f"向量检索 + BM25混合检索 (α={HYBRID_SEARCH_ALPHA})",
        "重排序模型": CROSS_ENCODER_MODEL_NAME,
        "生成模型": f"{GENERATOR_MODEL_OLLAMA}",
        "分词工具": "jieba (中文分词)"
    }


def process_thinking_content(text: str) -> str:
    """
    处理包含<think>标签的文本，将其转换为Markdown的可折叠详情框。
    这使得LLM的思考过程可以被整洁地展示在UI上。
    """
    if not isinstance(text, str):
        return "内容格式无法处理"

    # 使用正则表达式替换所有<think>...</think>片段
    def replace_think_tag(match):
        thinking_content = match.group(1).strip()
        return (
            f"\n\n<details>\n"
            f"<summary>🧠 思考过程（点击展开）</summary>\n\n"
            f"```\n{thinking_content}\n```\n\n"
            f"</details>\n\n"
        )

    # 匹配<think>...</think>标签（支持多行）
    pattern = r'<think>(.*?)</think>'
    result = re.sub(pattern, replace_think_tag, text, flags=re.DOTALL)

    return result.strip()


# --- 以下是与答案质量分析相关的辅助函数 ---

def extract_facts(text: str) -> dict:
    """从文本中提取关键事实的简单示例。"""
    facts = {}
    numbers = re.findall(r'\b\d{4}年|\b\d+%', text)
    if numbers:
        facts['关键数值'] = numbers
    return facts


def detect_conflicts(sources: list) -> bool:
    """基于提取的事实，检测来源之间是否存在矛盾。"""
    key_facts = {}
    for item in sources:
        text = item.get('text', '')
        facts = extract_facts(text)
        for fact, value in facts.items():
            if fact in key_facts and key_facts[fact] != value:
                logging.warning(f"检测到矛盾: 事实'{fact}'的值不同。旧值: {key_facts[fact]}, 新值: {value}")
                return True
            else:
                key_facts[fact] = value
    return False


def evaluate_source_credibility(source):
    """评估来源可信度"""
    credibility_scores = {
        "gov.cn": 0.9,
        "edu.cn": 0.85,
        "weixin": 0.7,
        "zhihu": 0.6,
        "baidu": 0.5
    }

    url = source.get('url', '')
    if not url:
        return 0.5  # 默认中等可信度

    domain_match = re.search(r'//([^/]+)', url)
    if not domain_match:
        return 0.5

    domain = domain_match.group(1)

    # 检查是否匹配任何已知域名
    for known_domain, score in credibility_scores.items():
        if known_domain in domain:
            return score

    return 0.5  # 默认中等可信度