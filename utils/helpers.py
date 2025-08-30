# utils/helpers.py
import socket
import requests
import re
import json
import logging
import logic

# 导入配置以访问API URL和模型信息
from config import (
    OLLAMA_CHECK_URL, OLLAMA_TAGS_URL, GENERATOR_MODEL_OLLAMA,
    EMBED_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, HYBRID_SEARCH_ALPHA,
    CROSS_ENCODER_MODEL_NAME, CONNECTION_TIMEOUT, REQUEST_TIMEOUT,
    APP_HOST, OLLAMA_EMBED_MODEL,
    # 新增：云端密钥用于放宽环境校验
    SILICONFLOW_API_KEY, DEEPSEEK_API_KEY, DASHSCOPE_API_KEY
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
    检查运行环境是否满足基本要求：
    - 优先检测本地 Ollama 服务与主生成模型；
    - 如果本地不可用，但检测到任一云端密钥（SiliconFlow/DeepSeek/阿里云），则允许以“仅云端模式”继续启动；
    - 仅当两者都不可用时才阻止启动。
    """
    cloud_keys_present = any([bool(SILICONFLOW_API_KEY), bool(DEEPSEEK_API_KEY), bool(DASHSCOPE_API_KEY)])

    try:
        # 1) 检查 Ollama 服务（可选）
        try:
            response = requests.get(
                OLLAMA_TAGS_URL.replace("/api/tags", ""),  # 访问根路径
                proxies={"http": None, "https": None},
                timeout=CONNECTION_TIMEOUT
            )
            ollama_online = (response.status_code == 200)
        except requests.exceptions.RequestException:
            ollama_online = False

        if ollama_online:
            print("✅ Ollama服务连接正常。")
            # 2) 检查所需本地生成模型是否存在（仅作提示，不再强制）
            try:
                model_check_response = requests.post(
                    OLLAMA_CHECK_URL,
                    json={"name": GENERATOR_MODEL_OLLAMA},
                    timeout=REQUEST_TIMEOUT,
                    proxies={"http": None, "https": None}
                )
                if model_check_response.status_code == 200:
                    print(f"✅ 本地生成模型已就绪: {GENERATOR_MODEL_OLLAMA}")
                else:
                    print(f"⚠️ 未找到本地生成模型: {GENERATOR_MODEL_OLLAMA}。如选择 'ollama' 可能失败，建议执行 'ollama pull {GENERATOR_MODEL_OLLAMA}'。")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ 检查本地生成模型时发生网络错误: {e}。如计划使用本地模型，请确认服务可用。")

            # 本地服务在线即视为环境可用
            return True
        else:
            print("⚠️ 未检测到 Ollama 服务在线。")
            if cloud_keys_present:
                print("🟨 检测到云端API密钥，将启用‘仅云端模式’，可在UI中选择 SiliconFlow / DeepSeek / 阿里云 进行生成。")
                return True
            else:
                print("❌ 未检测到任何云端API密钥，且本地Ollama不可用。请至少满足以下条件之一：\n"
                      "  1) 启动Ollama并下载所需本地模型；\n"
                      "  2) 在环境变量中配置任一云端密钥（SILICONFLOW_API_KEY / DEEPSEEK_API_KEY / DASHSCOPE_API_KEY）。")
                return False

    except Exception as e:
        print(f"❌ 环境检查发生异常: {e}")
        # 若存在云端密钥，仍可放行（容错）
        if cloud_keys_present:
            print("🟨 虽然环境检查异常，但检测到云端密钥，将尝试以‘仅云端模式’继续启动。")
            return True
        return False


def is_embedding_model_available(model_name: str = None) -> bool:
    """
    检查指定的嵌入模型是否已通过 Ollama 下载并可用。
    借鉴了 check_environment 函数的风格，使用 /api/show API。

    Args:
        model_name (str): 要检查的嵌入模型名称，如果不指定则使用配置中的默认嵌入模型。

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
        "嵌入模型": OLLAMA_EMBED_MODEL,
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

    # 替换所有的<think>标签
    processed_text = re.sub(
        r'<think>(.*?)</think>',
        replace_think_tag,
        text,
        flags=re.DOTALL
    )

    return processed_text


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


def get_system_runtime_info() -> dict:
    """
    获取系统运行时信息，包括服务状态、模型可用性等
    """
    info = {
        "Ollama服务状态": "🔴 离线",
        "嵌入模型状态": "🔴 不可用",
        "生成模型状态": "🔴 不可用",
        "系统环境": "未检查"
    }

    try:
        # 检查Ollama服务
        response = requests.get(
            OLLAMA_TAGS_URL.replace("/api/tags", ""),
            proxies={"http": None, "https": None},
            timeout=CONNECTION_TIMEOUT
        )
        if response.status_code == 200:
            info["Ollama服务状态"] = "🟢 在线"

            # 检查嵌入模型
            if is_embedding_model_available(OLLAMA_EMBED_MODEL):
                info["嵌入模型状态"] = "🟢 可用"
            else:
                info["嵌入模型状态"] = "🔴 不可用"

            # 检查生成模型
            try:
                model_response = requests.post(
                    OLLAMA_CHECK_URL,
                    json={"name": GENERATOR_MODEL_OLLAMA},
                    timeout=REQUEST_TIMEOUT,
                    proxies={"http": None, "https": None}
                )
                if model_response.status_code == 200:
                    info["生成模型状态"] = "🟢 可用"
                else:
                    info["生成模型状态"] = "🔴 不可用"
            except Exception:
                info["生成模型状态"] = "🔴 不可用"

            info["系统环境"] = "🟢 正常"
        else:
            info["系统环境"] = "🔴 Ollama服务异常"

    except Exception as e:
        info["系统环境"] = f"🔴 连接失败: {str(e)[:50]}..."

    return info


def get_system_statistics() -> dict:
    """
    获取系统统计信息
    """
    try:
        faiss_manager = logic.faiss_manager

        doc_count = len(set(meta.get('source', '') for meta in faiss_manager.faiss_metadatas_map.values()))
        chunk_count = len(faiss_manager.faiss_id_order_for_index)

        total_chars = sum(len(content) for content in faiss_manager.faiss_contents_map.values())

        return {
            "文档数量": f"{doc_count} 个",
            "文本块数量": f"{chunk_count} 个",
            "总字符数": f"{total_chars:,} 字符",
            "平均块大小": f"{total_chars // max(chunk_count, 1):,} 字符/块" if chunk_count > 0 else "0 字符/块"
        }
    except Exception as e:
        return {
            "文档数量": "获取失败",
            "文本块数量": "获取失败",
            "总字符数": "获取失败",
            "平均块大小": "获取失败"
        }
