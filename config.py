# config.py
import os
import yaml
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 上下文预算（可被环境变量覆盖）
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "8192"))
# 简易 token 估算：中文建议 1.0~1.2；英文 0.6~1.0。取偏保守值以防超限。
TOKENS_PER_CHAR = float(os.getenv("TOKENS_PER_CHAR", "0.8"))

# 读取 YAML 配置
with open("config_models.yaml", "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

# --- Model Configurations ---
EMBED_MODEL_NAME = _config["models"]["embedding"]["sentence_transformers"]
CROSS_ENCODER_MODEL_NAME = _config["models"]["reranker"]
GENERATOR_MODEL_OLLAMA = _config["models"]["generator"]["main"]
GENERATOR_MODEL_OLLAMA_LIGHT = _config["models"]["generator"]["light"]
GENERATOR_MODEL_SILICONFLOW = _config["models"]["siliconflow"]
# 新增：DeepSeek 与 阿里云模型名
GENERATOR_MODEL_DEEPSEEK = _config["models"].get("deepseek")
GENERATOR_MODEL_ALIYUN = _config["models"].get("aliyun")

OLLAMA_EMBED_MODEL = _config["models"]["embedding"]["ollama"]

# --- RAG Pipeline Parameters ---
CHUNK_SIZE = _config["chunking"]["chunk_size"]
CHUNK_OVERLAP = _config["chunking"]["chunk_overlap"]
# 混合检索中语义检索的权重 (alpha)
HYBRID_SEARCH_ALPHA = _config["retrieval"]["hybrid_search_alpha"]
# 检索和重排序返回的 top_k 数量
RETRIEVER_TOP_K = _config["retrieval"]["retriever_top_k"]
RERANKER_TOP_K = _config["retrieval"]["reranker_top_k"]
RECURSIVE_RETRIEVAL_MAX_ITERATIONS = _config["retrieval"]["max_iterations"]

# --- System & Environment ---
# 禁用 oneDNN 优化 (避免某些CPU上的警告)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 设置不走代理的地址
os.environ['NO_PROXY'] = _config["system"]["no_proxy"]
# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = _config["system"]["hf_endpoint"]

# --- Search Engine ---
SEARCH_ENGINE = _config["search"]["engine"]

# --- Rerank Method ---
RERANK_METHOD = os.getenv("RERANK_METHOD", _config["rerank_method"])

# --- API Keys & Endpoints ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# SiliconFlow
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", _config["endpoints"]["siliconflow"])

# DeepSeek（OpenAI兼容）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", _config["endpoints"].get("deepseek", "https://api.deepseek.com/v1/chat/completions"))

# 阿里云 DashScope 兼容
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
ALIYUN_API_URL = os.getenv("ALIYUN_API_URL", _config["endpoints"].get("aliyun", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"))

# 本地 Ollama
OLLAMA_API_URL = _config["endpoints"]["ollama_generate"]
OLLAMA_CHECK_URL = _config["endpoints"]["ollama_show"]
OLLAMA_TAGS_URL = _config["endpoints"]["ollama_tags"]
SERPAPI_URL = _config["endpoints"]["serpapi"]

# --- API 调用配置 ---
SILICONFLOW_TEMPERATURE = _config["api"]["siliconflow"]["temperature"]
SILICONFLOW_MAX_TOKENS = _config["api"]["siliconflow"]["max_tokens"]
SILICONFLOW_TIMEOUT = _config["api"]["siliconflow"]["timeout"]
SILICONFLOW_STREAM = _config["api"]["siliconflow"]["stream"]

# DeepSeek
DEEPSEEK_TEMPERATURE = _config["api"].get("deepseek", {}).get("temperature", 0.7)
DEEPSEEK_MAX_TOKENS = _config["api"].get("deepseek", {}).get("max_tokens", 1536)
DEEPSEEK_TIMEOUT = _config["api"].get("deepseek", {}).get("timeout", 120)
DEEPSEEK_STREAM = _config["api"].get("deepseek", {}).get("stream", True)

# 阿里云
ALIYUN_TEMPERATURE = _config["api"].get("aliyun", {}).get("temperature", 0.7)
ALIYUN_MAX_TOKENS = _config["api"].get("aliyun", {}).get("max_tokens", 1536)
ALIYUN_TIMEOUT = _config["api"].get("aliyun", {}).get("timeout", 120)
ALIYUN_STREAM = _config["api"].get("aliyun", {}).get("stream", True)

# Ollama
OLLAMA_TEMPERATURE = _config["api"]["ollama"]["temperature"]
OLLAMA_TIMEOUT = _config["api"]["ollama"]["timeout"]
OLLAMA_STREAM = _config["api"]["ollama"]["stream"]

# 查询生成相关配置
QUERY_GENERATION_TEMPERATURE = _config["api"]["ollama"]["query_generation"]["temperature"]
QUERY_GENERATION_MAX_TOKENS = _config["api"]["ollama"]["query_generation"]["max_tokens"]
QUERY_GENERATION_TIMEOUT = _config["api"]["ollama"]["query_generation"]["timeout"]

# --- 网络配置 ---
HTTP_RETRIES = _config["network"]["http_retries"]
CONNECTION_TIMEOUT = _config["network"]["connection_timeout"]
REQUEST_TIMEOUT = _config["network"]["request_timeout"]
SERPAPI_TIMEOUT = _config["network"]["serpapi_timeout"]
RERANKER_TIMEOUT = _config["network"]["reranker_timeout"]

# --- 搜索配置 ---
SEARCH_NUM_RESULTS = _config["search"]["num_results"]
SEARCH_LANGUAGE = _config["search"]["language"]
SEARCH_COUNTRY = _config["search"]["country"]

# --- 应用配置 ---
APP_PORT_START = _config["app"]["port_range"]["start"]
APP_PORT_END = _config["app"]["port_range"]["end"]
APP_HOST = _config["app"]["host"]

# --- 日志配置 ---
LOGGING_LEVEL = _config["logging"]["level"]
LOGGING_FORMAT = _config["logging"]["format"]
