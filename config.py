# config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# --- API Keys & Endpoints ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_CHECK_URL = "http://localhost:11434/api/show"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


# --- Model Configurations ---
# TODO: (改进方向) 更复杂的模型管理
# 思路: 未来可以考虑将模型配置定义为字典或类，方便管理多个不同用途的模型。
# 例如: MODELS = { 'embedding': 'all-MiniLM-L6-v2', 'reranker': '...', 'generator_small': 'deepseek-r1:1.5b' }
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
# 如果需要支持中文，可以切换为 'shibing624/text2vec-base-chinese'
CROSS_ENCODER_MODEL_NAME = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
GENERATOR_MODEL_OLLAMA = "deepseek-r1:7b" # 用于生成答案的主模型
GENERATOR_MODEL_OLLAMA_LIGHT = "deepseek-r1:1.5b" # 用于内部判断、评分的轻量模型
GENERATOR_MODEL_SILICONFLOW = "Pro/deepseek-ai/DeepSeek-R1"


# --- RAG Pipeline Parameters ---
# TODO: (改进方向) 更精细化的文本切分策略
# 当前配置: 使用固定的 chunk_size 和 chunk_overlap。
# 改进思路:
# 1. 语义切分: 引入如 'semantic-text-splitter' 库，它使用 embedding 模型来决定切分点，确保语义完整性。
# 2. 结构化切分: 针对Markdown或HTML，可以基于标题（#, ##）、列表（-）或表格进行切分，保留文档结构。
#    - 这需要在 `document_processor.py` 中实现新的切分逻辑。
# 3. 动态调整: 根据文档类型或内容密度动态调整 chunk_size。
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40

# --- Text Splitting Strategy Configuration ---
# Available strategies: "recursive", "semantic", "sentence_aware"
TEXT_SPLITTING_STRATEGY = os.getenv("TEXT_SPLITTING_STRATEGY", "recursive")

# Semantic chunking parameters (when using semantic strategy)
SEMANTIC_CHUNK_SIZE = int(os.getenv("SEMANTIC_CHUNK_SIZE", "600"))  # Slightly larger for semantic coherence
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.7"))  # Minimum similarity to merge chunks

# Sentence-aware chunking parameters
SENTENCE_CHUNK_SIZE = int(os.getenv("SENTENCE_CHUNK_SIZE", "500"))  # Target size for sentence-aware chunks
SENTENCE_OVERLAP = int(os.getenv("SENTENCE_OVERLAP", "50"))  # Overlap in characters for sentence chunks

# 混合检索中语义检索的权重 (alpha)
HYBRID_SEARCH_ALPHA = 0.7

# 检索和重排序返回的 top_k 数量
RETRIEVER_TOP_K = 10
RERANKER_TOP_K = 5
RECURSIVE_RETRIEVAL_MAX_ITERATIONS = 3


# --- System & Environment ---
# 禁用 oneDNN 优化 (避免某些CPU上的警告)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 设置不走代理的地址
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- Search Engine ---
SEARCH_ENGINE = "google" # 可选 'bing', 'baidu' 等

# --- Rerank Method ---
# 可选值: "cross_encoder", "llm"
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")