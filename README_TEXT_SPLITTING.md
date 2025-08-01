# Enhanced Text Splitting Strategies

本项目现在支持多种文本切分策略，以提高RAG系统的语义完整性和检索准确率。

## 可用的切分策略

### 1. Recursive Character Splitter (递归字符切分器) - 默认
- **策略名称**: `recursive`
- **描述**: 基于字符数量和分隔符的传统切分方法
- **优点**: 快速、稳定、资源消耗低
- **缺点**: 可能在语义边界处断裂

### 2. Sentence-Aware Splitter (句子感知切分器) - 推荐
- **策略名称**: `sentence_aware` 
- **描述**: 尊重句子边界的智能切分方法
- **优点**: 
  - 保持语义完整性
  - 支持中英文混合文本
  - 减少上下文碎片化
  - 改善RAG检索效果
- **适用场景**: 大多数文档处理场景

### 3. Semantic Splitter (语义切分器) - 实验性
- **策略名称**: `semantic`
- **描述**: 基于语义相似度的高级切分方法
- **优点**: 最佳的语义连贯性
- **缺点**: 需要嵌入模型，计算资源消耗较高
- **适用场景**: 对语义完整性要求极高的场景

## 配置方法

### 环境变量配置

```bash
# 设置切分策略
export TEXT_SPLITTING_STRATEGY=sentence_aware

# 句子感知切分器参数
export SENTENCE_CHUNK_SIZE=500          # 目标块大小
export SENTENCE_OVERLAP=50              # 块之间的重叠

# 语义切分器参数  
export SEMANTIC_CHUNK_SIZE=600          # 目标块大小
export SEMANTIC_SIMILARITY_THRESHOLD=0.7  # 语义相似度阈值

# 递归切分器参数（默认）
# 使用现有的 CHUNK_SIZE=400 和 CHUNK_OVERLAP=40
```

### 配置示例

#### 推荐配置 - 平衡性能和效果
```bash
export TEXT_SPLITTING_STRATEGY=sentence_aware
export SENTENCE_CHUNK_SIZE=400
export SENTENCE_OVERLAP=50
```

#### 高质量配置 - 最佳语义完整性
```bash
export TEXT_SPLITTING_STRATEGY=semantic
export SEMANTIC_CHUNK_SIZE=600
export SEMANTIC_SIMILARITY_THRESHOLD=0.7
```

#### 高性能配置 - 最快处理速度
```bash
export TEXT_SPLITTING_STRATEGY=recursive
# 使用默认参数
```

## 使用说明

### 1. 在代码中使用

```python
from core.document_processor import split_text
from sentence_transformers import SentenceTransformer

# 准备文本
text = "您的文档内容..."

# 基础使用（递归或句子感知）
chunks = split_text(text)

# 语义切分（需要嵌入模型）
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chunks = split_text(text, embedding_model=embed_model)
```

### 2. 直接使用切分器

```python
from core.enhanced_splitters import get_enhanced_splitter

# 创建句子感知切分器
splitter = get_enhanced_splitter(
    strategy="sentence_aware",
    chunk_size=400,
    overlap=50
)

chunks = splitter.split_text(text)
```

## 性能对比

| 策略 | 处理速度 | 语义完整性 | 资源消耗 | 推荐场景 |
|------|----------|------------|----------|----------|
| recursive | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | 大批量处理 |
| sentence_aware | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 通用推荐 |
| semantic | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 高质量需求 |

## 错误处理和回退机制

系统具有健壮的错误处理机制：

1. **语义切分失败**: 自动回退到句子感知切分
2. **句子感知切分失败**: 自动回退到递归字符切分
3. **嵌入模型不可用**: 自动切换到不需要模型的策略

## 监控和调试

启用详细日志来监控切分效果：

```python
import logging
logging.basicConfig(level=logging.INFO)

# 切分过程会输出日志信息，包括：
# - 使用的切分策略
# - 生成的块数量
# - 任何错误和回退情况
```

## 最佳实践

1. **首次使用**: 建议使用 `sentence_aware` 策略，它在性能和效果间取得良好平衡
2. **大文档处理**: 对于包含完整段落的长文档，使用 `semantic` 策略获得最佳效果
3. **实时应用**: 对于需要快速响应的应用，使用 `recursive` 策略
4. **混合语言**: 处理中英文混合内容时，`sentence_aware` 策略表现最佳
5. **参数调优**: 根据你的文档特点调整块大小和重叠参数

## 更新日志

- **v1.0**: 添加句子感知和语义切分策略
- **v1.0**: 支持配置化策略选择
- **v1.0**: 完善的错误处理和回退机制