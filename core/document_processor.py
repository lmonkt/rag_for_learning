# core/document_processor.py
import os
import time
from io import StringIO
from datetime import datetime
from pdfminer.high_level import extract_text_to_fp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging


# TODO: (改进方向) 多元数据源接入
# 思路:
# 1. 在 `extract_text_from_file` 函数中，通过文件后缀名判断文件类型。
# 2. 对每种类型调用不同的解析库。例如:
#    - .txt: 直接 file.read()
#    - .docx: 使用 `python-docx` 库
#    - .md: 使用 `markdown` 库，甚至可以解析其结构
# 3. 未来可以添加函数，从URL或API（如Notion）直接获取数据。
def extract_text_from_file(filepath: str) -> str:
    """根据文件路径提取文本内容，目前仅支持PDF。"""
    if filepath.lower().endswith('.pdf'):
        output = StringIO()
        with open(filepath, 'rb') as file:
            extract_text_to_fp(file, output)
        return output.getvalue()
    # 在此添加对其他文件类型的支持
    # elif filepath.lower().endswith('.txt'):
    #     with open(filepath, 'r', encoding='utf-8') as file:
    #         return file.read()
    else:
        raise ValueError(f"不支持的文件类型: {os.path.basename(filepath)}")


def split_text(text: str) -> list[str]:
    """将长文本切分为小块。"""
    # TODO: (改进方向) 更精细化的文本切分策略
    # 思路:
    # 1. 这里是实现新切分策略的核心位置。
    # 2. 可以替换 `RecursiveCharacterTextSplitter` 为其他高级切分器。
    #    例如，基于spaCy的句子切分器，或基于模型的语义切分器。
    # 3. 示例:
    #    from langchain_experimental.text_splitter import SemanticChunker
    #    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    #    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    #    text_splitter = SemanticChunker(embeddings)
    #    chunks = text_splitter.create_documents([text]) -> List of Document objects
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    return text_splitter.split_text(text)


class FileProcessor:
    """用于跟踪文件处理状态。"""

    def __init__(self):
        self.processed_files = {}

    def clear_files(self):
        self.processed_files = {}

    def add_file(self, file_name):
        self.processed_files[file_name] = {
            'status': '等待处理',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chunks': 0
        }

    def update_status(self, file_name, status, chunks=None):
        if file_name in self.processed_files:
            self.processed_files[file_name]['status'] = status
            if chunks is not None:
                self.processed_files[file_name]['chunks'] = chunks

    def get_file_list(self):
        return [
            f"📄 {fname} | {info['status']}"
            for fname, info in self.processed_files.items()
        ]


def process_files_to_chunks(files: list, file_processor: FileProcessor, progress=None) -> tuple[list, list, list]:
    """处理上传的文件列表，返回所有文本块、元数据和ID。"""
    all_new_chunks = []
    all_new_metadatas = []
    all_new_original_ids = []
    total_files = len(files)

    for idx, file in enumerate(files, 1):
        file_name = os.path.basename(file.name)
        if progress is not None:
            progress((idx - 1) / total_files, desc=f"处理文件 {idx}/{total_files}: {file_name}")
        file_processor.add_file(file_name)

        try:
            text = extract_text_from_file(file.name)
            if not text.strip():
                raise ValueError("文档内容为空或无法提取文本")

            chunks = split_text(text)
            doc_id = f"doc_{int(time.time())}_{idx}"

            current_file_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            current_file_metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]

            all_new_chunks.extend(chunks)
            all_new_metadatas.extend(current_file_metadatas)
            all_new_original_ids.extend(current_file_ids)

            file_processor.update_status(file_name, "处理完成", len(chunks))
        except Exception as e:
            logging.error(f"处理文件 {file_name} 时出错: {e}")
            file_processor.update_status(file_name, f"处理失败: {e}")

    return all_new_chunks, all_new_metadatas, all_new_original_ids