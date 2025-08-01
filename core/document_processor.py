# core/document_processor.py
import os
import time
from io import StringIO
from datetime import datetime
from pdfminer.high_level import extract_text_to_fp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
from docx import Document
import markdown


def extract_text_from_file(filepath: str) -> str:
    """根据文件路径提取文本内容，支持多种文件格式。"""
    file_extension = filepath.lower().split('.')[-1]
    
    try:
        if file_extension == 'pdf':
            output = StringIO()
            with open(filepath, 'rb') as file:
                extract_text_to_fp(file, output)
            return output.getvalue()
        
        elif file_extension == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        
        elif file_extension == 'docx':
            doc = Document(filepath)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        
        elif file_extension == 'md':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                md_content = file.read()
            # 将markdown转换为纯文本
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
        
        elif file_extension in ['xlsx', 'xls']:
            # 读取Excel文件的所有工作表
            excel_file = pd.ExcelFile(filepath)
            text_parts = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                # 将DataFrame转换为文本，保留结构信息
                text_parts.append(f"工作表: {sheet_name}\n")
                text_parts.append(df.to_string(index=False))
                text_parts.append("\n\n")
            return '\n'.join(text_parts)
        
        else:
            raise ValueError(f"不支持的文件类型: .{file_extension}")
            
    except Exception as e:
        raise ValueError(f"处理文件 {os.path.basename(filepath)} 时出错: {str(e)}")


def extract_text_from_url(url: str) -> str:
    """从URL提取文本内容。"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # 检测内容类型
        content_type = response.headers.get('content-type', '').lower()
        
        if 'application/json' in content_type:
            # JSON API响应
            json_data = response.json()
            return str(json_data)
        elif 'text/plain' in content_type:
            # 纯文本
            return response.text
        else:
            # HTML内容，使用BeautifulSoup提取文本
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除script和style标签
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 提取纯文本
            text = soup.get_text()
            # 清理多余的空白字符
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
    except Exception as e:
        raise ValueError(f"从URL {url} 提取内容时出错: {str(e)}")


def extract_text_from_api(api_url: str, headers: dict = None, params: dict = None) -> str:
    """从API端点提取文本内容。"""
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'application/json' in content_type:
            json_data = response.json()
            # 如果是JSON，尝试提取有意义的文本字段
            if isinstance(json_data, dict):
                text_fields = []
                for key, value in json_data.items():
                    if isinstance(value, str) and len(value) > 10:  # 假设有意义的文本长度 > 10
                        text_fields.append(f"{key}: {value}")
                return '\n'.join(text_fields)
            else:
                return str(json_data)
        else:
            return response.text
            
    except Exception as e:
        raise ValueError(f"从API {api_url} 提取内容时出错: {str(e)}")


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