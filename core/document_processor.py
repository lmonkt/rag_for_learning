# core/document_processor.py
import os
import time
from io import StringIO
from datetime import datetime
from pdfminer.high_level import extract_text_to_fp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging


def extract_text_from_file(filepath: str) -> str:
    """根据文件路径提取文本内容，支持PDF、TXT、DOCX、MD、Excel文件。"""
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
            try:
                from docx import Document
                doc = Document(filepath)
                text = []
                for paragraph in doc.paragraphs:
                    text.append(paragraph.text)
                return '\n'.join(text)
            except ImportError:
                raise ValueError("处理DOCX文件需要安装python-docx库")
        
        elif file_extension == 'md':
            try:
                import markdown
                from bs4 import BeautifulSoup
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    md_content = file.read()
                # 将markdown转换为纯文本
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
            except ImportError:
                raise ValueError("处理Markdown文件需要安装markdown和beautifulsoup4库")
        
        elif file_extension in ['xlsx', 'xls']:
            try:
                import pandas as pd
                # 读取Excel文件的所有工作表
                excel_file = pd.ExcelFile(filepath)
                text_parts = []
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    # 将DataFrame转换为文本
                    sheet_text = f"工作表: {sheet_name}\n"
                    sheet_text += df.to_string(index=False)
                    text_parts.append(sheet_text)
                return '\n\n'.join(text_parts)
            except ImportError:
                raise ValueError("处理Excel文件需要安装pandas和openpyxl库")
        
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")
            
    except Exception as e:
        if "需要安装" in str(e):
            raise e
        else:
            raise ValueError(f"处理文件 {os.path.basename(filepath)} 时出错: {str(e)}")


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