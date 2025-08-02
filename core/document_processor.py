# core/document_processor.py
import os
import time
from io import StringIO
from datetime import datetime
from pdfminer.high_level import extract_text_to_fp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging


# 新增的文件处理库
try:
    from docx import Document  # python-docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl  # Excel处理
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    from pptx import Presentation  # PowerPoint处理
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    import markdown  # Markdown处理
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


def extract_text_from_file(filepath: str) -> str:
    """根据文件路径提取文本内容，支持多种文件格式。"""
    file_ext = filepath.lower().split('.')[-1]

    try:
        if file_ext == 'pdf':
            # PDF文件处理
            output = StringIO()
            with open(filepath, 'rb') as file:
                extract_text_to_fp(file, output)
            return output.getvalue()

        elif file_ext == 'txt':
            # 纯文本文件处理
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()

        elif file_ext in ['doc', 'docx']:
            # Word文档处理
            if not DOCX_AVAILABLE:
                raise ImportError("需要安装 python-docx 库: pip install python-docx")
            doc = Document(filepath)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            # 处理表格中的文本
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text.append(cell.text)
            return '\n'.join(text)

        elif file_ext in ['xls', 'xlsx']:
            # Excel文件处理
            if not EXCEL_AVAILABLE:
                raise ImportError("需要安装 openpyxl 库: pip install openpyxl")
            workbook = openpyxl.load_workbook(filepath, data_only=True)
            text = []
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text.append(f"工作表: {sheet_name}")
                for row in sheet.iter_rows(values_only=True):
                    row_text = '\t'.join([str(cell) if cell is not None else '' for cell in row])
                    if row_text.strip():
                        text.append(row_text)
                text.append('')  # 工作表之间的分隔
            return '\n'.join(text)

        elif file_ext in ['ppt', 'pptx']:
            # PowerPoint文件处理
            if not PPTX_AVAILABLE:
                raise ImportError("需要安装 python-pptx 库: pip install python-pptx")
            prs = Presentation(filepath)
            text = []
            for slide_num, slide in enumerate(prs.slides, 1):
                text.append(f"幻灯片 {slide_num}:")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text.append(shape.text)
                text.append('')  # 幻灯片之间的分隔
            return '\n'.join(text)

        elif file_ext in ['md', 'markdown']:
            # Markdown文件处理
            with open(filepath, 'r', encoding='utf-8') as file:
                md_content = file.read()

            if MARKDOWN_AVAILABLE:
                # 如果有markdown库，可以转换为HTML再提取纯文本
                import re
                html = markdown.markdown(md_content)
                # 简单的HTML标签移除
                text = re.sub('<[^<]+?>', '', html)
                return text
            else:
                # 直接返回markdown原文
                return md_content

        elif file_ext in ['csv']:
            # CSV文件处理
            import csv
            text = []
            with open(filepath, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    text.append('\t'.join(row))
            return '\n'.join(text)

        elif file_ext in ['json']:
            # JSON文件处理
            import json
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return json.dumps(data, ensure_ascii=False, indent=2)

        elif file_ext in ['html', 'htm']:
            # HTML文件处理
            with open(filepath, 'r', encoding='utf-8') as file:
                html_content = file.read()
            # 简单的HTML标签移除
            import re
            text = re.sub('<[^<]+?>', '', html_content)
            # 移除多余的空白字符
            text = re.sub('\s+', ' ', text).strip()
            return text

        else:
            # 尝试作为纯文本处理
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    return file.read()
            except UnicodeDecodeError:
                # 如果UTF-8解码失败，尝试其他编码
                encodings = ['gbk', 'gb2312', 'latin-1']
                for encoding in encodings:
                    try:
                        with open(filepath, 'r', encoding=encoding) as file:
                            return file.read()
                    except UnicodeDecodeError:
                        continue
                raise ValueError(f"无法解码文件: {os.path.basename(filepath)}")

    except Exception as e:
        logging.error(f"处理文件 {filepath} 时出错: {e}")
        raise ValueError(f"处理文件 {os.path.basename(filepath)} 时出错: {str(e)}")


def get_supported_file_types() -> str:
    """返回支持的文件类型列表。"""
    supported_types = [
        "PDF (.pdf)",
        "纯文本 (.txt)",
        "Word文档 (.doc, .docx)" + (" - 需要安装 python-docx" if not DOCX_AVAILABLE else ""),
        "Excel表格 (.xls, .xlsx)" + (" - 需要安装 openpyxl" if not EXCEL_AVAILABLE else ""),
        "PowerPoint (.ppt, .pptx)" + (" - 需要安装 python-pptx" if not PPTX_AVAILABLE else ""),
        "Markdown (.md, .markdown)" + (" - 需要安装 markdown" if not MARKDOWN_AVAILABLE else ""),
        "CSV (.csv)",
        "JSON (.json)",
        "HTML (.html, .htm)"
    ]
    return "\n".join(f"• {t}" for t in supported_types)


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