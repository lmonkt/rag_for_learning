# ui.py
import logging

import gradio as gr

# 导入业务逻辑函数，UI的交互将调用这些函数
import logic
from utils.helpers import get_system_models_info, process_thinking_content

chunk_data_cache = []

# 将原始文件中的 process_chat 逻辑封装为UI的事件处理器
def chat_interface(question, history, enable_web_search, model_choice):
    """
    处理聊天输入的包装函数。
    它接收UI组件的状态，调用后端逻辑，并以流式方式更新UI。
    """
    if not question or not question.strip():
        # 如果history是None，初始化它
        history = history or []
        history.append({'role': 'user', 'content': question})
        history.append({'role': 'assistant', 'content': "提问不能为空，请输入有效的问题。"})
        # 注意: 当使用yield时，函数会变成一个生成器，所以我们需要用一个循环来驱动它
        # 这里为了简单，直接返回最终状态
        return history, ""
    history = history or []
    history.append({'role': 'user', 'content': question})
    history.append({'role': 'assistant', 'content': "🧠 思考中，请稍候..."})
    # 添加一个空的assistant回复，稍后填充
    yield history, ""

    try:
        logging.info(f"开始处理问题: '{question}'")
        response_generator = logic.answer_question_stream(question, enable_web_search, model_choice)

        # 流式更新聊天机器人的回答
        final_response = ""
        for response_chunk, status_message in response_generator:
            final_response = response_chunk  # 保存最新的完整回复
            history[-1]['content'] = final_response  # 更新最后一条消息的内容
            yield history, ""  # 每次收到新的内容块时，都更新UI

        # 确保处理思维链等最终格式
        history[-1]['content'] = process_thinking_content(final_response)
        logging.info(f"问题处理完成。")

    except Exception as e:
        logging.error(f"处理聊天时发生严重错误: {e}", exc_info=True)
        # 将错误信息显示在UI上
        error_message = f"抱歉，处理时遇到错误：\n\n{str(e)}"
        history[-1]['content'] = error_message

    # 最终更新一次UI
    yield history, ""


def get_document_chunks_for_ui(progress=None):
    """
    从后端逻辑层获取文档分块数据，并处理成适合UI展示的格式。
    """
    global chunk_data_cache

    # 关键改动：我们不再使用全局变量，而是通过 logic 模块访问已封装好的 faiss_manager
    faiss_manager = logic.faiss_manager
    if not faiss_manager.faiss_id_order_for_index:
        chunk_data_cache = []
        return [], "知识库中没有文档，请先上传并处理文档。"

    if progress is not None:
        progress(0.5, desc="正在组织分块数据...")

    doc_groups = {}
    for doc_id in faiss_manager.faiss_id_order_for_index:
        doc = faiss_manager.faiss_contents_map.get(doc_id, "")
        meta = faiss_manager.faiss_metadatas_map.get(doc_id, {})
        if not doc: continue

        source = meta.get('source', '未知来源')
        if source not in doc_groups:
            doc_groups[source] = []

        chunk_info = {
            "content": doc,
            "char_count": len(doc)
        }
        doc_groups[source].append(chunk_info)

    # 准备用于UI展示的数据
    result_for_df = []  # 用于Dataframe的数据
    result_for_cache = []  # 用于缓存的完整数据

    for source, chunks in doc_groups.items():
        for i, chunk in enumerate(chunks):
            # 添加到缓存
            result_for_cache.append({
                "来源": source,
                "序号": f"{i + 1}/{len(chunks)}",
                "字符数": chunk["char_count"],
                "内容预览": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "完整内容": chunk["content"]
            })
            # 添加到Dataframe的显示列表
            result_for_df.append([
                source,
                f"{i + 1}/{len(chunks)}",
                chunk["char_count"],
                chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            ])

    chunk_data_cache = result_for_cache
    summary = f"总计 {len(chunk_data_cache)} 个文本块，来自 {len(doc_groups)} 个文档。"

    if progress is not None:
        progress(1.0, desc="数据加载完成!")

    return result_for_df, summary


def show_chunk_details(evt: gr.SelectData):
    """
    当用户在Dataframe中点击某一行时，显示该分块的完整内容。
    """
    # evt.index[0] 是选中行的行号
    if evt.index[0] < len(chunk_data_cache):
        selected_chunk = chunk_data_cache[evt.index[0]]
        return selected_chunk.get("完整内容", "内容加载失败")
    return "未找到选中的分块"

def create_ui():
    """
    创建并返回Gradio应用的UI界面。
    """
    # 将原始文件中的UI布局代码 (`with gr.Blocks...`) 整体移动到这里
    with gr.Blocks(title="智能文档问答系统", css="""
    /* 全局主题变量 */
    :root[data-theme="light"] {
        --text-color: #2c3e50;
        --bg-color: #ffffff;
        --panel-bg: #f8f9fa;
        --border-color: #e9ecef;
        --success-color: #4CAF50;
        --error-color: #f44336;
        --primary-color: #2196F3;
        --secondary-bg: #ffffff;
        --hover-color: #e9ecef;
        --chat-user-bg: #e3f2fd;
        --chat-assistant-bg: #f5f5f5;
    }

    :root[data-theme="dark"] {
        --text-color: #e0e0e0;
        --bg-color: #1a1a1a;
        --panel-bg: #2d2d2d;
        --border-color: #404040;
        --success-color: #81c784;
        --error-color: #e57373;
        --primary-color: #64b5f6;
        --secondary-bg: #2d2d2d;
        --hover-color: #404040;
        --chat-user-bg: #1e3a5f;
        --chat-assistant-bg: #2d2d2d;
    }

    /* 全局样式 */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        margin: 0;
        padding: 0;
        overflow-x: hidden;
        width: 100vw;
        height: 100vh;
    }

    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 1% !important;
        color: var(--text-color);
        background-color: var(--bg-color);
        min-height: 100vh;
    }
    
    /* 确保标签内容撑满 */
    .tabs.svelte-710i53 {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }

    /* 主题切换按钮 */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        background: var(--panel-bg);
        color: var(--text-color);
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .theme-toggle:hover {
        background: var(--hover-color);
    }

    /* 面板样式 */
    .left-panel {
        padding-right: 20px;
        border-right: 1px solid var(--border-color);
        background: var(--bg-color);
        width: 100%;
    }

    .right-panel {
        height: 100vh;
        background: var(--bg-color);
        width: 100%;
    }

    /* 文件列表样式 */
    .file-list {
        margin-top: 10px;
        padding: 12px;
        background: var(--panel-bg);
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }

    /* 答案框样式 */
    .answer-box {
        min-height: 500px !important;
        background: var(--panel-bg);
        border-radius: 8px;
        padding: 16px;
        font-size: 15px;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }

    /* 输入框样式 */
    textarea {
        background: var(--panel-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 14px !important;
    }

    /* 按钮样式 */
    button.primary {
        background: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    button.primary:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* 标题和文本样式 */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }

    .footer-note {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 13px;
        margin-top: 12px;
    }

    /* 加载和进度样式 */
    #loading, .progress-text {
        color: var(--text-color);
    }

    /* 聊天记录样式 */
    .chat-container {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 16px;
        max-height: 80vh;
        height: 80vh !important;
        overflow-y: auto;
        background: var(--bg-color);
    }

    .chat-message {
        padding: 12px 16px;
        margin: 8px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.5;
    }

    .chat-message.user {
        background: var(--chat-user-bg);
        margin-left: 32px;
        border-top-right-radius: 4px;
    }

    .chat-message.assistant {
        background: var(--chat-assistant-bg);
        margin-right: 32px;
        border-top-left-radius: 4px;
    }

    .chat-message .timestamp {
        font-size: 12px;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 4px;
    }

    .chat-message .content {
        white-space: pre-wrap;
    }

    /* 按钮组样式 */
    .button-row {
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }

    .clear-button {
        background: var(--error-color) !important;
    }

    /* API配置提示样式 */
    .api-info {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        background: var(--panel-bg);
        border: 1px solid var(--border-color);
    }

    /* 新增: 数据可视化卡片样式 */
    .model-card {
        background: var(--panel-bg);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--border-color);
        margin-bottom: 16px;
    }

    .model-card h3 {
        margin-top: 0;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 8px;
    }

    .model-item {
        display: flex;
        margin-bottom: 8px;
    }

    .model-item .label {
        flex: 1;
        font-weight: 500;
    }

    .model-item .value {
        flex: 2;
    }

    /* 数据表格样式 */
    .chunk-table {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    .chunk-table th, .chunk-table td {
        border: 1px solid var(--border-color);
        padding: 8px;
    }

    .chunk-detail-box {
        min-height: 200px;
        padding: 16px;
        background: var(--panel-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-family: monospace;
        white-space: pre-wrap;
        overflow-y: auto;
    }
    """) as demo:  # CSS与原文件相同，此处省略
        gr.Markdown("# 🧠 智能文档问答系统")

        with gr.Tabs() as tabs:
            # --- 问答对话 Tab ---
            with gr.TabItem("💬 问答对话"):
                with gr.Row(equal_height=True):
                    # 左侧操作面板
                    with gr.Column(scale=5):
                        gr.Markdown("## 📂 文档处理区")
                        file_input = gr.File(label="上传文档", file_types=[".pdf", ".txt", ".docx", ".md", ".xlsx", ".xls"], file_count="multiple")
                        upload_btn = gr.Button("🚀 开始处理", variant="primary")
                        upload_status = gr.Textbox(label="处理状态", interactive=False, lines=2)
                        file_list = gr.Textbox(label="已处理文件", interactive=False, lines=3)

                        gr.Markdown("## ❓ 提问区")
                        question_input = gr.Textbox(label="输入问题", lines=3, placeholder="请输入您的问题...")
                        with gr.Row():
                            web_search_checkbox = gr.Checkbox(label="启用联网搜索", value=False)
                            model_choice = gr.Dropdown(choices=["ollama", "siliconflow"], value="ollama",
                                                       label="模型选择")
                        with gr.Row():
                            ask_btn = gr.Button("🔍 开始提问", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", scale=1)

                    # 右侧对话区
                    with gr.Column(scale=7):
                        gr.Markdown("## 📝 对话记录")
                        chatbot = gr.Chatbot(label="对话历史", height=600, show_label=False,type='messages')
                        gr.Markdown("<div class='footer-note'>*回答生成可能需要一些时间，请耐心等待。</div>")

            # --- 分块可视化 Tab ---
            with gr.TabItem("📊 分块可视化"):
                with gr.Column():
                    gr.Markdown("## 📄 文档分块检视")
                    gr.Markdown(
                        "在这里，你可以查看上传文档后，文本被切分成的所有小块（Chunks）。这有助于理解RAG系统检索的资料来源。")

                    with gr.Row():
                        refresh_chunks_btn = gr.Button("🔄 刷新分块数据", variant="primary")

                    chunks_status = gr.Markdown("点击按钮以加载分块数据...")

                    # 用于显示分块数据的表格
                    chunks_data = gr.Dataframe(
                        headers=["来源文档", "分块序号", "字符数", "内容预览"],
                        datatype=["str", "str", "number", "str"],
                        row_count=(10, "dynamic"),
                        wrap=True
                    )

                    # 用于显示单个分块的完整内容
                    chunk_detail_text = gr.Textbox(
                        label="分块详情",
                        placeholder="点击上方表格中的任意行，在此处查看其完整内容...",
                        lines=10,
                        interactive=False,
                        show_copy_button=True
                    )

        # ----------------------------------------------------
        # 绑定UI事件到后端逻辑
        # ----------------------------------------------------

        # 1. 上传按钮点击事件
        upload_btn.click(
            fn=logic.process_uploaded_files,  # 直接调用logic中的函数
            inputs=[file_input],
            outputs=[upload_status, file_list],
            show_progress="full"  # Gradio内置的进度条
        )

        # 2. 提问按钮点击事件
        ask_btn.click(
            fn=chat_interface,  # 调用上面定义的UI包装函数
            inputs=[question_input, chatbot, web_search_checkbox, model_choice],
            outputs=[chatbot, question_input]  # 更新chatbot，并清空输入框
        )

        # 3. 回车键提问事件
        question_input.submit(
            fn=chat_interface,
            inputs=[question_input, chatbot, web_search_checkbox, model_choice],
            outputs=[chatbot, question_input]
        )

        # 3. 清空按钮
        clear_btn.click(fn=lambda: None, inputs=[], outputs=[chatbot])

        # --- FIX: 绑定分块可视化Tab的事件 ---

        # 4. 刷新分块数据按钮
        refresh_chunks_btn.click(
            fn=get_document_chunks_for_ui,
            inputs=[],
            outputs=[chunks_data, chunks_status]
        )

        # 5. 表格行点击事件
        chunks_data.select(
            fn=show_chunk_details,
            inputs=[],  # `select` 事件会自动传递一个 `SelectData` 对象
            outputs=[chunk_detail_text]
        )

    return demo