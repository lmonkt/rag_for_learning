# app.py
import webbrowser

# 导入UI创建函数
from ui import create_ui

# 导入逻辑层和工具函数
import logic
from utils import helpers


def main():
    """
    主程序入口。
    """
    # 1. 检查环境依赖
    print("🚀 正在启动智能文档问答系统...")
    print("Step 1/4: 检查系统环境...")
    if not helpers.check_environment():
        print("🔴 环境检查失败，程序退出。")
        exit(1)
    print("✅ 环境检查通过。")

    # 2. 查找可用端口
    print("Step 2/4: 检查服务端口...")
    ports_to_try = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports_to_try if helpers.is_port_available(p)), None)

    if not selected_port:
        print("🔴 所有预设端口 (17995-17999) 都已被占用，请检查并释放端口。")
        exit(1)
    print(f"✅ 将在端口 {selected_port} 启动服务。")

    # 3. 初始化模型 (这是一个耗时操作，所以在启动UI前完成)
    print("Step 3/4: 初始化核心模型 (可能需要一些时间)...")
    logic.initialize_models()
    print("✅ 模型初始化完成。")

    # 4. 创建并启动Gradio应用
    print("Step 4/4: 构建并启动Web界面...")
    app_url = f"http://127.0.0.1:{selected_port}"
    print(f"🌐 系统即将就绪，请在浏览器中打开: {app_url}")

    demo = create_ui()

    # 尝试自动打开浏览器
    try:
        webbrowser.open(app_url)
    except Exception as e:
        print(f"自动打开浏览器失败: {e}，请手动复制链接访问。")

    # 启动服务
    demo.launch(
        server_port=selected_port,
        server_name="0.0.0.0",  # 允许局域网内其他设备访问
        show_error=True
    )


if __name__ == "__main__":
    main()