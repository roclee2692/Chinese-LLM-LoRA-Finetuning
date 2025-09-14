@echo off
echo 🌐 启动中文LLM LoRA微调框架Web界面...
echo 📍 当前目录: %CD%

:: 激活虚拟环境
echo ⚡ 激活虚拟环境...
call ".\llm-lora\Scripts\activate.bat"

:: 确认环境
echo 🔍 验证Python路径...
where python

:: 检查Gradio
echo 📦 检查Gradio...
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"

:: 启动Web界面
echo 🚀 启动Web界面...
echo 📡 界面将在 http://127.0.0.1:7860 启动
python start_gradio_demo.py

echo ✅ Web界面已关闭
pause off
echo 启动Gradio演示界面...
call .\llm-lora\Scriptsctivate.bat
python demo/gradio_demo.py
pause
