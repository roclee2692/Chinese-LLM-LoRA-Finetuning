@echo off
echo 🌐 启动中文LLM LoRA微调框架Web界面...
echo 📍 当前目录: %CD%

:: 激活虚拟环境
echo ⚡ 激活虚拟环境...
call ".\llm-lora\Scripts\activate.bat"

:: 确认环境
echo 🔍 验证Python路径...
where python

:: 启动Gradio界面
echo 🚀 启动Web界面...
echo 📱 界面将在浏览器中自动打开
echo 🔗 手动访问: http://localhost:7860
echo.

python quick_demo.py

echo.
echo 🎉 演示结束！
pause