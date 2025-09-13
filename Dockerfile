# Dockerfile for Chinese-LLM-LoRA-Finetuning

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p data/raw data/processed results/models results/logs results/evaluation

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 暴露端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# 启动命令
CMD ["python", "demo/gradio_demo.py", "--host", "0.0.0.0", "--port", "7860"]