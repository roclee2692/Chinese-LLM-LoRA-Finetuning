#!/bin/bash

# 中文大语言模型LoRA微调 - 训练启动脚本
# 使用方法: ./run_training.sh [配置文件路径]

set -e

# 默认配置文件
CONFIG_FILE=${1:-"configs/chatglm3_lora.yaml"}

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo "使用方法: $0 [配置文件路径]"
    exit 1
fi

echo "=========================================="
echo "中文大语言模型LoRA微调训练"
echo "配置文件: $CONFIG_FILE"
echo "开始时间: $(date)"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU训练（速度较慢）"
fi

# 检查必要的Python包
echo "检查Python环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
python -c "import peft; print(f'PEFT版本: {peft.__version__}')"

echo ""

# 创建必要的目录
mkdir -p results/models
mkdir -p results/logs
mkdir -p data/processed

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# 开始训练
echo "开始训练..."
python src/train.py --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "训练完成时间: $(date)"
echo "=========================================="