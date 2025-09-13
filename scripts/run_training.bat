@echo off
REM 中文大语言模型LoRA微调 - Windows训练启动脚本
REM 使用方法: run_training.bat [配置文件路径]

setlocal enabledelayedexpansion

REM 默认配置文件
set CONFIG_FILE=%1
if "%CONFIG_FILE%"=="" set CONFIG_FILE=configs\chatglm3_lora.yaml

REM 检查配置文件是否存在
if not exist "%CONFIG_FILE%" (
    echo 错误: 配置文件不存在: %CONFIG_FILE%
    echo 使用方法: %0 [配置文件路径]
    exit /b 1
)

echo ==========================================
echo 中文大语言模型LoRA微调训练
echo 配置文件: %CONFIG_FILE%
echo 开始时间: %date% %time%
echo ==========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python
    exit /b 1
)

REM 检查CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo 警告: 未检测到NVIDIA GPU，将使用CPU训练（速度较慢）
) else (
    echo GPU信息:
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo.
)

REM 检查必要的Python包
echo 检查Python环境...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo 错误: PyTorch未安装
    exit /b 1
)

python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')" 2>nul
if errorlevel 1 (
    echo 错误: Transformers未安装
    exit /b 1
)

python -c "import peft; print(f'PEFT版本: {peft.__version__}')" 2>nul
if errorlevel 1 (
    echo 错误: PEFT未安装
    exit /b 1
)

echo.

REM 创建必要的目录
if not exist "results\models" mkdir results\models
if not exist "results\logs" mkdir results\logs
if not exist "data\processed" mkdir data\processed

REM 设置环境变量
set PYTHONPATH=%PYTHONPATH%;%cd%\src
if "%CUDA_VISIBLE_DEVICES%"=="" set CUDA_VISIBLE_DEVICES=0

REM 开始训练
echo 开始训练...
python src\train.py --config "%CONFIG_FILE%"

echo.
echo ==========================================
echo 训练完成时间: %date% %time%
echo ==========================================

pause