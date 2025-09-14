#!/usr/bin/env python3
"""
环境设置脚本
运行: python setup_environment.py
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """创建项目目录结构"""
    
    directories = [
        "configs",
        "data/raw",
        "data/processed", 
        "src",
        "scripts",
        "notebooks",
        "demo",
        "results/models",
        "results/logs", 
        "results/evaluation",
        "tests",
        "docs"
    ]
    
    print("📁 创建项目目录结构...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   创建目录: {directory}")
    
    # 创建空的__init__.py文件
    init_files = ["src/__init__.py", "tests/__init__.py"]
    for init_file in init_files:
        Path(init_file).touch()
        print(f"   创建文件: {init_file}")

def create_config_files():
    """创建配置文件"""
    
    # 创建基础配置文件
    config_files = {
        ".env.example": """# Environment Variables Template
# 复制此文件为 .env 并填入实际值

# Weights & Biases
WANDB_PROJECT=chinese-llm-lora
WANDB_ENTITY=your_username

# Hugging Face
HF_TOKEN=your_huggingface_token

# Model paths
MODEL_CACHE_DIR=./cache
OUTPUT_DIR=./results
""",

        "requirements-full.txt": """# Complete requirements for production
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
transformers>=4.35.0
tokenizers>=0.14.0
accelerate>=0.24.0
peft>=0.6.0
datasets>=2.14.0
huggingface-hub>=0.19.0
bitsandbytes>=0.41.0
sentencepiece>=0.1.99
protobuf>=4.24.0
evaluate>=0.4.0
nltk>=3.8.0
rouge-score>=0.1.2
sacrebleu>=2.3.0
wandb>=0.16.0
tensorboard>=2.15.0
gradio>=4.0.0
streamlit>=1.28.0
jupyter>=1.0.0
notebook>=7.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
omegaconf>=2.3.0
pyyaml>=6.0.0
tqdm>=4.66.0
rich>=13.6.0
packaging>=23.2
psutil>=5.9.0
""",

        "configs/quick_test.yaml": """# Quick Test Configuration for 8GB GPU
model:
  name: "THUDM/chatglm3-6b"
  trust_remote_code: true
  torch_dtype: "bfloat16"
  device_map: "auto"
  load_in_8bit: true  # Enable 8-bit quantization for 8GB GPU

lora:
  r: 8  # Reduced rank for smaller memory footprint
  lora_alpha: 16
  target_modules: 
    - "query_key_value"
    - "dense"
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  output_dir: "./results/quick_test"
  num_train_epochs: 1
  per_device_train_batch_size: 1  # Small batch size for 8GB GPU
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8  # Larger accumulation steps
  learning_rate: 2e-4
  warmup_steps: 50
  logging_steps: 5
  save_steps: 100
  eval_steps: 100
  max_seq_length: 256  # Shorter sequences for memory efficiency
  dataloader_num_workers: 0
  remove_unused_columns: false
  
optimizer:
  name: "adamw_torch"
  weight_decay: 0.01
  
scheduler:
  name: "cosine"
  
data:
  dataset_name: "BelleGroup/train_0.5M_CN"
  validation_split: 0.1
  max_samples: 1000  # Very small dataset for quick test
"""
    }
    
    print("\n⚙️  创建配置文件...")
    for file_path, content in config_files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   创建文件: {file_path}")

def create_gitkeep_files():
    """在空目录中创建.gitkeep文件"""
    empty_dirs = [
        "data/raw",
        "data/processed",
        "results/models",
        "results/logs",
        "results/evaluation",
        "tests",
        "docs"
    ]
    
    print("\n📄 创建.gitkeep文件...")
    for directory in empty_dirs:
        gitkeep_path = Path(directory) / ".gitkeep"
        gitkeep_path.touch()
        print(f"   创建文件: {gitkeep_path}")

def create_batch_scripts():
    """创建批处理脚本"""
    scripts = {
        "activate_env.bat": """@echo off
echo 激活虚拟环境...
call .\llm-lora\Scripts\activate.bat
echo 环境已激活！虚拟环境: llm-lora
echo 使用 'deactivate' 退出环境
cmd /k
""",
        
        "run_quick_test.bat": """@echo off
echo 开始快速测试...
call .\llm-lora\Scripts\activate.bat
python src/train.py --config configs/quick_test.yaml
pause
""",
        
        "start_gradio_demo.bat": """@echo off
echo 启动Gradio演示界面...
call .\llm-lora\Scripts\activate.bat
python demo/gradio_demo.py
pause
"""
    }
    
    print("\n🔧 创建批处理脚本...")
    for script_name, content in scripts.items():
        with open(script_name, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   创建文件: {script_name}")

def main():
    print("🚀 开始设置项目环境...")
    
    # 检查是否在项目根目录
    if not Path("README.md").exists():
        print("❌ 请在项目根目录运行此脚本")
        sys.exit(1)
    
    create_project_structure()
    create_config_files() 
    create_gitkeep_files()
    create_batch_scripts()
    
    print("\n✅ 项目环境设置完成!")
    print("\n📋 后续步骤:")
    print("1. 复制 .env.example 为 .env 并配置API tokens")
    print("2. 运行 'activate_env.bat' 激活虚拟环境")
    print("3. 运行 'run_quick_test.bat' 进行快速测试")
    print("4. 运行 'start_gradio_demo.bat' 启动Web界面")
    print("\n🎯 推荐的下一步:")
    print("- 运行快速测试验证环境: python src/train.py --config configs/quick_test.yaml")
    print("- 启动Gradio界面: python demo/gradio_demo.py")

if __name__ == "__main__":
    main()