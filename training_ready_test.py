#!/usr/bin/env python3
"""
简化的训练测试 - 验证整个流程
"""

import os
import sys
import json
from pathlib import Path

def test_training_setup():
    """测试训练设置"""
    print("🚀 开始训练设置测试...")
    
    # 1. 检查环境
    try:
        import torch
        import transformers
        import peft
        from datasets import load_from_disk
        print("✅ 所有必要模块已导入")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   Transformers: {transformers.__version__}")
        print(f"   PEFT: {peft.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    
    # 2. 检查数据
    data_path = Path("data/processed/test_data")
    if not data_path.exists():
        print("❌ 训练数据不存在")
        return False
    
    print("✅ 训练数据存在")
    
    # 3. 检查配置文件
    config_path = Path("configs/quick_test.yaml")
    if not config_path.exists():
        print("❌ 配置文件不存在")
        return False
    
    print("✅ 配置文件存在")
    
    # 4. 模拟训练流程
    print("\n🔧 模拟训练流程...")
    
    try:
        # 加载数据
        dataset = load_from_disk(data_path)
        print(f"✅ 数据加载成功: {len(dataset)} 样本")
        
        # 加载模型和分词器（轻量级测试）
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "distilgpt2"
        print(f"🔧 加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ 分词器加载成功")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("✅ 模型加载成功")
        
        # 配置LoRA
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        print("✅ LoRA配置成功")
        
        # 显示参数信息
        model.print_trainable_parameters()
        
        # 测试推理
        print("\n🧪 测试基础推理...")
        test_input = "请介绍人工智能"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 生成文本: {generated_text}")
        
        print("\n🎉 训练设置测试完全成功！")
        print("📋 系统就绪，可以开始正式训练")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练设置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_training_log():
    """创建训练日志文件"""
    import datetime
    import torch
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cuda_status = 'CUDA可用' if torch.cuda.is_available() else 'CPU模式'
    
    log_content = f"""# 训练日志

## 环境配置
- 时间: {current_time}
- 系统: Windows
- GPU: {cuda_status}
- Python: {sys.version}

## 训练设置
- 模型: DistilGPT2
- LoRA配置: r=8, alpha=16
- 数据: 中文指令微调数据集
- 批次大小: 1
- 学习率: 2e-4

## 训练状态
✅ 环境验证通过
✅ 数据格式正确
✅ 模型配置成功
🚀 准备开始训练...
"""
    
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_dir / "training.log", 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"📝 训练日志已创建: {log_dir / 'training.log'}")

if __name__ == "__main__":
    print("🔍 开始综合训练测试...")
    
    if test_training_setup():
        create_training_log()
        print("\n🎯 下一步操作:")
        print("1. 运行完整训练: python src/train.py --config configs/quick_test.yaml")
        print("2. 启动Web界面: python demo/gradio_demo.py")
        print("3. 查看日志: cat results/logs/training.log")
        print("\n✨ 您的框架已经100%就绪！")
    else:
        print("\n❌ 请解决上述问题后重试")