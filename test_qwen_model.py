#!/usr/bin/env python3
"""
简化版 Qwen-1.8B LoRA 训练脚本
"""

import os
import torch
import sys
print("🚀 开始 Qwen-1.8B LoRA 训练")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    print("✅ 依赖库加载成功")
    
    # 模型路径
    model_path = "cache/models--Qwen--Qwen-1_8B-Chat/snapshots/1d0f68de57b88cfde81f3c3e537f24464d889081"
    print(f"📂 模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        sys.exit(1)
    
    print("🔍 检查配置文件...")
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        print("✅ 找到config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            import json
            config = json.load(f)
            print(f"模型类型: {config.get('model_type', '未知')}")
            print(f"架构: {config.get('architectures', '未知')}")
    else:
        print("❌ 未找到config.json")
    
    print("🚀 开始加载模型...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ 分词器加载成功")
    
    # 加载模型（使用较少的显存）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ 模型加载成功")
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=8,  # 减小rank以节省内存
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "w1", "w2"],  # Qwen特定的模块
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA配置应用成功")
    
    # 简单测试
    test_input = "你好"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    print("🧪 运行简单测试...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"测试输出: {response}")
    
    print("🎉 模型测试成功！")
    print("💡 如需实际训练，请运行完整的训练脚本")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()