#!/usr/bin/env python3
"""
Qwen-1.8B LoRA 实际训练脚本
基于测试成功的配置进行实际训练
"""

import os
import torch
import json
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def main():
    print("🚀 开始 Qwen-1.8B LoRA 实际训练")
    print(f"⏰ 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 模型路径
    model_path = "cache/models--Qwen--Qwen-1_8B-Chat/snapshots/1d0f68de57b88cfde81f3c3e537f24464d889081"
    output_dir = "results/models/qwen-1.8b-lora-training"
    
    print(f"📂 模型路径: {model_path}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载分词器
    print("📝 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ 分词器加载成功")
    
    # 2. 加载模型
    print("🤖 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ 模型加载成功")
    
    # 3. 配置LoRA
    print("⚙️ 配置LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "w1", "w2"],
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA配置成功")
    
    # 4. 准备训练数据
    print("📚 准备训练数据...")
    
    # 示例对话数据
    train_data = [
        {"instruction": "你好", "output": "你好！我是Qwen，很高兴为您服务。"},
        {"instruction": "请介绍一下你自己", "output": "我是Qwen，一个由阿里云开发的大型语言模型。"},
        {"instruction": "什么是人工智能？", "output": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"},
        {"instruction": "请解释机器学习", "output": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。"},
        {"instruction": "深度学习是什么？", "output": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑处理信息。"},
        {"instruction": "Python是什么？", "output": "Python是一种高级编程语言，以其简洁易读的语法而闻名。"},
        {"instruction": "如何学习编程？", "output": "学习编程需要选择一门语言开始，多练习项目，理解算法和数据结构。"},
        {"instruction": "LoRA是什么？", "output": "LoRA是低秩适应方法，用于高效地微调大型语言模型。"}
    ]
    
    # 扩展数据集
    extended_data = train_data * 50  # 重复50次以增加训练样本
    
    def format_instruction(sample):
        return f"<|im_start|>system\\n你是一个有用的AI助手。<|im_end|>\\n<|im_start|>user\\n{sample['instruction']}<|im_end|>\\n<|im_start|>assistant\\n{sample['output']}<|im_end|>"
    
    def tokenize_function(examples):
        texts = [format_instruction(ex) for ex in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # 创建数据集
    dataset = Dataset.from_list(extended_data)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function([x]),
        batched=False,
        remove_columns=dataset.column_names
    )
    
    print(f"📊 训练数据量: {len(tokenized_dataset)} 样本")
    
    # 5. 训练配置
    print("⚙️ 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        dataloader_num_workers=0,  # 避免Windows上的多进程问题
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to=None,
        run_name="qwen-1.8b-lora-training",
        load_best_model_at_end=False
    )
    
    # 6. 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 7. 创建训练器
    print("🏗️ 创建训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 8. 开始训练
    print("🎯 开始训练...")
    print(f"💾 输出目录: {output_dir}")
    print(f"📊 批量大小: {training_args.per_device_train_batch_size}")
    print(f"🔄 训练轮数: {training_args.num_train_epochs}")
    print(f"📚 数据量: {len(tokenized_dataset)} 样本")
    print("⚡ 开始训练过程...")
    
    try:
        trainer.train()
        print("🎉 训练完成！")
        
        # 保存模型
        print("💾 保存模型...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # 保存训练配置
        config_info = {
            "model_name": "Qwen-1.8B",
            "training_method": "LoRA",
            "lora_config": {
                "r": lora_config.r,
                "alpha": lora_config.lora_alpha,
                "dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules
            },
            "training_args": {
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate
            },
            "completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "dataset_size": len(tokenized_dataset)
        }
        
        with open(os.path.join(output_dir, "training_info.json"), 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 训练完成！模型已保存到: {output_dir}")
        print("🔍 运行 python simple_monitor.py 查看训练结果")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()