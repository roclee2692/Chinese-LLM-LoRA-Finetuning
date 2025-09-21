#!/usr/bin/env python3
"""
Qwen-1.8B LoRA 训练脚本
自动生成的训练脚本
"""

import os
import sys
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import json
from datetime import datetime

def setup_model_and_tokenizer():
    """设置模型和分词器"""
    model_path = "cache/models--Qwen--Qwen-1_8B-Chat/snapshots/1d0f68de57b88cfde81f3c3e537f24464d889081"
    
    print(f"🚀 加载模型: {model_path}")
    
    # 配置量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """准备训练数据"""
    # 使用示例数据
    train_data = [
        {"instruction": "你好", "output": "你好！我是Qwen，很高兴为您服务。有什么我可以帮助您的吗？"},
        {"instruction": "请介绍一下你自己", "output": "我是Qwen，一个由阿里云开发的大型语言模型。我可以帮助您回答问题、创作内容、翻译文本等多种任务。"},
        {"instruction": "什么是人工智能？", "output": "人工智能（AI）是指让机器模拟人类智能的技术，包括学习、推理、感知、语言理解等能力。"},
        {"instruction": "请解释机器学习", "output": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式，无需明确编程就能做出预测或决策。"},
        {"instruction": "深度学习是什么？", "output": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑处理信息的方式，特别擅长处理图像、语音和文本等复杂数据。"}
    ]
    
    def format_instruction(sample):
        return f"<|im_start|>system\n你是一个有用的AI助手。<|im_end|>\n<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"
    
    def tokenize_function(examples):
        texts = [format_instruction(ex) for ex in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # 创建数据集
    dataset = Dataset.from_list(train_data * 100)  # 复制数据以增加训练样本
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function([x]),
        batched=False,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    """主训练函数"""
    print("🚀 开始 Qwen-1.8B LoRA 训练")
    print(f"⏰ 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置输出目录
    output_dir = "results/models/qwen-1.8b-lora-chat"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/logs/qwen-1.8b-lora-chat", exist_ok=True)
    
    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer()
    
    # 准备数据集
    train_dataset = prepare_dataset(tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=0.0002,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="results/logs/qwen-1.8b-lora-chat",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to=None,  # 禁用wandb等
        run_name="qwen-1.8b-lora-training"
    )
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("📚 开始训练...")
    trainer.train()
    
    # 保存模型
    print("💾 保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("✅ 训练完成！")
    print(f"📁 模型保存在: {output_dir}")

if __name__ == "__main__":
    main()
