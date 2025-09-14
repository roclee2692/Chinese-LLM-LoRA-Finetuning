#!/usr/bin/env python3
"""
快速训练测试脚本
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

def quick_training_test():
    """快速训练测试"""
    print("🚀 开始快速训练测试...")
    
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    
    # 1. 加载模型和分词器
    print("\n📦 加载模型和分词器...")
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # 2. 配置LoRA
    print("\n⚙️  配置LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. 加载修复后的数据
    print("\n📥 加载数据...")
    data_path = Path("data/processed/test_data")
    
    if data_path.exists():
        dataset = load_from_disk(data_path)
        print(f"✅ 成功加载{len(dataset)}个样本")
        
        # 分割数据
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        print(f"训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}")
    else:
        print("❌ 找不到预处理数据，请先运行 fix_data_format.py")
        return False
    
    # 4. 配置训练参数
    print("\n⚙️  配置训练参数...")
    training_args = TrainingArguments(
        output_dir="./results/quick_test",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=device == "cuda",
        report_to="none",  # 不使用wandb
    )
    
    # 5. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 6. 创建训练器
    print("\n🏋️  创建训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 7. 开始训练
    print("\n🚀 开始训练...")
    try:
        trainer.train()
        print("✅ 训练完成！")
        
        # 8. 保存模型
        trainer.save_model()
        print("💾 模型已保存")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_training_test()
    if success:
        print("\n🎉 快速训练测试成功！")
        print("📋 下一步可以:")
        print("1. 测试模型推理")
        print("2. 运行完整训练")
        print("3. 启动Web界面")
    else:
        print("\n❌ 训练测试失败，请检查错误信息")