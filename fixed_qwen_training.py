# -*- coding: utf-8 -*-
"""
Fixed Qwen-1.8B LoRA Training Script (Windows Compatible)
Optimized for RTX 4060 with proper encoding handling
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime

# Set environment variables for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add better import handling
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from transformers import DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main training function with proper encoding"""

    # Clear console and show start message (ASCII compatible)
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("Starting Qwen-1.8B LoRA Training")
    print("=" * 60)

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    output_dir = os.path.join(base_dir, "results", "qwen-1.8b-lora")

    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Model setup
    model_name = "Qwen/Qwen-1_8B-Chat"

    print("\nLoading model and tokenizer...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./cache"
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./cache",
        load_in_4bit=True,
    )

    # LoRA configuration optimized for RTX 4060
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create simple training dataset
    train_data = []

    # Add Chinese conversation examples
    examples = [
        {
            "instruction": "请介绍一下人工智能",
            "input": "",
            "output": "人工智能(AI)是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。AI包括机器学习、深度学习、自然语言处理等多个领域。"
        },
        {
            "instruction": "解释什么是机器学习",
            "input": "",
            "output": "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下从数据中学习和改进。通过算法分析大量数据，机器可以识别模式并做出预测。"
        },
        {
            "instruction": "用Python写一个简单的Hello World程序",
            "input": "",
            "output": "```python\nprint('Hello, World!')\n```\n这是最简单的Python程序，使用print函数输出字符串。"
        },
        {
            "instruction": "什么是深度学习",
            "input": "",
            "output": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。它在图像识别、语音识别和自然语言处理等领域取得了突破性进展。"
        }
    ] * 100  # Repeat to create more training data

    for example in examples:
        # Format as instruction
        text = f"### 指令:\n{example['instruction']}\n\n### 回答:\n{example['output']}"

        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        # Create labels (same as input_ids for causal LM)
        inputs["labels"] = inputs["input_ids"].clone()

        train_data.append({
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["labels"].squeeze()
        })

    # Create dataset
    train_dataset = Dataset.from_list(train_data)

    print(f"Training dataset size: {len(train_dataset)}")

    # Training arguments optimized for RTX 4060
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        warmup_steps=50,
        weight_decay=0.01,
        dataloader_num_workers=0,
        report_to=None
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    start_time = time.time()

    try:
        # Start training
        trainer.train()

        # Calculate training time
        training_time = time.time() - start_time

        print(f"\nTraining completed!")
        print(f"Training time: {training_time/60:.1f} minutes")

        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        print(f"Model saved to: {output_dir}")

        # Create training summary
        summary = {
            "model_name": model_name,
            "training_time": training_time,
            "output_dir": output_dir,
            "training_samples": len(train_dataset),
            "epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "lora_rank": lora_config.r,
            "timestamp": datetime.now().isoformat()
        }

        summary_file = os.path.join(output_dir, "training_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Training summary saved to: {summary_file}")

        return True

    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Training failed!")
        print("="*60)
        sys.exit(1)