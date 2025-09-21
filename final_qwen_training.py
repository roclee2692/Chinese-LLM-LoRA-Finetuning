"""
Final Qwen-1.8B LoRA Training Script
Simplified approach - no padding, use DataCollatorForSeq2Seq
"""

import os
import sys
import torch

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
from datetime import datetime

def main():
    print("=" * 60)
    print("Final Qwen-1.8B LoRA Training Script")
    print("=" * 60)

    # Setup paths
    model_path = "cache/models--Qwen--Qwen-1_8B-Chat/snapshots/1d0f68de57b88cfde81f3c3e537f24464d889081"
    output_dir = "results/models/qwen-1.8b-lora-final"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")

    # Check if model exists locally
    if not os.path.exists(model_path):
        print("Local model not found, using online model...")
        model_path = "Qwen/Qwen-1_8B-Chat"

    # Load tokenizer (simple approach)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir="./cache"
    )

    # Set pad token to eos token (simple and safe)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Vocab size: {len(tokenizer)}")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir="./cache"
    )

    # LoRA config - using correct Qwen target modules
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # Only target attention layers for simplicity
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create simple dataset (no padding, let DataCollator handle it)
    print("Creating dataset...")
    examples = [
        {"text": "Hello, how are you?", "response": "I'm doing well, thank you!"},
        {"text": "What is AI?", "response": "AI is artificial intelligence."},
        {"text": "Explain Python", "response": "Python is a programming language."},
        {"text": "What is machine learning?", "response": "ML is a subset of AI."},
    ] * 50  # 200 examples

    train_data = []
    for ex in examples:
        # Simple format - just concat
        full_text = f"User: {ex['text']}\nAssistant: {ex['response']}"

        # Tokenize without padding
        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=200,
            return_tensors="pt"
        )

        train_data.append({
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["input_ids"].squeeze()
        })

    train_dataset = Dataset.from_list(train_data)
    print(f"Dataset size: {len(train_dataset)}")

    # Training args (very conservative)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=5,
        logging_steps=5,
        save_steps=20,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=True,
        report_to=None,
        optim="adamw_torch"
    )

    # Use DataCollatorForSeq2Seq which handles padding properly
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

    print("Starting training...")
    start_time = datetime.now()

    try:
        # Train!
        trainer.train()

        # Save
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        end_time = datetime.now()
        duration = end_time - start_time

        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Duration: {duration}")
        print(f"Model saved to: {output_dir}")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("SUCCESS: Training completed!")
    else:
        print("FAILED: Training had errors!")
        sys.exit(1)