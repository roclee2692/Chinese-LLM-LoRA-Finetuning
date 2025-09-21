"""
Ultimate Qwen-1.8B LoRA Training Script
Final solution with manual pad token handling
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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
from datetime import datetime

def main():
    print("=" * 60)
    print("ULTIMATE Qwen-1.8B LoRA Training Script")
    print("Final solution for all issues")
    print("=" * 60)

    # Setup paths
    model_path = "cache/models--Qwen--Qwen-1_8B-Chat/snapshots/1d0f68de57b88cfde81f3c3e537f24464d889081"
    output_dir = "results/models/qwen-1.8b-lora-ultimate"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")

    # Check if model exists locally
    if not os.path.exists(model_path):
        print("Using online model...")
        model_path = "Qwen/Qwen-1_8B-Chat"

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir="./cache"
    )

    # Debug tokenizer info
    print(f"Original vocab size: {len(tokenizer)}")
    print(f"Original EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"Original PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # Fix tokenizer by manually setting tokens
    # Qwen uses <|endoftext|> as EOS token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = 151643  # Known EOS token ID for Qwen

    # Set pad token to EOS token (safest approach)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Fixed EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"Fixed PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

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

    # LoRA config
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # Simple target for Qwen
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset - NO PADDING in tokenization, let collator handle it
    print("Creating dataset...")
    examples = [
        "Hello, how are you today?",
        "What is artificial intelligence?",
        "Python is a programming language.",
        "Machine learning is exciting!",
        "Deep learning uses neural networks.",
    ] * 40  # 200 examples

    train_data = []
    for text in examples:
        # Simple text - no special formatting to avoid issues
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=100,  # Very short to avoid issues
            return_tensors="pt"
        )

        train_data.append({
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["input_ids"].squeeze()
        })

    train_dataset = Dataset.from_list(train_data)
    print(f"Dataset size: {len(train_dataset)}")

    # Training args - very conservative
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=2,
        logging_steps=2,
        save_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=True,
        report_to=None,
        optim="adamw_torch",
        max_steps=10  # Very short training for testing
    )

    # Use DataCollatorForLanguageModeling - simpler than Seq2Seq
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not MLM
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    print(f"Training for {training_args.max_steps} steps")
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
        print("SUCCESS! TRAINING COMPLETED!")
        print(f"Duration: {duration}")
        print(f"Model saved to: {output_dir}")
        print("=" * 60)

        # Save summary
        summary = {
            "status": "SUCCESS",
            "model": "Qwen-1.8B-Chat",
            "lora_rank": lora_config.r,
            "training_samples": len(train_dataset),
            "training_steps": training_args.max_steps,
            "duration_seconds": duration.total_seconds(),
            "output_dir": output_dir,
            "timestamp": datetime.now().isoformat()
        }

        with open(os.path.join(output_dir, "training_success.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        print("Training summary saved!")
        return True

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nFINAL RESULT: SUCCESS!")
        print("Qwen-1.8B LoRA training completed successfully!")
    else:
        print("\nFINAL RESULT: FAILED!")
        sys.exit(1)