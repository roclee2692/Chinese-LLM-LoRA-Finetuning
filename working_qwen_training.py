"""
Qwen-1.8B LoRA Training Script - Windows Compatible
No emoji or special characters
"""

import os
import sys
import torch
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
    print("Starting Qwen-1.8B LoRA Training")
    print("=" * 60)

    # Clear Windows encoding issues
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    # Setup paths
    model_path = "cache/models--Qwen--Qwen-1_8B-Chat/snapshots/1d0f68de57b88cfde81f3c3e537f24464d889081"
    output_dir = "results/models/qwen-1.8b-lora-working"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {model_path}")
    print(f"Output directory: {output_dir}")

    # Check if model exists
    if not os.path.exists(model_path):
        print("ERROR: Model path does not exist!")
        print("Downloading model first...")
        model_path = "Qwen/Qwen-1_8B-Chat"

    # Configure quantization for RTX 4060
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
        cache_dir="./cache"
    )

    # Fix padding token issue for Qwen
    if tokenizer.pad_token is None:
        # For Qwen, add a proper pad token
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        print(f"Added pad token: {tokenizer.pad_token}")

    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

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

    # Resize model embeddings if we added new tokens
    if tokenizer.pad_token == '<|pad|>':
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to: {len(tokenizer)}")

    # Check model structure for LoRA target modules
    print("Model structure (first few layers):")
    for name, module in model.named_modules():
        if any(target in name for target in ['attn', 'proj', 'linear']):
            print(f"  {name}: {type(module)}")
            if len(name.split('.')) <= 4:  # Only show top-level structure
                break

    # LoRA configuration - using common Qwen target modules
    # Based on inspection, Qwen uses different naming
    lora_config = LoraConfig(
        r=8,  # Lower rank for stability
        lora_alpha=16,
        lora_dropout=0.1,
        # Common target modules for Qwen models
        target_modules=["c_attn", "c_proj", "w1", "w2"],  # Standard names
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )

    print("Applying LoRA...")
    try:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except ValueError as e:
        print(f"LoRA target module error: {e}")
        print("Trying alternative target modules...")

        # Alternative target modules for different Qwen versions
        lora_config.target_modules = ["attn"]  # Simplest option
        try:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ValueError as e2:
            print(f"Still error: {e2}")
            print("Using automatic target detection...")
            # Let's find all linear layers
            target_modules = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    layer_name = name.split('.')[-1]
                    if layer_name not in target_modules:
                        target_modules.append(layer_name)

            print(f"Found linear layers: {target_modules[:10]}")
            lora_config.target_modules = target_modules[:4]  # Use first 4
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    # Create simple training dataset
    print("Preparing dataset...")
    train_data = []

    # Simple Chinese examples
    examples = [
        {"input": "Hello", "output": "Hello! How can I help you?"},
        {"input": "What is AI?", "output": "AI stands for Artificial Intelligence."},
        {"input": "Explain machine learning", "output": "Machine learning is a subset of AI."},
        {"input": "What is Python?", "output": "Python is a programming language."},
    ] * 50  # Repeat for more data

    for example in examples:
        # Format for Qwen
        text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"

        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=256,  # Shorter for stability
            padding="max_length",
            return_tensors="pt"
        )

        train_data.append({
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["input_ids"].squeeze()
        })

    train_dataset = Dataset.from_list(train_data)
    print(f"Dataset size: {len(train_dataset)}")

    # Training arguments optimized for RTX 4060
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for testing
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  # Reduced
        learning_rate=1e-4,  # Conservative
        warmup_steps=10,
        logging_steps=5,
        save_steps=25,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        bf16=True,
        gradient_checkpointing=True,
        report_to=None,
        optim="adamw_torch"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
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
        # Train
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

        # Save training summary
        summary = {
            "model": "Qwen-1.8B-Chat",
            "lora_rank": lora_config.r,
            "training_samples": len(train_dataset),
            "duration_seconds": duration.total_seconds(),
            "output_dir": output_dir,
            "timestamp": datetime.now().isoformat()
        }

        with open(os.path.join(output_dir, "training_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        return True

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print("\nFinal result:", "SUCCESS" if success else "FAILED")