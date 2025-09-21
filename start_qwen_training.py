#!/usr/bin/env python3
"""
启动 Qwen-1.8B LoRA 训练脚本
使用预设的配置启动实际的大模型训练
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import datetime
import argparse

def check_environment():
    """检查训练环境"""
    print("🔍 检查训练环境...")
    
    # 检查GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 未检测到NVIDIA GPU")
            return False
        print("✅ GPU检查通过")
    except FileNotFoundError:
        print("❌ nvidia-smi 不可用")
        return False
    
    # 检查CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA不可用")
            return False
        print(f"✅ CUDA可用，检测到 {torch.cuda.device_count()} 个GPU")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    return True

def prepare_training_config():
    """准备训练配置"""
    config = {
        "model_name_or_path": "cache/models--Qwen--Qwen-1_8B-Chat",
        "dataset_name": "data/processed",
        "output_dir": "results/models/qwen-1.8b-lora-chat",
        "logging_dir": "results/logs/qwen-1.8b-lora-chat",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "max_seq_length": 512,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 100,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "use_lora": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM"
    }
    
    return config

def create_training_script(config):
    """创建训练脚本"""
    script_content = f'''#!/usr/bin/env python3
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
    model_path = "{config['model_name_or_path']}"
    
    print(f"🚀 加载模型: {{model_path}}")
    
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
        r={config['lora_r']},
        lora_alpha={config['lora_alpha']},
        lora_dropout={config['lora_dropout']},
        target_modules={config['target_modules']},
        task_type=TaskType.{config['task_type']}
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """准备训练数据"""
    # 使用示例数据
    train_data = [
        {{"instruction": "你好", "output": "你好！我是Qwen，很高兴为您服务。有什么我可以帮助您的吗？"}},
        {{"instruction": "请介绍一下你自己", "output": "我是Qwen，一个由阿里云开发的大型语言模型。我可以帮助您回答问题、创作内容、翻译文本等多种任务。"}},
        {{"instruction": "什么是人工智能？", "output": "人工智能（AI）是指让机器模拟人类智能的技术，包括学习、推理、感知、语言理解等能力。"}},
        {{"instruction": "请解释机器学习", "output": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式，无需明确编程就能做出预测或决策。"}},
        {{"instruction": "深度学习是什么？", "output": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑处理信息的方式，特别擅长处理图像、语音和文本等复杂数据。"}}
    ]
    
    def format_instruction(sample):
        return f"<|im_start|>system\\n你是一个有用的AI助手。<|im_end|>\\n<|im_start|>user\\n{{sample['instruction']}}<|im_end|>\\n<|im_start|>assistant\\n{{sample['output']}}<|im_end|>"
    
    def tokenize_function(examples):
        texts = [format_instruction(ex) for ex in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length={config['max_seq_length']},
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
    print(f"⏰ 训练开始时间: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    
    # 设置输出目录
    output_dir = "{config['output_dir']}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("{config['logging_dir']}", exist_ok=True)
    
    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer()
    
    # 准备数据集
    train_dataset = prepare_dataset(tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size={config['per_device_train_batch_size']},
        gradient_accumulation_steps={config['gradient_accumulation_steps']},
        num_train_epochs={config['num_train_epochs']},
        learning_rate={config['learning_rate']},
        warmup_ratio={config['warmup_ratio']},
        weight_decay={config['weight_decay']},
        logging_dir="{config['logging_dir']}",
        logging_steps={config['logging_steps']},
        save_steps={config['save_steps']},
        save_total_limit={config['save_total_limit']},
        load_best_model_at_end={config['load_best_model_at_end']},
        dataloader_num_workers={config['dataloader_num_workers']},
        remove_unused_columns={config['remove_unused_columns']},
        optim="{config['optim']}",
        lr_scheduler_type="{config['lr_scheduler_type']}",
        bf16={str(config['bf16']).lower()},
        tf32={str(config['tf32']).lower()},
        gradient_checkpointing={str(config['gradient_checkpointing']).lower()},
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
    print(f"📁 模型保存在: {{output_dir}}")

if __name__ == "__main__":
    main()
'''
    
    return script_content

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动 Qwen-1.8B LoRA 训练")
    parser.add_argument("--dry-run", action="store_true", help="只生成脚本不执行训练")
    parser.add_argument("--monitor", action="store_true", help="训练后启动监控")
    args = parser.parse_args()
    
    print("🚀 Qwen-1.8B LoRA 训练启动器")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请确保GPU和CUDA正常工作")
        return
    
    # 准备配置
    config = prepare_training_config()
    print("✅ 训练配置准备完成")
    
    # 创建训练脚本
    script_content = create_training_script(config)
    script_path = Path("train_qwen_lora.py")
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 训练脚本已生成: {script_path}")
    
    # 保存配置
    config_path = Path("configs/qwen_training_config.json")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 配置文件已保存: {config_path}")
    
    if args.dry_run:
        print("🔍 干运行模式 - 脚本已生成但不会执行训练")
        print(f"要开始训练，请运行: python {script_path}")
        return
    
    # 执行训练
    print("\n🚀 开始执行训练...")
    print("⚠️  注意: 这个训练可能需要较长时间，请确保有足够的GPU内存")
    
    # 确认是否继续
    response = input("是否继续执行训练？(y/N): ")
    if response.lower() != 'y':
        print("❌ 训练已取消")
        return
    
    try:
        # 使用虚拟环境中的Python执行训练脚本
        python_exe = Path("llm-lora/Scripts/python.exe")
        if python_exe.exists():
            cmd = [str(python_exe), str(script_path)]
        else:
            cmd = ["python", str(script_path)]
        
        print(f"📝 执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print("✅ 训练完成！")
        
        if args.monitor:
            print("\n🔍 启动监控...")
            monitor_cmd = [str(python_exe), "simple_monitor.py"] if python_exe.exists() else ["python", "simple_monitor.py"]
            subprocess.run(monitor_cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")

if __name__ == "__main__":
    main()