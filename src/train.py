"""
训练脚本
支持多种中文大语言模型的LoRA微调
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import wandb

from data_preprocessing import DataProcessor
from utils import (
    setup_logging,
    load_config,
    get_model_config,
    create_optimizer_and_scheduler
)

logger = logging.getLogger(__name__)


class LoRATrainer:
    """LoRA训练器类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.datasets = {}
        
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        model_config = self.config['model']
        model_name = model_config['model_name']
        
        logger.info(f"正在加载模型: {model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=model_config.get('trust_remote_code', True),
            cache_dir=model_config.get('cache_dir', './cache')
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        quantization_config = None
        if self.config.get('quantization', {}).get('load_in_4bit', False):
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=model_config.get('trust_remote_code', True),
            cache_dir=model_config.get('cache_dir', './cache'),
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if quantization_config else torch.bfloat16,
            device_map="auto"
        )
        
        # 准备模型用于量化训练
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("模型和分词器加载完成")
    
    def setup_lora(self):
        """设置LoRA"""
        lora_config = self.config['lora']
        model_type = self.config['model']['model_type']
        
        # 获取目标模块
        target_modules = lora_config['target_modules']
        if isinstance(target_modules, dict):
            target_modules = target_modules.get(model_type, list(target_modules.values())[0])
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=target_modules,
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias']
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA配置完成")
    
    def prepare_datasets(self):
        """准备数据集"""
        data_config = self.config.get('data', {})

        processor = DataProcessor(
            max_length=data_config.get('max_seq_length', 512)
        )

        # 准备数据集
        dataset_name = data_config.get('dataset_name', 'belle')
        train_file = data_config.get('train_file')
        validation_file = data_config.get('validation_file')

        if dataset_name == "custom" and train_file:
            # 使用自定义数据文件
            logger.info(f"加载自定义训练数据: {train_file}")

            # 加载训练集
            train_dataset = processor.load_custom_data(train_file)
            train_dataset = train_dataset.map(processor.format_instruction_data, num_proc=1)

            # 如果有验证文件，加载验证集
            if validation_file and os.path.exists(validation_file):
                logger.info(f"加载验证数据: {validation_file}")
                val_dataset = processor.load_custom_data(validation_file)
                val_dataset = val_dataset.map(processor.format_instruction_data, num_proc=1)

                self.datasets = {
                    'train': train_dataset,
                    'validation': val_dataset
                }
            else:
                self.datasets = {'train': train_dataset}

            # 分词
            if self.tokenizer:
                logger.info("正在对数据进行分词...")
                for key in self.datasets:
                    self.datasets[key] = self.datasets[key].map(
                        lambda examples: processor.tokenize_function(examples, self.tokenizer),
                        batched=True,
                        num_proc=1
                    )
            # 分词后清理列，避免collator收到非tensor列
            if self.tokenizer:
                for key in self.datasets:
                    ds_tok = self.datasets[key]
                    keep_cols = [c for c in ["input_ids", "attention_mask", "labels"] if c in ds_tok.column_names]
                    if keep_cols:
                        drop_cols = [c for c in ds_tok.column_names if c not in keep_cols]
                        if drop_cols:
                            ds_tok = ds_tok.remove_columns(drop_cols)
                    self.datasets[key] = ds_tok

        else:
            # 使用原有的数据集加载方式
            self.datasets = processor.prepare_dataset(
                dataset_name=dataset_name,
                custom_path=train_file,
                tokenizer=self.tokenizer,
                train_split=0.9
            )

        logger.info(f"数据集准备完成，训练集: {len(self.datasets['train'])}条")
        if 'validation' in self.datasets:
            logger.info(f"验证集: {len(self.datasets['validation'])}条")
    
    def create_trainer(self) -> Trainer:
        """创建训练器"""
        training_config = self.config['training']
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            warmup_ratio=training_config.get('warmup_ratio', 0.03),
            logging_steps=training_config.get('logging_steps', 100),
            save_steps=training_config.get('save_steps', 500),
            eval_steps=training_config.get('eval_steps', 500),
            eval_strategy="steps" if 'validation' in self.datasets else "no",
            save_strategy="steps",
            load_best_model_at_end=True if 'validation' in self.datasets else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=training_config.get('bf16', True),
            fp16=training_config.get('fp16', False),
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            remove_unused_columns=True,
            report_to=training_config.get('report_to', []),
            seed=training_config.get('seed', 42),
            data_seed=training_config.get('data_seed', 42),
            group_by_length=training_config.get('group_by_length', True)
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=True
        )
        
        # 回调函数
        callbacks = []
        if 'validation' in self.datasets:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            ))
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets.get('validation'),
            callbacks=callbacks
        )
        
        return trainer
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        # 设置W&B
        wandb_config = self.config.get('wandb', {})
        if wandb_config and 'tensorboard' not in self.config['training'].get('report_to', []):
            wandb.init(
                project=wandb_config.get('project', 'chinese-llm-lora'),
                name=wandb_config.get('name'),
                config=self.config,
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', '')
            )
        
        # 创建训练器
        trainer = self.create_trainer()
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        trainer.save_state()
        
        # 保存分词器
        self.tokenizer.save_pretrained(self.config['training']['output_dir'])
        
        logger.info("训练完成")
        
        # 关闭W&B
        if wandb.run:
            wandb.finish()
    
    def evaluate(self):
        """评估模型"""
        if 'validation' not in self.datasets:
            logger.warning("没有验证集，跳过评估")
            return
        
        trainer = self.create_trainer()
        eval_results = trainer.evaluate()
        
        logger.info(f"评估结果: {eval_results}")
        return eval_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中文大语言模型LoRA微调")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='从检查点恢复训练'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='仅评估模型'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = LoRATrainer(config)
    
    # 设置模型和分词器
    trainer.setup_model_and_tokenizer()
    
    # 设置LoRA
    trainer.setup_lora()
    
    # 准备数据集
    trainer.prepare_datasets()
    
    if args.eval_only:
        # 仅评估
        trainer.evaluate()
    else:
        # 训练
        trainer.train()
        
        # 训练后评估
        if 'validation' in trainer.datasets:
            trainer.evaluate()


if __name__ == "__main__":
    main()
