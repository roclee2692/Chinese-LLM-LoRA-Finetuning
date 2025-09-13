"""
数据预处理模块
用于处理中文指令数据集，支持多种格式的数据转换和清洗
"""

import json
import pandas as pd
import jieba
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器类"""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.instruction_template = "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:\n{output}"
        
    def load_belle_dataset(self, dataset_path: Optional[str] = None) -> Dataset:
        """加载Belle数据集"""
        try:
            if dataset_path:
                dataset = load_dataset('json', data_files=dataset_path)
            else:
                # 使用Hugging Face上的Belle数据集
                dataset = load_dataset("BelleGroup/train_0.5M_CN")
            logger.info(f"成功加载Belle数据集，共{len(dataset['train'])}条数据")
            return dataset['train']
        except Exception as e:
            logger.error(f"加载Belle数据集失败: {e}")
            raise
    
    def load_alpaca_chinese_dataset(self) -> Dataset:
        """加载中文Alpaca数据集"""
        try:
            dataset = load_dataset("shibing624/alpaca-zh")
            logger.info(f"成功加载中文Alpaca数据集，共{len(dataset['train'])}条数据")
            return dataset['train']
        except Exception as e:
            logger.error(f"加载中文Alpaca数据集失败: {e}")
            raise
    
    def load_custom_data(self, file_path: str) -> Dataset:
        """加载自定义数据集"""
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.suffix == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        dataset = Dataset.from_list(data)
        logger.info(f"成功加载自定义数据集，共{len(dataset)}条数据")
        return dataset
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not isinstance(text, str):
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:""''()（）【】《》<>]', '', text)
        
        return text
    
    def format_instruction_data(self, example: Dict) -> Dict:
        """格式化指令数据"""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        # 清洗文本
        instruction = self.clean_text(instruction)
        input_text = self.clean_text(input_text)
        output = self.clean_text(output)
        
        # 格式化为统一模板
        if input_text:
            formatted_text = self.instruction_template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        else:
            formatted_text = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
        
        return {
            'text': formatted_text,
            'instruction': instruction,
            'input': input_text,
            'output': output,
            'length': len(formatted_text)
        }
    
    def filter_by_length(self, example: Dict) -> bool:
        """根据长度过滤数据"""
        return len(example.get('text', '')) <= self.max_length * 2  # 预留一些空间
    
    def tokenize_function(self, examples: Dict, tokenizer) -> Dict:
        """分词函数"""
        texts = examples['text']
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_overflowing_tokens=False,
        )
        
        # 添加labels（用于训练）
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def prepare_dataset(
        self, 
        dataset_name: str = "belle",
        custom_path: Optional[str] = None,
        tokenizer=None,
        train_split: float = 0.9
    ) -> Dict[str, Dataset]:
        """准备训练数据集"""
        
        # 加载数据集
        if dataset_name == "belle":
            dataset = self.load_belle_dataset(custom_path)
        elif dataset_name == "alpaca":
            dataset = self.load_alpaca_chinese_dataset()
        elif dataset_name == "custom":
            if not custom_path:
                raise ValueError("使用自定义数据集时必须提供文件路径")
            dataset = self.load_custom_data(custom_path)
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_name}")
        
        # 格式化数据
        logger.info("正在格式化数据...")
        dataset = dataset.map(
            self.format_instruction_data,
            num_proc=4,
            desc="格式化数据"
        )
        
        # 过滤数据
        logger.info("正在过滤数据...")
        dataset = dataset.filter(
            self.filter_by_length,
            num_proc=4,
            desc="过滤数据"
        )
        
        # 分词（如果提供了tokenizer）
        if tokenizer:
            logger.info("正在分词...")
            dataset = dataset.map(
                lambda examples: self.tokenize_function(examples, tokenizer),
                batched=True,
                num_proc=4,
                desc="分词"
            )
        
        # 划分训练集和验证集
        if train_split < 1.0:
            dataset = dataset.train_test_split(
                train_size=train_split,
                seed=42
            )
            return {
                'train': dataset['train'],
                'validation': dataset['test']
            }
        else:
            return {'train': dataset}
    
    def save_processed_data(self, dataset: Dataset, save_path: str):
        """保存处理后的数据"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(dataset.to_list(), f, ensure_ascii=False, indent=2)
        elif save_path.suffix == '.jsonl':
            with open(save_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            # 保存为Hugging Face Dataset格式
            dataset.save_to_disk(save_path)
        
        logger.info(f"数据已保存到: {save_path}")
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict:
        """获取数据集统计信息"""
        lengths = [len(item['text']) for item in dataset]
        
        stats = {
            'total_samples': len(dataset),
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'median_length': sorted(lengths)[len(lengths) // 2]
        }
        
        logger.info(f"数据集统计信息: {stats}")
        return stats


def main():
    """主函数，用于测试数据处理"""
    processor = DataProcessor(max_length=512)
    
    # 处理Belle数据集示例
    try:
        datasets = processor.prepare_dataset(
            dataset_name="belle",
            train_split=0.9
        )
        
        # 获取统计信息
        train_stats = processor.get_dataset_statistics(datasets['train'])
        val_stats = processor.get_dataset_statistics(datasets['validation'])
        
        print(f"训练集统计: {train_stats}")
        print(f"验证集统计: {val_stats}")
        
        # 保存处理后的数据
        processor.save_processed_data(
            datasets['train'], 
            './data/processed/train_belle.jsonl'
        )
        processor.save_processed_data(
            datasets['validation'], 
            './data/processed/val_belle.jsonl'
        )
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")


if __name__ == "__main__":
    main()