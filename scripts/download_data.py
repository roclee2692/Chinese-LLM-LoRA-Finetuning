#!/usr/bin/env python3
"""
数据下载脚本
用于下载各种中文指令数据集
"""

import os
import json
import requests
import argparse
from pathlib import Path
from typing import Dict, List
import logging

# 尝试导入相关库
try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"某些库未安装: {e}")
    print("请安装: pip install datasets huggingface_hub")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 预定义的数据集配置
        self.datasets_config = {
            "belle": {
                "huggingface_name": "BelleGroup/train_0.5M_CN",
                "description": "Belle中文指令数据集（50万条）",
                "size": "约500MB",
                "format": "jsonl"
            },
            "belle_1m": {
                "huggingface_name": "BelleGroup/train_1M_CN",
                "description": "Belle中文指令数据集（100万条）",
                "size": "约1GB",
                "format": "jsonl"
            },
            "alpaca_chinese": {
                "huggingface_name": "shibing624/alpaca-zh",
                "description": "中文Alpaca数据集",
                "size": "约100MB",
                "format": "json"
            },
            "firefly": {
                "huggingface_name": "YeungNLP/firefly-train-1.1M",
                "description": "流萤(Firefly)中文对话数据集",
                "size": "约800MB",
                "format": "jsonl"
            },
            "cot_chinese": {
                "huggingface_name": "QingyiSi/Alpaca-CoT",
                "description": "中文思维链数据集",
                "size": "约300MB",
                "format": "json"
            }
        }
    
    def list_available_datasets(self):
        """列出可用的数据集"""
        print("可用的数据集:")
        print("-" * 80)
        for name, config in self.datasets_config.items():
            print(f"名称: {name}")
            print(f"描述: {config['description']}")
            print(f"大小: {config['size']}")
            print(f"格式: {config['format']}")
            print("-" * 80)
    
    def download_belle_dataset(self, dataset_name: str = "belle") -> str:
        """下载Belle数据集"""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        config = self.datasets_config[dataset_name]
        huggingface_name = config["huggingface_name"]
        
        logger.info(f"正在下载 {dataset_name} 数据集...")
        logger.info(f"来源: {huggingface_name}")
        
        try:
            # 使用datasets库下载
            dataset = load_dataset(huggingface_name)
            
            # 保存到本地
            output_file = self.data_dir / f"{dataset_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset['train']:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"数据集下载完成: {output_file}")
            logger.info(f"数据条数: {len(dataset['train'])}")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise
    
    def download_custom_dataset(self, url: str, filename: str) -> str:
        """下载自定义数据集"""
        output_file = self.data_dir / filename
        
        logger.info(f"正在从 {url} 下载数据集...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r下载进度: {progress:.1f}%", end="")
            
            print(f"\n数据集下载完成: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            if output_file.exists():
                output_file.unlink()
            raise
    
    def verify_dataset(self, file_path: str) -> Dict:
        """验证数据集格式"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        logger.info(f"正在验证数据集: {file_path}")
        
        sample_count = 0
        valid_count = 0
        errors = []
        
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        sample_count += 1
                        try:
                            item = json.loads(line.strip())
                            
                            # 检查必要字段
                            if self._validate_item(item):
                                valid_count += 1
                            else:
                                errors.append(f"行 {line_num}: 缺少必要字段")
                                
                        except json.JSONDecodeError as e:
                            errors.append(f"行 {line_num}: JSON解析错误 - {e}")
                        
                        # 只检查前1000行
                        if sample_count >= 1000:
                            break
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for i, item in enumerate(data[:1000]):
                            sample_count += 1
                            if self._validate_item(item):
                                valid_count += 1
                            else:
                                errors.append(f"项目 {i}: 缺少必要字段")
            
            validation_result = {
                "file_path": str(file_path),
                "total_checked": sample_count,
                "valid_items": valid_count,
                "invalid_items": sample_count - valid_count,
                "validity_rate": valid_count / sample_count if sample_count > 0 else 0,
                "errors": errors[:10]  # 只显示前10个错误
            }
            
            logger.info(f"验证完成: {valid_count}/{sample_count} 条数据有效")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            raise
    
    def _validate_item(self, item: Dict) -> bool:
        """验证单个数据项"""
        # 检查基本字段
        required_fields = ['instruction', 'output']
        
        for field in required_fields:
            if field not in item or not item[field]:
                return False
        
        # 检查字段类型
        if not isinstance(item['instruction'], str) or not isinstance(item['output'], str):
            return False
        
        return True
    
    def create_sample_dataset(self, size: int = 100) -> str:
        """创建示例数据集"""
        sample_data = []
        
        # 预定义的示例数据
        examples = [
            {
                "instruction": "请介绍一下中国的首都。",
                "input": "",
                "output": "中国的首都是北京。北京是中华人民共和国的政治、文化中心，也是重要的国际都市。"
            },
            {
                "instruction": "解释什么是机器学习。",
                "input": "",
                "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。通过算法和统计模型，机器学习系统可以从数据中识别模式并做出预测。"
            },
            {
                "instruction": "将以下文本翻译成英文。",
                "input": "今天天气很好。",
                "output": "The weather is very nice today."
            },
            {
                "instruction": "请写一首关于春天的诗。",
                "input": "",
                "output": "春风拂面花满枝，\n绿柳摇曳鸟儿啼。\n阳光明媚照大地，\n万物复苏展生机。"
            }
        ]
        
        # 重复示例数据直到达到指定数量
        for i in range(size):
            example = examples[i % len(examples)].copy()
            example['id'] = i
            sample_data.append(example)
        
        # 保存示例数据集
        output_file = self.data_dir / "sample_dataset.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"示例数据集创建完成: {output_file}")
        logger.info(f"数据条数: {len(sample_data)}")
        
        return str(output_file)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中文数据集下载工具")
    parser.add_argument('--dataset', type=str, help='要下载的数据集名称')
    parser.add_argument('--list', action='store_true', help='列出可用的数据集')
    parser.add_argument('--url', type=str, help='自定义数据集URL')
    parser.add_argument('--filename', type=str, help='保存的文件名')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='数据保存目录')
    parser.add_argument('--verify', type=str, help='验证指定的数据集文件')
    parser.add_argument('--sample', type=int, help='创建示例数据集（指定数据条数）')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.list:
        downloader.list_available_datasets()
    
    elif args.dataset:
        try:
            downloader.download_belle_dataset(args.dataset)
        except Exception as e:
            logger.error(f"下载失败: {e}")
    
    elif args.url and args.filename:
        try:
            downloader.download_custom_dataset(args.url, args.filename)
        except Exception as e:
            logger.error(f"下载失败: {e}")
    
    elif args.verify:
        try:
            result = downloader.verify_dataset(args.verify)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"验证失败: {e}")
    
    elif args.sample:
        try:
            downloader.create_sample_dataset(args.sample)
        except Exception as e:
            logger.error(f"创建示例数据集失败: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()