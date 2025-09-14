"""
数据预处理模块
用于处理常见指令微调数据集，支持格式修复、转换与清洗
"""

import json
import pandas as pd
import re
import argparse
from pathlib import Path
from typing import Dict, Optional
from datasets import Dataset, load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""

    def __init__(self, max_length: int = 512, fix_format: bool = False):
        self.max_length = max_length
        self.fix_format = fix_format
        self.instruction_template = "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回复:\n{output}"

    @staticmethod
    def _to_str(value) -> str:
        """将任意值安全转为字符串，避免嵌套导致的collator报错"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)

    def _normalize_example_keys(self, example: Dict) -> Dict:
        """兼容不同数据源的字段命名，返回标准键: instruction/input/output"""
        if not isinstance(example, dict):
            return {"instruction": "", "input": "", "output": ""}

        instruction = (
            example.get("instruction")
            or example.get("prompt")
            or example.get("query")
            or example.get("question")
            or example.get("user")
            or ""
        )
        input_text = (
            example.get("input")
            or example.get("context")
            or example.get("history")
            or ""
        )
        output = (
            example.get("output")
            or example.get("response")
            or example.get("answer")
            or example.get("assistant")
            or example.get("target")
            or ""
        )

        # 处理会话结构，如 {conversations: [{role: user, content: ...},{role: assistant, content: ...}]}
        if not instruction and isinstance(example.get("conversations"), list):
            conv = example["conversations"]
            user_turns = [turn.get("content") or turn.get("value") for turn in conv if (turn.get("role") or turn.get("from")) in ("user", "human")]
            asst_turns = [turn.get("content") or turn.get("value") for turn in conv if (turn.get("role") or turn.get("from")) in ("assistant", "gpt")]
            if user_turns:
                instruction = user_turns[0]
                if len(user_turns) > 1:
                    input_text = "\n".join(user_turns[1:])
            if asst_turns:
                output = asst_turns[0]

        return {
            "instruction": self._to_str(instruction),
            "input": self._to_str(input_text),
            "output": self._to_str(output),
        }

    def load_belle_dataset(self, dataset_path: Optional[str] = None) -> Dataset:
        """加载Belle数据集"""
        try:
            if dataset_path:
                dataset = load_dataset('json', data_files=dataset_path)
            else:
                dataset = load_dataset("BelleGroup/train_0.5M_CN")
            logger.info(f"成功加载Belle数据集，共{len(dataset['train'])}条样本")
            return dataset['train']
        except Exception as e:
            logger.error(f"加载Belle数据集失败: {e}")
            raise

    def load_alpaca_chinese_dataset(self) -> Dataset:
        """加载中文Alpaca数据集"""
        try:
            dataset = load_dataset("shibing624/alpaca-zh")
            logger.info(f"成功加载中文Alpaca数据集，共{len(dataset['train'])}条样本")
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
                    if line.strip():
                        data.append(json.loads(line.strip()))
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        dataset = Dataset.from_list(data)
        logger.info(f"成功加载自定义数据集，共{len(dataset)}条样本")
        return dataset

    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not isinstance(text, str):
            return ""

        # 去除多余空白字符
        text = re.sub(r'\s+', ' ', text).strip()

        # 去除控制字符，保留常见中英文与标点（宽松避免误删）
        text = re.sub(r"[\u0000-\u001f]", "", text)
        return text

    def _to_str(self, value) -> str:
        """将任何值转换为字符串，处理嵌套字典/列表"""
        if isinstance(value, dict):
            # 如果是字典，尝试提取主要内容
            if 'text' in value:
                return str(value['text'])
            elif 'content' in value:
                return str(value['content'])
            else:
                # 取第一个字符串值
                for v in value.values():
                    if isinstance(v, str) and v.strip():
                        return v
                return str(value)
        elif isinstance(value, list):
            # 如果是列表，连接所有字符串元素
            return ' '.join(str(item) for item in value if str(item).strip())
        else:
            return str(value) if value is not None else ''

    def _normalize_example_keys(self, example: Dict) -> Dict:
        """规范化样本字段名"""
        # 常见的字段名映射
        key_mapping = {
            'instruction': ['instruction', 'prompt', 'question', 'input_text'],
            'input': ['input', 'context', 'input_context', ''],  
            'output': ['output', 'response', 'answer', 'target', 'output_text']
        }
        
        normalized = {}
        for standard_key, possible_keys in key_mapping.items():
            value = ''
            for key in possible_keys:
                if key in example and example[key]:
                    value = self._to_str(example[key])
                    break
            normalized[standard_key] = value
            
        return normalized

    def format_instruction_data(self, example: Dict) -> Dict:
        """格式化指令数据，修复嵌套并标准化结构"""
        # 规范化字段名和内容
        normalized = self._normalize_example_keys(example)
        
        instruction = self.clean_text(normalized.get('instruction', ''))
        input_text = self.clean_text(normalized.get('input', ''))
        output = self.clean_text(normalized.get('output', ''))

        if input_text:
            formatted_text = self.instruction_template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        else:
            formatted_text = f"### 指令:\n{instruction}\n\n### 回复:\n{output}"

        return {
            'text': formatted_text,
            'instruction': instruction,
            'input': input_text,
            'output': output,
            'length': len(formatted_text)
        }

    def filter_by_length(self, example: Dict) -> bool:
        """按长度过滤样本"""
        return len(example.get('text', '')) <= self.max_length * 2  # 留出一定余量

    def tokenize_function(self, examples: Dict, tokenizer) -> Dict:
        """分词函数，处理批量数据"""
        # 确保输入是列表格式
        texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_overflowing_tokens=False,
        )
        
        # 确保labels存在且格式正确
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
                raise ValueError("使用自定义数据集时需要提供文件路径")
            dataset = self.load_custom_data(custom_path)
        else:
            raise ValueError(f"不支持的数据集名称: {dataset_name}")

        # 格式化样本
        logger.info("正在格式化样本...")
        dataset = dataset.map(
            self.format_instruction_data,
            num_proc=4,
            desc="格式化样本"
        )

        # 过滤样本
        logger.info("正在过滤样本...")
        dataset = dataset.filter(
            self.filter_by_length,
            num_proc=4,
            desc="过滤样本"
        )

        # 分词（如提供tokenizer）
        if tokenizer:
            logger.info("正在分词...")
            dataset = dataset.map(
                lambda examples: self.tokenize_function(examples, tokenizer),
                batched=True,
                num_proc=4,
                desc="分词"
            )
            keep_cols = [c for c in ["input_ids", "attention_mask", "labels"] if c in dataset.column_names]
            if keep_cols:
                drop_cols = [c for c in dataset.column_names if c not in keep_cols]
                if drop_cols:
                    dataset = dataset.remove_columns(drop_cols)

        # 切分
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
            dataset.save_to_disk(save_path)

        logger.info(f"数据已保存到: {save_path}")

    def get_dataset_statistics(self, dataset: Dataset) -> Dict:
        """获取数据集统计信息"""
        lengths = [len(item['text']) for item in dataset]

        stats = {
            'total_samples': len(dataset),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'median_length': sorted(lengths)[len(lengths) // 2] if lengths else 0
        }

        logger.info(f"数据集统计信息: {stats}")
        return stats


def main():
    """命令行入口：格式修复与数据预处理"""
    parser = argparse.ArgumentParser(description="数据预处理与格式修复")
    parser.add_argument("--dataset", type=str, default="belle", choices=["belle", "alpaca", "custom"], help="数据集名称")
    parser.add_argument("--dataset_path", type=str, default=None, help="自定义数据集文件路径(json/jsonl/csv)")
    parser.add_argument("--fix_format", action="store_true", help="启用格式修复与字段拉平")
    parser.add_argument("--max_samples", type=int, default=None, help="最多采样条数")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--train_split", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--out_train", type=str, default="./data/processed/train.jsonl", help="训练集输出路径")
    parser.add_argument("--out_val", type=str, default="./data/processed/val.jsonl", help="验证集输出路径")

    args = parser.parse_args()

    processor = DataProcessor(max_length=args.max_length, fix_format=args.fix_format)

    try:
        datasets = processor.prepare_dataset(
            dataset_name=args.dataset,
            custom_path=args.dataset_path,
            tokenizer=None,
            train_split=args.train_split
        )

        # 可选下采样
        if args.max_samples and args.max_samples > 0:
            for key in list(datasets.keys()):
                ds = datasets[key]
                if len(ds) > args.max_samples:
                    datasets[key] = ds.select(range(args.max_samples))

        # 统计
        for split, ds in datasets.items():
            stats = processor.get_dataset_statistics(ds)
            logger.info(f"{split} 集统计: {stats}")

        # 保存
        if 'train' in datasets:
            processor.save_processed_data(datasets['train'], args.out_train)
        if 'validation' in datasets:
            processor.save_processed_data(datasets['validation'], args.out_val)

    except Exception as e:
        logger.error(f"数据处理失败: {e}")


if __name__ == "__main__":
    main()

