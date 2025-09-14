#!/usr/bin/env python3
"""
数据格式修复和测试脚本
运行: python fix_data_format.py
"""

import sys
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# 添加src到路径
sys.path.append(str(Path(__file__).parent / "src"))
from data_preprocessing import DataProcessor

def test_data_format():
    """测试数据格式是否正确"""
    print("🔧 开始数据格式诊断和修复...")
    
    # 初始化数据处理器
    processor = DataProcessor(
        max_length=256,
        fix_format=True
    )
    
    try:
        # 1. 创建本地测试数据
        print("\n📥 创建本地测试数据...")
        test_data = [
            {
                "instruction": "请介绍人工智能",
                "input": "",
                "output": "人工智能（AI）是计算机科学的一个分支，旨在创造能够执行通常需要人类智能才能完成的任务的机器。"
            },
            {
                "instruction": "翻译以下英文",
                "input": "Hello world",
                "output": "你好世界"
            },
            {
                "instruction": "解释什么是深度学习",
                "input": "",
                "output": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的高级抽象。"
            },
            {
                "instruction": "计算数学题",
                "input": "2 + 3 = ?",
                "output": "2 + 3 = 5"
            },
            {
                "instruction": "写一首关于春天的诗",
                "input": "",
                "output": "春风吹绿江南岸，万物复苏展新颜。桃花朵朵迎朝阳，燕子归来报春还。"
            }
        ] * 10  # 重复10次以创建更多数据
        
        # 创建数据集
        from datasets import Dataset
        dataset = Dataset.from_list(test_data)
        
        print(f"✅ 成功创建{len(dataset)}个样本")
        
        # 2. 检查原始数据格式
        print("\n🔍 检查原始数据格式...")
        sample = dataset[0]
        print(f"样本字段: {list(sample.keys())}")
        print(f"样本内容预览: {str(sample)[:200]}...")
        
        # 3. 测试格式化函数
        print("\n🔧 测试数据格式化...")
        formatted_sample = processor.format_instruction_data(sample)
        print(f"格式化后字段: {list(formatted_sample.keys())}")
        print(f"格式化文本长度: {len(formatted_sample['text'])}")
        print(f"格式化文本预览: {formatted_sample['text'][:300]}...")
        
        # 4. 批量格式化测试
        print("\n⚡ 测试批量格式化...")
        formatted_dataset = dataset.map(
            processor.format_instruction_data,
            remove_columns=dataset.column_names,
            desc="格式化数据"
        )
        
        # 5. 过滤长度
        print("\n📏 测试长度过滤...")
        filtered_dataset = formatted_dataset.filter(
            processor.filter_by_length,
            desc="过滤长度"
        )
        
        print(f"过滤前: {len(formatted_dataset)}, 过滤后: {len(filtered_dataset)}")
        
        # 6. 测试分词
        print("\n🔤 测试分词...")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # 测试单个样本分词
        test_sample = {'text': [filtered_dataset[0]['text']]}
        tokenized = processor.tokenize_function(test_sample, tokenizer)
        print(f"分词结果字段: {list(tokenized.keys())}")
        print(f"input_ids长度: {len(tokenized['input_ids'][0])}")
        print(f"labels长度: {len(tokenized['labels'][0])}")
        
        # 7. 批量分词测试
        print("\n⚡ 测试批量分词...")
        tokenized_dataset = filtered_dataset.map(
            lambda examples: processor.tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=filtered_dataset.column_names,
            desc="分词处理"
        )
        
        print(f"✅ 成功分词{len(tokenized_dataset)}个样本")
        
        # 8. 保存修复后的数据
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存训练和验证数据
        if len(tokenized_dataset) >= 10:
            # 分割数据：80%训练，20%验证
            train_size = int(0.8 * len(tokenized_dataset))
            train_data = tokenized_dataset.select(range(train_size))
            val_data = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
            
            # 保存为JSONL格式
            train_data.to_json(output_dir / "train.jsonl")
            val_data.to_json(output_dir / "val.jsonl")
            
            print(f"\n💾 已保存训练数据: {len(train_data)}个样本 -> {output_dir / 'train.jsonl'}")
            print(f"💾 已保存验证数据: {len(val_data)}个样本 -> {output_dir / 'val.jsonl'}")
        else:
            # 如果数据太少，全部用作训练数据
            tokenized_dataset.to_json(output_dir / "train.jsonl")
            tokenized_dataset.to_json(output_dir / "val.jsonl")  # 复制一份作为验证数据
            
            print(f"\n💾 数据较少，全部用作训练: {len(tokenized_dataset)}个样本")
        
        # 保存少量样本用于测试
        test_data = tokenized_dataset.select(range(min(20, len(tokenized_dataset))))
        test_data.save_to_disk(output_dir / "test_data")
        
        print(f"💾 已保存测试数据到: {output_dir / 'test_data'}")
        
        print("\n🎉 数据格式修复完成！")
        print("📋 修复总结:")
        print(f"  - 原始样本: {len(dataset)}")
        print(f"  - 格式化后: {len(formatted_dataset)}")
        print(f"  - 长度过滤后: {len(filtered_dataset)}")
        print(f"  - 分词后: {len(tokenized_dataset)}")
        if len(tokenized_dataset) >= 10:
            train_size = int(0.8 * len(tokenized_dataset))
            print(f"  - 训练数据: {train_size}")
            print(f"  - 验证数据: {len(tokenized_dataset) - train_size}")
        print(f"  - 测试数据: {len(test_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据格式修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_test_data():
    """创建简单的测试数据"""
    print("\n🔧 创建简单测试数据...")
    
    test_data = [
        {
            "instruction": "请介绍人工智能",
            "input": "",
            "output": "人工智能（AI）是计算机科学的一个分支，旨在创造能够执行通常需要人类智能才能完成的任务的机器。"
        },
        {
            "instruction": "翻译以下英文",
            "input": "Hello world",
            "output": "你好世界"
        },
        {
            "instruction": "解释什么是深度学习",
            "input": "",
            "output": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的高级抽象。"
        }
    ]
    
    # 保存为JSON文件
    output_path = Path("data/raw/simple_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 创建简单测试数据: {output_path}")
    return output_path

if __name__ == "__main__":
    print("🚀 开始数据格式诊断和修复...")
    
    # 首先尝试修复网络数据
    success = test_data_format()
    
    if not success:
        print("\n⚠️ 网络数据处理失败，创建本地测试数据...")
        simple_data_path = create_simple_test_data()
        print(f"✅ 可以使用本地数据进行测试: {simple_data_path}")
    
    print("\n📋 下一步操作:")
    print("1. python src/train.py --config configs/quick_test.yaml")
    print("2. 检查训练是否能正常开始")
    print("3. 如果仍有问题，使用本地简单数据测试")