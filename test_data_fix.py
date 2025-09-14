#!/usr/bin/env python3
"""
验证数据修复结果
"""

import json
from pathlib import Path
from datasets import load_from_disk

def test_data_fix():
    """测试数据修复结果"""
    print("🔍 验证数据修复结果...")
    
    # 检查修复后的数据
    data_path = Path("data/processed/test_data")
    
    if not data_path.exists():
        print("❌ 找不到修复后的数据")
        return False
    
    try:
        # 加载数据
        dataset = load_from_disk(data_path)
        print(f"✅ 成功加载 {len(dataset)} 个样本")
        
        # 检查数据结构
        sample = dataset[0]
        print(f"\n📋 数据字段: {list(sample.keys())}")
        
        # 检查必要字段
        required_fields = ['input_ids', 'attention_mask', 'labels']
        missing_fields = [field for field in required_fields if field not in sample]
        
        if missing_fields:
            print(f"❌ 缺少必要字段: {missing_fields}")
            return False
        
        print("✅ 所有必要字段都存在")
        
        # 检查数据类型和长度
        print(f"\n📊 数据统计:")
        print(f"  - input_ids 长度: {len(sample['input_ids'])}")
        print(f"  - attention_mask 长度: {len(sample['attention_mask'])}")
        print(f"  - labels 长度: {len(sample['labels'])}")
        
        # 检查是否有嵌套字典问题
        for i, sample in enumerate(dataset[:3]):
            for key, value in sample.items():
                if isinstance(value, dict):
                    print(f"⚠️  样本 {i} 的字段 '{key}' 仍然是字典: {value}")
                    return False
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    print(f"⚠️  样本 {i} 的字段 '{key}' 包含嵌套字典: {value[0]}")
                    return False
        
        print("✅ 没有发现嵌套字典问题")
        
        # 显示样本内容
        print(f"\n📝 样本内容预览:")
        print(f"input_ids (前10个): {sample['input_ids'][:10]}")
        print(f"attention_mask (前10个): {sample['attention_mask'][:10]}")
        print(f"labels (前10个): {sample['labels'][:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_training_config():
    """创建轻量级训练配置"""
    config = {
        "model": {
            "name": "distilgpt2",
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "device_map": "auto"
        },
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["c_attn", "c_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training": {
            "output_dir": "./results/quick_test",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-4,
            "warmup_steps": 10,
            "logging_steps": 1,
            "save_steps": 100,
            "evaluation_strategy": "steps",
            "eval_steps": 20,
            "max_steps": 50,  # 限制步数进行快速测试
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "report_to": "none"
        }
    }
    
    # 保存配置
    config_path = Path("configs/test_fixed_data.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"💾 已创建测试配置: {config_path}")
    return config_path

if __name__ == "__main__":
    print("🚀 开始验证数据修复结果...")
    
    # 验证数据
    if test_data_fix():
        print("\n🎉 数据修复验证成功！")
        
        # 创建测试配置
        try:
            config_path = create_training_config()
            print(f"\n📋 下一步可以运行:")
            print(f"python src/train.py --config {config_path}")
        except Exception as e:
            print(f"⚠️  创建配置文件失败: {e}")
            print("请手动使用 configs/quick_test.yaml")
        
        print("\n✅ 数据格式问题已解决，可以开始训练了！")
        
    else:
        print("\n❌ 数据验证失败，请检查修复过程")