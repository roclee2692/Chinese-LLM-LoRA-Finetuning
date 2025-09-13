"""
工具函数模块
提供各种辅助功能，包括配置加载、日志设置、模型工具等
"""

import os
import yaml
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=getattr(logging, level.upper()),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log', encoding='utf-8')
        ]
    )


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    return config


def save_config(config: Dict, save_path: str):
    """保存配置文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
            yaml.safe_dump(config, f, ensure_ascii=False, indent=2)
        elif save_path.suffix == '.json':
            json.dump(config, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的配置文件格式: {save_path.suffix}")


def get_model_config(model_type: str) -> Dict:
    """获取模型配置"""
    model_configs = {
        'chatglm3': {
            'model_name': 'THUDM/chatglm3-6b',
            'target_modules': ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
            'max_length': 2048
        },
        'qwen': {
            'model_name': 'Qwen/Qwen-7B-Chat',
            'target_modules': ['c_attn', 'c_proj', 'w1', 'w2'],
            'max_length': 8192
        },
        'baichuan2': {
            'model_name': 'baichuan-inc/Baichuan2-7B-Chat',
            'target_modules': ['W_pack', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            'max_length': 4096
        },
        'yi': {
            'model_name': '01-ai/Yi-6B-Chat',
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            'max_length': 4096
        }
    }
    
    if model_type not in model_configs:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model_configs[model_type]


def create_optimizer_and_scheduler(model, training_args):
    """创建优化器和学习率调度器"""
    # 这里可以根据需要自定义优化器
    # 当前使用transformers的默认实现
    pass


def count_parameters(model) -> Dict[str, int]:
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_percentage': 100 * trainable_params / total_params
    }


def format_number(num: Union[int, float]) -> str:
    """格式化数字显示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def get_gpu_memory_usage() -> Dict[str, float]:
    """获取GPU内存使用情况"""
    if not torch.cuda.is_available():
        return {"error": "CUDA不可用"}
    
    memory_stats = {}
    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        
        memory_stats[device] = {
            "allocated": allocated,
            "cached": cached,
            "total": total,
            "free": total - cached
        }
    
    return memory_stats


def create_model_card(
    model_name: str,
    dataset_name: str,
    training_config: Dict,
    eval_results: Optional[Dict] = None
) -> str:
    """创建模型卡片"""
    card_content = f"""
# {model_name} - Chinese LoRA Fine-tuned

## 模型描述
这是一个使用LoRA技术在中文数据集上微调的大语言模型。

## 训练详情
- **基础模型**: {training_config.get('model', {}).get('model_name', 'Unknown')}
- **数据集**: {dataset_name}
- **训练轮数**: {training_config.get('training', {}).get('num_train_epochs', 'Unknown')}
- **学习率**: {training_config.get('training', {}).get('learning_rate', 'Unknown')}
- **LoRA rank**: {training_config.get('lora', {}).get('r', 'Unknown')}
- **LoRA alpha**: {training_config.get('lora', {}).get('lora_alpha', 'Unknown')}

## 使用方法
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型和分词器
base_model = AutoModelForCausalLM.from_pretrained("base_model_name")
tokenizer = AutoTokenizer.from_pretrained("base_model_name")

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "path_to_lora_adapter")

# 生成文本
inputs = tokenizer("你好", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
"""
    
    if eval_results:
        card_content += f"""
## 评估结果
{json.dumps(eval_results, indent=2, ensure_ascii=False)}
"""
    
    return card_content


def save_model_card(
    model_name: str,
    dataset_name: str,
    training_config: Dict,
    save_path: str,
    eval_results: Optional[Dict] = None
):
    """保存模型卡片"""
    card_content = create_model_card(model_name, dataset_name, training_config, eval_results)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(card_content)


def ensure_dir(path: Union[str, Path]):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """查找最新的检查点"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
    if not checkpoints:
        return None
    
    # 按检查点编号排序
    checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
    return str(checkpoints[-1])


def validate_config(config: Dict) -> bool:
    """验证配置文件"""
    required_keys = ['model', 'training']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必需的键: {key}")
    
    # 验证模型配置
    model_config = config['model']
    if 'model_name' not in model_config:
        raise ValueError("模型配置缺少model_name")
    
    # 验证训练配置
    training_config = config['training']
    required_training_keys = ['output_dir', 'num_train_epochs', 'per_device_train_batch_size']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"训练配置缺少必需的键: {key}")
    
    return True


class ModelSizeEstimator:
    """模型大小估算器"""
    
    @staticmethod
    def estimate_memory_usage(
        model_size: str,
        batch_size: int = 1,
        sequence_length: int = 512,
        precision: str = "float16"
    ) -> Dict[str, float]:
        """估算内存使用量（GB）"""
        
        # 模型参数数量（亿）
        size_map = {
            "6b": 6,
            "7b": 7,
            "13b": 13,
            "30b": 30,
            "65b": 65
        }
        
        if model_size.lower() not in size_map:
            raise ValueError(f"不支持的模型大小: {model_size}")
        
        params_billion = size_map[model_size.lower()]
        
        # 精度字节数
        precision_bytes = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int8": 1,
            "int4": 0.5
        }
        
        bytes_per_param = precision_bytes.get(precision, 2)
        
        # 模型权重内存
        model_memory = params_billion * 1e9 * bytes_per_param / (1024**3)
        
        # 激活值内存（粗略估算）
        activation_memory = batch_size * sequence_length * 4096 * 4 / (1024**3)  # 假设隐藏层大小4096
        
        # 梯度内存（训练时）
        gradient_memory = model_memory
        
        # 优化器状态内存（Adam）
        optimizer_memory = model_memory * 2
        
        return {
            "model_memory": model_memory,
            "activation_memory": activation_memory,
            "gradient_memory": gradient_memory,
            "optimizer_memory": optimizer_memory,
            "total_training_memory": model_memory + activation_memory + gradient_memory + optimizer_memory,
            "total_inference_memory": model_memory + activation_memory
        }