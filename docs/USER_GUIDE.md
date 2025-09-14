# 📖 用户使用指南

欢迎使用 Chinese-LLM-LoRA-Finetuning 框架！本指南将详细介绍如何使用框架的各项功能。

## 📋 目录

- [快速入门](#-快速入门)
- [配置详解](#-配置详解)
- [数据准备](#-数据准备)
- [模型训练](#-模型训练)
- [Web界面使用](#-web界面使用)
- [模型评估](#-模型评估)
- [故障排除](#-故障排除)
- [高级功能](#-高级功能)

---

## 🚀 快速入门

### 第一次使用

1. **环境检查**
```bash
# 检查Python版本 (需要3.8+)
python --version

# 检查CUDA (可选，用于GPU加速)
nvidia-smi
```

2. **安装框架**
```bash
# 克隆项目
git clone https://github.com/roclee2692/Chinese-LLM-LoRA-Finetuning.git
cd Chinese-LLM-LoRA-Finetuning

# 运行自动安装脚本
python setup_environment.py
```

3. **验证安装**
```bash
# 运行环境验证
python verify_installation.py

# 预期输出
✅ Python版本: 3.11.5
✅ PyTorch: 2.5.1+cu121  
✅ CUDA可用: True
✅ 显存: 8.0GB
✅ 所有依赖包已正确安装
```

4. **快速体验**
```bash
# 运行快速测试 (约3分钟)
python src/train.py --config configs/quick_test.yaml

# 启动Web界面
python demo/gradio_demo.py
```

---

## ⚙️ 配置详解

### 配置文件结构

框架使用YAML格式的配置文件，位于 `configs/` 目录：

```
configs/
├── quick_test.yaml          # 快速测试配置
├── production_training.yaml # 生产环境配置  
├── chatglm3_lora.yaml      # ChatGLM3专用配置
├── lightweight_training.yaml # 轻量级训练配置
└── model_config.yaml       # 基础模型配置
```

### 基础配置说明

#### 模型配置
```yaml
model:
  model_name: "distilgpt2"      # HuggingFace模型名称
  model_type: "gpt2"            # 模型类型
  trust_remote_code: false      # 是否信任远程代码
  torch_dtype: "auto"           # 数据类型 (auto/float16/bfloat16)
  device_map: "auto"            # 设备映射策略
```

**常用模型名称:**
- `distilgpt2` - 快速测试用
- `THUDM/chatglm3-6b` - 中文对话
- `Qwen/Qwen-7B-Chat` - 通用任务
- `baichuan-inc/Baichuan2-7B-Chat` - 中文理解

#### LoRA配置
```yaml
lora:
  r: 8                          # LoRA秩 (建议4-32)
  lora_alpha: 16               # LoRA缩放参数 (通常为2*r)
  target_modules:              # 目标模块
    - "c_attn"                 # 注意力层
    - "c_proj"                 # 投影层
  lora_dropout: 0.1            # Dropout率
  bias: "none"                 # 偏置处理 (none/all/lora_only)
  task_type: "CAUSAL_LM"       # 任务类型
```

**参数选择指南:**
- **r=4**: 最小参数，快速训练
- **r=8**: 平衡性能和效率 (推荐)
- **r=16**: 更好性能，需要更多资源
- **r=32**: 最佳性能，高资源需求

#### 训练配置
```yaml
training:
  output_dir: "./results/my_model"    # 输出目录
  num_train_epochs: 3                 # 训练轮数
  per_device_train_batch_size: 1      # 每设备批次大小
  gradient_accumulation_steps: 8      # 梯度累积步数
  learning_rate: 2e-4                # 学习率
  weight_decay: 0.01                 # 权重衰减
  logging_steps: 10                  # 日志间隔
  save_steps: 500                    # 保存间隔
  eval_steps: 500                    # 评估间隔
  max_seq_length: 512                # 最大序列长度
  warmup_ratio: 0.1                  # 预热比例
  lr_scheduler_type: "cosine"        # 学习率调度器
```

---

## 📊 数据准备

### 数据格式要求

框架支持标准的指令微调数据格式：

```json
{
  "instruction": "请解释什么是人工智能",
  "input": "",
  "output": "人工智能是让计算机模拟人类智能的技术..."
}
```

或对话格式：

```json
{
  "conversations": [
    {
      "from": "human", 
      "value": "你好，请介绍一下自己"
    },
    {
      "from": "gpt",
      "value": "您好！我是一个AI助手..."
    }
  ]
}
```

### 数据预处理

1. **自动数据下载**
```bash
# 下载常用中文数据集
python scripts/download_data.py --dataset alpaca_chinese

# 可选数据集:
# - alpaca_chinese: 中文Alpaca数据
# - belle: BELLE中文指令数据  
# - firefly: 流萤中文对话数据
```

2. **数据格式修复**
```bash
# 修复数据格式问题
python fix_data_format.py --input data/raw/your_data.json --output data/processed/

# 支持的修复功能:
# ✅ 嵌套字典展平
# ✅ 编码格式转换
# ✅ 字段名标准化
# ✅ 数据类型转换
```

3. **数据质量检查**
```bash
# 运行数据分析
jupyter notebook notebooks/data_analysis.ipynb

# 或使用命令行工具
python src/utils.py --check_data data/processed/train.json
```

### 自定义数据集

如果您有自己的数据集，请按以下步骤处理：

1. **转换为标准格式**
```python
import json

# 示例转换脚本
def convert_to_standard_format(your_data):
    standard_data = []
    for item in your_data:
        standard_item = {
            "instruction": item["question"],  # 替换为您的字段名
            "input": "",
            "output": item["answer"]          # 替换为您的字段名
        }
        standard_data.append(standard_item)
    return standard_data

# 保存为JSON文件
with open("data/processed/my_dataset.json", "w", encoding="utf-8") as f:
    json.dump(standard_data, f, ensure_ascii=False, indent=2)
```

2. **数据分割**
```python
# 使用内置工具分割数据
from src.utils import split_dataset

train_data, val_data, test_data = split_dataset(
    "data/processed/my_dataset.json",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

---

## 🎯 模型训练

### 训练流程

1. **选择配置**
```bash
# 快速测试 (5分钟，验证流程)
python src/train.py --config configs/quick_test.yaml

# 轻量级训练 (1小时，小规模数据)  
python src/train.py --config configs/lightweight_training.yaml

# 生产环境训练 (数小时，完整数据集)
python src/train.py --config configs/production_training.yaml
```

2. **监控训练**

训练过程中可以通过多种方式监控：

**终端输出:**
```
Step 10/100: loss=3.45, lr=2.0e-4, time=00:32
Step 20/100: loss=3.12, lr=2.0e-4, time=01:05  
Step 30/100: loss=2.98, lr=2.0e-4, time=01:38
```

**TensorBoard:**
```bash
# 启动TensorBoard
tensorboard --logdir results/logs --port 6006
```

**Weights & Biases (可选):**
```bash
# 安装wandb
pip install wandb

# 登录并配置
wandb login
```

3. **训练参数调优**

根据您的硬件配置调整参数：

**8GB GPU (RTX 4060):**
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 256
  fp16: true
```

**16GB GPU (RTX 4080):**
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_seq_length: 512
  fp16: true
```

**24GB+ GPU (RTX 4090):**
```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  max_seq_length: 1024
  bf16: true
```

### 训练技巧

1. **显存优化**
```yaml
# 启用梯度检查点
gradient_checkpointing: true

# 使用DeepSpeed ZeRO
deepspeed_config: "configs/ds_config.json"

# 8-bit优化器
optim: "adamw_8bit"
```

2. **训练稳定性**
```yaml
# 梯度裁剪
max_grad_norm: 1.0

# 学习率预热
warmup_ratio: 0.1

# 标签平滑
label_smoothing_factor: 0.1
```

3. **收敛加速**
```yaml
# 自适应学习率
lr_scheduler_type: "cosine_with_restarts"

# 早停策略
early_stopping_patience: 5
early_stopping_threshold: 0.01
```

---

## 🌐 Web界面使用

### 启动界面

```bash
# 基本启动
python demo/gradio_demo.py

# 自定义端口
python demo/gradio_demo.py --port 7860

# 公开访问
python demo/gradio_demo.py --share

# 指定模型
python demo/gradio_demo.py --model_path results/models/my_model
```

### 界面功能

#### 1. 模型对话
- **单轮对话**: 输入问题，获取回答
- **多轮对话**: 支持上下文记忆
- **参数调节**: 调整temperature、top_p等参数

#### 2. 模型对比
- **并排对比**: 同时测试多个模型
- **性能评估**: 自动计算BLEU、ROUGE分数
- **批量测试**: 上传测试文件进行批量评估

#### 3. 系统信息
- **环境检查**: 显示Python、PyTorch版本
- **硬件状态**: GPU使用率、内存占用
- **模型信息**: 参数量、量化状态

#### 4. 配置管理
- **在线编辑**: 直接修改训练配置
- **配置下载**: 保存自定义配置
- **预设模板**: 选择预定义配置

### 使用技巧

1. **对话优化**
```python
# 调整生成参数获得更好效果
temperature: 0.7     # 控制随机性
top_p: 0.9          # 核采样
max_length: 512     # 最大生成长度
repetition_penalty: 1.1  # 重复惩罚
```

2. **批量评估**
```json
# 准备测试文件 test_cases.json
[
  {
    "instruction": "解释量子计算的基本原理",
    "expected": "量子计算是基于量子力学原理..."
  },
  {
    "instruction": "比较Python和Java的区别", 
    "expected": "Python和Java都是流行的编程语言..."
  }
]
```

---

## 📈 模型评估

### 自动评估

```bash
# 基础评估
python src/evaluate.py --model_path results/models/my_model

# 详细评估
python src/evaluate.py \
  --model_path results/models/my_model \
  --test_file data/processed/test.json \
  --output_file results/evaluation/detailed_report.json

# 对比评估
python src/evaluate.py \
  --models results/models/model1 results/models/model2 \
  --compare
```

### 评估指标

框架支持多种评估指标：

1. **自动指标**
   - **BLEU**: 机器翻译质量评估
   - **ROUGE**: 文本摘要质量评估  
   - **BERTScore**: 基于BERT的语义相似度
   - **Perplexity**: 语言模型困惑度

2. **人工评估**
   - **流畅度**: 语言表达的自然程度
   - **相关性**: 回答与问题的相关程度
   - **有用性**: 回答的实用价值
   - **安全性**: 内容的安全和合规性

### 交互式评估

```bash
# 启动交互式评估
python src/evaluate.py --model_path results/models/my_model --interactive

# 示例会话:
>>> 请介绍一下北京的天气特点
模型回答: 北京属于温带大陆性季风气候，四季分明...
>>> 评分 (1-10): 8
>>> 评语: 回答准确但可以更详细
```

---

## 🔧 故障排除

### 常见问题及解决方案

#### 1. 环境问题

**Q: ModuleNotFoundError: No module named 'torch'**
```bash
# 解决方案: 重新安装PyTorch
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Q: CUDA out of memory**
```bash
# 解决方案: 调整训练参数
per_device_train_batch_size: 1  # 减小批次大小
gradient_accumulation_steps: 8  # 增加梯度累积
max_seq_length: 256            # 减小序列长度
fp16: true                     # 启用混合精度
```

#### 2. 数据问题

**Q: 数据格式错误**
```bash
# 解决方案: 运行数据修复脚本
python fix_data_format.py --input your_data.json

# 或手动检查数据格式
python -c "
import json
with open('your_data.json') as f:
    data = json.load(f)
    print(f'数据条数: {len(data)}')
    print(f'示例数据: {data[0]}')
"
```

**Q: 编码错误**
```bash
# 解决方案: 转换文件编码
python -c "
import codecs
with codecs.open('input.txt', 'r', 'gbk') as f:
    content = f.read()
with codecs.open('output.txt', 'w', 'utf-8') as f:
    f.write(content)
"
```

#### 3. 训练问题

**Q: 训练损失不下降**
```yaml
# 解决方案: 调整学习率和优化器
learning_rate: 1e-4  # 降低学习率
warmup_ratio: 0.1    # 增加预热
optim: "adamw_torch" # 使用不同优化器
```

**Q: 训练过程中断**
```yaml
# 解决方案: 启用断点续训
resume_from_checkpoint: "results/checkpoint-1000"
save_steps: 100  # 更频繁保存
```

#### 4. 推理问题

**Q: 生成内容重复**
```python
# 解决方案: 调整生成参数
generation_config = {
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "temperature": 0.8
}
```

**Q: 生成速度慢**
```python
# 解决方案: 优化推理设置
model.half()  # 使用半精度
torch.backends.cudnn.benchmark = True  # 启用cudnn优化
```

### 调试工具

1. **详细日志**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **性能分析**
```bash
# 安装性能分析工具
pip install py-spy

# 分析Python程序
py-spy record -o profile.svg -- python src/train.py --config configs/quick_test.yaml
```

3. **内存监控**
```bash
# 安装内存监控工具
pip install memory_profiler

# 监控内存使用
mprof run python src/train.py --config configs/quick_test.yaml
mprof plot
```

---

## 🚀 高级功能

### 1. 多GPU训练

```bash
# 使用DistributedDataParallel
torchrun --nproc_per_node=2 src/train.py --config configs/multi_gpu.yaml

# 使用DeepSpeed
deepspeed src/train.py --config configs/deepspeed.yaml --deepspeed configs/ds_config.json
```

### 2. 量化训练

```yaml
# QLoRA配置
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
```

### 3. 自定义模型

```python
# 注册自定义模型
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyCustomModel(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 自定义实现

# 在train.py中使用
AutoModelForCausalLM.register("my_model", MyCustomModel)
```

### 4. 高级数据处理

```python
# 自定义数据处理器
from src.data_preprocessing import BaseDataProcessor

class MyDataProcessor(BaseDataProcessor):
    def process(self, raw_data):
        # 自定义处理逻辑
        return processed_data

# 在配置中指定
data:
  processor: "MyDataProcessor"
  processor_kwargs:
    special_tokens: ["<|user|>", "<|assistant|>"]
```

---

## 📞 获取帮助

如果本指南没有解决您的问题，请通过以下方式获取帮助：

1. **查看文档**
   - [README.md](../README.md) - 项目概述
   - [TRAINING_REPORT.md](TRAINING_REPORT.md) - 训练实验报告
   - [FAQ.md](FAQ.md) - 常见问题解答

2. **社区支持**
   - [GitHub Issues](https://github.com/roclee2692/Chinese-LLM-LoRA-Finetuning/issues)
   - [GitHub Discussions](https://github.com/roclee2692/Chinese-LLM-LoRA-Finetuning/discussions)

3. **贡献代码**
   - [CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南
   - [开发者文档](DEVELOPER.md) - 开发者指南

---

**最后更新**: 2024年12月19日  
**版本**: v1.0.0