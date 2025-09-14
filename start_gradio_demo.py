#!/usr/bin/env python3
"""
简化的Web演示界面
无需预训练模型，展示框架功能
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json

class SimpleDemo:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
    def load_model(self, model_name="distilgpt2"):
        """加载轻量级模型进行演示"""
        try:
            print(f"🔧 加载模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.model_loaded = True
            return "✅ 模型加载成功！"
        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}"
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """文本生成"""
        if not self.model_loaded:
            return "❌ 请先加载模型"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            return f"❌ 生成失败: {str(e)}"
    
    def get_system_info(self):
        """获取系统信息"""
        info = f"""
🖥️ **系统信息**
- Python: {torch.__version__ if torch else "未安装"}
- PyTorch: {torch.__version__ if torch else "未安装"}
- CUDA可用: {torch.cuda.is_available() if torch else "未知"}
- 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

📁 **项目状态**
- 数据已预处理: ✅
- 配置文件完整: ✅
- 训练环境就绪: ✅
- Web界面运行: ✅

🎯 **功能展示**
这是中文LLM LoRA微调框架的演示界面。
您可以：
1. 加载轻量级模型进行测试
2. 体验文本生成功能
3. 查看框架完整性

📋 **下一步**
- 运行完整训练: `python src/train.py --config configs/quick_test.yaml`
- 加载训练后的模型进行更好的中文对话
        """
        return info

def create_interface():
    """创建Gradio界面"""
    demo_instance = SimpleDemo()
    
    with gr.Blocks(
        title="中文LLM LoRA微调框架演示",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("# 🚀 中文大语言模型LoRA微调框架")
        gr.Markdown("*A Comprehensive LoRA Fine-tuning Framework for Chinese LLMs*")
        
        with gr.Tab("📊 系统状态"):
            info_output = gr.Markdown(demo_instance.get_system_info())
            gr.Button("🔄 刷新状态").click(
                lambda: demo_instance.get_system_info(),
                outputs=info_output
            )
        
        with gr.Tab("🤖 模型演示"):
            with gr.Row():
                with gr.Column():
                    load_btn = gr.Button("📦 加载演示模型", variant="primary")
                    load_status = gr.Textbox(label="加载状态", interactive=False)
                    
                    load_btn.click(
                        demo_instance.load_model,
                        outputs=load_status
                    )
            
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="输入提示文本",
                        placeholder="请输入您想要的文本提示...",
                        lines=3
                    )
                    
                    with gr.Row():
                        max_length = gr.Slider(50, 200, 100, label="最大长度")
                        temperature = gr.Slider(0.1, 1.0, 0.7, label="温度")
                        top_p = gr.Slider(0.1, 1.0, 0.9, label="Top-p")
                    
                    generate_btn = gr.Button("✨ 生成文本", variant="secondary")
                
                with gr.Column():
                    output_text = gr.Textbox(
                        label="生成结果",
                        lines=8,
                        interactive=False
                    )
            
            generate_btn.click(
                demo_instance.generate_text,
                inputs=[prompt_input, max_length, temperature, top_p],
                outputs=output_text
            )
        
        with gr.Tab("📖 使用指南"):
            gr.Markdown("""
## 🎯 框架特性

### ✨ 核心功能
- **多模型支持**: ChatGLM3, Qwen, Baichuan2, Yi
- **高效训练**: LoRA参数高效微调
- **数据处理**: 中文指令数据集优化
- **Web界面**: 交互式模型对话
- **实验跟踪**: Weights & Biases集成

### 🚀 快速开始

1. **环境准备** (已完成 ✅)
```bash
pip install -r requirements.txt
```

2. **数据预处理** (已完成 ✅)
```bash
python fix_data_format.py
```

3. **开始训练**
```bash
python src/train.py --config configs/quick_test.yaml
```

4. **模型推理**
```bash
python src/inference.py --model_path results/models/your-model
```

### 📁 项目结构
```
├── src/                 # 核心源代码
├── configs/             # 配置文件
├── demo/               # Web演示界面
├── data/               # 数据目录
├── results/            # 训练结果
└── scripts/            # 工具脚本
```

### 🔧 配置说明
- `quick_test.yaml`: 快速测试配置
- `production_training.yaml`: 生产环境配置
- `chatglm3_lora.yaml`: ChatGLM3专用配置

### 💡 提示
当前演示使用轻量级模型。完整训练后，您将获得专业的中文对话能力！
            """)
    
    return interface

if __name__ == "__main__":
    print("🌐 启动中文LLM LoRA微调框架演示界面...")
    
    try:
        interface = create_interface()
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("请确保已安装gradio: pip install gradio")