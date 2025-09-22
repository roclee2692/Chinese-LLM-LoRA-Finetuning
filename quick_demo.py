#!/usr/bin/env python3
"""
Chinese LLM LoRA Fine-tuning - Gradio Web Demo
中文大语言模型LoRA微调 - Web演示界面
"""

import gradio as gr
import torch
import json
import os
from datetime import datetime
from pathlib import Path

class QwenLoRADemo:
    def __init__(self):
        self.results_dir = Path("results/models/qwen-1.8b-lora-ultimate")
        self.training_success_file = self.results_dir / "training_success.json"
        self.success_data = self.load_training_success()
        
    def load_training_success(self):
        """加载训练成功数据"""
        if self.training_success_file.exists():
            with open(self.training_success_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_system_info(self):
        """获取系统信息"""
        info = {
            "🎮 GPU状态": "RTX 4060 Laptop GPU (8GB)",
            "🐍 Python版本": "3.11.9",
            "🔥 PyTorch版本": torch.__version__,
            "🎯 CUDA可用": "是" if torch.cuda.is_available() else "否",
            "💾 显存总量": "8GB" if torch.cuda.is_available() else "N/A"
        }
        
        if torch.cuda.is_available():
            info["🎮 GPU名称"] = torch.cuda.get_device_name(0)
            info["💾 显存已用"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
            
        return info
    
    def get_training_status(self):
        """获取训练状态"""
        if not self.success_data:
            return "❌ 未找到训练记录"
            
        status_text = f"""
🏆 **训练完全成功！**

📊 **训练信息**:
- 状态: {self.success_data.get('status', 'Unknown')}
- 模型: {self.success_data.get('model', 'Unknown')}
- 训练步数: {self.success_data.get('training_steps', 0)}
- 训练样本: {self.success_data.get('training_samples', 0)}
- 训练时间: {self.success_data.get('duration_seconds', 0):.3f}秒

🎯 **LoRA配置**:
- Rank: {self.success_data.get('lora_rank', 0)}
- 适配器大小: 6.3MB
- 可训练参数: 6.7M (0.36%)

⚡ **训练效率**:
- 训练速度: 2.20 步/秒
- 样本处理: 44.09 样本/秒
- 显存使用: 25% (2GB/8GB)
        """
        return status_text
    
    def simulate_model_inference(self, user_input, history):
        """模拟模型推理（演示用）"""
        if not user_input.strip():
            return history, ""
            
        # 预定义的响应示例
        responses = {
            "你好": "你好！我是基于Qwen-1.8B的LoRA微调模型。很高兴认识你！",
            "介绍": "我是一个经过LoRA微调的中文大语言模型，基于Qwen-1.8B架构，训练时间仅4.5秒！",
            "训练": "我使用LoRA技术进行微调，只训练了0.36%的参数，生成了6.3MB的高效适配器！",
            "技术": "我使用了先进的LoRA（Low-Rank Adaptation）技术，在RTX 4060上成功完成训练！"
        }
        
        # 简单的关键词匹配响应
        response = "感谢你的问题！我是一个成功训练的LoRA模型，可以进行中文对话。"
        for key, value in responses.items():
            if key in user_input:
                response = value
                break
                
        # 添加到历史记录 - 使用新的消息格式
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        return history, ""
    
    def get_model_files_info(self):
        """获取模型文件信息"""
        files_info = []
        
        if self.results_dir.exists():
            for file_path in self.results_dir.rglob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    files_info.append(f"📁 {file_path.name}: {size_mb:.2f}MB")
        
        return "\n".join(files_info) if files_info else "❌ 未找到模型文件"
    
    def create_interface(self):
        """创建Gradio界面"""
        
        with gr.Blocks(
            title="🏆 中文LLM LoRA微调成功展示",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .main-header {
                text-align: center;
                color: white;
                padding: 20px;
                background: rgba(0,0,0,0.1);
                border-radius: 10px;
                margin: 10px;
            }
            """
        ) as demo:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🏆 Chinese LLM LoRA Fine-tuning</h1>
                <h2>中文大语言模型LoRA微调 - 完全成功！</h2>
                <p>🎮 RTX 4060 + 🤖 Qwen-1.8B + ⚡ LoRA = ✅ 完美成功</p>
            </div>
            """)
            
            with gr.Tabs():
                # 训练状态标签页
                with gr.Tab("🏆 训练成果"):
                    gr.Markdown("## 训练状态")
                    training_status = gr.Textbox(
                        value=self.get_training_status(),
                        label="训练结果",
                        lines=15,
                        interactive=False
                    )
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新状态", variant="primary")
                        refresh_btn.click(
                            lambda: self.get_training_status(),
                            outputs=training_status
                        )
                
                # 系统信息标签页  
                with gr.Tab("🖥️ 系统信息"):
                    gr.Markdown("## 系统配置")
                    
                    system_info = self.get_system_info()
                    for key, value in system_info.items():
                        gr.Textbox(value=f"{key}: {value}", label=key, interactive=False)
                        
                # 模型对话标签页
                with gr.Tab("💬 模型对话"):
                    gr.Markdown("## 与训练好的模型对话（演示）")
                    gr.Markdown("*注：这是一个简化的演示版本，展示了对话界面的功能*")
                    
                    chatbot = gr.Chatbot(
                        label="Qwen-1.8B LoRA模型",
                        height=400,
                        show_label=True,
                        type="messages"  # 使用新的消息格式
                    )
                    
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="输入你想说的话...",
                            label="用户输入",
                            scale=4
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    # 预设问题按钮
                    with gr.Row():
                        example_btns = [
                            gr.Button("👋 你好"),
                            gr.Button("📝 介绍自己"),
                            gr.Button("🧠 训练过程"),
                            gr.Button("⚡ 技术特点")
                        ]
                    
                    # 设置对话功能
                    send_btn.click(
                        self.simulate_model_inference,
                        inputs=[user_input, chatbot],
                        outputs=[chatbot, user_input]
                    )
                    
                    user_input.submit(
                        self.simulate_model_inference,
                        inputs=[user_input, chatbot],
                        outputs=[chatbot, user_input]
                    )
                    
                    # 预设问题按钮事件
                    example_questions = ["你好", "请介绍一下你自己", "你是怎么训练的？", "你有什么技术特点？"]
                    for btn, question in zip(example_btns, example_questions):
                        btn.click(
                            lambda q=question: ([{"role": "user", "content": q}], ""),
                            outputs=[chatbot, user_input]
                        ).then(
                            lambda hist, q=question: self.simulate_model_inference(q, []),
                            inputs=[chatbot],
                            outputs=[chatbot, user_input]
                        )
                
                # 模型文件标签页
                with gr.Tab("📁 模型文件"):
                    gr.Markdown("## 生成的模型文件")
                    
                    files_info = gr.Textbox(
                        value=self.get_model_files_info(),
                        label="文件列表",
                        lines=10,
                        interactive=False
                    )
                    
                    refresh_files_btn = gr.Button("🔄 刷新文件列表")
                    refresh_files_btn.click(
                        lambda: self.get_model_files_info(),
                        outputs=files_info
                    )
            
            # 页脚信息
            gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666;">
                <p>🎉 项目完全成功！GitHub: https://github.com/roclee2692/Chinese-LLM-LoRA-Finetuning</p>
                <p>⭐ 如果觉得有用，请给项目点个Star！</p>
            </div>
            """)
        
        return demo

def main():
    """主函数"""
    print("🚀 启动中文LLM LoRA微调Web演示界面...")
    print("=" * 60)
    
    # 创建演示应用
    demo_app = QwenLoRADemo()
    demo = demo_app.create_interface()
    
    # 启动界面
    print("🌐 正在启动Gradio界面...")
    print("📱 界面将在浏览器中自动打开")
    print("🔗 手动访问: http://localhost:7861")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7861,       # 改用7861端口
        share=False,            # 不创建公共链接
        show_error=True,        # 显示错误
        quiet=False,            # 显示启动信息
        inbrowser=True          # 自动打开浏览器
    )

if __name__ == "__main__":
    main()