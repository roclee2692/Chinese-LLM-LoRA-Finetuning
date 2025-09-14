"""
基于Gradio的中文大语言模型LoRA微调演示界面
提供模型推理、对比和交互功能
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 尝试导入相关库
try:
    import gradio as gr
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import yaml
except ImportError as e:
    print(f"某些库未安装: {e}")
    print("请安装: pip install gradio torch transformers peft")

# 导入自定义模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from inference import ModelInference, ChatBot
    from utils import load_config
except ImportError as e:
    print(f"无法导入自定义模块: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradioDemo:
    """Gradio演示界面类"""
    
    def __init__(self):
        self.models = {}  # 存储加载的模型
        self.current_model = None
        self.chat_history = []
        self.max_history_length = 10
        
        # 预定义的示例
        self.examples = [
            ["请介绍一下人工智能的发展历史。", ""],
            ["解释什么是大语言模型？", ""],
            ["请写一首关于春天的诗。", ""],
            ["翻译以下文本", "Hello, how are you?"],
            ["总结以下内容的要点", "人工智能是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"],
        ]
    
    def load_model(self, model_path: str, base_model_path: str = "") -> str:
        """加载模型"""
        try:
            logger.info(f"正在加载模型: {model_path}")
            
            # 创建模型推理器
            inference = ModelInference(
                model_path=model_path,
                base_model_path=base_model_path if base_model_path else None
            )
            inference.load_model()
            
            # 存储模型
            model_name = Path(model_path).name
            self.models[model_name] = inference
            self.current_model = model_name
            
            return f"✅ 模型 {model_name} 加载成功！"
            
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def generate_response(
        self, 
        instruction: str, 
        input_text: str, 
        model_name: str,
        max_length: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float
    ) -> str:
        """生成回复"""
        try:
            if not instruction.strip():
                return "⚠️ 请输入指令内容"
            
            if model_name not in self.models:
                return "❌ 请先加载模型"
            
            inference = self.models[model_name]
            
            # 生成回复
            if input_text.strip():
                response = inference.instruction_following(
                    instruction=instruction,
                    input_text=input_text
                )
            else:
                response = inference.generate(
                    prompt=f"### 指令:\n{instruction}\n\n### 回答:\n",
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
            
            return response
            
        except Exception as e:
            error_msg = f"❌ 生成失败: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def chat_response(
        self,
        message: str,
        history: List[List[str]],
        model_name: str,
        max_length: int,
        temperature: float,
        top_p: float
    ) -> Tuple[str, List[List[str]]]:
        """聊天回复"""
        try:
            if not message.strip():
                return "", history
            
            if model_name not in self.models:
                return "❌ 请先加载模型", history
            
            inference = self.models[model_name]
            
            # 构建对话历史
            chat_history = []
            for user_msg, bot_msg in history:
                chat_history.append({"user": user_msg, "assistant": bot_msg})
            
            # 生成回复
            response = inference.chat(
                message=message,
                history=chat_history
            )
            
            # 更新历史
            history.append([message, response])
            
            # 限制历史长度
            if len(history) > self.max_history_length:
                history = history[-self.max_history_length:]
            
            return "", history
            
        except Exception as e:
            error_msg = f"❌ 聊天失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, history
    
    def compare_models(
        self,
        instruction: str,
        input_text: str,
        model1_name: str,
        model2_name: str,
        max_length: int,
        temperature: float
    ) -> Tuple[str, str]:
        """模型对比"""
        try:
            if not instruction.strip():
                return "⚠️ 请输入指令内容", "⚠️ 请输入指令内容"
            
            response1 = "❌ 模型1未加载"
            response2 = "❌ 模型2未加载"
            
            # 模型1生成
            if model1_name in self.models:
                inference1 = self.models[model1_name]
                if input_text.strip():
                    response1 = inference1.instruction_following(instruction, input_text)
                else:
                    response1 = inference1.generate(
                        f"### 指令:\n{instruction}\n\n### 回答:\n",
                        max_new_tokens=max_length,
                        temperature=temperature
                    )
            
            # 模型2生成
            if model2_name in self.models:
                inference2 = self.models[model2_name]
                if input_text.strip():
                    response2 = inference2.instruction_following(instruction, input_text)
                else:
                    response2 = inference2.generate(
                        f"### 指令:\n{instruction}\n\n### 回答:\n",
                        max_new_tokens=max_length,
                        temperature=temperature
                    )
            
            return response1, response2
            
        except Exception as e:
            error_msg = f"❌ 对比失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, error_msg
    
    def get_model_list(self) -> List[str]:
        """获取已加载的模型列表"""
        return list(self.models.keys())
    
    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        
        with gr.Blocks(
            title="中文大语言模型LoRA微调演示",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .header {
                text-align: center;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            """
        ) as demo:
            
            # 标题
            gr.HTML("""
                <div class="header">
                    <h1>🚀 中文大语言模型LoRA微调演示</h1>
                    <p>支持多模型加载、推理对比和交互式聊天</p>
                </div>
            """)
            
            # 模型管理选项卡
            with gr.Tab("🔧 模型管理"):
                with gr.Row():
                    with gr.Column():
                        model_path_input = gr.Textbox(
                            label="模型路径",
                            placeholder="输入模型路径，如: ./results/models/chatglm3-lora",
                            lines=1
                        )
                        base_model_path_input = gr.Textbox(
                            label="基础模型路径（可选，用于LoRA）",
                            placeholder="如: THUDM/chatglm3-6b",
                            lines=1
                        )
                        load_btn = gr.Button("🔄 加载模型", variant="primary")
                    
                    with gr.Column():
                        load_status = gr.Textbox(
                            label="加载状态",
                            lines=3,
                            interactive=False
                        )
                        loaded_models = gr.Textbox(
                            label="已加载模型",
                            lines=3,
                            interactive=False
                        )
                
                load_btn.click(
                    fn=self.load_model,
                    inputs=[model_path_input, base_model_path_input],
                    outputs=[load_status]
                ).then(
                    fn=lambda: "\n".join(self.get_model_list()),
                    outputs=[loaded_models]
                )
            
            # 单模型推理选项卡
            with gr.Tab("💬 模型推理"):
                with gr.Row():
                    with gr.Column(scale=2):
                        instruction_input = gr.Textbox(
                            label="指令",
                            placeholder="请输入指令，如：请介绍一下人工智能",
                            lines=3
                        )
                        input_text_input = gr.Textbox(
                            label="输入内容（可选）",
                            placeholder="如果指令需要处理特定内容，请在此输入",
                            lines=3
                        )
                        
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                label="选择模型",
                                choices=[],
                                interactive=True
                            )
                            generate_btn = gr.Button("✨ 生成回复", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 生成参数")
                        max_length_slider = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=256,
                            step=50,
                            label="最大长度"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="创造性（Temperature）"
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="多样性（Top-p）"
                        )
                        repetition_penalty_slider = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.1,
                            step=0.05,
                            label="重复惩罚"
                        )
                
                response_output = gr.Textbox(
                    label="模型回复",
                    lines=8,
                    interactive=False
                )
                
                # 示例
                gr.Examples(
                    examples=self.examples,
                    inputs=[instruction_input, input_text_input],
                    label="💡 示例"
                )
                
                generate_btn.click(
                    fn=self.generate_response,
                    inputs=[
                        instruction_input,
                        input_text_input,
                        model_dropdown,
                        max_length_slider,
                        temperature_slider,
                        top_p_slider,
                        repetition_penalty_slider
                    ],
                    outputs=[response_output]
                )
            
            # 对话聊天选项卡
            with gr.Tab("💭 对话聊天"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=400,
                            type="messages"
                        )
                        
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="输入消息",
                                placeholder="请输入您的消息...",
                                lines=2,
                                scale=4
                            )
                            chat_btn = gr.Button("发送", variant="primary", scale=1)
                            clear_btn = gr.Button("清空", scale=1)
                    
                    with gr.Column(scale=1):
                        chat_model_dropdown = gr.Dropdown(
                            label="选择模型",
                            choices=[],
                            interactive=True
                        )
                        
                        gr.Markdown("### ⚙️ 聊天参数")
                        chat_max_length = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=200,
                            step=25,
                            label="回复最大长度"
                        )
                        chat_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.5,
                            value=0.7,
                            step=0.1,
                            label="创造性"
                        )
                        chat_top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="多样性"
                        )
                
                chat_btn.click(
                    fn=self.chat_response,
                    inputs=[
                        chat_input,
                        chatbot,
                        chat_model_dropdown,
                        chat_max_length,
                        chat_temperature,
                        chat_top_p
                    ],
                    outputs=[chat_input, chatbot]
                )
                
                clear_btn.click(
                    fn=lambda: ([], ""),
                    outputs=[chatbot, chat_input]
                )
                
                # 回车发送
                chat_input.submit(
                    fn=self.chat_response,
                    inputs=[
                        chat_input,
                        chatbot,
                        chat_model_dropdown,
                        chat_max_length,
                        chat_temperature,
                        chat_top_p
                    ],
                    outputs=[chat_input, chatbot]
                )
            
            # 模型对比选项卡
            with gr.Tab("⚖️ 模型对比"):
                with gr.Row():
                    with gr.Column():
                        compare_instruction = gr.Textbox(
                            label="指令",
                            placeholder="输入要对比的指令",
                            lines=2
                        )
                        compare_input = gr.Textbox(
                            label="输入内容（可选）",
                            lines=2
                        )
                        
                        with gr.Row():
                            model1_dropdown = gr.Dropdown(
                                label="模型1",
                                choices=[],
                                interactive=True
                            )
                            model2_dropdown = gr.Dropdown(
                                label="模型2", 
                                choices=[],
                                interactive=True
                            )
                        
                        with gr.Row():
                            compare_max_length = gr.Slider(50, 500, 200, label="最大长度")
                            compare_temperature = gr.Slider(0.1, 1.5, 0.7, label="创造性")
                        
                        compare_btn = gr.Button("🔄 开始对比", variant="primary")
                
                with gr.Row():
                    model1_output = gr.Textbox(
                        label="模型1回复",
                        lines=8,
                        interactive=False
                    )
                    model2_output = gr.Textbox(
                        label="模型2回复",
                        lines=8,
                        interactive=False
                    )
                
                compare_btn.click(
                    fn=self.compare_models,
                    inputs=[
                        compare_instruction,
                        compare_input,
                        model1_dropdown,
                        model2_dropdown,
                        compare_max_length,
                        compare_temperature
                    ],
                    outputs=[model1_output, model2_output]
                )
            
            # 更新模型列表的函数
            def update_model_choices():
                choices = self.get_model_list()
                return (
                    gr.Dropdown(choices=choices),
                    gr.Dropdown(choices=choices),
                    gr.Dropdown(choices=choices),
                    gr.Dropdown(choices=choices)
                )
            
            # 定期更新模型选择器
            demo.load(
                fn=update_model_choices,
                outputs=[model_dropdown, chat_model_dropdown, model1_dropdown, model2_dropdown]
            )
            
            # 页脚信息
            gr.HTML("""
                <div style="text-align: center; margin-top: 20px; padding: 10px; 
                           background-color: #f8f9fa; border-radius: 5px;">
                    <p>🔗 项目地址: <a href="https://github.com/roclee2692/Chinese-LLM-LoRA-Finetuning" target="_blank">GitHub</a></p>
                    <p>📖 使用说明: 先在"模型管理"中加载模型，然后在其他选项卡中使用</p>
                </div>
            """)
        
        return demo


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="中文大语言模型LoRA微调Gradio演示")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器地址')
    parser.add_argument('--port', type=int, default=7860, help='端口号')
    parser.add_argument('--share', action='store_true', help='创建公共链接')
    parser.add_argument('--model_path', type=str, help='默认加载的模型路径')
    parser.add_argument('--base_model_path', type=str, help='基础模型路径')
    
    args = parser.parse_args()
    
    # 创建演示界面
    demo_app = GradioDemo()
    
    # 如果指定了模型路径，则自动加载
    if args.model_path:
        logger.info(f"自动加载模型: {args.model_path}")
        result = demo_app.load_model(args.model_path, args.base_model_path or "")
        print(result)
    
    # 创建界面
    demo = demo_app.create_interface()
    
    # 启动服务
    print("启动Gradio演示界面...")
    print(f"地址: http://{args.host}:{args.port}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()