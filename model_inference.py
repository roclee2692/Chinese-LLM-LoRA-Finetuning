#!/usr/bin/env python3
"""
真实模型推理模块
加载训练好的Qwen LoRA模型进行推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json
from pathlib import Path

class QwenLoRAInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
        # 模型路径配置
        self.base_model_path = "cache/models--Qwen--Qwen-1_8B-Chat/snapshots/1d0f68de57b88cfde81f3c3e537f24464d889081"
        self.lora_adapter_path = "results/models/qwen-1.8b-lora-ultimate"
        
    def load_model(self):
        """加载基础模型和LoRA适配器"""
        try:
            print("🚀 开始加载Qwen-1.8B基础模型...")
            
            # 检查模型路径
            if not os.path.exists(self.base_model_path):
                print(f"❌ 基础模型路径不存在: {self.base_model_path}")
                return False
                
            if not os.path.exists(self.lora_adapter_path):
                print(f"❌ LoRA适配器路径不存在: {self.lora_adapter_path}")
                return False
            
            # 加载tokenizer
            print("📝 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, 
                trust_remote_code=True,
                pad_token='<|endoftext|>'
            )
            
            # 加载基础模型
            print("🧠 加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 加载LoRA适配器
            print("⚡ 加载LoRA适配器...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_adapter_path,
                torch_dtype=torch.float16
            )
            
            print("✅ 模型加载成功！")
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return False
    
    def generate_response(self, user_input, max_length=512, temperature=0.7):
        """生成模型回复"""
        if not self.model_loaded:
            if not self.load_model():
                return "❌ 模型加载失败，无法生成回复"
        
        try:
            # 构建对话格式
            messages = [
                {"role": "system", "content": "你是一个有用的AI助手，经过LoRA微调，请用中文回答问题。"},
                {"role": "user", "content": user_input}
            ]
            
            # 使用chat template格式化输入
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # 生成回复
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_p=0.8,
                    repetition_penalty=1.1
                )
            
            # 解码输出
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 清理输出
            response = response.strip()
            if not response:
                response = "抱歉，我暂时无法生成合适的回复。"
            
            return response
            
        except Exception as e:
            print(f"❌ 生成回复时出错: {str(e)}")
            return f"生成回复时出现错误: {str(e)}"
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            "model_loaded": self.model_loaded,
            "device": str(self.device),
            "base_model": "Qwen-1.8B-Chat",
            "lora_adapter": "qwen-1.8b-lora-ultimate",
            "status": "✅ 已加载" if self.model_loaded else "❌ 未加载"
        }
        return info

# 全局模型实例
model_inference = None

def get_model_instance():
    """获取模型实例（单例模式）"""
    global model_inference
    if model_inference is None:
        model_inference = QwenLoRAInference()
    return model_inference

def chat_with_model(user_input):
    """与模型对话的简单接口"""
    model = get_model_instance()
    return model.generate_response(user_input)

def test_model():
    """测试模型功能"""
    print("🧪 测试模型功能...")
    model = get_model_instance()
    
    test_questions = [
        "你好，请介绍一下自己",
        "你是什么模型？",
        "请解释一下LoRA技术",
        "你会中文吗？"
    ]
    
    for question in test_questions:
        print(f"\n❓ 用户: {question}")
        response = model.generate_response(question)
        print(f"🤖 模型: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_model()