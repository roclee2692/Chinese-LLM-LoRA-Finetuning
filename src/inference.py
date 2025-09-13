"""
模型推理模块
提供模型加载和推理功能
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# 尝试导入推理相关的库
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        GenerationConfig
    )
    from peft import PeftModel
except ImportError as e:
    logging.warning(f"某些推理库未安装: {e}")

logger = logging.getLogger(__name__)


class ModelInference:
    """模型推理类"""
    
    def __init__(
        self, 
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
    def load_model(self):
        """加载模型和分词器"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            
            # 确定模型路径
            model_to_load = self.base_model_path or self.model_path
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_to_load,
                trust_remote_code=True,
                cache_dir="./cache"
            )
            
            # 设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            if self.base_model_path:
                # 加载基础模型 + LoRA适配器
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True,
                    cache_dir="./cache"
                )
                
                # 加载LoRA适配器
                self.model = PeftModel.from_pretrained(
                    base_model, 
                    self.model_path,
                    torch_dtype=torch.float16
                )
            else:
                # 直接加载完整模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True,
                    cache_dir="./cache"
                )
            
            # 设置评估模式
            self.model.eval()
            
            # 设置生成配置
            self.generation_config = GenerationConfig(
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """生成回复"""
        if self.model is None:
            self.load_model()
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = inputs['input_ids'].shape[1]
        
        # 移动到设备
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新生成的部分
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: str = "你是一个有用的AI助手。"
    ) -> str:
        """聊天模式"""
        if history is None:
            history = []
        
        # 构建对话历史
        conversation = f"### 系统:\n{system_prompt}\n\n"
        
        for turn in history:
            conversation += f"### 用户:\n{turn['user']}\n\n"
            conversation += f"### 助手:\n{turn['assistant']}\n\n"
        
        conversation += f"### 用户:\n{message}\n\n### 助手:\n"
        
        # 生成回复
        response = self.generate(conversation)
        
        return response
    
    def instruction_following(
        self,
        instruction: str,
        input_text: str = ""
    ) -> str:
        """指令跟随模式"""
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"
        
        response = self.generate(prompt)
        return response
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **generation_kwargs
    ) -> List[str]:
        """批量生成"""
        if self.model is None:
            self.load_model()
        
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = []
            
            for prompt in batch_prompts:
                try:
                    response = self.generate(prompt, **generation_kwargs)
                    batch_responses.append(response)
                except Exception as e:
                    logger.error(f"生成失败: {e}")
                    batch_responses.append("")
            
            responses.extend(batch_responses)
            
            if i + batch_size < len(prompts):
                logger.info(f"批量生成进度: {i + batch_size}/{len(prompts)}")
        
        return responses
    
    def save_responses(
        self,
        prompts: List[str],
        responses: List[str],
        output_file: str
    ):
        """保存生成结果"""
        results = []
        for prompt, response in zip(prompts, responses):
            results.append({
                "prompt": prompt,
                "response": response
            })
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")


class ChatBot:
    """聊天机器人类"""
    
    def __init__(self, model_inference: ModelInference):
        self.inference = model_inference
        self.history = []
        self.system_prompt = "你是一个有用、诚实且无害的AI助手。"
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示"""
        self.system_prompt = prompt
    
    def chat(self, message: str) -> str:
        """聊天"""
        response = self.inference.chat(
            message, 
            self.history, 
            self.system_prompt
        )
        
        # 更新历史
        self.history.append({
            "user": message,
            "assistant": response
        })
        
        return response
    
    def clear_history(self):
        """清空历史"""
        self.history = []
    
    def get_history(self) -> List[Dict]:
        """获取历史"""
        return self.history.copy()
    
    def save_history(self, file_path: str):
        """保存对话历史"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "system_prompt": self.system_prompt,
                "history": self.history
            }, f, ensure_ascii=False, indent=2)
    
    def load_history(self, file_path: str):
        """加载对话历史"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.system_prompt = data.get("system_prompt", self.system_prompt)
            self.history = data.get("history", [])


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型推理工具")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--base_model_path', type=str, help='基础模型路径（用于LoRA）')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    parser.add_argument('--interactive', action='store_true', help='交互式聊天')
    parser.add_argument('--instruction', type=str, help='指令')
    parser.add_argument('--input', type=str, help='输入文本')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = ModelInference(
        args.model_path, 
        args.base_model_path, 
        args.device
    )
    
    if args.interactive:
        # 交互式聊天
        chatbot = ChatBot(inference)
        
        print("=== 中文大语言模型聊天 ===")
        print("输入'quit'退出，'clear'清空历史，'system <prompt>'设置系统提示")
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    chatbot.clear_history()
                    print("对话历史已清空")
                    continue
                elif user_input.startswith('system '):
                    system_prompt = user_input[7:]
                    chatbot.set_system_prompt(system_prompt)
                    print(f"系统提示已设置: {system_prompt}")
                    continue
                
                response = chatbot.chat(user_input)
                print(f"助手: {response}")
                
            except KeyboardInterrupt:
                print("\n聊天已终止")
                break
            except Exception as e:
                print(f"生成出错: {e}")
    
    elif args.instruction:
        # 指令模式
        response = inference.instruction_following(
            args.instruction, 
            args.input or ""
        )
        print(f"回答: {response}")
    
    else:
        print("请指定交互模式或提供指令")


if __name__ == "__main__":
    main()