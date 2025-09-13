"""
模型评估模块
提供多种评估指标和方法
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# 尝试导入评估相关的库
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    from datasets import Dataset
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import jieba
except ImportError as e:
    logging.warning(f"某些评估库未安装: {e}")

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, base_model_path: str = None):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path or self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            if self.base_model_path:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                # 加载LoRA适配器
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.model.eval()
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """生成回复"""
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """计算BLEU分数"""
        try:
            # 中文分词
            ref_tokens = list(jieba.cut(reference))
            hyp_tokens = list(jieba.cut(hypothesis))
            
            # 计算BLEU分数
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu(
                [ref_tokens], 
                hyp_tokens, 
                smoothing_function=smoothing
            )
            
            return bleu_score
        except Exception as e:
            logger.warning(f"BLEU计算失败: {e}")
            return 0.0
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """计算ROUGE分数"""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE计算失败: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def evaluate_dataset(
        self, 
        test_data: List[Dict],
        output_file: Optional[str] = None
    ) -> Dict:
        """评估数据集"""
        if self.model is None:
            self.load_model()
        
        results = []
        bleu_scores = []
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        logger.info(f"开始评估，共{len(test_data)}条数据")
        
        for i, item in enumerate(test_data):
            if i % 10 == 0:
                logger.info(f"评估进度: {i}/{len(test_data)}")
            
            # 构建输入
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            reference = item.get('output', '')
            
            if input_text:
                prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n"
            else:
                prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"
            
            # 生成回复
            try:
                hypothesis = self.generate_response(prompt)
                
                # 计算指标
                bleu = self.calculate_bleu(reference, hypothesis)
                rouge = self.calculate_rouge(reference, hypothesis)
                
                bleu_scores.append(bleu)
                rouge_scores['rouge1'].append(rouge['rouge1'])
                rouge_scores['rouge2'].append(rouge['rouge2'])
                rouge_scores['rougeL'].append(rouge['rougeL'])
                
                result = {
                    'instruction': instruction,
                    'input': input_text,
                    'reference': reference,
                    'hypothesis': hypothesis,
                    'bleu': bleu,
                    'rouge1': rouge['rouge1'],
                    'rouge2': rouge['rouge2'],
                    'rougeL': rouge['rougeL']
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"评估第{i}条数据时出错: {e}")
                continue
        
        # 计算平均分数
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_rouge1 = np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0
        avg_rouge2 = np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0
        avg_rougeL = np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0
        
        evaluation_summary = {
            'num_samples': len(results),
            'avg_bleu': avg_bleu,
            'avg_rouge1': avg_rouge1,
            'avg_rouge2': avg_rouge2,
            'avg_rougeL': avg_rougeL,
            'detailed_results': results
        }
        
        # 保存结果
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估结果已保存到: {output_path}")
        
        # 打印摘要
        logger.info(f"评估完成！")
        logger.info(f"平均BLEU分数: {avg_bleu:.4f}")
        logger.info(f"平均ROUGE-1分数: {avg_rouge1:.4f}")
        logger.info(f"平均ROUGE-2分数: {avg_rouge2:.4f}")
        logger.info(f"平均ROUGE-L分数: {avg_rougeL:.4f}")
        
        return evaluation_summary
    
    def interactive_evaluation(self):
        """交互式评估"""
        if self.model is None:
            self.load_model()
        
        print("=== 交互式模型评估 ===")
        print("输入'quit'退出")
        
        while True:
            try:
                instruction = input("\n请输入指令: ").strip()
                if instruction.lower() == 'quit':
                    break
                
                input_text = input("请输入输入内容（可选，直接回车跳过）: ").strip()
                
                if input_text:
                    prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n"
                else:
                    prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"
                
                print("\n正在生成回复...")
                response = self.generate_response(prompt)
                
                print(f"\n模型回复:\n{response}")
                
                # 可选的人工评估
                rating = input("\n请对回复质量打分（1-5分，直接回车跳过）: ").strip()
                if rating.isdigit():
                    print(f"您的评分: {rating}/5")
                
            except KeyboardInterrupt:
                print("\n评估已终止")
                break
            except Exception as e:
                print(f"评估出错: {e}")


def load_test_data(file_path: str) -> List[Dict]:
    """加载测试数据"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.suffix == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    return data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型评估工具")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--base_model_path', type=str, help='基础模型路径（用于LoRA）')
    parser.add_argument('--test_data', type=str, help='测试数据文件路径')
    parser.add_argument('--output_file', type=str, help='结果输出文件路径')
    parser.add_argument('--interactive', action='store_true', help='交互式评估')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, args.base_model_path)
    
    if args.interactive:
        # 交互式评估
        evaluator.interactive_evaluation()
    elif args.test_data:
        # 数据集评估
        test_data = load_test_data(args.test_data)
        evaluator.evaluate_dataset(test_data, args.output_file)
    else:
        print("请指定测试数据文件或使用交互式模式")


if __name__ == "__main__":
    main()