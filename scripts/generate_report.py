#!/usr/bin/env python3
"""
模型评估报告生成器
用于生成详细的评估报告和可视化图表
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False
except ImportError as e:
    print(f"某些可视化库未安装: {e}")
    print("请安装: pip install matplotlib seaborn pandas")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_evaluation_results(self) -> Dict:
        """加载评估结果"""
        results = {}
        
        # 查找所有评估结果文件
        for file_path in self.results_dir.glob("*.json"):
            if "evaluation" in file_path.name or "eval" in file_path.name:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results[file_path.stem] = data
                        logger.info(f"加载评估结果: {file_path}")
                except Exception as e:
                    logger.warning(f"无法加载 {file_path}: {e}")
        
        return results
    
    def generate_metrics_summary(self, results: Dict) -> Dict:
        """生成指标摘要"""
        summary = {
            "models": [],
            "metrics": {
                "bleu": [],
                "rouge1": [],
                "rouge2": [],
                "rougeL": []
            }
        }
        
        for model_name, result in results.items():
            summary["models"].append(model_name)
            summary["metrics"]["bleu"].append(result.get("avg_bleu", 0))
            summary["metrics"]["rouge1"].append(result.get("avg_rouge1", 0))
            summary["metrics"]["rouge2"].append(result.get("avg_rouge2", 0))
            summary["metrics"]["rougeL"].append(result.get("avg_rougeL", 0))
        
        return summary
    
    def create_metrics_comparison_chart(self, summary: Dict):
        """创建指标对比图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('模型评估指标对比', fontsize=16, fontweight='bold')
            
            metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL']
            metric_names = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                ax = axes[i // 2, i % 2]
                
                values = summary["metrics"][metric]
                models = summary["models"]
                
                bars = ax.bar(models, values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
                ax.set_title(f'{name} 分数对比', fontweight='bold')
                ax.set_ylabel('分数')
                ax.set_ylim(0, max(values) * 1.1 if values else 1)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # 旋转x轴标签
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = self.output_dir / "metrics_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"指标对比图表已保存: {chart_path}")
            
        except Exception as e:
            logger.error(f"创建指标对比图表失败: {e}")
    
    def create_detailed_metrics_heatmap(self, results: Dict):
        """创建详细指标热力图"""
        try:
            # 准备数据
            data = []
            for model_name, result in results.items():
                data.append([
                    result.get("avg_bleu", 0),
                    result.get("avg_rouge1", 0),
                    result.get("avg_rouge2", 0),
                    result.get("avg_rougeL", 0)
                ])
            
            if not data:
                logger.warning("没有数据用于创建热力图")
                return
            
            # 创建DataFrame
            df = pd.DataFrame(
                data,
                index=list(results.keys()),
                columns=['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            )
            
            # 创建热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                df,
                annot=True,
                cmap='YlOrRd',
                fmt='.3f',
                cbar_kws={'label': '分数'},
                square=True
            )
            
            plt.title('模型评估指标热力图', fontsize=16, fontweight='bold')
            plt.ylabel('模型')
            plt.xlabel('评估指标')
            plt.tight_layout()
            
            # 保存图表
            heatmap_path = self.output_dir / "metrics_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"指标热力图已保存: {heatmap_path}")
            
        except Exception as e:
            logger.error(f"创建指标热力图失败: {e}")
    
    def analyze_score_distribution(self, results: Dict):
        """分析分数分布"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('评估分数分布分析', fontsize=16, fontweight='bold')
            
            metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL']
            metric_names = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                ax = axes[i // 2, i % 2]
                
                # 收集所有模型的该指标详细分数
                all_scores = []
                for model_name, result in results.items():
                    if 'detailed_results' in result:
                        scores = [item.get(metric.replace('rouge', 'rouge'), 0) 
                                for item in result['detailed_results']]
                        all_scores.extend(scores)
                
                if all_scores:
                    ax.hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.axvline(np.mean(all_scores), color='red', linestyle='--', 
                              label=f'平均值: {np.mean(all_scores):.3f}')
                    ax.set_title(f'{name} 分数分布')
                    ax.set_xlabel('分数')
                    ax.set_ylabel('频次')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{name} 分数分布（无数据）')
            
            plt.tight_layout()
            
            # 保存图表
            dist_path = self.output_dir / "score_distribution.png"
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"分数分布图已保存: {dist_path}")
            
        except Exception as e:
            logger.error(f"分析分数分布失败: {e}")
    
    def generate_html_report(self, results: Dict, summary: Dict):
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>中文大语言模型评估报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .metric-card {{ background-color: #ecf0f1; padding: 20px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #3498db; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .highlight {{ background-color: #e8f6f3; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 中文大语言模型LoRA微调评估报告</h1>
        
        <h2>📊 评估概览</h2>
        <div class="metric-card">
            <p><strong>评估模型数量:</strong> {len(summary['models'])}</p>
            <p><strong>评估时间:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>评估指标:</strong> BLEU, ROUGE-1, ROUGE-2, ROUGE-L</p>
        </div>
        
        <h2>🏆 最佳模型表现</h2>
        """
        
        # 找出最佳模型
        if summary['models']:
            best_bleu_idx = np.argmax(summary['metrics']['bleu'])
            best_rouge1_idx = np.argmax(summary['metrics']['rouge1'])
            best_rougeL_idx = np.argmax(summary['metrics']['rougeL'])
            
            html_content += f"""
            <table>
                <tr>
                    <th>指标</th>
                    <th>最佳模型</th>
                    <th>分数</th>
                </tr>
                <tr class="highlight">
                    <td>BLEU</td>
                    <td>{summary['models'][best_bleu_idx]}</td>
                    <td>{summary['metrics']['bleu'][best_bleu_idx]:.4f}</td>
                </tr>
                <tr class="highlight">
                    <td>ROUGE-1</td>
                    <td>{summary['models'][best_rouge1_idx]}</td>
                    <td>{summary['metrics']['rouge1'][best_rouge1_idx]:.4f}</td>
                </tr>
                <tr class="highlight">
                    <td>ROUGE-L</td>
                    <td>{summary['models'][best_rougeL_idx]}</td>
                    <td>{summary['metrics']['rougeL'][best_rougeL_idx]:.4f}</td>
                </tr>
            </table>
            """
        
        html_content += """
        <h2>📈 指标对比图表</h2>
        <div class="chart-container">
            <img src="metrics_comparison.png" alt="指标对比图表">
        </div>
        
        <div class="chart-container">
            <img src="metrics_heatmap.png" alt="指标热力图">
        </div>
        
        <div class="chart-container">
            <img src="score_distribution.png" alt="分数分布图">
        </div>
        
        <h2>📋 详细评估结果</h2>
        <table>
            <tr>
                <th>模型名称</th>
                <th>BLEU</th>
                <th>ROUGE-1</th>
                <th>ROUGE-2</th>
                <th>ROUGE-L</th>
                <th>样本数量</th>
            </tr>
        """
        
        # 添加详细结果
        for i, model in enumerate(summary['models']):
            result = results[model]
            html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{summary['metrics']['bleu'][i]:.4f}</td>
                <td>{summary['metrics']['rouge1'][i]:.4f}</td>
                <td>{summary['metrics']['rouge2'][i]:.4f}</td>
                <td>{summary['metrics']['rougeL'][i]:.4f}</td>
                <td>{result.get('num_samples', 'N/A')}</td>
            </tr>
            """
        
        html_content += """
        </table>
        
        <div class="footer">
            <p>📄 报告生成于 中文大语言模型LoRA微调框架</p>
            <p>🔗 项目地址: <a href="https://github.com/roclee2692/Chinese-LLM-LoRA-Finetuning">GitHub</a></p>
        </div>
    </div>
</body>
</html>
        """
        
        # 保存HTML报告
        report_path = self.output_dir / "evaluation_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML评估报告已生成: {report_path}")
    
    def generate_report(self):
        """生成完整的评估报告"""
        logger.info("开始生成评估报告...")
        
        # 加载评估结果
        results = self.load_evaluation_results()
        
        if not results:
            logger.warning("未找到评估结果文件")
            return
        
        # 生成指标摘要
        summary = self.generate_metrics_summary(results)
        
        # 创建图表
        self.create_metrics_comparison_chart(summary)
        self.create_detailed_metrics_heatmap(results)
        self.analyze_score_distribution(results)
        
        # 生成HTML报告
        self.generate_html_report(results, summary)
        
        logger.info(f"评估报告生成完成，保存在: {self.output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估报告生成器")
    parser.add_argument(
        '--results_dir', 
        type=str, 
        required=True, 
        help='评估结果目录'
    )
    
    args = parser.parse_args()
    
    # 生成报告
    generator = EvaluationReportGenerator(args.results_dir)
    generator.generate_report()


if __name__ == "__main__":
    main()