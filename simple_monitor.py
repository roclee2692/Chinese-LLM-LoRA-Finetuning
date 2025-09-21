#!/usr/bin/env python3
"""
简化版训练进度监控脚本
实时监控模型训练状态并生成可视化报告
"""

import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import psutil
import subprocess
import os

# 设置matplotlib后端和字体
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']  # 使用系统支持的字体
plt.rcParams['axes.unicode_minus'] = False

class SimpleTrainingMonitor:
    """简化版训练监控器"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.results_dir = self.project_root / "results"
        self.models_dir = self.results_dir / "models"
        
    def get_system_metrics(self):
        """获取系统指标"""
        try:
            # CPU和内存
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # GPU信息
            gpu_info = {"gpu_util": 0, "gpu_memory": 0, "gpu_temp": 0}
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_data = result.stdout.strip().split(', ')
                    gpu_info = {
                        "gpu_util": float(gpu_data[0]),
                        "gpu_memory": float(gpu_data[1]) / float(gpu_data[2]) * 100,
                        "gpu_temp": float(gpu_data[3])
                    }
            except:
                pass
            
            return {
                "timestamp": datetime.datetime.now(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                **gpu_info
            }
        except Exception as e:
            print(f"警告: 获取系统指标失败: {e}")
            return None
    
    def scan_training_logs(self):
        """扫描训练日志"""
        training_data = []
        
        if not self.models_dir.exists():
            return training_data
        
        # 扫描所有模型目录
        for model_dir in self.models_dir.glob("*"):
            if model_dir.is_dir():
                trainer_state_file = model_dir / "trainer_state.json"
                if trainer_state_file.exists():
                    try:
                        with open(trainer_state_file, 'r', encoding='utf-8') as f:
                            trainer_state = json.load(f)
                        
                        model_name = model_dir.name
                        log_history = trainer_state.get('log_history', [])
                        
                        for entry in log_history:
                            if 'loss' in entry:
                                training_data.append({
                                    'model': model_name,
                                    'step': entry.get('step', 0),
                                    'epoch': entry.get('epoch', 0),
                                    'loss': entry.get('loss', 0),
                                    'learning_rate': entry.get('learning_rate', 0),
                                    'grad_norm': entry.get('grad_norm', 0),
                                    'timestamp': datetime.datetime.now()
                                })
                    except Exception as e:
                        print(f"警告: 读取训练日志失败 {trainer_state_file}: {e}")
        
        return training_data
    
    def check_training_status(self):
        """检查训练状态"""
        status = {
            "models_found": 0,
            "active_training": False,
            "completed_models": [],
            "total_steps": 0,
            "total_epochs": 0
        }
        
        if not self.models_dir.exists():
            return status
        
        for model_dir in self.models_dir.glob("*"):
            if model_dir.is_dir() and model_dir.name != ".gitkeep":
                status["models_found"] += 1
                
                trainer_state_file = model_dir / "trainer_state.json"
                if trainer_state_file.exists():
                    try:
                        with open(trainer_state_file, 'r') as f:
                            trainer_state = json.load(f)
                        
                        model_info = {
                            "name": model_dir.name,
                            "global_step": trainer_state.get('global_step', 0),
                            "max_steps": trainer_state.get('max_steps', 0),
                            "epoch": trainer_state.get('epoch', 0),
                            "num_train_epochs": trainer_state.get('num_train_epochs', 0),
                            "is_completed": trainer_state.get('global_step', 0) >= trainer_state.get('max_steps', 1)
                        }
                        
                        status["completed_models"].append(model_info)
                        status["total_steps"] += model_info["global_step"]
                        status["total_epochs"] += model_info["epoch"]
                        
                        # 检查是否有训练在进行
                        if not model_info["is_completed"]:
                            # 检查最近的修改时间
                            last_modified = trainer_state_file.stat().st_mtime
                            if time.time() - last_modified < 300:  # 5分钟内修改过
                                status["active_training"] = True
                                
                    except Exception as e:
                        print(f"警告: 检查训练状态失败: {e}")
        
        return status
    
    def generate_report(self):
        """生成训练报告"""
        print("=" * 60)
        print("Chinese LLM LoRA 训练状态报告")
        print("=" * 60)
        
        # 检查训练状态
        status = self.check_training_status()
        print(f"\n发现模型: {status['models_found']} 个")
        print(f"训练进行中: {'是' if status['active_training'] else '否'}")
        print(f"总训练步数: {status['total_steps']}")
        print(f"总训练轮数: {status['total_epochs']:.1f}")
        
        if status['completed_models']:
            print(f"\n模型详情:")
            for model in status['completed_models']:
                completion = model['global_step'] / max(model['max_steps'], 1) * 100
                print(f"• {model['name']}: {completion:.1f}% 完成 ({model['global_step']}/{model['max_steps']} 步)")
        
        # 系统状态
        sys_metrics = self.get_system_metrics()
        if sys_metrics:
            print(f"\n系统状态:")
            print(f"GPU使用率: {sys_metrics['gpu_util']:.1f}%")
            print(f"GPU显存: {sys_metrics['gpu_memory']:.1f}%")
            print(f"GPU温度: {sys_metrics['gpu_temp']:.1f}°C")
            print(f"CPU使用率: {sys_metrics['cpu_percent']:.1f}%")
            print(f"内存使用: {sys_metrics['memory_used_gb']:.1f}GB / {sys_metrics['memory_total_gb']:.1f}GB ({sys_metrics['memory_percent']:.1f}%)")
        
        # 获取训练数据
        training_history = self.scan_training_logs()
        if training_history:
            print(f"\n训练历史:")
            df = pd.DataFrame(training_history)
            for model, group in df.groupby('model'):
                if len(group) > 0:
                    latest = group.iloc[-1]
                    print(f"• {model}: 最新损失 {latest['loss']:.4f}, 学习率 {latest['learning_rate']:.2e}")
        
        return status, sys_metrics, training_history
    
    def create_plots(self, training_history, sys_metrics):
        """创建可视化图表"""
        if not training_history:
            print("没有训练数据可供可视化")
            return
        
        try:
            df = pd.DataFrame(training_history)
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Chinese LLM LoRA Training Monitor', fontsize=16, fontweight='bold')
            
            # 训练损失图
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            
            for model, group in df.groupby('model'):
                if len(group) > 0:
                    ax1.plot(group['step'], group['loss'], 'o-', label=f'{model}', linewidth=2, markersize=4)
            ax1.legend()
            
            # 学习率图
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True, alpha=0.3)
            
            for model, group in df.groupby('model'):
                if len(group) > 0:
                    ax2.plot(group['step'], group['learning_rate'], 'o-', label=f'{model}', linewidth=2, markersize=4)
            ax2.legend()
            
            # 系统状态
            if sys_metrics:
                ax3.bar(['GPU Usage', 'GPU Memory', 'CPU Usage'], 
                       [sys_metrics['gpu_util'], sys_metrics['gpu_memory'], sys_metrics['cpu_percent']])
                ax3.set_title('System Usage (%)')
                ax3.set_ylabel('Percentage')
                
                # 添加数值标签
                for i, v in enumerate([sys_metrics['gpu_util'], sys_metrics['gpu_memory'], sys_metrics['cpu_percent']]):
                    ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
            
            # 训练进度
            status = self.check_training_status()
            if status['completed_models']:
                model_names = []
                completions = []
                for model in status['completed_models']:
                    model_names.append(model['name'])
                    completion = model['global_step'] / max(model['max_steps'], 1) * 100
                    completions.append(completion)
                
                ax4.barh(model_names, completions)
                ax4.set_title('Training Progress (%)')
                ax4.set_xlabel('Completion Percentage')
                
                # 添加数值标签
                for i, v in enumerate(completions):
                    ax4.text(v + 1, i, f'{v:.1f}%', ha='left', va='center')
            
            plt.tight_layout()
            
            # 保存图表
            output_file = self.results_dir / "training_monitor_report.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n图表已保存到: {output_file}")
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            print(f"创建图表时出错: {e}")

def main():
    """主函数"""
    print("Chinese LLM LoRA Training Monitor - Simple Version")
    print("=" * 60)
    
    # 创建监控器
    monitor = SimpleTrainingMonitor()
    
    # 生成报告
    status, sys_metrics, training_history = monitor.generate_report()
    
    # 创建可视化
    if training_history:
        print("\n正在生成可视化图表...")
        monitor.create_plots(training_history, sys_metrics)
    else:
        print("\n没有找到训练数据，无法生成图表")
        print("提示: 请确保训练已经开始并且有 trainer_state.json 文件生成")
    
    # 提供下一步建议
    if not status['active_training'] and status['models_found'] == 0:
        print("\n建议:")
        print("1. 检查是否已经开始训练")
        print("2. 确认训练脚本是否正常运行")
        print("3. 查看 results/models/ 目录是否有模型输出")
    elif not status['active_training'] and status['models_found'] > 0:
        print("\n检测到已完成的训练，但没有正在进行的训练")
        print("如果需要启动新的训练，请运行相应的训练脚本")
    
    print(f"\n监控完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()