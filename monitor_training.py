#!/usr/bin/env python3
"""
训练进度监控和可视化脚本
实时监控模型训练状态并生成可视化图表
"""

import json
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import psutil
import GPUtil
import os
import subprocess

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.results_dir = self.project_root / "results"
        self.logs_dir = self.results_dir / "logs"
        self.models_dir = self.results_dir / "models"
        
        # 创建日志目录
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史数据
        self.training_history = []
        self.system_metrics = []
        
        # 设置图表
        self.setup_plots()
        
    def setup_plots(self):
        """设置可视化图表"""
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('🚀 Chinese LLM LoRA 训练实时监控', fontsize=16, fontweight='bold')
        
        # 训练损失图
        self.ax1.set_title('📈 训练损失变化')
        self.ax1.set_xlabel('训练步数')
        self.ax1.set_ylabel('损失值')
        self.ax1.grid(True, alpha=0.3)
        
        # 学习率图
        self.ax2.set_title('⚡ 学习率调度')
        self.ax2.set_xlabel('训练步数')
        self.ax2.set_ylabel('学习率')
        self.ax2.grid(True, alpha=0.3)
        
        # GPU使用率图
        self.ax3.set_title('🎮 GPU状态监控')
        self.ax3.set_xlabel('时间')
        self.ax3.set_ylabel('使用率 (%)')
        self.ax3.grid(True, alpha=0.3)
        
        # 训练速度图
        self.ax4.set_title('🏃 训练速度统计')
        self.ax4.set_xlabel('训练步数')
        self.ax4.set_ylabel('步数/秒')
        self.ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def get_system_metrics(self):
        """获取系统指标"""
        try:
            # CPU和内存
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # GPU信息
            gpu_info = {"gpu_util": 0, "gpu_memory": 0, "gpu_temp": 0}
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info = {
                        "gpu_util": gpu.load * 100,
                        "gpu_memory": gpu.memoryUtil * 100,
                        "gpu_temp": gpu.temperature
                    }
            except:
                # 如果GPUtil失败，尝试nvidia-smi
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True)
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
            print(f"⚠️ 获取系统指标失败: {e}")
            return None
    
    def scan_training_logs(self):
        """扫描训练日志"""
        training_data = []
        
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
                        print(f"⚠️ 读取训练日志失败 {trainer_state_file}: {e}")
        
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
                        print(f"⚠️ 检查训练状态失败: {e}")
        
        return status
    
    def update_plots(self, frame):
        """更新图表"""
        # 获取系统指标
        sys_metrics = self.get_system_metrics()
        if sys_metrics:
            self.system_metrics.append(sys_metrics)
            # 只保留最近100个数据点
            if len(self.system_metrics) > 100:
                self.system_metrics = self.system_metrics[-100:]
        
        # 获取训练数据
        self.training_history = self.scan_training_logs()
        
        # 清除旧图表
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        # 重新设置标题和标签
        self.ax1.set_title('📈 训练损失变化')
        self.ax1.set_xlabel('训练步数')
        self.ax1.set_ylabel('损失值')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('⚡ 学习率调度')
        self.ax2.set_xlabel('训练步数')
        self.ax2.set_ylabel('学习率')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('🎮 GPU状态监控')
        self.ax3.set_xlabel('时间')
        self.ax3.set_ylabel('使用率 (%)')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title('🏃 训练速度统计')
        self.ax4.set_xlabel('时间')
        self.ax4.set_ylabel('状态')
        self.ax4.grid(True, alpha=0.3)
        
        # 绘制训练数据
        if self.training_history:
            df = pd.DataFrame(self.training_history)
            
            # 按模型分组绘制
            for model, group in df.groupby('model'):
                if len(group) > 0:
                    # 损失曲线
                    self.ax1.plot(group['step'], group['loss'], 'o-', label=f'{model}', linewidth=2, markersize=4)
                    
                    # 学习率曲线
                    self.ax2.plot(group['step'], group['learning_rate'], 'o-', label=f'{model}', linewidth=2, markersize=4)
            
            self.ax1.legend()
            self.ax2.legend()
        
        # 绘制系统指标
        if self.system_metrics:
            df_sys = pd.DataFrame(self.system_metrics)
            times = df_sys['timestamp']
            
            # GPU使用率
            self.ax3.plot(times, df_sys['gpu_util'], 'r-', label='GPU使用率', linewidth=2)
            self.ax3.plot(times, df_sys['gpu_memory'], 'b-', label='GPU显存', linewidth=2)
            self.ax3.plot(times, df_sys['cpu_percent'], 'g-', label='CPU使用率', linewidth=2)
            self.ax3.legend()
            
            # 格式化时间轴
            self.ax3.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            plt.setp(self.ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 显示训练状态
        status = self.check_training_status()
        status_text = f"""📊 训练状态总览:
        
🔍 发现模型: {status['models_found']} 个
🔄 训练进行中: {'是' if status['active_training'] else '否'}
📈 总训练步数: {status['total_steps']}
🎯 总训练轮数: {status['total_epochs']:.1f}

💾 已完成模型:"""
        
        for model in status['completed_models']:
            completion = model['global_step'] / max(model['max_steps'], 1) * 100
            status_text += f"""
• {model['name']}: {completion:.1f}% ({model['global_step']}/{model['max_steps']} 步)"""
        
        self.ax4.text(0.05, 0.95, status_text, transform=self.ax4.transAxes, 
                     verticalalignment='top', fontsize=10, 
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        self.ax4.set_xlim(0, 1)
        self.ax4.set_ylim(0, 1)
        self.ax4.axis('off')
        
        # 更新总标题显示当前时间
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        gpu_status = "🟢 GPU空闲"
        if self.system_metrics:
            latest = self.system_metrics[-1]
            if latest['gpu_util'] > 50:
                gpu_status = f"🔥 GPU训练中 ({latest['gpu_util']:.1f}%)"
            elif latest['gpu_util'] > 10:
                gpu_status = f"🟡 GPU使用中 ({latest['gpu_util']:.1f}%)"
        
        self.fig.suptitle(f'🚀 Chinese LLM LoRA 训练监控 - {current_time} - {gpu_status}', 
                         fontsize=14, fontweight='bold')
        
    def start_monitoring(self, interval=5000):
        """开始监控"""
        print("🚀 启动训练监控系统...")
        print(f"📊 监控间隔: {interval/1000}秒")
        print("🔍 监控目录:", self.results_dir.absolute())
        print("💡 按 Ctrl+C 停止监控")
        
        # 创建动画
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=interval, blit=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n⏹️ 监控已停止")
        
        return ani

def main():
    """主函数"""
    print("🎯 Chinese LLM LoRA 训练监控系统")
    print("=" * 50)
    
    # 检查环境
    try:
        import matplotlib
        import pandas
        print("✅ 依赖检查完成")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install matplotlib pandas psutil GPUtil")
        return
    
    # 创建监控器
    monitor = TrainingMonitor()
    
    # 首次状态检查
    status = monitor.check_training_status()
    print(f"\n📊 当前状态:")
    print(f"🔍 发现模型: {status['models_found']} 个")
    print(f"🔄 训练进行中: {'是' if status['active_training'] else '否'}")
    print(f"📈 总训练步数: {status['total_steps']}")
    
    if status['completed_models']:
        print(f"\n💾 模型详情:")
        for model in status['completed_models']:
            completion = model['global_step'] / max(model['max_steps'], 1) * 100
            print(f"• {model['name']}: {completion:.1f}% 完成")
    
    # 系统状态
    sys_metrics = monitor.get_system_metrics()
    if sys_metrics:
        print(f"\n🖥️ 系统状态:")
        print(f"🎮 GPU使用率: {sys_metrics['gpu_util']:.1f}%")
        print(f"💾 GPU显存: {sys_metrics['gpu_memory']:.1f}%")
        print(f"🧠 CPU使用率: {sys_metrics['cpu_percent']:.1f}%")
        print(f"📊 内存使用: {sys_metrics['memory_used_gb']:.1f}GB / {sys_metrics['memory_total_gb']:.1f}GB")
    
    print(f"\n🚀 启动实时监控界面...")
    
    # 开始监控
    monitor.start_monitoring(interval=5000)  # 5秒更新一次

if __name__ == "__main__":
    main()