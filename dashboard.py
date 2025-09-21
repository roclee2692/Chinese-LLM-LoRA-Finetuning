#!/usr/bin/env python3
"""
Chinese LLM LoRA 训练综合状态仪表板
提供训练状态概览、系统监控和操作建议
"""

import json
import subprocess
import psutil
from pathlib import Path
import datetime
import time

class TrainingDashboard:
    """训练状态仪表板"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.results_dir = self.project_root / "results"
        self.models_dir = self.results_dir / "models"
        self.cache_dir = self.project_root / "cache"
        
    def get_emoji_status(self, condition):
        """根据条件返回表情符号"""
        return "✅" if condition else "❌"
    
    def get_gpu_info(self):
        """获取GPU信息"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                return {
                    "name": gpu_data[0],
                    "memory_total": int(gpu_data[1]),
                    "memory_used": int(gpu_data[2]),
                    "memory_free": int(gpu_data[3]),
                    "utilization": int(gpu_data[4]),
                    "temperature": int(gpu_data[5])
                }
        except:
            pass
        return None
    
    def check_models_status(self):
        """检查模型状态"""
        status = {
            "qwen_downloaded": False,
            "qwen_size": 0,
            "distilgpt2_downloaded": False,
            "chatglm3_downloaded": False,
            "completed_trainings": [],
            "total_models": 0
        }
        
        # 检查缓存的模型
        if (self.cache_dir / "models--Qwen--Qwen-1_8B-Chat").exists():
            status["qwen_downloaded"] = True
            # 计算模型大小
            qwen_path = self.cache_dir / "models--Qwen--Qwen-1_8B-Chat"
            try:
                total_size = sum(f.stat().st_size for f in qwen_path.rglob('*') if f.is_file())
                status["qwen_size"] = total_size / (1024**3)  # GB
            except:
                pass
        
        if (self.cache_dir / "models--distilgpt2").exists():
            status["distilgpt2_downloaded"] = True
            
        if (self.cache_dir / "models--THUDM--chatglm3-6b").exists():
            status["chatglm3_downloaded"] = True
        
        # 检查训练结果
        if self.models_dir.exists():
            for model_dir in self.models_dir.glob("*"):
                if model_dir.is_dir() and model_dir.name != ".gitkeep":
                    status["total_models"] += 1
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
                                "is_completed": trainer_state.get('global_step', 0) >= trainer_state.get('max_steps', 1),
                                "last_modified": trainer_state_file.stat().st_mtime
                            }
                            status["completed_trainings"].append(model_info)
                        except:
                            pass
        
        return status
    
    def get_system_resources(self):
        """获取系统资源信息"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_total": memory.total / (1024**3),
            "memory_used": memory.used / (1024**3),
            "memory_percent": memory.percent,
            "disk_total": disk.total / (1024**3),
            "disk_used": disk.used / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100
        }
    
    def check_training_readiness(self):
        """检查训练准备状态"""
        checks = {
            "gpu_available": False,
            "cuda_available": False,
            "pytorch_installed": False,
            "transformers_installed": False,
            "peft_installed": False,
            "qwen_model_ready": False,
            "disk_space_ok": False,
            "memory_sufficient": False
        }
        
        # GPU检查
        gpu_info = self.get_gpu_info()
        checks["gpu_available"] = gpu_info is not None
        
        # CUDA检查
        try:
            import torch
            checks["cuda_available"] = torch.cuda.is_available()
            checks["pytorch_installed"] = True
        except ImportError:
            pass
        
        # 库检查
        try:
            import transformers
            checks["transformers_installed"] = True
        except ImportError:
            pass
        
        try:
            import peft
            checks["peft_installed"] = True
        except ImportError:
            pass
        
        # 模型检查
        models_status = self.check_models_status()
        checks["qwen_model_ready"] = models_status["qwen_downloaded"]
        
        # 资源检查
        system = self.get_system_resources()
        checks["disk_space_ok"] = system["disk_percent"] < 90
        checks["memory_sufficient"] = system["memory_total"] >= 16  # 至少16GB内存
        
        return checks
    
    def display_dashboard(self):
        """显示完整仪表板"""
        print("=" * 80)
        print("🚀 Chinese LLM LoRA Fine-tuning Dashboard")
        print("=" * 80)
        print(f"📅 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 训练准备状态
        print("\n📋 训练准备状态检查:")
        print("-" * 40)
        readiness = self.check_training_readiness()
        
        print(f"{self.get_emoji_status(readiness['gpu_available'])} GPU可用")
        print(f"{self.get_emoji_status(readiness['cuda_available'])} CUDA可用")
        print(f"{self.get_emoji_status(readiness['pytorch_installed'])} PyTorch已安装")
        print(f"{self.get_emoji_status(readiness['transformers_installed'])} Transformers已安装")
        print(f"{self.get_emoji_status(readiness['peft_installed'])} PEFT已安装")
        print(f"{self.get_emoji_status(readiness['qwen_model_ready'])} Qwen模型已下载")
        print(f"{self.get_emoji_status(readiness['disk_space_ok'])} 磁盘空间充足")
        print(f"{self.get_emoji_status(readiness['memory_sufficient'])} 内存充足")
        
        ready_count = sum(readiness.values())
        total_checks = len(readiness)
        print(f"\n📊 准备就绪: {ready_count}/{total_checks} ({ready_count/total_checks*100:.1f}%)")
        
        # GPU状态
        print("\n🎮 GPU状态:")
        print("-" * 40)
        gpu_info = self.get_gpu_info()
        if gpu_info:
            print(f"💻 型号: {gpu_info['name']}")
            print(f"💾 显存: {gpu_info['memory_used']}MB / {gpu_info['memory_total']}MB ({gpu_info['memory_used']/gpu_info['memory_total']*100:.1f}%)")
            print(f"⚡ 使用率: {gpu_info['utilization']}%")
            print(f"🌡️  温度: {gpu_info['temperature']}°C")
            
            # GPU状态指示
            if gpu_info['utilization'] > 80:
                gpu_status = "🔥 高负载运行"
            elif gpu_info['utilization'] > 30:
                gpu_status = "⚡ 使用中"
            else:
                gpu_status = "😴 空闲"
            print(f"📊 状态: {gpu_status}")
        else:
            print("❌ 无法获取GPU信息")
        
        # 系统资源
        print("\n💻 系统资源:")
        print("-" * 40)
        system = self.get_system_resources()
        print(f"🧠 CPU使用率: {system['cpu_percent']:.1f}%")
        print(f"💾 内存使用: {system['memory_used']:.1f}GB / {system['memory_total']:.1f}GB ({system['memory_percent']:.1f}%)")
        print(f"💿 磁盘使用: {system['disk_used']:.1f}GB / {system['disk_total']:.1f}GB ({system['disk_percent']:.1f}%)")
        
        # 模型状态
        print("\n🤖 模型状态:")
        print("-" * 40)
        models_status = self.check_models_status()
        
        print(f"{self.get_emoji_status(models_status['qwen_downloaded'])} Qwen-1.8B-Chat: ", end="")
        if models_status['qwen_downloaded']:
            print(f"已下载 ({models_status['qwen_size']:.1f}GB)")
        else:
            print("未下载")
        
        print(f"{self.get_emoji_status(models_status['distilgpt2_downloaded'])} DistilGPT2: ", end="")
        print("已下载" if models_status['distilgpt2_downloaded'] else "未下载")
        
        print(f"{self.get_emoji_status(models_status['chatglm3_downloaded'])} ChatGLM3-6B: ", end="")
        print("已下载" if models_status['chatglm3_downloaded'] else "未下载")
        
        # 训练历史
        print("\n📚 训练历史:")
        print("-" * 40)
        if models_status['completed_trainings']:
            for training in models_status['completed_trainings']:
                completion = training['global_step'] / max(training['max_steps'], 1) * 100
                last_modified = datetime.datetime.fromtimestamp(training['last_modified'])
                age = datetime.datetime.now() - last_modified
                
                status_icon = "✅" if training['is_completed'] else "🔄"
                print(f"{status_icon} {training['name']}: {completion:.1f}% ({training['global_step']}/{training['max_steps']} 步)")
                print(f"    📅 最后更新: {last_modified.strftime('%Y-%m-%d %H:%M:%S')} ({age.days}天前)")
        else:
            print("📭 暂无训练记录")
        
        # 操作建议
        print("\n💡 操作建议:")
        print("-" * 40)
        
        if ready_count == total_checks:
            print("🎉 所有检查通过！可以开始训练")
            print("📝 运行命令: python start_qwen_training.py")
            print("🔍 监控训练: python simple_monitor.py")
        else:
            print("⚠️ 存在问题需要解决:")
            if not readiness['gpu_available']:
                print("   • 检查GPU驱动安装")
            if not readiness['cuda_available']:
                print("   • 安装CUDA或检查PyTorch CUDA版本")
            if not readiness['qwen_model_ready']:
                print("   • 下载Qwen模型: 运行 python download_models.py")
            if not readiness['memory_sufficient']:
                print("   • 建议至少16GB内存用于大模型训练")
        
        # 快捷操作
        print("\n🔧 快捷操作:")
        print("-" * 40)
        print("1. 🏋️  开始Qwen训练: python start_qwen_training.py")
        print("2. 📊 查看训练状态: python simple_monitor.py")
        print("3. 🧪 快速测试训练: python quick_training_test.py")
        print("4. 🔍 系统诊断: python verify_installation.py")
        
        print("\n" + "=" * 80)

def main():
    """主函数"""
    dashboard = TrainingDashboard()
    dashboard.display_dashboard()

if __name__ == "__main__":
    main()