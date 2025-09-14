#!/usr/bin/env python3
"""
验证环境安装脚本
运行: python verify_installation.py
"""

import sys
import importlib
from packaging import version

def check_package(package_name, min_version=None):
    """检查包是否安装以及版本"""
    try:
        module = importlib.import_module(package_name)
        installed_version = getattr(module, '__version__', 'Unknown')
        
        if min_version and installed_version != 'Unknown':
            if version.parse(installed_version) < version.parse(min_version):
                print(f"❌ {package_name}: {installed_version} (需要 >= {min_version})")
                return False
        
        print(f"✅ {package_name}: {installed_version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False

def main():
    print("🔍 检查Python环境...")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print("-" * 50)
    
    # 核心依赖检查
    packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.35.0"),
        ("peft", "0.6.0"),
        ("datasets", "2.14.0"),
        ("accelerate", "0.24.0"),
        ("bitsandbytes", None),
        ("gradio", "4.0.0"),
        ("wandb", None),
        ("pandas", "2.0.0"),
        ("numpy", "1.24.0"),
        ("matplotlib", "3.7.0"),
    ]
    
    all_good = True
    for package, min_ver in packages:
        if not check_package(package, min_ver):
            all_good = False
    
    print("-" * 50)
    
    # CUDA检查
    try:
        import torch
        print(f"🚀 CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
        else:
            print("⚠️  未检测到CUDA，将使用CPU模式")
    except ImportError:
        print("❌ PyTorch未正确安装")
        all_good = False
    
    print("-" * 50)
    
    # 内存检查
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 系统内存: {memory.total // (1024**3)}GB (可用: {memory.available // (1024**3)}GB)")
        
        if memory.total < 16 * (1024**3):
            print("⚠️  建议至少16GB内存进行模型训练")
    except ImportError:
        print("⚠️  无法检查系统内存")
    
    if all_good:
        print("\n🎉 环境检查通过！可以开始训练了！")
    else:
        print("\n❌ 环境存在问题，请检查上述错误")
    
    return all_good

if __name__ == "__main__":
    main()