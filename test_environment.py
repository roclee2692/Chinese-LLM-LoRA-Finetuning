#!/usr/bin/env python3
"""
环境独立的训练测试脚本
"""

def test_imports():
    """测试导入"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"🖥️  CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 未安装")
        return False
    
    try:
        import peft
        print(f"✅ PEFT: {peft.__version__}")
    except ImportError:
        print("❌ PEFT 未安装")
        return False
    
    try:
        from datasets import load_from_disk
        print("✅ Datasets 可用")
    except ImportError:
        print("❌ Datasets 未安装")
        return False
    
    return True

def main():
    print("🚀 测试环境和导入...")
    
    if test_imports():
        print("\n🎉 所有依赖都可用！")
        print("📋 可以运行完整训练:")
        print("1. 激活虚拟环境: .\\llm-lora\\Scripts\\Activate.ps1")
        print("2. 运行训练: python src/train.py --config configs/quick_test.yaml")
    else:
        print("\n❌ 部分依赖缺失，请检查虚拟环境")

if __name__ == "__main__":
    main()
