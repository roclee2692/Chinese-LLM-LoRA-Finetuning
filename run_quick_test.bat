@echo off
echo 开始快速测试...
call .\llm-lora\Scriptsctivate.bat
python src/train.py --config configs/quick_test.yaml
pause
