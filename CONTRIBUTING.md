# 🤝 贡献指南

感谢您对 Chinese-LLM-LoRA-Finetuning 项目的关注！我们欢迎所有形式的贡献，无论是代码、文档、问题反馈还是功能建议。

## 📋 贡献方式

### 🐛 报告问题
- 使用 GitHub Issues 报告 bug
- 提供详细的复现步骤
- 包含错误信息和环境信息

### 💡 提出建议
- 在 Issues 中标记为 `enhancement`
- 详细描述功能需求和使用场景
- 讨论实现方案的可行性

### 📝 改进文档
- 修正文档中的错误
- 添加使用示例和教程
- 翻译文档到其他语言

### 💻 贡献代码
- 修复已知 bug
- 实现新功能
- 优化性能和代码质量

## 🔧 开发环境设置

### 1. Fork 项目

点击项目页面右上角的 "Fork" 按钮，将项目 fork 到您的 GitHub 账户。

### 2. 克隆代码
   git checkout -b feature/your-feature-name
   ```

3. **安装开发依赖**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **进行更改**
   - 编写代码
   - 添加测试
   - 更新文档

5. **运行测试**
   ```bash
   pytest tests/
   ```

6. **代码格式化**
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   ```

7. **提交更改**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```

8. **推送分支**
   ```bash
   git push origin feature/your-feature-name
   ```

9. **创建Pull Request**

## 📝 代码规范

### Python代码风格
- 使用 [Black](https://github.com/psf/black) 进行代码格式化
- 使用 [isort](https://github.com/PyCQA/isort) 整理导入
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 风格指南
- 使用类型提示 (Type Hints)

### 提交信息格式
```
<type>: <description>

[optional body]

[optional footer]
```

类型可以是：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 代码重构
- `test`: 添加测试
- `chore`: 构建过程或辅助工具的变动

### 示例
```
feat: add support for Baichuan2 model

- Add Baichuan2 model configuration
- Update model loading logic
- Add corresponding tests

Closes #123
```

## 🧪 测试

请确保您的代码包含适当的测试：

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_data_preprocessing.py

# 运行覆盖率检查
pytest --cov=src tests/
```

## 📚 文档

如果您的更改涉及用户可见的功能：
- 更新相关的README文件
- 在docstring中添加详细的文档
- 如果需要，添加使用示例

## 🏷️ 版本管理

项目使用 [语义化版本](https://semver.org/lang/zh-CN/)：
- MAJOR: 不兼容的API变更
- MINOR: 向下兼容的功能新增
- PATCH: 向下兼容的问题修正

## 📋 Pull Request检查清单

在提交PR之前，请确保：

- [ ] 代码通过所有测试
- [ ] 代码遵循项目的编码规范
- [ ] 包含适当的测试用例
- [ ] 更新了相关文档
- [ ] 提交信息格式正确
- [ ] PR描述清楚地说明了更改内容

## 🎖️ 贡献者认可

我们会在项目中认可所有贡献者：
- 在README中列出贡献者
- 在发布说明中感谢贡献者
- 对重大贡献给予特别认可

## ❓ 获取帮助

如果您需要帮助：
- 创建issue进行讨论
- 发邮件至 your.email@example.com
- 查看现有的issue和PR

## 📄 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下授权。

---

再次感谢您的贡献！🙏