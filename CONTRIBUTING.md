# 贡献指南

感谢你对 NR-IQA 项目的关注！我们欢迎各种形式的贡献。

## 目录

- [报告问题](#报告问题)
- [提交代码](#提交代码)
- [代码风格](#代码风格)
- [文档](#文档)
- [测试](#测试)
- [提交 PR](#提交-pr)

---

## 报告问题

如果你发现 bug 或有改进建议，请提交 Issue：

### 提交 Bug 报告

```markdown
**描述 Bug**
清晰简洁地描述是什么 bug。

**复现步骤**
1. 执行...
2. 运行...
3. 看到...

**预期行为**
描述你期望的行为。

**实际行为**
描述实际发生的行为。

**环境**
- Python 版本: 
- OS: 
- 其他相关信息: 

**额外信息**
任何其他可能有用的信息（日志、堆栈跟踪等）。
```

### 提交功能建议

```markdown
**特性描述**
简洁描述你建议的新功能。

**使用场景**
描述在哪些场景下需要这个功能。

**建议实现**
如果你有想法，描述如何实现这个功能。

**其他信息**
任何其他相关信息。
```

---

## 提交代码

### 环境设置

1. **Fork 仓库**
   ```bash
   # 访问 GitHub 仓库，点击 Fork 按钮
   ```

2. **克隆仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NR-IQA-Semantic-Quality.git
   cd NR-IQA-Semantic-Quality
   ```

3. **创建开发分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b bugfix/your-bug-fix-name
   ```

4. **安装开发依赖**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8
   ```

### 开发工作流

1. **创建新分支**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **进行更改**
   - 遵循[代码风格](#代码风格)
   - 添加必要的测试
   - 更新文档

3. **提交更改**
   ```bash
   git add .
   git commit -m "Add: 清晰描述你的更改"
   ```

4. **推送到 Fork**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **创建 Pull Request**

---

## 代码风格

### Python 代码规范

我们遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 标准。

#### 命名约定

```python
# 类名使用 PascalCase
class ImageAnalyzer:
    pass

# 函数/变量名使用 snake_case
def compute_quality_score():
    pass

# 常量使用 UPPER_CASE
MAX_ITERATIONS = 1000
DEFAULT_THRESHOLD = 0.5
```

#### 代码格式化

使用 Black 进行自动格式化：

```bash
# 格式化整个项目
black src/ examples.py

# 格式化特定文件
black src/quality_index_model.py
```

#### 代码检查

使用 Flake8 进行代码检查：

```bash
# 检查整个项目
flake8 src/ examples.py

# 检查特定文件
flake8 src/utils.py
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def compute_quality(image: np.ndarray, prompt: Optional[str] = None) -> Dict[str, float]:
    """
    计算图像质量指标。
    
    对输入图像进行多维度质量评估，包括语义保真度、
    技术质量和结构完整性。
    
    Args:
        image: 输入图像，形状为 (H, W, 3) 或 (H, W)
        prompt: 文本提示词，用于语义分析 (可选)
        
    Returns:
        包含各项质量指标的字典：
        - 'overall_quality': 综合质量指数 ∈ [0, 1]
        - 'semantic_fidelity': 语义保真度
        - 'technical_quality': 技术质量
        - 'structural_integrity': 结构完整性
        
    Raises:
        ValueError: 如果图像格式无效
        TypeError: 如果输入类型错误
        
    Example:
        >>> image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        >>> prompt = "A beautiful sunset"
        >>> result = compute_quality(image, prompt)
        >>> print(result['overall_quality'])
        0.7234
    """
    pass
```

### 类型注解

在所有函数中使用类型注解：

```python
from typing import Dict, List, Optional, Tuple

def process_images(
    image_paths: List[str],
    prompts: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """处理图像列表."""
    pass
```

---

## 文档

### 修改 README

如果你的更改涉及新功能或 API 变化，请相应更新 README.md。

### 添加示例

新功能应该有相应的使用示例。更新 `examples.py` 文件。

### API 文档

在代码中添加详细的文档字符串，包括：
- 函数/类的描述
- 参数说明
- 返回值说明
- 异常说明
- 使用示例

---

## 测试

### 编写测试

为新功能添加单元测试：

```bash
# 创建测试文件
touch tests/test_new_feature.py
```

```python
import pytest
import numpy as np
from src.quality_index_model import NRIQAModel

def test_model_initialization():
    """测试模型初始化."""
    model = NRIQAModel(
        weight_semantic=0.35,
        weight_technical=0.40,
        weight_structural=0.25
    )
    assert model.weight_semantic == 0.35
    assert model.weight_technical == 0.40
    assert model.weight_structural == 0.25

def test_invalid_weights():
    """测试权重验证."""
    with pytest.raises(AssertionError):
        NRIQAModel(
            weight_semantic=0.5,
            weight_technical=0.3,
            weight_structural=0.3  # 和不为 1
        )
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_quality_model.py

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

---

## 提交 PR

### PR 检查清单

在提交 PR 前，请确保：

- [ ] 代码遵循风格指南（已运行 Black 和 Flake8）
- [ ] 添加了必要的测试
- [ ] 所有测试通过 (`pytest`)
- [ ] 更新了相关文档
- [ ] 提交信息清晰明确
- [ ] 没有添加不必要的依赖

### PR 信息模板

```markdown
## 描述
简洁描述这个 PR 的目的和内容。

## 类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 性能优化
- [ ] 文档更新
- [ ] 其他

## 相关 Issue
关联到相关 Issue（如有）：Fixes #123

## 更改内容
- 更改 1
- 更改 2
- 更改 3

## 测试
- [ ] 添加了相关测试
- [ ] 所有测试通过

## 截图/视频（如适用）
如果有视觉更改，请附加截图或视频。

## 破坏性更改
是否有任何破坏性的 API 更改？

## 其他信息
任何其他需要注意的信息。
```

---

## 提交信息规范

遵循以下格式编写提交信息：

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码风格（不改变功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 添加或更新测试
- `chore`: 其他更改（依赖、配置等）

### 示例

```
feat(semantic-parser): Add multi-language support

- Added support for Chinese text parsing
- Improved keyword extraction accuracy
- Added unit tests for new language support

Closes #456
```

---

## 开发最佳实践

### 1. 代码审查

- 在提交 PR 前进行自审
- 确保代码可读性
- 避免过度设计

### 2. 提交频率

- 频繁的小提交优于大的合并提交
- 每个提交应该是一个完整的、可工作的单元

### 3. 分支管理

```
main (稳定版本)
  ↓
develop (开发分支)
  ↓
feature/xxx (功能分支)
bugfix/xxx (修复分支)
```

### 4. 版本管理

遵循 [语义化版本](https://semver.org/lang/zh-CN/)：

- MAJOR: 不兼容的 API 更改
- MINOR: 向后兼容的新功能
- PATCH: 向后兼容的 Bug 修复

---

## 联系方式

- **Issues**: GitHub Issues
- **讨论**: GitHub Discussions
- **邮件**: FFbond-hao@example.com

---

## 致谢

感谢所有贡献者！你们的贡献使这个项目变得更好。

---

**再次感谢你的贡献！** 🎉
