# 安装指南

> **⚠️ 重要提示**: 本文档是基于源码分析由AI生成的。虽然我们努力确保准确性，但仍可能存在不一致或问题。我们正在积极改进和验证所有内容。如遇到任何问题，请及时报告。

## 中文安装指南

### 系统要求

- Python 3.9 或更高版本
- pip 包管理器

### 从 PyPI 安装（推荐）

```bash
pip install coords-nsga2
```

### 从源码安装

如果您想安装最新开发版本或修改代码：

```bash
git clone https://github.com/ZXF1001/coords-nsga2.git
cd coords-nsga2
pip install -e .
```

### 开发环境安装

如果您想参与开发：

```bash
git clone https://github.com/ZXF1001/coords-nsga2.git
cd coords-nsga2
pip install -e ".[test]"
```

### 验证安装

安装完成后，您可以通过以下方式验证安装：

```python
import coords_nsga2
print(coords_nsga2.__version__)
```

## 依赖包

### 必需依赖

- **numpy >= 1.23**: 数值计算库
- **tqdm >= 4**: 进度条显示
- **shapely >= 2**: 几何计算库

### 可选依赖

- **scipy**: 用于距离计算和其他科学计算
- **matplotlib**: 用于结果可视化

### 开发依赖

- **pytest >= 8.2**: 测试框架
- **pytest-cov >= 5**: 测试覆盖率
- **coverage[toml] >= 7.5**: 代码覆盖率
- **hypothesis >= 6.100**: 属性测试
- **ruff >= 0.11**: 代码格式化和检查
- **pre-commit >= 3.7**: Git钩子