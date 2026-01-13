# Qwen3 模型测试脚本使用指南

## 📦 可用脚本

### 1. `interactive_test.py` - 交互式对话测试

**功能**: 与 Qwen3 模型进行实时交互式对话

**使用方法**:
```bash
python3 interactive_test.py
```

**特性**:
- ✅ 实时交互式对话
- 📊 显示性能指标
- 🎨 彩色输出
- ⚠️ 完整的错误处理
- 🌐 支持中英文

**退出方式**: 输入 `quit`、`exit` 或 `q`

---

### 2. `quick_test.py` - 批量快速测试

**功能**: 自动运行多个测试用例并生成报告

**使用方法**:
```bash
python3 quick_test.py
```

**特性**:
- 🚀 批量测试 7 个预设提示
- 📊 自动生成测试报告
- ⏱️ 统计总耗时和平均耗时
- ✅ 计算成功率
- 🎯 包含中英文测试

**测试用例**:
1. 简单问候: "Hello, world!"
2. 英文问答: "What is the capital of France?"
3. 中文问答: "中国的首都是哪里？"
4. 代码生成: "Write a Python function..."
5. 概念解释: "Explain quantum computing..."
6. 中文问答: "什么是人工智能？"
7. 创意写作: "Write a haiku..."

---

## 🚀 快速开始

### 方式一: 交互式测试（推荐用于日常使用）

```bash
# 1. 进入项目目录
cd /Users/sunyue/workspace/0_mllm_v2

# 2. 运行交互式脚本
python3 interactive_test.py

# 3. 开始对话
💬 你: 你好！
🤖 Qwen3: 你好！有什么我可以帮助你的吗？
```

### 方式二: 批量测试（推荐用于验证模型）

```bash
# 1. 进入项目目录
cd /Users/sunyue/workspace/0_mllm_v2

# 2. 运行批量测试
python3 quick_test.py

# 3. 查看测试报告
============================================================
  Qwen3-0.6B-w4a32kai 快速批量测试
============================================================

[测试 1/7]
输入: Hello, world!
输出: Hello, world!
性能: TTFT=952138.00 μs Avg=14668.75 μs/token

...

测试总结
============================================================
总测试数: 7
成功数: 7
失败数: 0
总耗时: 45.23 秒
平均耗时: 6.46 秒/测试
成功率: 100.0%
```

---

## 📊 性能指标说明

### 交互式测试显示的指标

| 指标 | 说明 | 正常范围 |
|--------|------|---------|
| **Prefill 时间** | 首次处理输入的时间 | 100-200 ms |
| **Decode 时间** | 生成所有 token 的总时间 | 50-3000 ms |
| **TTFT** | Time to First Token | 100-1000 ms |
| **平均 Decode 时间** | 每个 token 的平均生成时间 | 10-20 ms |

### 批量测试显示的指标

| 指标 | 说明 |
|--------|------|
| **总测试数** | 运行的测试用例总数 |
| **成功数** | 成功完成的测试数 |
| **失败数** | 失败的测试数 |
| **总耗时** | 所有测试的总时间 |
| **平均耗时** | 每个测试的平均时间 |
| **成功率** | 成功测试的百分比 |

---

## ⚙️ 配置修改

### 修改模型路径

如果需要使用不同的模型，编辑脚本中的配置：

```python
# interactive_test.py 或 quick_test.py
MODEL_PATH = "models/Qwen3-0.6B-w4a32kai/model.mllm"
TOKENIZER_PATH = "models/Qwen3-0.6B-w4a32kai/tokenizer.json"
CONFIG_PATH = "models/Qwen3-0.6B-w4a32kai/config.json"
RUNNER_PATH = "./build-osx-accelerate/bin/mllm-qwen3-runner"
```

### 修改超时时间

如果推理较慢，可以增加超时时间：

```python
# interactive_test.py 第 60 行
stdout, stderr = process.communicate(input=prompt + "\nexit\n", timeout=240)  # 240秒

# quick_test.py 第 48 行
stdout, stderr = process.communicate(input=prompt + "\nexit\n", timeout=240)  # 240秒
```

### 修改测试用例

在 `quick_test.py` 中修改 `TEST_PROMPTS` 列表：

```python
TEST_PROMPTS = [
    "你的自定义提示 1",
    "Your custom prompt 2",
    # 添加更多...
]
```

---

## 🔧 故障排除

### 问题 1: 模型文件不存在

**错误信息**:
```
错误: 模型文件不存在: models/Qwen3-0.6B-w4a32kai/model.mllm
```

**解决方法**:
```bash
# 检查文件是否存在
ls -lh models/Qwen3-0.6B-w4a32kai/model.mllm

# 如果不存在，下载模型
# 参考 README 中的模型下载说明
```

### 问题 2: 推理程序不存在

**错误信息**:
```
错误: 推理程序不存在: ./build-osx-accelerate/bin/mllm-qwen3-runner
```

**解决方法**:
```bash
# 检查推理程序
ls -lh build-osx-accelerate/bin/mllm-qwen3-runner

# 如果不存在，需要编译
# 参考 README 中的编译说明
```

### 问题 3: 推理超时

**错误信息**:
```
错误: 推理超时
```

**解决方法**:
1. 增加超时时间（见配置修改部分）
2. 检查系统资源（CPU、内存）
3. 简化输入提示

### 问题 4: 输出乱码

**可能原因**:
- Tokenizer 和模型不匹配
- 模型文件损坏

**解决方法**:
```bash
# 检查 tokenizer 词汇表大小
python3 -c "
import json
with open('models/Qwen3-0.6B-w4a32kai/tokenizer.json', 'r') as f:
    tokenizer = json.load(f)
print(f'Vocab size: {len(tokenizer[\"model\"][\"vocab\"])}')
"

# 检查配置文件
cat models/Qwen3-0.6B-w4a32kai/config.json | grep vocab_size
```

---

## 💡 使用建议

### 交互式测试适用场景

- ✅ 日常对话测试
- ✅ 功能验证
- ✅ 性能调优
- ✅ 错误排查

### 批量测试适用场景

- ✅ 模型验证
- ✅ 性能基准测试
- ✅ 回归测试
- ✅ 质量评估

---

## 📝 示例输出

### 交互式测试示例

```
============================================================
  Qwen3-0.6B-w4a32kai 交互式测试
============================================================

模型路径: models/Qwen3-0.6B-w4a32kai/model.mllm
Tokenizer: models/Qwen3-0.6B-w4a32kai/tokenizer.json
配置文件: models/Qwen3-0.6B-w4a32kai/config.json

提示: 输入 'quit' 或 'exit' 退出程序

模型加载成功！开始交互式对话...

💬 你: Write a Python function to calculate factorial.

🔄 正在处理...

🤖 Qwen3: Here's a simple Python function that calculates factorial:

```python
def calculate_factorial(n):
    if n < 0:
        return 0
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

性能指标:
  Prefill 时间: 158695.00 μs ( 151.23 tokens/s)
  Decode 时间: 2969988.00 μs ( 56.57 tokens/s)
  TTFT: 158946.00 μs
  平均 Decode 时间:   17678.50 μs/token

💬 你: quit

再见！
```

---

## 🎯 总结

现在你有两个强大的测试工具：

1. **`interactive_test.py`** - 用于日常交互式对话
2. **`quick_test.py`** - 用于快速批量验证

选择适合你需求的脚本开始测试吧！🚀
