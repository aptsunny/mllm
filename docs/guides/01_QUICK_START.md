# 🚀 Qwen3 模型测试脚本 - 快速开始

## 📦 可用脚本

| 脚本 | 用途 | 推荐场景 |
|--------|--------|---------|
| **interactive_test.py** | 交互式对话 | 日常使用、功能验证 |
| **quick_test.py** | 批量测试 | 模型验证、性能测试 |

---

## 🎯 方式一：交互式对话测试

### 运行命令

```bash
cd /Users/sunyue/workspace/0_mllm_v2
python3 interactive_test.py
```

### 使用示例

```
============================================================
  Qwen3-0.6B-w4a32kai 交互式测试
============================================================

模型路径: models/Qwen3-0.6B-w4a32kai/model.mllm
Tokenizer: models/Qwen3-0.6B-w4a32kai/tokenizer.json
配置文件: models/Qwen3-0.6B-w4a32kai/config.json

提示: 输入 'quit' 或 'exit' 退出程序

模型加载成功！开始交互式对话...

💬 你: 你好！
🔄 正在处理...

🤖 Qwen3: 你好！有什么我可以帮助你的吗？

性能指标:
  Prefill 时间: 134239.00 μs (126.64 tokens/s)
  Decode 时间:   78098.00 μs ( 64.02 tokens/s)
  TTFT: 134513.00 μs
  平均 Decode 时间:   15619.60 μs/token

💬 你: quit

再见！
```

### 特性

- ✅ 实时交互式对话
- 📊 显示详细性能指标
- 🎨 彩色输出，易于阅读
- ⚠️ 完整的错误处理
- 🌐 支持中英文输入

---

## 🚀 方式二：批量快速测试

### 运行命令

```bash
cd /Users/sunyue/workspace/0_mllm_v2
python3 quick_test.py
```

### 使用示例

```
======================================================================
  Qwen3-0.6B-w4a32kai 快速批量测试
======================================================================

开始测试 7 个提示...

正在运行测试 1/7...

[测试 1/7]
输入: Hello, world!
输出: Hello, world!
性能: TTFT=952138.00 μs Avg=14668.75 μs/token

...

======================================================================
测试总结
======================================================================

总测试数: 7
成功数: 7
失败数: 0
总耗时: 45.23 秒
平均耗时: 6.46 秒/测试
成功率: 100.0%
```

### 测试用例

脚本会自动测试以下 7 个用例：

1. **简单问候**: "Hello, world!"
2. **英文问答**: "What is the capital of France?"
3. **中文问答**: "中国的首都是哪里？"
4. **代码生成**: "Write a Python function to calculate factorial..."
5. **概念解释**: "Explain quantum computing in simple terms."
6. **中文概念**: "什么是人工智能？"
7. **创意写作**: "Write a haiku about programming."

### 特性

- 🚀 自动运行 7 个测试用例
- 📊 自动生成测试报告
- ⏱️ 统计总耗时和平均耗时
- ✅ 计算成功率
- 🎯 包含中英文测试

---

## 💡 推荐使用流程

### 第一次使用（验证模型）

```bash
# 1. 运行批量测试验证模型
python3 quick_test.py

# 2. 检查测试报告
# 确保成功率 > 90%
```

### 日常使用（交互式对话）

```bash
# 1. 运行交互式脚本
python3 interactive_test.py

# 2. 开始对话
💬 你: 你的问题...
```

---

## ⚙️ 自定义配置

### 修改模型路径

编辑脚本中的配置部分：

```python
# 在 interactive_test.py 或 quick_test.py 中
MODEL_PATH = "models/Qwen3-0.6B-w4a32kai/model.mllm"
TOKENIZER_PATH = "models/Qwen3-0.6B-w4a32kai/tokenizer.json"
CONFIG_PATH = "models/Qwen3-0.6B-w4a32kai/config.json"
RUNNER_PATH = "./build-osx-accelerate/bin/mllm-qwen3-runner"
```

### 修改测试用例（仅限 quick_test.py）

```python
# 在 quick_test.py 中修改 TEST_PROMPTS
TEST_PROMPTS = [
    "你的自定义提示 1",
    "Your custom prompt 2",
    # 添加更多...
]
```

### 增加超时时间

如果推理较慢，可以增加超时：

```python
# interactive_test.py 第 60 行
stdout, stderr = process.communicate(input=prompt + "\nexit\n", timeout=240)

# quick_test.py 第 58 行
stdout, stderr = process.communicate(input=prompt + "\nexit\n", timeout=240)
```

---

## 🔧 故障排除

### 问题：模型文件不存在

```bash
# 检查文件
ls -lh models/Qwen3-0.6B-w4a32kai/model.mllm

# 应该显示: -rw-r--r-- 1 sunyue staff 1.5G ...
```

### 问题：推理程序不存在

```bash
# 检查程序
ls -lh build-osx-accelerate/bin/mllm-qwen3-runner

# 应该显示: -rwxr-xr-x 1 sunyue staff 357K ...
```

### 问题：推理超时

- 增加超时时间（见配置修改）
- 检查系统资源（CPU、内存）
- 简化输入提示

---

## 📊 性能指标说明

| 指标 | 说明 | 正常范围 |
|--------|------|---------|
| **TTFT** | Time to First Token（首字时间） | 100-1000 ms |
| **Prefill 时间** | 处理输入的时间 | 100-200 ms |
| **Decode 时间** | 生成所有 token 的总时间 | 50-3000 ms |
| **平均 Decode 时间** | 每个 token 的平均生成时间 | 10-20 ms |

---

## 🎯 快速开始

```bash
# 进入项目目录
cd /Users/sunyue/workspace/0_mllm_v2

# 选择一个脚本运行

# 交互式测试（推荐用于日常使用）
python3 interactive_test.py

# 批量测试（推荐用于验证模型）
python3 quick_test.py
```

---

## 📝 详细文档

- **交互式测试详细说明**: [INTERACTIVE_TEST_README.md](file:///Users/sunyue/workspace/0_mllm_v2/INTERACTIVE_TEST_README.md)
- **完整使用指南**: [TEST_SCRIPTS_GUIDE.md](file:///Users/sunyue/workspace/0_mllm_v2/TEST_SCRIPTS_GUIDE.md)

---

## ✨ 开始测试吧！

选择适合你需求的脚本，开始测试 Qwen3 模型！🚀
