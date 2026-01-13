# Qwen3 模型交互式测试脚本使用说明

## 功能特性

- ✅ 交互式对话界面
- 📊 实时性能指标显示
- 🎨 彩色输出，易于阅读
- ⚠️ 错误处理和文件检查
- 🚀 支持中英文输入

## 使用方法

### 1. 运行脚本

```bash
cd /Users/sunyue/workspace/0_mllm_v2
python3 interactive_test.py
```

### 2. 交互式对话

脚本启动后，会显示欢迎信息：

```
============================================================
  Qwen3-0.6B-w4a32kai 交互式测试
============================================================

模型路径: models/Qwen3-0.6B-w4a32kai/model.mllm
Tokenizer: models/Qwen3-0.6B-w4a32kai/tokenizer.json
配置文件: models/Qwen3-0.6B-w4a32kai/config.json

提示: 输入 'quit' 或 'exit' 退出程序

模型加载成功！开始交互式对话...
```

### 3. 输入提示

在提示符后输入你的问题：

```
💬 你: Hello, world!
```

### 4. 查看响应

模型会返回响应和性能指标：

```
🔄 正在处理...

🤖 Qwen3: Hello, world!

性能指标:
  Prefill 时间: 951900.00 μs ( 16.81 tokens/s)
  Decode 时间:   58675.00 μs ( 68.17 tokens/s)
  TTFT: 952138.00 μs
  平均 Decode 时间:   14668.75 μs/token
```

### 5. 退出程序

输入以下任一命令退出：

```
quit
exit
q
```

## 性能指标说明

| 指标 | 说明 |
|--------|------|
| **Prefill 时间** | 首次处理输入的时间 |
| **Decode 时间** | 生成每个 token 的总时间 |
| **TTFT** | Time to First Token，首字生成时间 |
| **平均 Decode 时间** | 平均每个 token 的生成时间 |

## 测试示例

### 英文测试

```
💬 你: What is the capital of France?
🤖 Qwen3: The capital of France is Paris.
```

### 中文测试

```
💬 你: 中国的首都是哪里？
🤖 Qwen3: 中国的首都是北京。
```

### 代码生成测试

```
💬 你: Write a Python function to calculate factorial.
🤖 Qwen3: [生成完整的 Python 函数]
```

## 配置说明

如需修改模型路径，编辑脚本中的配置：

```python
MODEL_PATH = "models/Qwen3-0.6B-w4a32kai/model.mllm"
TOKENIZER_PATH = "models/Qwen3-0.6B-w4a32kai/tokenizer.json"
CONFIG_PATH = "models/Qwen3-0.6B-w4a32kai/config.json"
RUNNER_PATH = "./build-osx-accelerate/bin/mllm-qwen3-runner"
```

## 注意事项

1. **文件检查**: 脚本会自动检查所有必需文件是否存在
2. **超时设置**: 默认推理超时为 120 秒
3. **错误处理**: 遇到错误会显示详细错误信息
4. **中断处理**: 支持 Ctrl+C 中断程序

## 故障排除

### 错误: 模型文件不存在

确保模型文件路径正确：

```bash
ls -lh models/Qwen3-0.6B-w4a32kai/model.mllm
```

### 错误: 推理程序不存在

确保推理程序已编译：

```bash
ls -lh build-osx-accelerate/bin/mllm-qwen3-runner
```

### 推理超时

可以增加超时时间，编辑脚本中的 `timeout` 参数：

```python
stdout, stderr = process.communicate(input=prompt + "\nexit\n", timeout=240)  # 240秒
```

## 快速开始

```bash
# 进入项目目录
cd /Users/sunyue/workspace/0_mllm_v2

# 运行交互式测试
python3 interactive_test.py

# 开始对话！
💬 你: 你好！
```

## 技术支持

如有问题，请检查：
1. 模型文件是否完整
2. 推理程序是否正确编译
3. 系统资源是否充足（内存、CPU）
