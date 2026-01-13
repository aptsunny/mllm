# MLLM 量化工具测试指南

## 📦 量化工具概述

`mllm-quantizer` 是 MLLM 框架提供的量化工具，用于将 FP16/FP32 模型转换为量化模型，以减小模型大小并提高推理速度。

---

## 🚀 快速开始

### 1. 运行测试脚本

```bash
cd /Users/sunyue/workspace/0_mllm_v2
python3 test_quantizer.py
```

### 2. 查看测试结果

脚本会自动：
- ✅ 检查量化工具是否存在
- 📖 显示使用说明
- 🎨 列出支持的量化类型
- 🧪 测试量化工具是否正常运行
- 📝 创建示例配置文件

---

## 📋 使用方法

### 基本命令格式

```bash
./build-osx-accelerate/bin/mllm-quantizer \
  <输入文件> \
  -c <配置文件> \
  -o <输出文件> \
  -iv <输入版本> \
  -ov <输出版本>
```

### 参数说明

| 参数 | 说明 | 必需 | 默认值 |
|--------|------|--------|---------|
| `<FILE>` | 输入模型文件路径 | ✅ | - |
| `-c, --config` | 量化配置文件路径 | ✅ | - |
| `-o, --output` | 输出模型文件路径 | ✅ | - |
| `-iv, --input_version` | 输入文件版本（v1 或 v2） | ❌ | v2 |
| `-ov, --output_version` | 输出文件版本（v1 或 v2） | ❌ | v2 |
| `-h, --help` | 显示帮助信息 | ❌ | - |

---

## 🎯 支持的量化类型

### 1. KAI w4a32（推荐用于 Apple Silicon）

- **描述**: 4-bit 权重量化，32-bit 激活
- **适用平台**: Apple Silicon (M1/M2/M3)
- **优点**: 推理速度快，内存占用小
- **实现**: `KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32`

### 2. KAI w4a8（推荐用于 Apple Silicon）

- **描述**: 4-bit 权重量化，8-bit 激活
- **适用平台**: Apple Silicon (M1/M2/M3)
- **优点**: 推理速度快，内存占用小
- **实现**: `KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk`

### 3. GGUF Q4_0（通用）

- **描述**: GGUF 格式 Q4 量化
- **适用平台**: 通用（x86, ARM）
- **优点**: 兼容性好，文件小
- **实现**: `GGUF_Q4_0`

### 4. GGUF Q8_0（通用）

- **描述**: GGUF 格式 Q8 量化
- **适用平台**: 通用（x86, ARM）
- **优点**: 精度高，兼容性好
- **实现**: `GGUF_Q8_0`

---

## 📝 配置文件格式

### 示例配置文件

```json
{
    "quantization": {
        "type": "w4a32",
        "weight_dtype": "qai8dxp",
        "activation_dtype": "qsi4c32p",
        "linear_impl_type": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32"
    }
}
```

### 配置参数说明

| 参数 | 说明 | 可选值 |
|--------|------|---------|
| `type` | 量化类型 | w4a32, w4a8, q4_0, q8_0 |
| `weight_dtype` | 权重数据类型 | qai8dxp, qai8dxp, qsi8d32p |
| `activation_dtype` | 激活数据类型 | qsi4c32p, qai4c32p |
| `linear_impl_type` | 线性层实现类型 | KaiLinear_f32_*, GGUF_* |

---

## 💡 使用示例

### 示例 1: 量化为 w4a32（推荐）

```bash
./build-osx-accelerate/bin/mllm-quantizer \
  models/Qwen3-0.6B-fp16.mllm \
  -c config_w4a32.json \
  -o models/Qwen3-0.6B-w4a32.mllm \
  -iv v2 \
  -ov v2
```

### 示例 2: 量化为 w4a8

```bash
./build-osx-accelerate/bin/mllm-quantizer \
  models/Qwen3-0.6B-fp16.mllm \
  -c config_w4a8.json \
  -o models/Qwen3-0.6B-w4a8.mllm \
  -iv v2 \
  -ov v2
```

### 示例 3: 量化为 GGUF Q4

```bash
./build-osx-accelerate/bin/mllm-quantizer \
  models/Qwen3-0.6B-fp16.mllm \
  -c config_q4.json \
  -o models/Qwen3-0.6B-q4.mllm \
  -iv v2 \
  -ov v2
```

---

## 📊 量化效果对比

| 量化类型 | 模型大小 | 推理速度 | 精度 | 推荐场景 |
|---------|---------|---------|--------|---------|
| **FP16** | 100% | 100% | 100% | 基准测试 |
| **w4a32** | ~50% | ~2x | ~95% | Apple Silicon 日常使用 |
| **w4a8** | ~50% | ~2x | ~90% | Apple Silicon 速度优先 |
| **Q4** | ~40% | ~2.5x | ~85% | 通用部署 |
| **Q8** | ~60% | ~1.5x | ~98% | 精度要求高 |

---

## 🔧 完整工作流程

### 步骤 1: 准备模型文件

确保你有未量化的模型文件（FP16 或 FP32 格式）：

```bash
# 检查模型文件
ls -lh models/Qwen3-0.6B-fp16.mllm

# 应该显示类似:
# -rw-r--r-- 1 sunyue staff 3.0G ...
```

### 步骤 2: 创建配置文件

使用测试脚本创建示例配置：

```bash
python3 test_quantizer.py

# 会创建: quantize_config_sample.json
```

然后根据需要修改配置：

```json
{
    "quantization": {
        "type": "w4a32",
        "weight_dtype": "qai8dxp",
        "activation_dtype": "qsi4c32p",
        "linear_impl_type": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32"
    }
}
```

### 步骤 3: 运行量化

```bash
./build-osx-accelerate/bin/mllm-quantizer \
  models/Qwen3-0.6B-fp16.mllm \
  -c quantize_config_sample.json \
  -o models/Qwen3-0.6B-w4a32.mllm \
  -iv v2 \
  -ov v2
```

### 步骤 4: 验证量化结果

使用参数检查工具验证：

```bash
./build-osx-accelerate/bin/mllm-params-inspector \
  models/Qwen3-0.6B-w4a32.mllm \
  v2
```

### 步骤 5: 测试推理

使用推理工具测试量化后的模型：

```bash
python3 interactive_test.py
```

---

## ⚠️ 注意事项

### 1. 输入文件要求

- ✅ 必须是有效的 MLLM 模型文件（.mllm）
- ✅ 必须是 FP16 或 FP32 格式
- ❌ 不能是已经量化的模型

### 2. 配置文件要求

- ✅ 必须是有效的 JSON 格式
- ✅ 必须包含 `quantization` 对象
- ✅ 参数必须与量化类型匹配

### 3. 平台兼容性

| 量化类型 | Apple Silicon | x86 | ARM |
|---------|--------------|------|-----|
| **KAI w4a32** | ✅ | ❌ | ❌ |
| **KAI w4a8** | ✅ | ❌ | ❌ |
| **GGUF Q4** | ✅ | ✅ | ✅ |
| **GGUF Q8** | ✅ | ✅ | ✅ |

### 4. 性能权衡

- **模型大小**: 量化后模型更小
- **推理速度**: 量化后推理更快
- **精度**: 量化会损失一定精度
- **内存占用**: 量化后内存占用更小

---

## 🚨 故障排除

### 问题 1: 量化工具不存在

**错误**:
```
错误: 量化工具不存在: ./build-osx-accelerate/bin/mllm-quantizer
```

**解决方法**:
```bash
# 检查工具是否存在
ls -lh build-osx-accelerate/bin/mllm-quantizer

# 如果不存在，需要编译
# 参考 README 中的编译说明
```

### 问题 2: 输入文件不存在

**错误**:
```
Error: No input file path provided
```

**解决方法**:
```bash
# 检查输入文件
ls -lh models/Qwen3-0.6B-fp16.mllm

# 确保路径正确
```

### 问题 3: 配置文件格式错误

**错误**:
```
Error: Config file parse error
```

**解决方法**:
```bash
# 验证 JSON 格式
python3 -c "
import json
with open('quantize_config.json', 'r') as f:
    json.load(f)
print('JSON 格式正确')
"
```

### 问题 4: 量化失败

**错误**:
```
Error: Quantization failed
```

**可能原因**:
- 输入模型格式不正确
- 配置参数不匹配
- 内存不足

**解决方法**:
- 检查输入模型格式
- 验证配置参数
- 增加系统内存

---

## 🎯 推荐配置

### Apple Silicon (M1/M2/M3) - 日常使用

```json
{
    "quantization": {
        "type": "w4a32",
        "weight_dtype": "qai8dxp",
        "activation_dtype": "qsi4c32p",
        "linear_impl_type": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32"
    }
}
```

### Apple Silicon - 速度优先

```json
{
    "quantization": {
        "type": "w4a8",
        "weight_dtype": "qai8dxp",
        "activation_dtype": "qsi4c32p",
        "linear_impl_type": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk"
    }
}
```

### 通用部署 - 兼容性优先

```json
{
    "quantization": {
        "type": "q4_0",
        "weight_dtype": "q4_0",
        "activation_dtype": "f32",
        "linear_impl_type": "GGUF_Q4_0"
    }
}
```

---

## 📚 相关文档

- **量化工具源码**: [tools/mllm-quantizer/](file:///Users/sunyue/workspace/0_mllm_v2/tools/mllm-quantizer/)
- **量化配置示例**: [examples/qwen3/config_0.6B_w4a32_kai.json](file:///Users/sunyue/workspace/0_mllm_v2/examples/qwen3/config_0.6B_w4a32_kai.json)
- **参数检查工具**: [mllm-params-inspector](file:///Users/sunyue/workspace/0_mllm_v2/build-osx-accelerate/bin/mllm-params-inspector)

---

## 🚀 快速开始

```bash
# 1. 进入项目目录
cd /Users/sunyue/workspace/0_mllm_v2

# 2. 运行量化测试脚本
python3 test_quantizer.py

# 3. 查看测试结果和示例配置

# 4. 根据需要修改配置文件

# 5. 运行量化命令
./build-osx-accelerate/bin/mllm-quantizer \
  <输入模型> \
  -c <配置文件> \
  -o <输出模型> \
  -iv v2 \
  -ov v2

# 6. 测试量化后的模型
python3 interactive_test.py
```

---

## ✨ 总结

量化工具 `mllm-quantizer` 提供了强大的模型量化功能：

- ✅ 支持多种量化类型
- 🎯 针对 Apple Silicon 优化
- 🌐 支持通用 GGUF 格式
- 📊 显著减小模型大小
- ⚡ 提高推理速度

选择适合你需求的量化类型，开始量化你的模型吧！🚀
