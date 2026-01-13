# Qwen3 模型下载和验证报告

## 执行日期
- **日期**: 2026-01-12
- **分支**: v2
- **环境**: macOS (Apple Silicon)

## 任务概述

1. ✅ 重命名不匹配的 qwen3 模型文件
2. ✅ 使用 modelscope 下载匹配的 Qwen3 模型
3. ✅ 使用 Hugging Face 下载完整的 Qwen3 模型
4. ✅ 验证下载的模型文件

## 执行步骤

### 步骤 1: 重命名不匹配的模型文件

**命令**:
```bash
cd models/qwen-3-0.6b-mllm && \
mv qwen-3-0.6b-fp32.mllm qwen-3-0.6b-fp32.mllm.mismatched && \
mv qwen-3-0.6b-q4_k.mllm qwen-3-0.6b-q4_k.mllm.mismatched
```

**结果**: ✅ 成功重命名

**重命名后的文件**:
- `qwen-3-0.6b-fp32.mllm.mismatched` (2.8GB)
- `qwen-3-0.6b-q4_k.mllm.mismatched` (1.2GB)

### 步骤 2: 使用 modelscope 下载模型

**尝试 1**: 使用 Python API
```bash
from modelscope import snapshot_download
model_dir = snapshot_download('mllmTeam/Qwen3-0.6B-w4a32kai', cache_dir='./models')
```

**结果**: ❌ 失败

**错误**:
```
ModuleNotFoundError: No module named 'zoneinfo'
```

**解决方案**: 安装 `backports.zoneinfo`

**尝试 2**: 使用命令行工具
```bash
modelscope download --model mllmTeam/Qwen3-0.6B-w4a32kai --local_dir ./models/Qwen3-0.6B-w4a32kai
```

**结果**: ❌ 失败

**错误**: 仍然出现 `zoneinfo` 模块错误

**尝试 3**: 使用 git clone
```bash
git clone https://www.modelscope.cn/mllmTeam/Qwen3-0.6B-w4a32kai.git ./models/Qwen3-0.6B-w4a32kai
```

**结果**: ✅ 成功

**下载的文件**:
```
total 32
-rw-r--r--@ 1 sunyue  staff   591B  1 12 15:38 config.json
-rw-r--r--@ 1 sunyue  staff   135B  1 12 15:38 model.mllm
-rw-r--r--@ 1 sunyue  staff   1.3K  1 12 15:38 README.md
-rw-r--r--@ 1 sunyue  staff   133B  1 12 15:38 tokenizer.json
```

**问题**: `model.mllm` 和 `tokenizer.json` 是 Git LFS 指针文件（只有 135B 和 133B），需要使用 Git LFS 来获取实际文件

### 步骤 3: 使用 Hugging Face 下载完整模型

**命令**:
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B-hf
```

**结果**: ✅ 成功

**下载的文件**:
```
total 2994368
-rw-r--r--@ 1 sunyue  staff   726B  1 12 15:40 config.json
-rw-r--r--@ 1 sunyue  staff   239B  1 12 15:40 generation_config.json
-rw-r--r--@ 1 sunyue  staff    11K  1 12 15:40 LICENSE
-rw-r--r--@ 1 sunyue  staff   1.6M  1 12 15:40 merges.txt
-rw-r--r--@ 1 sunyue  staff   1.4G  1 12 15:48 model.safetensors
-rw-r--r--@ 1 sunyue  staff    14K  1 12 15:40 README.md
-rw-r--r--@ 1 sunyue  staff   9.5K  1 12 15:40 tokenizer_config.json
-rw-r--r--@ 1 sunyue  staff    11M  1 12 15:40 tokenizer.json
-rw-r--r--@ 1 sunyue  staff   2.6M  1 12 15:40 vocab.json
```

**文件说明**:
- `model.safetensors` (1.4GB): 完整的模型权重文件
- `tokenizer.json` (11MB): SentencePiece tokenizer 配置
- `vocab.json` (2.6MB): 词汇表文件
- `config.json` (726B): 模型配置
- `generation_config.json` (239B): 生成配置
- `merges.txt` (1.6M): BPE 合并规则
- `tokenizer_config.json` (9.5K): Tokenizer 配置

## 模型配置分析

### config.json 内容

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

### 关键配置参数

| 参数 | 值 | 说明 |
|------|------|------|
| `vocab_size` | 151936 | 词汇表大小 |
| `hidden_size` | 1024 | 隐藏层大小 |
| `num_hidden_layers` | 28 | 隐藏层数 |
| `num_attention_heads` | 16 | 注意力头数 |
| `num_key_value_heads` | 8 | KV 头数 |
| `tie_word_embeddings` | true | 是否共享 embedding |
| `max_position_embeddings` | 40960 | 最大位置嵌入 |
| `rope_theta` | 1000000 | RoPE theta |
| `rms_norm_eps` | 1e-06 | RMS Norm epsilon |

## Tokenizer 分析

### tokenizer.json 分析

**基础 vocab 大小**: 151,643
**Added tokens 数量**: 26
**总 token 数量**: 151,669
**基础 vocab ID 范围**: [0, 151642]

### vocab.json 分析

**vocab.json 大小**: 151,643
**vocab.json 类型**: 字典

### 词汇表匹配检查

| 项目 | 数值 |
|------|------|
| **模型配置 vocab_size** | 151,936 |
| **tokenizer.json 总 token 数量** | 151,669 |
| **vocab.json 大小** | 151,643 |
| **差异** | **268 个 token** ❌ |

**结论**: ❌ Tokenizer 和模型配置不匹配

## 问题分析

### 问题 1: Tokenizer 词汇表不匹配

**现象**:
- 模型配置 vocab_size: 151,936
- Tokenizer 实际大小: 151,669
- 差异: 268 个 token

**影响**:
- 模型输出的 token ID 可能超出 tokenizer 的范围
- 导致解码失败，输出异常字符

**原因**:
- 可能是不同版本的 Qwen3 模型
- Tokenizer 可能是早期版本
- 模型权重可能使用完整的 151,936 个 token

**解决方案**:
1. **获取完整的 tokenizer**（推荐）
   - 从官方渠道下载包含所有 151,936 个 token 的 tokenizer
   - 检查是否有其他版本的 Qwen3 模型

2. **修改代码支持不匹配的 tokenizer**
   - 在解码时处理超出范围的 token ID
   - 添加警告或错误处理

3. **使用不同的模型版本**
   - 寻找与当前 tokenizer 匹配的 Qwen3 模型版本
   - 或者寻找与当前模型匹配的 tokenizer 版本

### 问题 2: ModelScope 模型文件是 LFS 指针

**现象**:
- `model.mllm` 只有 135B
- `tokenizer.json` 只有 133B
- 实际文件需要通过 Git LFS 获取

**原因**:
- ModelScope 使用 Git LFS 存储大文件
- Git clone 只下载了指针文件

**解决方案**:
1. **使用 Git LFS 工具**
   ```bash
   git lfs install
   git lfs pull
   ```

2. **使用 Hugging Face 下载**（已采用）
   - Hugging Face 提供完整的文件下载
   - 不需要额外的 Git LFS 配置

## 文件位置总结

### 不匹配的模型文件（已重命名）

| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| FP32 模型 | `models/qwen-3-0.6b-mllm/qwen-3-0.6b-fp32.mllm.mismatched` | 2.8GB | 不匹配 |
| Q4_K 模型 | `models/qwen-3-0.6b-mllm/qwen-3-0.6b-q4_k.mllm.mismatched` | 1.2GB | 不匹配 |

### ModelScope 下载的模型

| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| 配置文件 | `models/Qwen3-0.6B-w4a32kai/config.json` | 591B | 模型配置 |
| 模型文件 | `models/Qwen3-0.6B-w4a32kai/model.mllm` | 135B | LFS 指针 |
| Tokenizer | `models/Qwen3-0.6B-w4a32kai/tokenizer.json` | 133B | LFS 指针 |
| README | `models/Qwen3-0.6B-w4a32kai/README.md` | 1.3K | 说明文档 |

### Hugging Face 下载的模型（完整）

| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| 模型权重 | `models/Qwen3-0.6B-hf/model.safetensors` | 1.4GB | 完整权重 |
| Tokenizer | `models/Qwen3-0.6B-hf/tokenizer.json` | 11MB | SentencePiece |
| 词汇表 | `models/Qwen3-0.6B-hf/vocab.json` | 2.6MB | 词汇表 |
| 配置文件 | `models/Qwen3-0.6B-hf/config.json` | 726B | 模型配置 |
| 生成配置 | `models/Qwen3-0.6B-hf/generation_config.json` | 239B | 生成配置 |
| 合并规则 | `models/Qwen3-0.6B-hf/merges.txt` | 1.6M | BPE 合并 |
| Tokenizer 配置 | `models/Qwen3-0.6B-hf/tokenizer_config.json` | 9.5K | Tokenizer 配置 |

## 下一步建议

### 选项 1: 转换 Hugging Face 模型为 MLLM 格式

使用 `mllm-convertor` 将 Hugging Face 模型转换为 MLLM 格式：

```bash
# 安装 pymllm
bash ./scripts/install_pymllm.sh

# 转换模型
mllm-convertor \
  --input_path ./models/Qwen3-0.6B-hf/model.safetensors \
  --output_path ./models/Qwen3-0.6B-hf/qwen3-0.6b.mllm \
  --cfg_path ./models/Qwen3-0.6B-hf/config.json \
  --pipeline builtin_pipeline
```

### 选项 2: 寻找匹配的 Tokenizer

检查是否有其他版本的 Qwen3 模型，其 tokenizer 包含完整的 151,936 个 token：

1. 检查 ModelScope 上的其他 Qwen3 模型
2. 检查 Hugging Face 上的其他 Qwen3 模型
3. 联系 mllmTeam 获取正确的 tokenizer

### 选项 3: 修改代码支持不匹配的 Tokenizer

修改 MLLM 代码以处理 tokenizer 词汇表不匹配的情况：

1. 在解码时检查 token ID 是否在范围内
2. 添加警告或错误处理
3. 考虑使用默认字符替换未知 token

## 总结

### 成功部分

1. ✅ **重命名不匹配的模型文件**: 成功标记不匹配的模型
2. ✅ **从 Hugging Face 下载完整模型**: 获得了完整的模型文件
3. ✅ **验证模型配置**: 确认模型配置参数正确
4. ✅ **分析 Tokenizer**: 识别了词汇表不匹配问题

### 失败部分

1. ❌ **ModelScope 下载失败**: Git LFS 指针问题
2. ❌ **Tokenizer 词汇表不匹配**: 268 个 token 差异
3. ❌ **无法直接运行推理**: 需要先转换模型格式

### 核心问题

**Tokenizer 词汇表不匹配**是当前的主要问题：
- 模型期望 151,936 个 token
- Tokenizer 只有 151,669 个 token
- 差异 268 个 token 导致解码失败

### 建议解决方案

1. **转换 Hugging Face 模型为 MLLM 格式**（推荐）
2. **寻找匹配的 Tokenizer**
3. **修改代码支持不匹配的 Tokenizer**

## 附录

### A. 下载命令汇总

```bash
# 1. 重命名不匹配的模型
cd models/qwen-3-0.6b-mllm && \
mv qwen-3-0.6b-fp32.mllm qwen-3-0.6b-fp32.mllm.mismatched && \
mv qwen-3-0.6b-q4_k.mllm qwen-3-0.6b-q4_k.mllm.mismatched

# 2. 从 Hugging Face 下载完整模型
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  --resume-download Qwen/Qwen3-0.6B \
  --local-dir ./models/Qwen3-0.6B-hf

# 3. 验证 Tokenizer
python3 << 'EOF'
import json
with open('models/Qwen3-0.6B-hf/tokenizer.json', 'r') as f:
    data = json.load(f)
vocab = data.get('model', {}).get('vocab', {})
added_tokens = data.get('added_tokens', [])
print(f"总 token 数量: {len(vocab) + len(added_tokens)}")
EOF
```

### B. 相关文件

| 文件路径 | 说明 |
|---------|------|
| `models/qwen-3-0.6b-mllm/qwen-3-0.6b-fp32.mllm.mismatched` | 不匹配的 FP32 模型 |
| `models/qwen-3-0.6b-mllm/qwen-3-0.6b-q4_k.mllm.mismatched` | 不匹配的 Q4_K 模型 |
| `models/Qwen3-0.6B-w4a32kai/` | ModelScope 下载的模型（LFS 指针） |
| `models/Qwen3-0.6B-hf/` | Hugging Face 下载的完整模型 |
| `docs/Qwen3_0.6B_推理测试报告.md` | 之前的推理测试报告 |

---

**报告生成时间**: 2026-01-12
**报告版本**: 1.0
