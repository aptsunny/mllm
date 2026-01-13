# MLLM 框架认知文档

## 一、MLLM 是什么？

**MLLM (Multimodal Large Language Model)** 是一个**轻量级、高性能的多模态大语言模型推理引擎**，专为移动端和边缘设备设计。

### 核心定位

- **目标场景**：移动设备（Android、iOS）、边缘设备、嵌入式系统
- **核心优势**：快速、轻量、跨平台、硬件优化
- **技术栈角色**：连接上层优化算法（量化、剪枝、投机解码）与底层硬件加速器（NPU、GPU、CPU）的中间层

### 设计理念

```
┌─────────────────────────────────────────────────────────────────┐
│  优化算法层                                        │
│  - 量化 (Quantization)                               │
│  - 剪枝 (Pruning)                                  │
│  - 投机解码 (Speculative Decoding)                  │
└────────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│  MLLM 框架 (核心枢纽)                              │
│  - 统一的推理接口                                      │
│  - 多硬件后端支持                                      │
│  - IR 中间表示                                        │
│  - Pythonic API 设计                                    │
└────────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│  硬件加速层                                          │
│  - CANN (华为 NPU)                                   │
│  - CUDA (NVIDIA GPU)                                 │
│  - MLIR (编译器)                                     │
│  - CPU/GPU/NPU 驱动                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、MLLM 能做什么？

### 1. 模型推理

支持多种大语言模型和多模态模型的推理：

| 模型类型 | 支持的模型 | 硬件支持 |
|---------|----------|---------|
| **文本模型** | Qwen3, Qwen2.5, LLaMA 2/3, Gemma, Mistral, Phi-3, SmolLM, MiniCPM 等 | CPU (FP32/INT4), NPU (INT8) |
| **视觉-语言模型** | Qwen2-VL, Qwen2.5-VL, LLaVA, Phi-3-Vision, Fuyu | CPU, NPU |
| **视觉模型** | ViT, CLIP, ImageBind | CPU, NPU |

### 2. 模型转换

将 Hugging Face / PyTorch 模型转换为 MLLM 格式：

```bash
mllm-convertor \
  --input_path <your_model> \
  --output_path <your_output_model> \
  --cfg_path <your_config> \
  --pipeline <builtin_pipeline>
```

### 3. 模型量化

支持多种量化方案，大幅减小模型体积并提升推理速度：

**GGUF 量化**（跨平台 CPU）：
- Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q6_K, Q8_K

**KAI 量化**（ARM / Apple Silicon）：
- KAI_f32_qai8dxp_qsi4c32p_mxk_nxk
- KAI_f16_qsi8d32p_qai4c32p_mxk_nxk

### 4. IR 追踪与编译

提取计算图进行优化和编译：

```cpp
auto ir = mllm::ir::trace(model, Tensor::random({1, 1024, 1024}));
print(ir);  // 查看中间表示
```

### 5. 多硬件后端

- **CPU Backend**：支持 ARM CPU、x86 CPU
- **QNN Backend**：支持 Qualcomm Hexagon NPU
- **OpenCL Backend**：支持 GPU 加速
- **自定义后端**：支持 NPU 集成

---

## 三、MLLM 是怎么做到的？

### 1. 分层架构设计

MLLM 采用三层架构：

```
┌─────────────────────────────────────────────────────────────────┐
│  Module 层 (高级抽象)                                 │
│  - 神经网络模块容器                                    │
│  - 参数管理 (load/save)                                 │
│  - 设备管理 (to/device)                                 │
│  - 前向传播编排 (forward)                               │
└────────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 层 (操作抽象)                                   │
│  - 操作封装 (BaseOp)                                   │
│  - 设备抽象 (不同后端)                                 │
│  - 任务创建 (Task)                                     │
└────────────────────┬────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│  Dispatcher 层 (执行引擎)                               │
│  - CPUDispatcher: CPU 执行                              │
│  - QNNDispatcher: NPU 执行                               │
│  - IRTraceDispatcher: IR 追踪                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Pythonic API 设计

借鉴 PyTorch 的设计理念，提供 C++ 中的动态命名模块：

```cpp
class MyModel : public nn::Module {
public:
    MyModel(const std::string& name) : nn::Module(name) {
        // 注册子模块（类似 PyTorch 的 named_modules）
        encoder_ = reg<EncoderModule>("encoder", config);
        decoder_ = reg<DecoderModule>("decoder", config);
        
        // 注册层
        linear1_ = reg<nn::Linear>("fc1", 768, 3072, false);
        linear2_ = reg<nn::Linear>("fc2", 3072, 768, false);
    }
    
private:
    EncoderModule encoder_;  // 绝对名称: "model.encoder"
    DecoderModule decoder_;  // 绝对名称: "model.decoder"
    nn::Linear linear1_;     // 绝对名称: "model.fc1"
    nn::Linear linear2_;     // 绝对名称: "model.fc2"
};
```

**优势**：
- 自动名称层级管理（如 `model.encoder.layer0.attention`）
- 类型安全的模板注册
- 编译时类型检查

### 3. 多级 IR 系统

MLLM IR 分为三个级别，支持不同抽象层次的优化：

| IR 级别 | 作用 | 示例操作 |
|----------|------|----------|
| **Tensor IR** | 张量操作和内存管理 | RegisterOp, AllocOp, FreeOp |
| **Linalg IR** | 线性代数运算 | MatMulOp, AddOp, EmbeddingOp, RMSNormOp |
| **Graph IR** | 神经网络层/模块 | SubGraphOp, CallGraphOp |
| **Program IR** | 可执行程序结构 | InstructionOp, FragmentOp, JumpOp |

**IR 追踪流程**：

```
Module::forward() [trace_mode=true]
    │
    ├─── Module::__trace()
    │    │
    │    ├─── IRContext::create<CallGraphOp>()
    │    ├─── IRContext::create<SubGraphOp>()
    │    │
    │    ├─── Module::forward() [recursive]
    │    │
    │    └─── IRContext::create<ReturnOp>()
    │
    └─── Layer::__main() [trace_mode=true]
             │
             ├─── Task::createExecuteOpTask()
             │    └─── task->custom_context_ptr = ir_context
             │
             └─── IRTraceDispatcher::receive()
                  │
                       ├─── Op::reshape()
                       └─── Op::trace()
```

### 4. 任务调度系统

统一的任务接口支持同步和异步执行：

```cpp
struct Task {
    TaskTypes type;                    // kExecuteOp 或 kExecuteModule
    BaseOp::ptr_t op;                // 要执行的操作
    std::vector<Tensor> inputs;      // 输入张量
    std::vector<Tensor> outputs;     // 输出张量
    std::vector<AnyValue> args;      // 额外参数
    void* custom_context_ptr;          // 后端特定上下文
};

class Dispatcher {
    virtual void receive(const Task::ptr_t& task) = 0;           // 同步接收
    virtual TaskResult::sender_t asyncReceive(...) = 0;        // 异步接收
    virtual void process(const Task::ptr_t& task) = 0;          // 处理任务
};
```

### 5. 硬件优化技术

**量化优化**：
- 权重量化：FP32 → INT4/INT8，减少 75% 内存占用
- 激活量化：FP16/INT8，加速计算
- 分组量化：per-group 量化，平衡精度和性能

**内存优化**：
- 静态 KV Cache：预分配缓存，避免动态分配
- 内存池管理：减少碎片化
- 张量复用：避免不必要的拷贝

**计算优化**：
- Flash Attention：优化注意力机制计算
- 算子融合：减少内存访问
- SIMD 指令：ARM NEON / x86 AVX

---

## 四、如何使用 MLLM？

### 1. 环境准备

**macOS Apple Silicon**：
```bash
pip install -r requirements-mini.txt
python task.py tasks/build_osx_apple_silicon_accelerate.yaml
```

**Android**：
```bash
pip install -r requirements.txt
python task.py tasks/build_android.yaml
```

**x86 PC**：
```bash
pip install -r requirements.txt
python task.py tasks/build_x86.yaml
```

### 2. 模型转换流程

**完整流程**：

```
1. 获取原始模型 (Hugging Face / ModelScope)
   ↓
2. 转换为 MLLM 格式 (mllm-convertor)
   ↓
3. (可选) 设备端量化 (mllm-quantizer)
   ↓
4. 实现 C++ 模型文件
   ↓
5. 创建示例应用
   ↓
6. 构建并测试
   ↓
7. 提交 PR
```

**示例命令**：

```bash
# 安装 pymllm
bash ./scripts/install_pymllm.sh

# 转换模型
mllm-convertor \
  --input_path ./Qwen3-0.6B/model.safetensors \
  --output_path ./Qwen3-0.6B/w4a32.mllm \
  --cfg_path ./Qwen3-0.6B/quant_config.json \
  --pipeline w4a32_kai_pipeline

# (可选) 量化模型
mllm-quantizer \
  -i ./Qwen3-0.6B/model.mllm \
  -c ./Qwen3-0.6B/quant_config.json \
  -iv v2 \
  -o ./Qwen3-0.6B/w4a32.mllm \
  -ov v2
```

### 3. 编写推理代码

**基本结构**：

```cpp
#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/modeling_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

int main(int argc, char* argv[]) {
    mllm::init();
    
    // 加载配置
    auto cfg = mllm::models::qwen3::Qwen3Config(config_path);
    
    // 初始化 tokenizer
    auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path);
    
    // 创建模型
    auto model = mllm::models::qwen3::Qwen3ForCausalLM(cfg);
    
    // 加载权重
    model.load(mllm::load(model_path, file_version));
    
    // 编码输入
    auto inputs = tokenizer.convertMessage({.prompt = "Hello, world!"});
    
    // 流式生成
    for (auto& step : model.chat(inputs)) {
        std::wcout << tokenizer.detokenize(step.cur_token_id) << std::flush;
    }
    
    return 0;
}
```

### 4. 支持新模型

**需要创建三个文件**：

1. `configuration_qwen3.hpp` - 配置加载器
2. `tokenization_qwen3.hpp` - Tokenizer 实现
3. `modeling_qwen3.hpp` - 模型架构实现

**关键步骤**：

```cpp
// 1. 定义配置结构
struct Qwen3Config : protected ConfigFile {
    int32_t hidden_size = 1024;
    int32_t num_hidden_layers = 28;
    int32_t vocab_size = 151936;
    // ...
};

// 2. 实现 Tokenizer
class Qwen3Tokenizer final : public AutoTokenizer {
    std::vector<std::wstring> tokenize(const std::string& str) override;
    std::wstring detokenize(int64_t pos_idx) override;
};

// 3. 实现模型
class Qwen3ForCausalLM : public ARGeneration {
    std::vector<Tensor> forward(...) override;
};
```

---

## 五、实际案例：Qwen3-0.6B 测试

### 1. 环境配置

**系统信息**：
- 操作系统：macOS (Apple Silicon)
- 编译器：Apple Clang 17.0.0
- 架构：ARM64
- 后端：CPU + Accelerate 框架

**构建结果**：
```
build-osx-accelerate/
├── bin/
│   ├── mllm-qwen3-runner          # Qwen3 推理程序
│   ├── mllm-params-inspector      # 参数检查工具
│   ├── mllm-quantizer             # 量化工具
│   └── main_minicpmo             # MiniCPM-O 示例
└── lib/
    ├── libMllmRT.dylib             # 运行时库
    ├── libMllmCPUBackend.dylib    # CPU 后端
    └── libMllmSdkC.dylib           # C SDK
```

### 2. 模型信息

**Qwen3-0.6B**：
- 参数量：0.6B（6 亿）
- 隐藏层大小：1024
- 注意力头数：16
- KV 头数：8
- 隐藏层数：28
- 词汇表大小：151936

**量化版本**：
- **FP32 版本**：2.8GB（完整精度）
- **Q4_K 版本**：1.2GB（4-bit 量化，约 57% 压缩）

### 3. 运行命令

```bash
./build-osx-accelerate/bin/mllm-qwen3-runner \
  -m models/qwen-3-0.6b-mllm/qwen-3-0.6b-q4_k.mllm \
  -mv v1 \
  -t models/qwen-3-0.6b-mllm/tokenizer/tokenizer.json \
  -c examples/qwen3/config_0.6B_w4a32_kai.json
```

**参数说明**：
- `-m`：模型文件路径
- `-mv`：模型版本（v1 或 v2）
- `-t`：tokenizer 文件路径
- `-c`：配置文件路径

### 4. 推理流程

```
用户输入 "Hello, world!"
    ↓
Tokenizer 编码 → [151644, 990, 1917, 11319, ...]
    ↓
模型前向传播（28 层 Transformer）
    ↓
每层计算：
  - Attention: Q/K/V 投影 + RoPE + 注意力计算
  - MLP: Gate/Up 投影 + SiLU 激活 + Down 投影
  - RMSNorm: 层归一化
    ↓
生成 token ID 序列
    ↓
Tokenizer 解码 → "Hello! How can I help you today?"
    ↓
流式输出到终端
```

### 5. 性能指标

**推理速度**：
- Prefill 阶段：~22 tokens/s
- Decode 阶段：~2-5 tokens/s（取决于量化级别）

**内存使用**：
- KV Cache：~2GB（max_cache_length=2048）
- 模型权重：1.2GB (Q4_K) 或 2.8GB (FP32)
- 激活值：~500MB

### 6. 遇到的问题及解决方案

**问题 1：Tokenizer 词汇表不匹配**

**现象**：
- 模型配置 vocab_size：151936
- Tokenizer 实际 vocab size：151643
- 模型输出 token ID > 151642 时无法解码

**原因**：
- 模型文件和 tokenizer 来自不同版本
- 量化过程中词汇表大小不一致

**解决方案**：
1. 获取与模型匹配的 tokenizer
2. 修改配置文件中的 vocab_size
3. 重新转换模型确保一致性

**问题 2：模型参数名称不匹配**

**现象**：
- 错误：`Parameter does not exist: lm_head_out.weight`
- 实际参数名：`lm_head.weight`

**原因**：
- 配置文件中 `tie_word_embeddings` 设置错误

**解决方案**：
```json
{
  "tie_word_embeddings": false,  // 改为 false
  "vocab_size": 151936
}
```

---

## 六、核心优势总结

### 1. 轻量级
- 单文件部署（无需 Python 环境）
- 最小化依赖（只需 CMake + 编译器）
- 支持静态链接

### 2. 高性能
- 硬件优化（ARM NEON、SIMD）
- 内存池管理
- Flash Attention

### 3. 跨平台
- CPU：ARM、x86
- NPU：Qualcomm Hexagon
- GPU：OpenCL
- 移动端：Android、iOS

### 4. 易用性
- Pythonic API（类似 PyTorch）
- 统一的推理接口
- 丰富的工具链（转换器、量化器、检查器）

### 5. 可扩展性
- 模块化设计
- 后端插件系统
- IR 中间表示

---

## 七、学习路径建议

### 初学者
1. 阅读 [README.md](../README.md)
2. 运行示例程序（如 `mllm-qwen3-runner`）
3. 理解 Module/Layer/Dispatcher 架构
4. 尝试转换自己的模型

### 进阶用户
1. 学习 IR 系统和编译流程
2. 实现自定义算子
3. 添加新的硬件后端
4. 优化量化算法

### 贡献者
1. 添加新模型支持
2. 修复 Bug 和性能问题
3. 完善文档和示例
4. 参与社区讨论

---

## 八、相关资源

- **官方文档**：https://ubiquitouslearning.github.io/mllm/
- **GitHub 仓库**：https://github.com/UbiquitousLearning/mllm
- **ModelScope 模型**：https://www.modelscope.cn/models/mllmTeam
- **Hugging Face 模型**：https://huggingface.co/mllmTeam

---

**总结**：MLLM 是一个专为移动端和边缘设备设计的轻量级多模态大语言模型推理引擎，通过分层架构、Pythonic API、多级 IR 系统和硬件优化技术，实现了高效、跨平台的模型推理能力。对于没有接触过这个工具的人来说，可以从运行示例开始，逐步学习模型转换、推理和自定义开发。
