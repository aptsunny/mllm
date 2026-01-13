# w4a8 激活量化详解

## 📊 w4a8 配置解读

### 配置文件

```json
{
    "linear_impl_type": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32"
}
```

### 符号解析

| 部分 | 含义 | 说明 |
|------|------|------|
| **f32** | Float32 | 输入激活保持 FP32 精度 |
| **qai8dxp** | 量化激活 8-bit dot product | 激活量化为 8-bit |
| **qsi4c32p** | 量化权重 4-bit channel 32-bit packed | 权重量化为 4-bit |
| **mxk_nxk** | 矩阵布局 | M×K 矩阵乘以 K×N 矩阵 |
| **qai8dxp4x8** | 量化激活 8-bit，4×8 瓦片 | 激活按 4×8 瓦片量化 |
| **qsi4c32p8x8** | 量化权重 4-bit，8×8 瓦片 | 权重按 8×8 瓦片量化 |
| **4x8x32** | 输出瓦片大小 | 输出按 4×8×32 瓦片计算 |

---

## 🔍 激活量化算法

### w4a8 vs w4a32 的区别

| 特性 | w4a32 | w4a8 |
|------|--------|-------|
| **权重精度** | 4-bit | 4-bit |
| **激活精度** | 32-bit (FP32) | 8-bit (Int8) |
| **激活量化** | ❌ 不量化 | ✅ 量化 |
| **内存占用** | 中等 | 更低 |
| **计算速度** | 快 | 更快 |
| **精度损失** | 3-5% | 5-8% |

---

## 🎯 激活量化原理

### 1. 量化三元组（Triplet）

**qai8dxp** 的含义：

| 符号 | 含义 |
|------|------|
| **q** | Quantized（量化） |
| **a** | Asymmetric（非对称） |
| **i** | Integer（整数） |
| **8** | 8-bit |
| **dx** | Per dimension（按维度） |
| **p** | Packed（打包） |

**完整含义**：非对称 8-bit 按维度量化，打包存储

### 2. 非对称量化（Asymmetric Quantization）

**对称量化**（Symmetric）：
- Zero point = 0
- 范围：[-8, 7]（对于 8-bit）
- 公式：`quantized = round(value / scale)`
- 反量化：`dequantized = quantized * scale`

**非对称量化**（Asymmetric）：
- Zero point ≠ 0（可调整）
- 范围：[0, 255]（对于无符号 8-bit）
- 公式：`quantized = round((value - zero_point) / scale)`
- 反量化：`dequantized = quantized * scale + zero_point`

**为什么 w4a8 使用非对称量化**：
- ✅ 更好的精度保持（特别是对于 ReLU 等激活函数）
- ✅ 适应激活值的分布（通常是非对称的）
- ✅ 减少量化误差

---

## 🔄 激活量化流程

### 推理时的激活量化

#### 步骤 1: 前向传播

```python
# 输入激活（FP32）
activation_fp32 = tensor([0.1234, 0.5678, 0.9012, ...])
```

#### 步骤 2: 量化激活

```python
# 量化为 8-bit 非对称整数
activation_int8 = quantize_asymmetric(activation_fp32)

# 量化公式
zero_point = mean(activation_fp32)  # 计算零点
scale = (max(activation_fp32) - min(activation_fp32)) / 255  # 计算缩放因子
activation_int8 = round((activation_fp32 - zero_point) / scale)  # 量化
activation_int8 = clip(activation_int8, 0, 255)  # 限制到 [0, 255]
```

#### 步骤 3: 矩阵乘法

```cpp
// 使用量化后的激活进行矩阵乘法
// 激活：Int8 (8-bit)
// 权重：Int4 (4-bit，打包在 UInt8)
// 输出：Int32 (32-bit 累积)

void matmul_int8_int4(
    const int8_t* activation,  // 量化后的激活（8-bit）
    const uint8_t* weight,      // 量化后的权重（4-bit，打包）
    int32_t* output,           // 输出（32-bit 累积）
    int M, int K, int N
) {
    // 使用 ARM NEON 指令进行高效计算
    // 每个输出元素是 32-bit 累积
    // 最后需要反量化
}
```

#### 步骤 4: 反量化输出

```python
# 将 32-bit 累积反量化为 FP32
output_fp32 = dequantize_asymmetric(output_int32)

# 反量化公式
output_fp32 = output_int32 * scale + zero_point
```

---

## 📊 量化示例

### 示例 1: ReLU 激活

**原始 FP32 激活**：
```python
activation_fp32 = [0.1234, 0.5678, 0.9012, -0.3456, 0.7890]
```

**量化为 Int8**（非对称）：

```python
# 计算统计信息
min_val = -0.3456
max_val = 0.9012
zero_point = 0.0  # 假设 zero point 为 0
scale = (0.9012 - (-0.3456)) / 255 = 0.0049

# 量化
activation_int8 = [
    round((0.1234 - 0.0) / 0.0049) = 25,
    round((0.5678 - 0.0) / 0.0049) = 116,
    round((0.9012 - 0.0) / 0.0049) = 184,
    round((-0.3456 - 0.0) / 0.0049) = -71,
    round((0.7890 - 0.0) / 0.0049) = 161,
]

# 限制到 [0, 255]
activation_int8 = clip(activation_int8, 0, 255)
activation_int8 = [25, 116, 184, 0, 161]  # -71 被裁剪为 0
```

### 示例 2: GELU 激活

**原始 FP32 激活**：
```python
activation_fp32 = [-0.2345, 0.4567, 0.7890, 0.1234, -0.5678]
```

**量化为 Int8**（非对称）：

```python
# 计算统计信息
min_val = -0.5678
max_val = 0.7890
zero_point = 0.0
scale = (0.7890 - (-0.5678)) / 255 = 0.0053

# 量化
activation_int8 = [
    round((-0.2345 - 0.0) / 0.0053) = -44,
    round((0.4567 - 0.0) / 0.0053) = 86,
    round((0.7890 - 0.0) / 0.0053) = 149,
    round((0.1234 - 0.0) / 0.0053) = 23,
    round((-0.5678 - 0.0) / 0.0053) = -107,
]

# 限制到 [0, 255]
activation_int8 = clip(activation_int8, 0, 255)
activation_int8 = [0, 86, 149, 23, 0]  # -44 和 -107 被裁剪为 0
```

---

## 🎯 瓦片配置（Tile Configuration）

### qai8dxp4x8 的含义

**激活瓦片配置**：

| 参数 | 含义 | 值 |
|------|------|-----|
| **qai8dxp** | 量化激活 8-bit dot product | - |
| **4x8** | 激活瓦片大小 | 4×8 |
| **qsi4c32p** | 量化权重 4-bit channel 32-bit packed | - |
| **8x8** | 权重瓦片大小 | 8×8 |

**完整含义**：
- 激活按 4×8 瓦片处理
- 每个瓦片包含 32 个激活值
- 权重按 8×8 瓦片组织
- 使用 8-bit dot product 进行计算

### 瓦片优势

| 优势 | 说明 |
|------|------|
| **缓存友好** | 小瓦片大小（4×8）适合 CPU 缓存 |
| **并行度高** | 可以并行处理多个瓦片 |
| **内存效率** | 减少内存访问次数 |
| **硬件优化** | 针对 ARM NEON 指令集优化 |

---

## 💡 为什么激活量化能提升性能

### 1. 内存带宽节省

| 数据类型 | 每个元素大小 | 相对大小 |
|---------|------------|---------|
| **FP32** | 4 bytes | 100% |
| **Int8** | 1 byte | 25% |

**内存带宽节省**：75%

### 2. 计算速度提升

**FP32 矩阵乘法**：
- 需要 32-bit 浮点运算
- 每个操作需要多个周期

**Int8 × Int4 矩阵乘法**：
- 使用 8-bit 整数运算
- 可以使用 ARM NEON 指令并行处理
- 每个操作需要更少周期

**速度提升**：1.5-2x

### 3. 缓存利用率提升

**FP32 激活**：
- 每个元素 4 bytes
- 缓存行利用率低

**Int8 激活**：
- 每个元素 1 byte
- 缓存行利用率高
- 更多元素可以放入缓存

**缓存利用率提升**：2-3x

---

## ⚠️ 激活量化的挑战

### 1. 精度损失

**量化误差来源**：
- 量化误差：`round((value - zero_point) / scale)`
- 裁剪误差：超出范围的值被裁剪
- 累积误差：多次量化/反量化累积误差

**精度损失**：
- ReLU 激活：3-5%
- GELU 激活：5-8%
- Swish 激活：4-7%

### 2. 数值范围问题

**Int8 范围**：[0, 255]

**问题**：
- 负值需要映射到正范围
- 可能导致信息丢失
- 需要仔细选择 zero_point

**解决方案**：
- 使用非对称量化（允许 zero_point ≠ 0）
- 优化 zero_point 选择
- 使用缩放因子适应数据分布

### 3. 梯度传播问题

**训练时**：
- 使用 FP32 激活
- 梯度计算准确

**推理时**：
- 使用 Int8 激活
- 梯度计算不准确
- 需要特殊处理

**解决方案**：
- 使用直通估计器（Straight-Through Estimator, STE）
- 使用感知量化训练（Quantization-Aware Training, QAT）

---

## 🎯 w4a8 的优势

### 相比 w4a32

| 特性 | w4a32 | w4a8 | 提升 |
|------|--------|-------|------|
| **模型大小** | 50% | 50% | 相同 |
| **激活大小** | 100% (FP32) | 25% (Int8) | **75% ↓** |
| **内存占用** | 中等 | 更低 | **30-40% ↓** |
| **推理速度** | 1.8-2x | 2-2.5x | **10-25% ↑** |
| **精度损失** | 3-5% | 5-8% | 略高 |

### 适用场景

**w4a8 适合**：
- ✅ 边缘设备部署（内存受限）
- ✅ 移动端推理（电池受限）
- ✅ 实时应用（延迟敏感）
- ✅ 批量推理（吞吐量优先）

**w4a32 适合**：
- ✅ 精度要求高的场景
- ✅ 复杂推理任务
- ✅ 需要保持高精度的应用

---

## 📚 相关文档

- **KAI 量化头文件**: [mllm/backends/cpu/kernels/arm/linear/kai.hpp](file:///Users/sunyue/workspace/0_mllm_v2/mllm/backends/cpu/kernels/arm/linear/kai.hpp)
- **KAI 量化实现**: [mllm/backends/cpu/kernels/arm/linear/kai.cpp](file:///Users/sunyue/workspace/0_mllm_v2/mllm/backends/cpu/kernels/arm/linear/kai.cpp)
- **激活量化头文件**: [kai_lhs_quant_pack_qai8dxp_f32.h](file:///Users/sunyue/workspace/0_mllm_v2/mllm/backends/cpu/vendors/kleidiai/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h)
- **w4a8 配置示例**: [examples/qwen3/config_0.6B_w4a8_i8mm_kai.json](file:///Users/sunyue/workspace/0_mllm_v2/examples/qwen3/config_0.6B_w4a8_i8mm_kai.json)

---

## 🎯 总结

### w4a8 激活量化

**核心概念**：
- **权重量化**：4-bit 对称量化（与 w4a32 相同）
- **激活量化**：8-bit 非对称量化（新增）
- **存储格式**：UInt8（激活），UInt8（权重，打包）
- **计算格式**：Int8 × Int4 → Int32 累积

**量化流程**：
1. **前向传播**：计算 FP32 激活
2. **激活量化**：量化为 Int8（非对称）
3. **矩阵乘法**：使用 Int8 × Int4 进行计算
4. **反量化**：将 Int32 累积反量化为 FP32

**关键优势**：
- ✅ 激活内存占用减少 75%
- ✅ 内存带宽节省 75%
- ✅ 推理速度提升 10-25%
- ✅ 缓存利用率提升 2-3x
- ✅ 适合边缘设备部署

**权衡**：
- ⚠️ 精度损失略高（5-8% vs 3-5%）
- ⚠️ 数值范围问题（负值映射）
- ⚠️ 梯度传播复杂（需要 QAT）

**推荐场景**：
- ✅ 边缘设备部署
- ✅ 移动端推理
- ✅ 实时应用
- ✅ 批量推理

w4a8 是在保持 4-bit 权重量化的同时，通过量化激活来进一步提升性能和内存效率的优秀方案！
