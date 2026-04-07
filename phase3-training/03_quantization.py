"""
======================================================
第 3 课：模型量化 —— 让大模型跑在消费级显卡上
======================================================

核心问题：LLaMA-70B 用 FP32 需要 280GB 显存，怎么跑？
答案：把 FP32 (32位) 权重量化为 INT8 (8位) 甚至 INT4 (4位)

直觉：
  FP32: 3.141592653589793  → 精确但占空间
  INT8: 3                  → 粗略但只占 1/4 空间
  关键发现：模型对这种精度损失非常鲁棒！

显存对比（以 LLaMA-70B 为例）：
  FP32:  280 GB  → 需要 8×A100
  FP16:  140 GB  → 需要 4×A100
  INT8:   70 GB  → 需要 2×A100
  INT4:   35 GB  → 1×A100 或消费级显卡！

学习目标：
1. 理解浮点数和整数的表示方式
2. 掌握对称/非对称量化的数学原理
3. 从零实现量化和反量化
4. 理解量化误差与精度的权衡

运行方式：python3 03_quantization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: 浮点数回顾
# ============================================================

print("=" * 60)
print("Part 1: 浮点数格式回顾")
print("=" * 60)
print("""
计算机中的数字格式：

  FP32 (float32): 1位符号 + 8位指数 + 23位尾数 = 32位
    → 精度高，但每个数占 4 字节

  FP16 (float16): 1位符号 + 5位指数 + 10位尾数 = 16位
    → 精度中等，每个数占 2 字节
    → 范围较小，容易溢出

  BF16 (bfloat16): 1位符号 + 8位指数 + 7位尾数 = 16位
    → 精度低一点，但范围和 FP32 一样
    → 训练大模型的首选

  INT8: 8位整数，范围 [-128, 127] 或 [0, 255]
    → 每个数只占 1 字节
""")

x = torch.tensor([3.14159, -0.001, 100.5, 0.0001])
print("同一个张量在不同精度下的表示:")
print(f"  FP32:  {x}")
print(f"  FP16:  {x.half()}")
print(f"  BF16:  {x.bfloat16()}")
print(f"  INT8 不能直接转换，需要量化（这就是本课的重点）")
print(f"\n每个元素的字节数:")
print(f"  FP32: {x.element_size()} 字节")
print(f"  FP16: {x.half().element_size()} 字节")
print(f"  INT8: {x.to(torch.int8).element_size()} 字节")
print()


# ============================================================
# Part 2: 对称量化（Symmetric Quantization）
# ============================================================

print("=" * 60)
print("Part 2: 对称量化")
print("=" * 60)
print("""
把浮点数映射到 [-127, 127] 的整数范围：

  scale = max(|x|) / 127
  x_quantized = round(x / scale)
  x_dequantized = x_quantized * scale

特点：0 精确映射到 0，正负对称
""")


def symmetric_quantize(x, num_bits=8):
    """对称量化：FP32 → INT8"""
    qmin = -(2 ** (num_bits - 1) - 1)
    qmax = 2 ** (num_bits - 1) - 1

    abs_max = x.abs().max()
    scale = abs_max / qmax

    x_quantized = torch.clamp(torch.round(x / scale), qmin, qmax).to(torch.int8)
    return x_quantized, scale


def symmetric_dequantize(x_quantized, scale):
    """反量化：INT8 → FP32"""
    return x_quantized.float() * scale


weights = torch.randn(4, 4)
print("原始权重 (FP32):")
print(weights)

q_weights, scale = symmetric_quantize(weights)
print(f"\n量化后 (INT8), scale={scale:.6f}:")
print(q_weights)

dq_weights = symmetric_dequantize(q_weights, scale)
print(f"\n反量化后 (FP32):")
print(dq_weights)

error = (weights - dq_weights).abs()
print(f"\n量化误差:")
print(f"  平均误差: {error.mean():.6f}")
print(f"  最大误差: {error.max():.6f}")
print(f"  相对误差: {(error / weights.abs().clamp(min=1e-8)).mean():.4%}")
print()


# ============================================================
# Part 3: 非对称量化（Asymmetric Quantization）
# ============================================================

print("=" * 60)
print("Part 3: 非对称量化")
print("=" * 60)
print("""
把浮点数映射到 [0, 255] 的整数范围：

  scale = (max(x) - min(x)) / 255
  zero_point = round(-min(x) / scale)
  x_quantized = round(x / scale) + zero_point
  x_dequantized = (x_quantized - zero_point) * scale

特点：更好地利用整数范围，适合分布不对称的数据
""")


def asymmetric_quantize(x, num_bits=8):
    """非对称量化"""
    qmin = 0
    qmax = 2**num_bits - 1

    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = torch.round(-x_min / scale).clamp(qmin, qmax).to(torch.int32)

    x_quantized = torch.clamp(
        torch.round(x / scale) + zero_point, qmin, qmax
    ).to(torch.uint8)
    return x_quantized, scale, zero_point


def asymmetric_dequantize(x_quantized, scale, zero_point):
    """非对称反量化"""
    return (x_quantized.float() - zero_point.float()) * scale


relu_output = F.relu(torch.randn(4, 4))  # ReLU 后全是非负数
print("ReLU 输出 (全部 >= 0):")
print(relu_output)

q_sym, s_sym = symmetric_quantize(relu_output)
q_asym, s_asym, zp = asymmetric_quantize(relu_output)

err_sym = (relu_output - symmetric_dequantize(q_sym, s_sym)).abs().mean()
err_asym = (relu_output - asymmetric_dequantize(q_asym, s_asym, zp)).abs().mean()

print(f"\n对称量化误差:   {err_sym:.6f}")
print(f"非对称量化误差: {err_asym:.6f}")
print("→ 对于 ReLU 这种全非负输出，非对称量化误差更小")
print()


# ============================================================
# Part 4: 逐通道量化 vs 逐张量量化
# ============================================================

print("=" * 60)
print("Part 4: 逐通道 vs 逐张量量化")
print("=" * 60)
print("""
逐张量量化：整个张量共用一个 scale
  → 简单，但如果不同通道的数值范围差异大，精度差

逐通道量化：每个输出通道有自己的 scale
  → 精度更高，推荐用于权重量化
""")


def per_channel_quantize(weight, num_bits=8):
    """逐通道对称量化（对每行独立量化）"""
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = weight.abs().max(dim=1, keepdim=True)[0]
    scales = abs_max / qmax
    q_weight = torch.clamp(torch.round(weight / scales), -qmax, qmax).to(torch.int8)
    return q_weight, scales.squeeze()


weight = torch.randn(4, 8)
weight[0] *= 10  # 第0行数值范围大
weight[3] *= 0.01  # 第3行数值范围小
print("权重（不同行数值范围差异大）:")
print(f"  第0行范围: [{weight[0].min():.2f}, {weight[0].max():.2f}]")
print(f"  第3行范围: [{weight[3].min():.4f}, {weight[3].max():.4f}]")

q_per_tensor, s_tensor = symmetric_quantize(weight)
q_per_channel, s_channel = per_channel_quantize(weight)

err_per_tensor = (weight - symmetric_dequantize(q_per_tensor, s_tensor)).abs().mean()
dq_per_channel = q_per_channel.float() * s_channel.unsqueeze(1)
err_per_channel = (weight - dq_per_channel).abs().mean()

print(f"\n逐张量量化误差: {err_per_tensor:.6f}")
print(f"逐通道量化误差: {err_per_channel:.6f}")
print(f"精度提升:       {err_per_tensor/err_per_channel:.1f}x")
print()


# ============================================================
# Part 5: 量化一个完整模型
# ============================================================

print("=" * 60)
print("Part 5: 量化完整模型")
print("=" * 60)


class SimpleModel(nn.Module):
    def __init__(self, d_in=32, d_hidden=64, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QuantizedLinear(nn.Module):
    """量化版 Linear 层：权重存为 INT8，推理时反量化"""

    def __init__(self, original_linear):
        super().__init__()
        weight = original_linear.weight.data
        q_weight, scales = per_channel_quantize(weight)

        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scales", scales)
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data)
        else:
            self.bias = None

    def forward(self, x):
        dq_weight = self.q_weight.float() * self.scales.unsqueeze(1)
        output = F.linear(x, dq_weight, self.bias)
        return output


def quantize_model(model):
    """把模型中所有 Linear 层替换为量化版"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinear(module))
        else:
            quantize_model(module)
    return model


model = SimpleModel()
x_test = torch.randn(10, 32)

with torch.no_grad():
    output_fp32 = model(x_test)

def model_size_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for name, buf in model.named_buffers():
        total += buf.nelement() * buf.element_size()
    return total

size_fp32 = model_size_bytes(model)

q_model = quantize_model(model)

with torch.no_grad():
    output_int8 = q_model(x_test)

size_int8 = model_size_bytes(q_model)
diff = (output_fp32 - output_int8).abs()

print(f"FP32 模型大小: {size_fp32:,} 字节 ({size_fp32/1024:.1f} KB)")
print(f"INT8 模型大小: {size_int8:,} 字节 ({size_int8/1024:.1f} KB)")
print(f"压缩比:        {size_fp32/size_int8:.2f}x")
print(f"\n输出差异:")
print(f"  平均差异: {diff.mean():.6f}")
print(f"  最大差异: {diff.max():.6f}")
print("→ 输出几乎一样！这就是为什么量化可行")
print()


# ============================================================
# Part 6: 不同位数的量化对比
# ============================================================

print("=" * 60)
print("Part 6: 不同位数量化的精度对比")
print("=" * 60)

weight = torch.randn(64, 64)
print("原始权重 shape:", weight.shape)
print()

for bits in [8, 4, 2]:
    qmax = 2 ** (bits - 1) - 1
    abs_max = weight.abs().max()
    scale = abs_max / qmax
    q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
    dq = q * scale
    error = (weight - dq).abs()

    original_bytes = weight.numel() * 4
    quantized_bytes = weight.numel() * bits / 8

    print(f"INT{bits}:")
    print(f"  整数范围: [{-qmax}, {qmax}]")
    print(f"  平均误差: {error.mean():.6f}")
    print(f"  最大误差: {error.max():.6f}")
    print(f"  大小:     {original_bytes/1024:.1f}KB → {quantized_bytes/1024:.1f}KB")
    print(f"  压缩比:   {original_bytes/quantized_bytes:.0f}x")
    print()

print("""
实践中的选择：
  INT8:  精度损失极小，通用推荐
  INT4:  精度有轻微下降，但模型仍可用（llama.cpp 默认）
  INT2:  精度下降明显，仅适合极端场景
""")


# ============================================================
# Part 7: 量化感知训练 (QAT) 的概念
# ============================================================

print("=" * 60)
print("Part 7: 量化感知训练 (QAT) 简介")
print("=" * 60)
print("""
两种量化方式：

1. 训练后量化 (PTQ - Post-Training Quantization)
   - 训练完成后直接量化
   - 简单，但精度可能下降
   - 上面我们实现的就是 PTQ

2. 量化感知训练 (QAT - Quantization-Aware Training)
   - 训练时就模拟量化的效果
   - 前向传播：权重先量化再反量化（模拟量化噪声）
   - 反向传播：用 Straight-Through Estimator (STE)
   - 精度更好，但需要训练

QAT 的核心：fake quantization（伪量化）
""")


class FakeQuantize(torch.autograd.Function):
    """伪量化：前向量化+反量化，反向直通（STE）"""

    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmax = 2 ** (num_bits - 1) - 1
        scale = x.abs().max() / qmax
        x_q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        x_dq = x_q * scale
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


x = torch.randn(4, requires_grad=True)
x_fq = FakeQuantize.apply(x)
loss = x_fq.sum()
loss.backward()

print("伪量化示例:")
print(f"  输入:       {x.data}")
print(f"  伪量化后:   {x_fq.data}")
print(f"  梯度(STE):  {x.grad}")
print("→ 梯度直通，不被量化操作阻断")
print()


# ============================================================
# 练习
# ============================================================

print("=" * 60)
print("动手练习")
print("=" * 60)
print("""
练习 1：分组量化 (Group Quantization)
  不是逐通道，而是把每 32 个元素分一组，每组有自己的 scale
  实现 group_quantize(weight, group_size=32) 函数
  对比逐张量、逐通道、分组量化的误差

练习 2：量化模型的精度测试
  用 Part 5 的 SimpleModel，随机生成测试数据
  对比 FP32 和 INT8 模型在分类准确率上的差异

练习 3：混合精度量化
  模型的不同层对量化的敏感度不同
  尝试：第一层和最后一层用 FP16，中间层用 INT8
  对比全 INT8 和混合精度的输出差异
""")

print("=" * 60)
print("恭喜完成第 3 课！")
print("下一课我们将学习 RLHF —— 让模型变得有用且安全")
print("=" * 60)
