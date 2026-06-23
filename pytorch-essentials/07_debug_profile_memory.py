"""
第 7 课：调试 / 显存 / 算力（torch 版）
=====================================
你已经会搭模型、写训练循环了，但真正动手训练大模型时，80% 的时间花在
「为什么报错」「为什么 loss 变 nan」「这个模型能不能塞进我的显卡」「训练要多少算力」。
这节课把这些工程问题系统化：复现三类经典报错并解读、检测 nan/inf、
用「每参数约 16 字节」估显存、用「C≈6ND」估训练 FLOPs，最后用 torch.profiler 量耗时。

核心问题：
- shape / device / dtype 不匹配的报错长什么样？怎么从报错信息一眼定位？
- loss 变 nan/inf 怎么查？哪里最容易触发（除零、log(0)、exp 溢出）？
- 给定参数量 N，训练态要多少显存？为什么是「每参数 16 字节」而不是 4 字节？
- 训练一个模型大概要多少次浮点运算？为什么经验公式是 C≈6ND？
- 一个操作到底慢在哪？怎么用 profiler 量？

与大模型的关系：
- phase1 第 4 课你算过「显存账」：权重 + 梯度 + 优化器状态 + 激活。这里把它落成可调用的估算器。
- 6ND 是 OpenAI/Chinchilla 论文里规划训练算力、推算「需要多少卡多少天」的核心经验公式。
- nan 排查、device 对齐是每个 PyTorch 工程师的日常急救包。

前置：phase1 第 4 课（显存账 / 算力），本专项第 3 课（参数统计）、第 6 课（训练循环）
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 图片/产物锚定脚本自身目录


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
section("Part 1: 三类经典报错复现 —— 看一眼就知道哪错了")
# ============================================================
# 90% 的 PyTorch 报错都是这三类。我们用 try/except 主动触发并捕获，
# 把报错信息打出来「解剖」，以后线上遇到同样的字眼就能秒定位。

print("【报错 1】shape 不匹配（矩阵乘法 / 加法维度对不上）")
# matmul 要求 (a, k) @ (k, b)；这里第二个矩阵第一维写错成 5，k 对不上 → 报错。
try:
    A = torch.randn(3, 4)
    B = torch.randn(5, 6)      # 应该是 (4, ?) 才能和 A 相乘
    _ = A @ B
except RuntimeError as e:
    # 报错信息里会出现 "mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)"
    print("  捕获 RuntimeError:", str(e).splitlines()[0])
    print("  解读：3x4 @ 5x6，内维 4≠5。处理：检查每一步的 .shape，对齐内维。")

print("\n【报错 2】device 不一致（CPU 张量和 GPU 张量混算）")
# 大模型最高频的坑之一：模型在 cuda，输入忘了 .to(device)，或反过来。
# CPU-only 环境无法真正造出 cuda 张量，这里用注释说明 GPU 上的真实报错，
# 并用「构造一个 device 不一致」的概念演示优雅降级。
if torch.cuda.is_available():
    try:
        x_cpu = torch.randn(2, 3)                  # 在 CPU
        w_gpu = torch.randn(3, 4).cuda()           # 在 GPU
        _ = x_cpu @ w_gpu                          # 跨设备 → 报错
    except RuntimeError as e:
        print("  捕获 RuntimeError:", str(e).splitlines()[0])
    print("  解读：报错含 'Expected all tensors to be on the same device'。")
else:
    # CPU-only：真实 GPU 上跨设备相乘会抛
    #   RuntimeError: Expected all tensors to be on the same device,
    #   but found at least two devices, cuda:0 and cpu!
    print("  (本机无 GPU) GPU 上的真实报错形如：")
    print("    Expected all tensors to be on the same device, "
          "but found at least two devices, cuda:0 and cpu!")
    print("  解读：模型 .to('cuda') 后，每个 batch 的输入也必须 .to('cuda')。"
          "统一用一个 device 变量搬运。")

print("\n【报错 3】dtype 不对（long vs float / 索引要整数、CE 要 long 标签）")
# CrossEntropyLoss 的 target 必须是 long（类别索引），传 float 会报错；
# 还有：用浮点张量当下标索引也会报错。
try:
    logits = torch.randn(4, 3)                     # (batch, num_class)
    target_float = torch.tensor([0.0, 1.0, 2.0, 0.0])   # 错：是 float
    _ = F.cross_entropy(logits, target_float)
except (RuntimeError, ValueError) as e:
    print("  捕获:", type(e).__name__, "-", str(e).splitlines()[0])
    print("  解读：CE 的标签必须是 torch.long（类别索引）。"
          "处理：target = target.long()。")

# 修正后就能跑通
target_long = torch.tensor([0, 1, 2, 0], dtype=torch.long)
ok_loss = F.cross_entropy(logits, target_long)
print(f"  修正 dtype 后 cross_entropy = {ok_loss.item():.4f}（跑通）")


# ============================================================
section("Part 2: nan / inf 检测 —— loss 炸了去哪查")
# ============================================================
# 训练中 loss 突然变成 nan，几乎都源于：除零、log(0)=-inf、exp 溢出=+inf、
# 学习率过大导致梯度爆炸。第一步永远是「定位是哪个张量先坏的」。
# 工具：torch.isnan / torch.isinf（逐元素返回 bool），再 .any() 看有没有。

print("torch.isnan / torch.isinf 是逐元素的布尔检测：")
bad = torch.tensor([1.0, float("nan"), 2.0, float("inf"), -float("inf")])
print("  张量      :", bad.tolist())
print("  isnan     :", torch.isnan(bad).tolist())
print("  isinf     :", torch.isinf(bad).tolist())
print("  含 nan?   :", torch.isnan(bad).any().item())
print("  含 inf?   :", torch.isinf(bad).any().item())

print("\n触发演示 1：除零 → inf / nan")
z = torch.tensor([1.0, 0.0])
div = torch.tensor([1.0, 0.0]) / z      # 1/1=1, 0/0=nan
print("  [1,0] / [1,0] =", div.tolist(), "→ 0/0 得 nan")

print("\n触发演示 2：log(0) = -inf（softmax 后某概率为 0 再取 log 就炸）")
p = torch.tensor([0.0, 0.5, 1.0])
print("  log([0, .5, 1]) =", torch.log(p).tolist(), "→ log(0) = -inf")

print("\n排查思路（detect anomaly）：")
# torch.autograd.set_detect_anomaly(True) 会让反向传播在产生 nan 的那一步
# 直接抛错并指出是哪个 forward 算子，便于定位（代价是变慢，只在 debug 时开）。
print("  - 前向：在每个可疑张量后 assert not torch.isnan(t).any()")
print("  - 反向：with torch.autograd.detect_anomaly(): loss.backward()")
print("    它会把「哪一步 forward 制造了 nan 梯度」精确报出来（慢，调试专用）。")

# 实演 detect_anomaly：故意用 sqrt(负数) 制造 nan 梯度
def _anomaly_demo():
    x = torch.tensor([-1.0], requires_grad=True)
    try:
        with torch.autograd.detect_anomaly():
            y = torch.sqrt(x)          # sqrt(-1)=nan，其梯度也 nan
            y.backward()
    except RuntimeError as e:
        print("  detect_anomaly 捕获:", str(e).splitlines()[0])

_anomaly_demo()


# ============================================================
section("Part 3: 参数量统计 + 显存估算（每参数约 16 字节）")
# ============================================================
# phase1 第 4 课的「显存账」：训练时一个 fp32 参数要占多少字节？
#   权重 w        : 4 字节（fp32）
#   梯度 grad     : 4 字节（和权重同形状）
#   Adam 一阶动量 m: 4 字节
#   Adam 二阶动量 v: 4 字节
#   ─────────────────────────
#   合计           : 16 字节 / 参数（这还【不含】前向激活！激活另算）
# 所以「能不能训」的快速估算：显存(GB) ≈ 参数量 × 16 / 1e9。

def count_params(model: nn.Module):
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory_gb(num_params: int, bytes_per_param: int = 16) -> float:
    """训练态显存粗估（GB）。口径：用 1e9（十进制 GB），与论文/厂商标称一致。
    bytes_per_param=16 = fp32权重4 + 梯度4 + Adam m 4 + Adam v 4。
    纯推理（无梯度无优化器状态）可传 bytes_per_param=4。"""
    return num_params * bytes_per_param / 1e9


# 搭个小模型实测参数统计
demo_model = nn.Sequential(
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
)
tot, trn = count_params(demo_model)
print(f"demo 模型参数量：total={tot:,}  trainable={trn:,}")

# 冻结第一层，观察 trainable 下降（迁移学习/LoRA 常见）
for p in demo_model[0].parameters():
    p.requires_grad = False
tot2, trn2 = count_params(demo_model)
print(f"冻结第一层后：total={tot2:,}（不变）  trainable={trn2:,}（下降）")

print("\n给定参数量估训练显存（每参数 16 字节，1e9 口径）：")
for n, label in [(1.24e8, "GPT-2 small 124M"),
                 (1.3e9, "1.3B"),
                 (7e9, "7B")]:
    gb = estimate_memory_gb(int(n))
    print(f"  {label:<18} ≈ {gb:6.1f} GB（仅 权重+梯度+Adam 状态，未含激活）")
print("  → 7B 训练态显存就要 ~112GB（权重+梯度 56GB + Adam 优化器态 m+v 56GB），")
print("    单卡放不下，所以才需要 ZeRO / 量化 / 多卡切分。")
print("  对照混合精度（AMP，见第 6 课）：优化器三件套仍≈16 字节/参")
print("    （fp16 权重2 + fp32 master 副本4 + fp16 梯度2 + Adam fp32 状态8），")
print("    所以 AMP 省的主要是【激活】与算力/速度，不是这组常驻显存。")


# ============================================================
section("Part 4: 训练算力估算 —— C ≈ 6ND")
# ============================================================
# 经验公式（OpenAI Scaling Laws / Chinchilla）：
#   训练总浮点运算量 C ≈ 6 × N × D
#   N = 模型参数量，D = 训练 token 总数。
# 直觉：每个 token 走一遍前向≈2N FLOPs（每个参数一次乘一次加），
# 反向≈2×前向=4N，合计 6N FLOPs/token，乘以 D 个 token 即 6ND。

def estimate_flops(num_params: int, num_tokens: int) -> float:
    """训练总 FLOPs 粗估：C ≈ 6 * N * D。"""
    return 6.0 * num_params * num_tokens


print("C ≈ 6ND 估算几个真实规模：")
cases = [
    ("GPT-2 small 124M / 300B tok", int(1.24e8), int(3e11)),
    ("LLaMA-7B / 1T tok",           int(7e9),     int(1e12)),
    ("70B / 2T tok（Chinchilla 量）", int(7e10),   int(2e12)),
]
for label, N, D in cases:
    C = estimate_flops(N, D)
    # 折算成 A100（约 312 TFLOPS bf16，实际利用率打 5 折≈156 TFLOPS）需要多少天
    a100_eff = 156e12          # 有效 FLOP/s
    gpu_seconds = C / a100_eff
    gpu_days = gpu_seconds / 86400
    print(f"  {label:<30} C≈{C:.2e} FLOPs ≈ 单卡 A100 {gpu_days:,.0f} 卡天")
print("  → 用「卡天 / 卡数」就能反推训练周期，这就是算力预算的算法。")


# ============================================================
section("Part 5: profiler / 计时 —— 一个操作到底慢在哪")
# ============================================================
# 优化前先测量。两种工具：
#   1) 简单计时：time.perf_counter() 包住操作（注意先 warmup，GPU 还要 synchronize）。
#   2) torch.profiler：能拆出每个算子的 CPU/CUDA 耗时，定位瓶颈。
# CPU 上没有 cuda 事件，profiler 仍可用，只统计 CPU 侧。

a = torch.randn(512, 512)
b = torch.randn(512, 512)

# warmup：首次调用有一次性开销（分配/缓存），不计入。
for _ in range(3):
    _ = a @ b

N_ITER = 50
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = a @ b
t1 = time.perf_counter()
print(f"简单计时：512x512 matmul 平均 {1000 * (t1 - t0) / N_ITER:.3f} ms/次")

# torch.profiler：优雅处理（老版本可能没有 record_function，用 try 兜底）
try:
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("matmul_block"):
            for _ in range(N_ITER):
                _ = a @ b
    print("\ntorch.profiler 按 CPU 总耗时排序的 top 算子：")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
except Exception as e:
    print(f"\n(profiler 不可用，已优雅降级) 原因: {type(e).__name__}: {e}")
    print("可改用 time.perf_counter() 手动计时，效果同样能定位热点。")


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. 三类经典报错：shape 不匹配（看内维）、device 不一致（输入忘 .to(device)）、
     dtype 不对（CE 标签要 long）。读报错第一行就能定位 90% 的问题。
  2. loss 变 nan/inf：用 torch.isnan/isinf().any() 逐张量排查；常见源是除零、log(0)、exp 溢出；
     反向用 torch.autograd.detect_anomaly() 精确定位制造 nan 的那一步（慢，调试专用）。
  3. 训练显存 ≈ 参数量 × 16 字节（fp32 权重4 + 梯度4 + Adam 动量 4+4），未含激活；
     7B 光这部分就 ~112GB，这是要 ZeRO / 量化 / 多卡的根本原因。
  4. 训练算力 C ≈ 6ND（N 参数、D token）：前向 2N + 反向 4N = 6N FLOPs/token；
     除以有效算力即得卡天，用来做训练预算。
  5. 优化前先测量：time.perf_counter() 要 warmup（GPU 还要 synchronize）；
     torch.profiler 能拆出每个算子的耗时，定位真正的瓶颈。

下一课：恭喜你打通了 PyTorch 专项 7 课，可以正式进入 phase3（训练 / LoRA / 量化 / RLHF / 推理）了。
""")
