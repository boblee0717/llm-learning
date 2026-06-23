"""
第 1 课：Tensor 机制
====================
PyTorch 的一切都建立在 tensor 上。这节课把你在 phase0 用 NumPy 练过的
形状 / 广播 / reshape，翻译成 torch 的写法，并补上 NumPy 没有、但读懂
PyTorch 代码必须知道的几件事：storage / stride、view vs reshape、
contiguous、in-place 操作、dtype / device、和 numpy 的内存共享。

核心问题：
- tensor 和 numpy ndarray 到底像不像？哪里不一样？
- view 和 reshape 有什么区别？为什么 transpose 之后要 contiguous？
- 一块内存（storage）+ 一组 stride 怎么就变出了「不同形状」的视图？
- in-place 操作（带下划线的 add_）省内存，但为什么有时会坑到 autograd？
- tensor 和 numpy 互转时，什么时候共享内存、什么时候是拷贝？

与大模型的关系：
- 多头注意力的 reshape + transpose、KV Cache 的就地写入、混合精度的 dtype 切换，
  全都依赖你对 tensor 内存布局的理解。看不懂 .contiguous() 报错，就调不动模型。

前置：phase0 第 1~3 课（向量 / 形状 / axis / 广播 / reshape / 多头切分）
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import numpy as np
import torch


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
section("Part 1: 创建 tensor —— 和 numpy 几乎一一对应")
# ============================================================
# numpy 怎么写，torch 基本就怎么写，只是把 np 换成 torch。

print("zeros:", torch.zeros(2, 3).shape)
print("ones :", torch.ones(2, 3).shape)
print("arange:", torch.arange(6))                 # 0..5
print("randn:", torch.randn(2, 3).shape)          # 标准正态
print("from list:", torch.tensor([[1, 2], [3, 4]]))

# 关键区别 1：dtype。numpy 默认 float64，torch 默认 float32（深度学习够用且省一半内存）
a = torch.tensor([1.0, 2.0, 3.0])
print("\ntorch 默认浮点 dtype:", a.dtype)          # torch.float32
print("numpy 默认浮点 dtype:", np.array([1.0]).dtype)  # float64

# 整数张量：torch.long（int64）是索引/标签的标准类型（cross_entropy 的 target 必须是 long）
idx = torch.tensor([0, 2, 1])
print("整数张量默认 dtype:", idx.dtype)             # torch.int64 (long)

# dtype 转换
print("float→long:", a.long())                     # 截断小数
print("long→float:", idx.float())


# ============================================================
section("Part 2: device —— tensor 住在 CPU 还是 GPU 上")
# ============================================================
# 这是 torch 比 numpy 多出来的核心维度：tensor 有 device 属性。
# 所有参与运算的 tensor 必须在同一个 device，否则报 "Expected all tensors
# to be on the same device"。

device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前可用 device:", device)

x = torch.randn(2, 2)
print("默认在:", x.device)            # 默认 cpu
x_dev = x.to(device)                  # 搬到目标 device（GPU 上才有意义）
print("搬运后:", x_dev.device)

# 工程惯例：写 device-agnostic 代码，模型和数据都 .to(device)，
# 这样同一份代码 CPU 能调试、GPU 能加速。新建 tensor 时也可直接 device=...
y = torch.zeros(2, 2, device=device)
print("直接在目标 device 创建:", y.device)


# ============================================================
section("Part 3: 广播 —— 和 numpy 完全相同的规则")
# ============================================================
# 从右往左对齐，维度相等或其中一个为 1 即可广播。phase0 第 1 课练过。

X = torch.randn(4, 3)        # (batch=4, features=3)
bias = torch.randn(3)        # (3,) → 广播成 (4, 3)
print("X + bias:", (X + bias).shape, "（bias 在 batch 维上共享）")

# 一个常见技巧：用 None / unsqueeze 插入长度 1 的维度来触发广播
col = torch.arange(3).reshape(3, 1)   # (3, 1)
row = torch.arange(4).reshape(1, 4)   # (1, 4)
print("外积（广播）:\n", col * row)     # (3, 4)
print("unsqueeze 等价于 None 索引:",
      torch.equal(col.squeeze().unsqueeze(1), col))


# ============================================================
section("Part 4: storage 与 stride —— 形状只是「视图」")
# ============================================================
# 这是理解 view / transpose / contiguous 的钥匙。
# 一个 tensor = 一块连续的一维内存（storage）+ 形状(shape) + 步长(stride)。
# stride[i] 表示「在第 i 维上前进 1 步，要在底层内存里跳几个元素」。

t = torch.arange(12).reshape(3, 4)
print("t.shape :", t.shape)
print("t.stride:", t.stride())   # (4, 1)：行内走 1 个、换行跳 4 个
print("底层 storage 是连续的 0..11:", t.flatten().tolist())

# transpose 不复制数据，只是交换 shape 和 stride —— 同一块内存的另一种「看法」
tt = t.transpose(0, 1)           # (4, 3)
print("\n转置后 shape :", tt.shape)
print("转置后 stride:", tt.stride(), "← 步长被交换，没有复制内存")
print("两者共享同一块 storage:",
      t.untyped_storage().data_ptr() == tt.untyped_storage().data_ptr())


# ============================================================
section("Part 5: view vs reshape vs contiguous")
# ============================================================
# view：要求底层内存连续，零拷贝地换个形状；不连续会直接报错。
# reshape：能 view 就 view，不能就先 copy 成连续再 view（更省心但可能偷偷拷贝）。
# contiguous：把「逻辑顺序」实际重排成连续内存，代价是一次拷贝。

base = torch.arange(12)
print("view 连续张量 OK:", base.view(3, 4).shape)

tt = torch.arange(12).reshape(3, 4).transpose(0, 1)  # 不连续
print("\ntranspose 后 is_contiguous:", tt.is_contiguous())
try:
    tt.view(12)      # 不连续 → view 报错
except RuntimeError as e:
    print("不连续直接 view 报错:", str(e).splitlines()[0])

print("先 contiguous 再 view OK:", tt.contiguous().view(12).shape)
print("或者直接用 reshape（内部自动处理）:", tt.reshape(12).shape)

# 这正是 phase2 多头注意力里 `transpose(1,2).contiguous().view(B,T,C)` 的由来：
# 合并多头前必须 contiguous，否则 view 会报错。


# ============================================================
section("Part 6: in-place 操作 —— 省内存，但小心 autograd")
# ============================================================
# 带下划线的方法（add_ / mul_ / zero_ / copy_）会就地修改张量、不分配新内存。
# 训练大模型时 KV Cache、优化器更新都用 in-place 省显存。

z = torch.ones(3)
z.add_(5)            # z 自己变成 6，没有新建张量
print("in-place add_:", z)

z2 = z + 5           # 非 in-place：返回新张量，z 不变
print("非 in-place: z 仍是", z, "结果是", z2)

# 坑：对需要梯度的张量做 in-place，可能破坏反向传播需要的中间值 → 报错。
w = torch.randn(3, requires_grad=True)
try:
    w.add_(1.0)      # 对叶子张量 in-place → autograd 报错
except RuntimeError as e:
    print("\n对 requires_grad 的叶子 in-place 报错:", str(e).splitlines()[0])
print("→ 经验：模型 forward 里别对要求梯度的张量做 in-place，调试期尤其如此")


# ============================================================
section("Part 7: 和 numpy 互转 —— 何时共享内存")
# ============================================================
# CPU 上 tensor ↔ numpy 默认【共享同一块内存】（零拷贝），改一个另一个也变。
# 这很高效，但也容易埋坑：以为是拷贝、结果改了原数据。

np_arr = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(np_arr)     # 共享内存
np_arr[0] = 99.0
print("from_numpy 共享内存 → 改 numpy，tensor 也变:", t_from_np)

t2 = torch.ones(3)
n2 = t2.numpy()                          # 共享内存
t2.add_(1.0)
print(".numpy() 共享内存 → 改 tensor，numpy 也变:", n2)

# 想要独立副本就显式 clone / copy
t3 = torch.from_numpy(np.array([1.0, 2.0])).clone()
print("clone 后是独立副本:", t3)

# 注意：带梯度的张量不能直接 .numpy()，要先 .detach()
g = torch.randn(2, requires_grad=True)
print("带梯度张量转 numpy 需先 detach:", g.detach().numpy())


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. tensor ≈ numpy + (dtype 默认 float32) + (device) + (autograd)。
  2. shape 只是「视图」：一块连续 storage + 一组 stride，transpose 只换 stride 不拷贝。
  3. view 要求连续（不连续会报错），reshape 会在必要时自动 contiguous（拷贝）。
     → 多头注意力合并前的 `.transpose().contiguous().view()` 就是这个原因。
  4. in-place（下划线方法）省内存，但别对 requires_grad 的张量乱用。
  5. CPU 上 tensor ↔ numpy 默认共享内存，需要独立副本就 clone；带梯度先 detach。

下一课：autograd —— 让 PyTorch 自动算出你以前在 phase1 手写的那些梯度。
""")
