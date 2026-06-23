"""
======================================================
PyTorch 专项 / 第 1 课（自写版）：Tensor 机制
======================================================

用法：
1. 运行：python 01_tensor_basics_self_write.py
2. 按 TODO-1 到 TODO-6 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错
   （没填的 TODO 会返回 None，校验提示「未实现」，这是正常的）

目标：
- 用广播手写外积（不许用 torch.outer）
- 手写多头切分 split_heads 与多头合并 merge_heads（理解 contiguous 的必要性）
- 把不连续张量安全地变回连续视图
- 把带梯度的张量安全转成 numpy（detach）
- 用 in-place 操作就地修改张量（理解「同一块内存」）

对照：本课主课 01_tensor_basics.py，以及 phase0 第 3 课（多头切分）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")  # ValidationError 走 stderr，也要防乱码

import numpy as np
import torch

torch.manual_seed(0)


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    pass


def require_not_none(name, value):
    if value is None:
        raise ValidationError(f"{name} 未实现：结果是 None。")


def require_true(name, cond, hint=""):
    if not cond:
        raise ValidationError(f"{name} 条件不满足：{hint}")


def require_shape(name, actual, expected_shape):
    require_not_none(name, actual)
    if tuple(actual.shape) != tuple(expected_shape):
        raise ValidationError(
            f"{name} 形状不对：actual={tuple(actual.shape)}, expected={tuple(expected_shape)}"
        )


def require_close(name, actual, expected, atol=1e-5):
    require_not_none(name, actual)
    if not torch.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}"
        )


# ============================================================
section("TODO-1：用广播手写外积 outer_via_broadcast")
# ============================================================
# 输入 a: (m,)，b: (n,)，输出 (m, n)，其中 out[i, j] = a[i] * b[j]。
# 不许用 torch.outer —— 要用广播亲手拼出来。
#
# 提示：把 a 变成列向量 (m, 1)，把 b 变成行向量 (1, n)，相乘即广播成 (m, n)。
#   a[:, None]  或  a.unsqueeze(1)   → (m, 1)
#   b[None, :]  或  b.unsqueeze(0)   → (1, n)


def outer_via_broadcast(a, b):
    # TODO-1: 用广播返回 (m, n) 的外积
    return None


_a = torch.arange(1, 4, dtype=torch.float32)   # [1, 2, 3]
_b = torch.arange(1, 5, dtype=torch.float32)   # [1, 2, 3, 4]
_outer = outer_via_broadcast(_a, _b)
require_shape("TODO-1 外积形状", _outer, (3, 4))
require_close("TODO-1 数值", _outer, torch.outer(_a, _b))
print("outer_via_broadcast OK：\n", _outer)


# ============================================================
section("TODO-2：多头切分 split_heads")
# ============================================================
# 输入 x: (B, T, C)，n_head 整除 C。输出 (B, n_head, T, d_k)，d_k = C // n_head。
# 这就是多头注意力把 d_model 切成多个 head 的那一步。
#
# 提示（和 phase2 的 numpy reshape + transpose 是同一件事，只是多了 batch 维）：
#   1. view 成 (B, T, n_head, d_k)
#   2. transpose(1, 2) 把 n_head 提到 T 前面 → (B, n_head, T, d_k)


def split_heads(x, n_head):
    # TODO-2: 返回 (B, n_head, T, d_k)
    return None


_x = torch.randn(2, 5, 12)
_heads = split_heads(_x, 4)
require_shape("TODO-2 split_heads", _heads, (2, 4, 5, 3))
# 第 0 个 head 应该正好取 x 的前 d_k=3 列
require_close("TODO-2 head 0 内容", _heads[:, 0], _x[..., :3])
print("split_heads OK：(2, 5, 12) → (2, 4, 5, 3)")


# ============================================================
section("TODO-3：多头合并 merge_heads")
# ============================================================
# 输入 x: (B, n_head, T, d_k)，输出 (B, T, C)，C = n_head * d_k。
# 是 split_heads 的逆操作，也是注意力算完后把多头拼回去的那一步。
#
# 关键坑：transpose 之后内存不连续，直接 view 会报错，必须先 contiguous！
# 提示：x.transpose(1, 2).contiguous().view(B, T, C)


def merge_heads(x):
    # TODO-3: 返回 (B, T, C)，记得 transpose 后要 contiguous 再 view
    return None


_merged = merge_heads(_heads)
require_shape("TODO-3 merge_heads", _merged, (2, 5, 12))
# merge 应该是 split 的逆：还原回原始 x
require_close("TODO-3 merge 是 split 的逆", _merged, _x)
require_true("TODO-3 结果必须连续", _merged.is_contiguous(),
             "transpose 后要 .contiguous() 再 view，否则结果不连续")
print("merge_heads OK：(2, 4, 5, 3) → (2, 5, 12)，且还原回了原始张量")


# ============================================================
section("TODO-4：把不连续张量变回连续一维 flatten_contiguous")
# ============================================================
# 输入一个【转置过、因此不连续】的二维张量，返回它按行优先展平的一维张量。
# 直接 .view(-1) 会报错（不连续），你需要正确处理。
#
# 提示：用 .reshape(-1)（内部自动 contiguous），或 .contiguous().view(-1)。


def flatten_contiguous(x):
    # TODO-4: 返回展平后的一维连续张量
    return None


_t = torch.arange(6).reshape(2, 3).transpose(0, 1)   # (3, 2)，不连续
require_true("前置检查：输入确实不连续", not _t.is_contiguous(), "测试数据应是不连续的")
_flat = flatten_contiguous(_t)
require_shape("TODO-4 展平形状", _flat, (6,))
require_true("TODO-4 结果连续", _flat.is_contiguous(), "结果应是连续张量")
# 转置后按行优先展平：原 storage 是 0..5，转置后逻辑顺序是 [0,3,1,4,2,5]
require_close("TODO-4 按转置后的逻辑顺序展平", _flat,
              torch.tensor([0, 3, 1, 4, 2, 5]))
print("flatten_contiguous OK：", _flat.tolist())


# ============================================================
section("TODO-5：带梯度的张量安全转 numpy detach_to_numpy")
# ============================================================
# 输入一个 requires_grad=True 的张量，返回对应的 numpy 数组。
# 直接 .numpy() 会报错（带梯度的张量在计算图里），必须先 detach。
#
# 提示：t.detach().numpy()
#   detach 把张量从计算图里「摘下来」，得到一个不带梯度、共享数据的副本视图。


def detach_to_numpy(t):
    # TODO-5: 返回 numpy 数组
    return None


_g = torch.randn(3, requires_grad=True)
_np = detach_to_numpy(_g)
require_not_none("TODO-5 detach_to_numpy", _np)
require_true("TODO-5 返回的是 numpy 数组", isinstance(_np, np.ndarray),
             "应返回 numpy.ndarray")
require_close("TODO-5 数值一致", torch.from_numpy(_np), _g.detach())
print("detach_to_numpy OK：", _np)


# ============================================================
section("TODO-6：in-place 就地加法 add_in_place")
# ============================================================
# 对输入张量【就地】加上 value，并返回这个张量本身（不能新建张量）。
# 校验会检查：返回的是同一个对象（底层内存地址不变），且数值正确。
#
# 提示：用带下划线的 in-place 方法 t.add_(value)，它返回 t 自己。


def add_in_place(t, value):
    # TODO-6: 就地给 t 加上 value 并返回 t 本身
    return None


_orig = torch.ones(4)
_ptr_before = _orig.data_ptr()
_ret = add_in_place(_orig, 5.0)
require_not_none("TODO-6 add_in_place", _ret)
require_close("TODO-6 数值", _ret, torch.full((4,), 6.0))
require_true("TODO-6 必须是就地修改（同一块内存）",
             _ret.data_ptr() == _ptr_before,
             "应该用 in-place 的 add_，而不是 t = t + value（后者会新建张量）")
require_true("TODO-6 返回的是同一个对象", _ret is _orig,
             "in-place 方法返回张量自身")
print("add_in_place OK：就地把 [1,1,1,1] 变成", _ret.tolist())


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. 用广播拼出外积（列向量 × 行向量）
  2. split_heads / merge_heads 多头切分与合并
  3. 理解了 merge 时为什么必须 contiguous（transpose 破坏了内存连续性）
  4. 把不连续张量安全展平（reshape / contiguous().view）
  5. 带梯度张量安全转 numpy（detach）
  6. in-place 就地修改（同一块内存，省显存）

复盘三问：
  * view 和 reshape 的区别是什么？什么时候 view 会报错？
  * 为什么 transpose 之后内存不连续？它改的是 storage 还是 stride？
  * tensor ↔ numpy 什么时候共享内存？带梯度的张量为什么要先 detach？
""")
