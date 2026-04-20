"""
======================================================
phase0 / 第 2 课（自写版）：矩阵乘法与形状推断
======================================================

用法：
1. 运行：python3 02_matmul_and_shapes_self_write.py
2. 按 TODO-N 顺序补全 `xxx = None` 处的实现
3. 每补完一个 TODO 就运行一次，依靠 assert 校验即时纠错

口诀：(m, [k]) @ ([k], n) → (m, n)，方括号里的"内层维"被消掉。
"""

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    pass


def require_not_none(name, value):
    if value is None:
        raise ValidationError(f"{name} 未实现：结果是 None。")


def require_shape(name, actual, expected_shape):
    require_not_none(name, actual)
    if actual.shape != expected_shape:
        raise ValidationError(
            f"{name} 形状不对：actual={actual.shape}, expected={expected_shape}"
        )


def require_close(name, actual, expected, atol=1e-6):
    require_not_none(name, actual)
    if not np.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对：\nactual={actual}\nexpected={expected}"
        )


np.random.seed(42)


# ============================================================
section("TODO-1：基础矩阵乘法 (2,3) @ (3,4)")
# ============================================================
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)
B = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 0, 0]], dtype=float)

# 提示：用 @ 或 np.matmul
C = None  # TODO-1

expected = np.array([[4., 5., 1., 2.],
                     [10., 11., 4., 5.]])
require_shape("TODO-1 C", C, (2, 4))
require_close("TODO-1 C val", C, expected)
print("C =\n", C)


# ============================================================
section("TODO-2：用'行视角'重算 C 的第 0 行")
# ============================================================
# 不用 @，只用 np.dot 一次只算一个元素，把 C 的第 0 行手动凑出来。
# 提示：C[0, j] = A[0] · B[:, j]
c_row0 = np.zeros(4)
# TODO-2: 用 for 循环计算 c_row0[j] = A[0] @ B[:, j]
# for j in range(4):
#     c_row0[j] = ...

require_close("TODO-2 c_row0", c_row0, expected[0])
print("行视角 C[0] =", c_row0)


# ============================================================
section("TODO-3：Linear 层在 batched 输入上的 forward")
# ============================================================
# X: (B=2, T=4, d_in=8)，W: (d_in=8, d_out=5)，b: (5,)
# 求 Y = X @ W + b，形状 (2, 4, 5)
B_, T, d_in, d_out = 2, 4, 8, 5
X = np.random.randn(B_, T, d_in)
W = np.random.randn(d_in, d_out)
b = np.random.randn(d_out)

Y = None  # TODO-3

require_shape("TODO-3 Y", Y, (B_, T, d_out))
# 抽样验证：第 0 batch、第 0 token 应等于 X[0,0] @ W + b
require_close("TODO-3 sample", Y[0, 0], X[0, 0] @ W + b)
print("Y.shape =", Y.shape)


# ============================================================
section("TODO-4：注意力分数 Q @ K.T（最后两维转置）")
# ============================================================
# Q, K: (B=2, H=3, T=4, d_h=5)
# 期望 scores = Q @ K^T，形状 (2, 3, 4, 4)
# 提示：用 np.swapaxes(K, -1, -2) 而不是 K.T！
Q = np.random.randn(2, 3, 4, 5)
K = np.random.randn(2, 3, 4, 5)

scores = None  # TODO-4

require_shape("TODO-4 scores", scores, (2, 3, 4, 4))
# 验证：scores[0,0] 应等于 Q[0,0] @ K[0,0].T
require_close("TODO-4 sample", scores[0, 0], Q[0, 0] @ K[0, 0].T)
print("scores.shape =", scores.shape)


# ============================================================
section("TODO-5：用 einsum 重写注意力分数（不准用 swapaxes）")
# ============================================================
# 提示：'bhtd,bhsd->bhts'
scores_einsum = None  # TODO-5

require_shape("TODO-5 scores_einsum", scores_einsum, (2, 3, 4, 4))
require_close("TODO-5 一致", scores_einsum, scores)
print("einsum 与 swapaxes+@ 结果一致 ✓")


# ============================================================
section("TODO-6：完整 attention 输出 attn @ V")
# ============================================================
# 假设你已经从 scores 经过 softmax 得到 attn: (2, 3, 4, 4)
# 现在要算 out = attn @ V，其中 V: (2, 3, 4, 5)，out 形状 (2, 3, 4, 5)
def softmax_last(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

attn = softmax_last(scores / np.sqrt(5))
V = np.random.randn(2, 3, 4, 5)

out = None  # TODO-6

require_shape("TODO-6 out", out, (2, 3, 4, 5))
require_close("TODO-6 sample", out[0, 0], attn[0, 0] @ V[0, 0])
print("attention 输出形状 =", out.shape)


# ============================================================
section("TODO-7：写一个 'shape 推断器'")
# ============================================================
# 实现一个不真的做矩阵乘、只推断结果形状的函数。
# 规则：
#   - 最后两维按矩阵乘法：(..., m, k) @ (..., k, n) → (..., m, n)
#   - 内层 k 必须相等
#   - 前面的 batch 维按广播规则对齐（与第 1 课 TODO-7 同款）
def matmul_shape(shape_a, shape_b):
    """TODO-7：返回结果 shape（tuple），不合法时返回 None。"""
    # 实现思路：
    # 1) 取出 a 的最后两维 (m, ka) 和 b 的最后两维 (kb, n)
    # 2) 检查 ka == kb，否则返回 None
    # 3) 对 a/b 的前缀维按"右对齐 + 1 可广播"规则合并
    raise NotImplementedError("TODO-7 未完成：请实现 matmul_shape")


cases = [
    ((2, 3),       (3, 4),       (2, 4)),
    ((4, 10, 8),   (8, 16),      (4, 10, 16)),
    ((4, 2, 10, 8),(4, 2, 8, 16),(4, 2, 10, 16)),
    ((1, 10, 8),   (4, 2, 8, 16),(4, 2, 10, 16)),  # 广播 batch 维
    ((4, 3),       (5, 7),       None),             # 内层不一致
]
for sa, sb, expected in cases:
    got = matmul_shape(sa, sb)
    if got != expected:
        raise ValidationError(
            f"TODO-7 错：matmul_shape({sa}, {sb}) 应为 {expected}，得到 {got}"
        )
    print(f"  {sa} @ {sb} -> {got}")

print("\n第 2 课自写练习全部通过！可以进入第 3 课。")
