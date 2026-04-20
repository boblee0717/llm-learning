"""
======================================================
phase0 / 第 3 课（自写版）：reshape / transpose / split
======================================================

用法：
1. 运行：python3 03_reshape_transpose_split_self_write.py
2. 按 TODO-N 顺序补全 `xxx = None` 处的实现
3. 每补完一个 TODO 就运行一次，依靠 assert 校验即时纠错

本课的核心场景是"多头注意力的形状变换"，所有 TODO 都对应 phase2 的真实写法。
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
        raise ValidationError(f"{name} 数值不对")


np.random.seed(7)


# ============================================================
section("TODO-1：把 (12,) 的 1D 数组变成 (3, 4)")
# ============================================================
x = np.arange(12)
# 提示：reshape
m = None  # TODO-1

require_shape("TODO-1 m", m, (3, 4))
require_close("TODO-1 m val", m, np.array([[0, 1, 2, 3],
                                            [4, 5, 6, 7],
                                            [8, 9, 10, 11]]))
print("m =\n", m)


# ============================================================
section("TODO-2：转置 (B=2, T=3, d=4) → (B=2, d=4, T=3)")
# ============================================================
y = np.arange(24).reshape(2, 3, 4)
# 提示：用 np.swapaxes(y, -1, -2) 或 y.transpose(0, 2, 1)
y_t = None  # TODO-2

require_shape("TODO-2 y_t", y_t, (2, 4, 3))
# 验证：y_t[b, i, j] == y[b, j, i]
require_close("TODO-2 sample", y_t[0, 0, 1], y[0, 1, 0])
print("y_t.shape =", y_t.shape)


# ============================================================
section("TODO-3：多头切分 (B, T, d) → (B, H, T, d_h)")
# ============================================================
# 这是 phase2 多头注意力的关键一步。
# 步骤：
#   1) reshape: (B, T, d) → (B, T, H, d_h)，其中 d = H * d_h
#   2) transpose: → (B, H, T, d_h)
B, T, d, H = 2, 4, 12, 3
d_h = d // H
X = np.arange(B * T * d).reshape(B, T, d).astype(float)

def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """TODO-3：把最后一维拆成 (H, d_h)，再把 H 换到 T 前面。"""
    # B_, T_, d_ = x.shape
    # d_h_ = d_ // num_heads
    # step1 = x.reshape(...)
    # return step1.transpose(...)
    raise NotImplementedError("TODO-3 未完成：请实现 split_heads")


X_split = split_heads(X, H)
require_shape("TODO-3 X_split", X_split, (B, H, T, d_h))
# 验证：第 0 个头的第 0 个 token 应该是 X[0, 0, :d_h]
require_close("TODO-3 head0 token0", X_split[0, 0, 0], X[0, 0, :d_h])
print("X_split.shape =", X_split.shape)


# ============================================================
section("TODO-4：多头合并 (B, H, T, d_h) → (B, T, d)")
# ============================================================
# 是 split_heads 的逆操作：先 transpose 把 H 换回 T 后面，再 reshape 合并。
def merge_heads(x: np.ndarray) -> np.ndarray:
    """TODO-4：把 (B, H, T, d_h) 合并回 (B, T, H*d_h)。"""
    # step1 = x.transpose(...)
    # B_, T_, H_, d_h_ = step1.shape
    # return step1.reshape(...)
    raise NotImplementedError("TODO-4 未完成：请实现 merge_heads")


X_back = merge_heads(X_split)
require_shape("TODO-4 X_back", X_back, (B, T, d))
require_close("TODO-4 与原 X 一致", X_back, X)
print("split + merge 之后能还原 ✓")


# ============================================================
section("TODO-5：W_qkv 一次算出 Q/K/V，再 split 回来")
# ============================================================
# 在 GPT 实现里常见：用一个 (d, 3d) 的大矩阵一次得到拼接的 QKV。
B, T, d = 2, 4, 8
X = np.random.randn(B, T, d)
W_qkv = np.random.randn(d, 3 * d)

# 第 1 步：算出 QKV 拼接 (B, T, 3d)
QKV = None  # TODO-5-1

require_shape("TODO-5-1 QKV", QKV, (B, T, 3 * d))

# 第 2 步：把最后一维 split 成 3 份 Q / K / V，每份 (B, T, d)
# 提示：np.split(QKV, 3, axis=-1)
Q, K, V = None, None, None  # TODO-5-2

require_shape("TODO-5-2 Q", Q, (B, T, d))
require_shape("TODO-5-2 K", K, (B, T, d))
require_shape("TODO-5-2 V", V, (B, T, d))
# 抽样验证：Q + K + V 拼起来应等于 QKV
require_close("TODO-5-2 拼回", np.concatenate([Q, K, V], axis=-1), QKV)
print("Q,K,V 形状 =", Q.shape, K.shape, V.shape)


# ============================================================
section("TODO-6：外积构造一个 (3, 5) 的 'unit table'")
# ============================================================
# 给定 u = [1, 2, 3]，v = [10, 20, 30, 40, 50]
# 求外积 outer，使 outer[i, j] = u[i] * v[j]
u = np.array([1, 2, 3])
v = np.array([10, 20, 30, 40, 50])

outer = None  # TODO-6

require_shape("TODO-6 outer", outer, (3, 5))
require_close("TODO-6 [1,2]", outer[1, 2], 2 * 30)
require_close("TODO-6 [2,4]", outer[2, 4], 3 * 50)
print("outer =\n", outer)


# ============================================================
section("TODO-7：手写 mask 屏蔽（softmax 前减去 1e9）")
# ============================================================
# 给定 scores: (T, T) 和因果 mask（下三角为 1，上三角为 0）。
# 要求：把 mask 为 0 的位置，scores 的对应位置变成 -1e9（用减法实现）。
# 提示：(1 - mask) * 1e9，再用 scores - 这一项。
T = 5
scores = np.random.randn(T, T)
mask = np.tril(np.ones((T, T)))   # 下三角

scores_masked = None  # TODO-7

require_shape("TODO-7 scores_masked", scores_masked, (T, T))
# 上三角位置（mask=0）应非常小
require_close("TODO-7 上三角被压低", scores_masked[0, 1] < -1e8, True, atol=0)
# 下三角位置应保持原值
require_close("TODO-7 下三角不动", scores_masked[1, 0], scores[1, 0])
print("masked 后第 0 行 =", scores_masked[0])

print("\n第 3 课自写练习全部通过！可以进入第 4 课。")
