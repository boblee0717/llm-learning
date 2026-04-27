"""
======================================================
phase0 / 第 1 课（自写版）：向量、形状与轴
======================================================

用法：
1. 运行：python3 01_vectors_and_axes_self_write.py
2. 按 TODO-N 顺序补全 `xxx = None` 处的实现
3. 每补完一个 TODO 就运行一次，依靠 assert 校验即时纠错

提示：所有 TODO 都只用 numpy，不要导入新的库。
"""

import sys
from itertools import zip_longest
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    """统一的学习脚手架校验错误。"""


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


# ============================================================
section("TODO-0：搞清楚 (3,) / (1,3) / (3,1) 到底有什么不一样")
# ============================================================
# 背景：np.array([1,2,3]).shape 是 (3,) 而不是 (1,3)。
# 这是因为 numpy 严格区分 "ndim（几维）"：
#   - (3,)   是 1 维向量，没有行/列概念
#   - (1, 3) 是 2 维矩阵，恰好只有一行
#   - (3, 1) 是 2 维矩阵，恰好只有一列
# numpy 不会自动把 1 维补成 2 维，否则你就丢失了"我就是个一维向量"的表达能力，
# 而且后面 axis、keepdims、广播的语义都会乱。
#
# 本题：从一个 1 维向量 u 出发，用 4 种不同方式构造出对应的 2 维形状，
# 然后亲眼看看它们的转置（.T）行为差在哪里。

u = np.array([1, 2, 3])

# 提示：
#   - u_row_a：用 reshape 得到 (1, 3)
#   - u_row_b：用 None / np.newaxis 索引得到 (1, 3)，写法 u[None, :]
#   - u_col_a：用 reshape 得到 (3, 1)
#   - u_col_b：用 None / np.newaxis 索引得到 (3, 1)，写法 u[:, None]
u_row_a = u.reshape(1, 3)   # TODO-0-row-a
u_row_b = u[None, :]  # TODO-0-row-b
u_col_a = u.reshape(3, 1)  # TODO-0-col-a
u_col_b = u[:, None]  # TODO-0-col-b

require_shape("TODO-0 u_row_a", u_row_a, (1, 3))
require_shape("TODO-0 u_row_b", u_row_b, (1, 3))
require_shape("TODO-0 u_col_a", u_col_a, (3, 1))
require_shape("TODO-0 u_col_b", u_col_b, (3, 1))

print("u.shape       =", u.shape, "  ndim =", u.ndim)
print("u.T.shape     =", u.T.shape, "  <- 1 维数组转置还是它自己！")
print("u_row_a.shape =", u_row_a.shape, "  u_row_a.T.shape =", u_row_a.T.shape)
print("u_col_a.shape =", u_col_a.shape, "  u_col_a.T.shape =", u_col_a.T.shape)

if u.T.shape != (3,):
    raise ValidationError("u 是 1 维，转置后形状仍应为 (3,)")
if u_row_a.T.shape != (3, 1):
    raise ValidationError("(1,3) 行向量转置后应为 (3,1) 列向量")


# ============================================================
section("TODO-1：把一维向量 v 变成行向量和列向量")
# ============================================================
v = np.array([1, 2, 3, 4])
# 提示：用 reshape，分别得到 (1,4) 和 (4,1)。
row = v.reshape(1, 4) # TODO-1-row
col = v.reshape(4, 1) # TODO-1-col

require_shape("TODO-1 row", row, (1, 4))
require_shape("TODO-1 col", col, (4, 1))
print("row =", row, "  col.shape =", col.shape)


# ============================================================
section("TODO-2：用 @ 计算两个向量的点积")
# ============================================================
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
# 提示：np.dot(a, b) 或 a @ b 都行，结果应是 32.0
dot = np.dot(a, b) # TODO-2

require_close("TODO-2 dot", dot, 32.0)
print("dot(a, b) =", dot)


# ============================================================
section("TODO-3：计算余弦相似度 cos(a, b) = (a·b) / (|a||b|)")
# ============================================================
# 提示：
#   - 范数用 np.linalg.norm
#   - 结果应在 0.97 左右
cos_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) # TODO-3

require_close("TODO-3 cos_ab", cos_ab, 0.9746318461970762, atol=1e-6)
print("cos(a, b) =", cos_ab)


# ============================================================
section("TODO-4：按行求和与按列求和")
# ============================================================
x = np.array([[1, 2, 3],
              [4, 5, 6]])

# 提示：np.sum 配合 axis 参数。
sum_by_row = np.sum(x, axis=1) # TODO-4-row    形状 (2,)，期望 [6, 15]
sum_by_col = np.sum(x, axis=0) # TODO-4-col    形状 (3,)，期望 [5, 7, 9]

require_shape("TODO-4 row", sum_by_row, (2,))
require_shape("TODO-4 col", sum_by_col, (3,))
require_close("TODO-4 row val", sum_by_row, np.array([6, 15]))
require_close("TODO-4 col val", sum_by_col, np.array([5, 7, 9]))
print("行和:", sum_by_row, "  列和:", sum_by_col)


# ============================================================
section("TODO-4a：3D 张量上的 axis（先在脑子里推 shape，再写代码）")
# ============================================================
# 把 t 看作 (batch=2, seq_len=3, hidden=4) —— 这是 LLM 里最典型的 shape。
# 数据故意构造得很规整，方便你心算验证。
t = np.arange(24).reshape(2, 3, 4)
# t[0] =
#   [[ 0,  1,  2,  3],
#    [ 4,  5,  6,  7],
#    [ 8,  9, 10, 11]]
# t[1] =
#   [[12, 13, 14, 15],
#    [16, 17, 18, 19],
#    [20, 21, 22, 23]]

# 提示（先不要看下面的期望值，自己用"聚合方向法"推一下 shape）：
#   axis=0  消掉 batch 维
#   axis=1  消掉 seq 维 → 每个 batch 把 3 个 token 加起来 → (batch, hidden)
#   axis=-1 等价 axis=2，消掉 hidden 维 → 每个 token 把 4 个 feature 加起来 → (batch, seq)

sum_axis0  = np.sum(t, axis=0)  # TODO-4a-1：np.sum(t, axis=0)，shape 应该是 (3, 4)
sum_axis1  = np.sum(t, axis=1)  # TODO-4a-2：每个 batch 的 token 之和，shape 应该是 (2, 4)
sum_axisM1 = np.sum(t, axis=-1)  # TODO-4a-3：每个 token 的 feature 之和，shape 应该是 (2, 3)

require_shape("TODO-4a sum_axis0",  sum_axis0,  (3, 4))
require_shape("TODO-4a sum_axis1",  sum_axis1,  (2, 4))
require_shape("TODO-4a sum_axisM1", sum_axisM1, (2, 3))

# 数值期望（你可以用上面的 t[0] / t[1] 心算验证）
require_close(
    "TODO-4a sum_axis0 val", sum_axis0,
    np.array([[12, 14, 16, 18],
              [20, 22, 24, 26],
              [28, 30, 32, 34]]),
)
require_close(
    "TODO-4a sum_axis1 val", sum_axis1,
    np.array([[12, 15, 18, 21],
              [48, 51, 54, 57]]),
)
require_close(
    "TODO-4a sum_axisM1 val", sum_axisM1,
    np.array([[ 6, 22, 38],
              [54, 70, 86]]),
)
print("TODO-4a 通过：3D 上的 axis=0/1/-1 都对")


# ============================================================
section("TODO-4b：keepdims 的 shape 区别 + 多轴聚合")
# ============================================================
# 同一个 t (2, 3, 4)。
# 要求：
#   1) 不带 keepdims 时 shape 应该是 (2, 4)
#   2) 带 keepdims=True 时 shape 应该是 (2, 1, 4)  —— 中间那一维被保留为 1
#   3) 在 axis=(0, 1) 上同时聚合，shape 应该是 (4,)
#      —— 一次抠掉两条轴：消掉 batch 也消掉 seq，只剩 hidden

without_kd = np.sum(t, axis=1)  # TODO-4b-1：np.sum(t, axis=1)
with_kd    = np.sum(t, axis=1, keepdims=True) # TODO-4b-2：np.sum(t, axis=1, keepdims=True)
sum_two    = np.sum(t, axis=(0, 1)) # TODO-4b-3：np.sum(t, axis=(0, 1))

require_shape("TODO-4b without_kd", without_kd, (2, 4))
require_shape("TODO-4b with_kd",    with_kd,    (2, 1, 4))
require_shape("TODO-4b sum_two",    sum_two,    (4,))

# with_kd 去掉中间那个 1，应该和 without_kd 一模一样
require_close("TODO-4b with vs without", with_kd.squeeze(axis=1), without_kd)
# sum_two 应当等于 without_kd 在 batch 维再求和 → 也等于 axis=0 的结果再 axis=0 一次
require_close("TODO-4b sum_two val", sum_two, without_kd.sum(axis=0))
print("TODO-4b 通过：keepdims 多了一个长度 1 的占位轴，axis=(0,1) 一次干掉两维")


# ============================================================
section("TODO-4c：mean / max / argmax 都吃同一套 axis 规则")
# ============================================================
# 模拟一个 (batch=2, num_classes=5) 的 logits，让你练 attention/分类里最常见的几个聚合。
logits = np.array([
    [2.0, 1.0, 0.1, -1.0,  3.0],
    [0.5, 0.5, 4.0,  2.0, -2.0],
])

# 提示：
#   - 每行平均：axis=-1，shape (2,)
#   - 每行最大值：axis=-1，shape (2,)
#   - 每行最大值的下标（预测的类别）：np.argmax，axis=-1，shape (2,)
row_mean   = np.mean(logits, axis=-1) # TODO-4c-1
row_max    = np.max(logits, axis=-1) # TODO-4c-2
row_argmax = np.argmax(logits, axis=-1) # TODO-4c-3

require_shape("TODO-4c row_mean",   row_mean,   (2,))
require_shape("TODO-4c row_max",    row_max,    (2,))
require_shape("TODO-4c row_argmax", row_argmax, (2,))

require_close("TODO-4c row_mean val", row_mean, np.array([1.02, 1.0]))
require_close("TODO-4c row_max  val", row_max,  np.array([3.0, 4.0]))
# argmax 是整数，用普通比较即可
if not np.array_equal(row_argmax, np.array([4, 2])):
    raise ValidationError(
        f"TODO-4c row_argmax 错：actual={row_argmax}, expected=[4 2]"
    )
print("TODO-4c 通过：predicted classes =", row_argmax)


# ============================================================
section("TODO-5：实现行级 softmax（必须用 keepdims）")
# ============================================================
# 给定 scores: (B, n)，对每一行做 softmax，使每行和为 1。
# 要求步骤：
#   1) 减去每行最大值（数值稳定）
#   2) 取指数
#   3) 除以每行的指数和
# 提示：np.max / np.sum 都要用 axis=1 + keepdims=True。

def softmax_rowwise(scores: np.ndarray) -> np.ndarray:
    """TODO-5：对最后一维做 softmax。"""
    row_max = np.max(scores, axis=-1, keepdims=True)
    exp_s   = np.exp(scores - row_max)
    return  exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    raise NotImplementedError("TODO-5 未完成：请实现 softmax_rowwise")


scores = np.array([[1.0, 2.0, 3.0],
                   [1.0, 1.0, 1.0]])
probs = softmax_rowwise(scores)
require_shape("TODO-5 probs", probs, (2, 3))
require_close("TODO-5 row sum", probs.sum(axis=1), np.array([1.0, 1.0]))
require_close(
    "TODO-5 first row",
    probs[0],
    np.array([0.09003057, 0.24472847, 0.66524096]),
    atol=1e-6,
)
print("softmax 结果 =\n", probs)


# ============================================================
section("TODO-6：广播加 bias")
# ============================================================
# 给定 X: (B=2, T=4, d=5)，bias: (5,)
# 要求 Y = X + bias，形状仍为 (2, 4, 5)。
np.random.seed(0)
X = np.random.randn(2, 4, 5)
bias = np.arange(5).astype(float)
print("bias shape =", bias.shape)

# 提示：直接加号即可，但请先在脑子里推一下广播规则。
Y = X + bias # TODO-6

require_shape("TODO-6 Y", Y, (2, 4, 5))
# 第 0 个 batch 第 0 个 token 加完后应等于 X[0,0] + bias
require_close("TODO-6 first token", Y[0, 0], X[0, 0] + bias)
print("Y.shape =", Y.shape)


# ============================================================
section("TODO-7：判断两个 shape 能否广播（不报错就是能）")
# ============================================================
# 不要用 try/except，请用形状逻辑判断。
# 规则：从右往左对齐两个 shape，对应维要么相等，要么其中之一为 1。
# 缺位的维当成 1。
def can_broadcast(shape_a, shape_b) -> bool:
    """TODO-7：返回 True/False。"""
    # 提示：先把两个 shape 反转，然后逐位检查；缺位用 1 兜底。
    for a, b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        # TODO：在这里写一个判断，不满足就 return False
        if a != b and a != 1 and b != 1:
            return False
    return True
    raise NotImplementedError("TODO-7 未完成：请实现 can_broadcast")


cases = [
    ((2, 3, 4), (3, 4),     True),
    ((2, 3, 4), (1, 4),     True),
    ((2, 3, 4), (4,),       True),
    ((2, 3, 4), (2, 4),     False),   # 第 -2 维 3 vs 2 冲突
    ((5, 1, 6), (1, 4, 6),  True),
    ((5, 2, 6), (1, 4, 6),  False),

    # ---- 5 维 case，模拟真实 attention/视频/批量矩阵场景 ----
    # (B, H, T, T, D) vs (1, 1, 1, 1, D)：bias 在前 4 维都共享，最后一维匹配
    ((2, 8, 16, 16, 64), (1, 1, 1, 1, 64), True),

    # 5 维 vs 3 维：左边自动补 1 → (1, 1, 4, 5, 6) vs (2, 3, 4, 5, 6)
    # 每对都满足 "相等或一个是 1"
    ((2, 3, 4, 5, 6), (4, 5, 6),           True),

    # 5 维 vs 5 维：多处 1 ↔ N 的"互相帮忙"扩张
    # (5,1,6,1,8) vs (1,4,1,7,8) → 每位都合法 → 结果 (5,4,6,7,8)
    ((5, 1, 6, 1, 8), (1, 4, 1, 7, 8),     True),

    # 5 维冲突：第 -3 维 6 vs 7 都不是 1 又不相等 → 报错
    ((5, 1, 6, 1, 8), (1, 4, 7, 1, 8),     False),

    # 5 维 vs 4 维：左边补 1 后第 -4 维 3 vs 2 冲突
    # 对齐为 (2,3,4,5,6) vs (1,2,4,5,6)，第二位 3 vs 2 冲突
    ((2, 3, 4, 5, 6), (2, 4, 5, 6),        False),

    # LLM 经典：attention 加 mask
    # scores: (batch, heads, T, T)  mask: (1, 1, T, T)
    ((2, 8, 16, 16), (1, 1, 16, 16),       True),
]
for sa, sb, expected in cases:
    got = can_broadcast(sa, sb)
    if got != expected:
        raise ValidationError(
            f"TODO-7 错：can_broadcast({sa}, {sb}) 应为 {expected}，得到 {got}"
        )
    print(f"  {sa} vs {sb} -> {got}")
print("TODO-7 全部通过")

print("\n第 1 课自写练习全部通过！可以进入第 2 课。")
