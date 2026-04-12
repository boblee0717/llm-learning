"""
======================================================
第 1 课（自写版）：NumPy 基础 —— 大模型的数据基石
======================================================

用法：
1. 运行：python3 01_numpy_basics_self_write.py
2. 按 TODO 顺序补全关键实现
3. 每补完一段就运行，观察输出与校验结果
"""

import numpy as np


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    """统一的学习脚手架校验错误。"""


def shape_or_none(x):
    return None if x is None else getattr(x, "shape", None)


def require_not_none(name, value, hint):
    if value is None:
        raise ValidationError(f"{name} 错误：结果是 None。{hint}")


def require_close(name, actual, expected, hint="", atol=1e-6):
    try:
        if not np.allclose(actual, expected, atol=atol):
            raise ValidationError(
                f"{name} 错误：数值不正确。\nactual={actual}\nexpected={expected}\n{hint}"
            )
    except TypeError as err:
        raise ValidationError(f"{name} 错误：类型不正确。{hint}\n底层错误: {err}") from err


def require_shape(name, actual, expected_shape, hint=""):
    real_shape = shape_or_none(actual)
    if real_shape != expected_shape:
        raise ValidationError(
            f"{name} 错误：shape 不正确。actual_shape={real_shape}, expected_shape={expected_shape}\n{hint}"
        )


def reference_softmax(x: np.ndarray) -> np.ndarray:
    x_stable = x - np.max(x)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x)


# ============================================================
# 第一部分：从标量到张量（已给出）
# ============================================================
section("第一部分：从标量到张量")

scalar = np.array(42)
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2, 3], [4, 5, 6]])
tensor_3d = np.random.randn(2, 3, 4)

print(f"标量 shape: {scalar.shape}")
print(f"向量 shape: {vector.shape}")
print(f"矩阵 shape: {matrix.shape}")
print(f"3维张量 shape: {tensor_3d.shape}")
print()


# ============================================================
# 第二部分：核心运算（TODO）
# ============================================================
section("第二部分：核心运算（TODO）")

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print(f"逐元素加法: {a + b}")
print(f"逐元素乘法: {a * b}")

# TODO-1: 用 np.dot(a, b) 计算点积，结果应该是 32.0
# dot = ...
dot = None
print(f"点积: {dot}")

W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
x = np.array([1.0, 2.0, 3.0])

# TODO-2: 用矩阵乘法计算 output = x @ W
# output = ...
output = None
print(f"矩阵乘法输出: {output}")
print()


# ============================================================
# 第三部分：广播机制（TODO）
# ============================================================
section("第三部分：广播机制（TODO）")

mat = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])

# TODO-3: 使用广播计算 result = mat + bias
# result = ...
result = None
print("广播结果:")
print(result)
print()


# ============================================================
# 第四部分：Softmax（TODO）
# ============================================================
section("第四部分：实现 Softmax（TODO）")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    TODO-4:
    1) 做数值稳定处理：x - np.max(x)
    2) 计算指数
    3) 归一化（除以指数和）
    """
    # x_stable = ...
    # exp_x = ...
    # return ...
    raise NotImplementedError("TODO-4 未完成：请实现 softmax")
logits = np.array([2.0, 1.0, 0.5, -1.0])
softmax_error = None
probs = None
try:
    probs = softmax(logits)
except Exception as err:  # noqa: BLE001
    softmax_error = err

print(f"logits: {logits}")
if probs is not None:
    print(f"softmax: {np.round(probs, 4)}")
    print(f"sum: {np.sum(probs):.6f}")
else:
    print(f"softmax: 执行失败 -> {softmax_error}")
print()

# 补充：axis 和 keepdims（给后面 TODO-8 做铺垫）
axis_demo = np.array([
    [1.0, 0.5, 1.5],
    [0.2, 1.2, 1.4],
])  # shape (2, 3)
row_max_no_keep = np.max(axis_demo, axis=1)  # shape (2,)
row_max_keep = np.max(axis_demo, axis=1, keepdims=True)  # shape (2, 1)

print("【补充概念】axis 与 keepdims")
print(f"axis_demo.shape={axis_demo.shape}")
print(f"np.max(axis_demo, axis=1).shape={row_max_no_keep.shape}  -> 压缩了一个维度")
print(f"np.max(axis_demo, axis=1, keepdims=True).shape={row_max_keep.shape}  -> 保留维度")
print("axis=1 表示每一行单独处理；keepdims=True 让后续广播更直观")
print()

# 补充：axis=0/1/2、axis=(1,2) 与 keepdims 的形状差异
x3d_demo = np.array([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[101, 102, 103, 104], [105, 106, 107, 108], [109, 110, 111, 112]],
])  # shape (2, 3, 4)
print("【补充概念】axis 的具体含义")
print(f"x3d_demo.shape={x3d_demo.shape}")
print(f"sum axis=0 -> shape {np.sum(x3d_demo, axis=0).shape}")
print(f"sum axis=1 -> shape {np.sum(x3d_demo, axis=1).shape}")
print(f"sum axis=2 -> shape {np.sum(x3d_demo, axis=2).shape}")
print(f"sum axis=(1,2), keepdims=False -> shape {np.sum(x3d_demo, axis=(1,2)).shape}")
print(
    f"sum axis=(1,2), keepdims=True  -> shape {np.sum(x3d_demo, axis=(1,2), keepdims=True).shape}"
)
print("→ axis=(1,2) 表示同时压缩后两维；keepdims=True 会把它们保留成长度 1")
print()

# 反例：不加 keepdims 时可能广播失败
bad_demo = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])  # (2,3)
bad_row_max = np.max(bad_demo, axis=1)  # (2,)
print("【反例】广播失败演示")
print(f"bad_demo.shape={bad_demo.shape}, bad_row_max.shape={bad_row_max.shape}")
try:
    _ = bad_demo - bad_row_max
except ValueError as err:
    print(f"预期报错: {err}")
good_row_max = np.max(bad_demo, axis=1, keepdims=True)  # (2,1)
print(f"good_row_max.shape={good_row_max.shape}")
print(f"bad_demo - good_row_max =\n{bad_demo - good_row_max}")
print()


# ============================================================
# 第五部分：动手练习（TODO）
# ============================================================
section("第五部分：动手练习（TODO）")

# 练习 1：余弦相似度
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([2.0, 4.0, 6.0])
v3 = np.array([-1.0, -2.0, -3.0])

# TODO-5: 计算 cos_sim_12 和 cos_sim_13
# cos_sim_12 = ...
# cos_sim_13 = ...
cos_sim_12 = None
cos_sim_13 = None

print(f"cos(v1, v2) = {cos_sim_12}")
print(f"cos(v1, v3) = {cos_sim_13}")
print()


# 练习 2：矩阵批量运算
tokens = np.random.randn(3, 4)
weights = np.random.randn(4, 2)

# TODO-6: 计算 out = tokens @ weights
# out = ...
out = None
print(
    f"tokens.shape={tokens.shape}, weights.shape={weights.shape}, out.shape={shape_or_none(out)}"
)
print()


# 练习 3：简化 Attention 分数
query = np.array([1.0, 0.5])
keys = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

# TODO-7:
# 1) 计算 scores（每个 key 与 query 点积）
# 2) 计算 attn_weights = softmax(scores)
# scores = ...
# attn_weights = ...
scores = None
attn_weights = None

print(f"scores={scores}")
print(f"attn_weights={attn_weights}")
print()


# 练习 4：多 Query 的 Attention 分数（query 改为 2x2）
queries = np.array([[1.0, 0.5], [0.2, 1.2]])  # shape (2, 2)
keys2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # shape (3, 2)

# TODO-8:
# 1) 先算打分矩阵：score_matrix = queries @ keys2.T
#    - queries.shape = (2, 2)，表示 2 个 query，每个 2 维
#    - keys2.T.shape = (2, 3)，表示 3 个 key（转置后用于右乘）
#    - 所以 score_matrix.shape = (2, 3)
#      含义：第 i 行是第 i 个 query 对 3 个 key 的分数
# 2) 再做“逐行 softmax”（不是整个矩阵一起 softmax）
#    - 沿最后一维/列方向归一化：axis=1（等价 axis=-1）
#    - 每一行归一化后都应满足 sum == 1
# 3) 推荐数值稳定写法：
#    exp_scores = np.exp(score_matrix - np.max(score_matrix, axis=1, keepdims=True))
#    attn_matrix = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
# score_matrix = ...
# attn_matrix = ...
score_matrix = None
exp_scores = np.exp(score_matrix - np.max(score_matrix, axis=1, keepdims=True))
attn_matrix = None

print(f"queries.shape={queries.shape}, keys2.shape={keys2.shape}")
print(f"score_matrix.shape={shape_or_none(score_matrix)}")
print(f"score_matrix=\n{score_matrix}")
print(f"attn_matrix=\n{attn_matrix}")
print()


def validate_all() -> None:
    """校验模块：逐项检查 TODO 是否正确。"""
    require_not_none("TODO-1", dot, "请使用 np.dot(a, b) 计算点积。")
    require_close("TODO-1", dot, 32.0, "点积应为 32.0。")

    require_not_none("TODO-2", output, "请使用 x @ W 计算矩阵乘法。")
    require_close("TODO-2", output, np.array([2.2, 2.8]), "矩阵乘法结果应为 [2.2, 2.8]。")

    require_not_none("TODO-3", result, "请使用广播 result = mat + bias。")
    require_close(
        "TODO-3",
        result,
        np.array([[11, 22, 33], [14, 25, 36]]),
        "广播结果应与逐行加偏置一致。",
    )

    if softmax_error is not None:
        raise ValidationError(f"TODO-4 错误：softmax 执行失败。底层错误: {softmax_error}")
    require_not_none("TODO-4", probs, "请返回 softmax 概率向量。")
    require_close("TODO-4", np.sum(probs), 1.0, "softmax 概率和应为 1。")
    require_close("TODO-4", probs, reference_softmax(logits), "softmax 数值与参考实现不一致。")

    require_not_none("TODO-5", cos_sim_12, "请计算 cos(v1, v2)。")
    require_not_none("TODO-5", cos_sim_13, "请计算 cos(v1, v3)。")
    require_close("TODO-5", cos_sim_12, 1.0, "v1 与 v2 同方向，相似度应接近 1。")
    require_close("TODO-5", cos_sim_13, -1.0, "v1 与 v3 反方向，相似度应接近 -1。")

    require_not_none("TODO-6", out, "请计算 out = tokens @ weights。")
    require_shape("TODO-6", out, (3, 2), "输出应是 3 个 token 映射到 2 维后的结果。")
    require_close("TODO-6", out, tokens @ weights, "矩阵乘法结果与参考计算不一致。")

    require_not_none("TODO-7", scores, "请先计算 query 与每个 key 的点积。")
    require_close("TODO-7", scores, np.array([1.0, 0.5, 1.5]), "scores 应为 [1.0, 0.5, 1.5]。")
    require_not_none("TODO-7", attn_weights, "请计算 attn_weights = softmax(scores)。")
    require_close("TODO-7", np.sum(attn_weights), 1.0, "注意力权重和应为 1。")
    require_close("TODO-7", attn_weights, reference_softmax(scores), "attn_weights 与参考 softmax 不一致。")

    require_not_none("TODO-8", score_matrix, "请计算 score_matrix = queries @ keys2.T。")
    require_shape("TODO-8", score_matrix, (2, 3), "score_matrix 的 shape 应为 (2, 3)。")
    require_close(
        "TODO-8",
        score_matrix,
        np.array([[1.0, 0.5, 1.5], [0.2, 1.2, 1.4]]),
        "score_matrix 数值不正确，请检查矩阵乘法方向是否是 queries @ keys2.T。",
    )
    require_not_none("TODO-8", attn_matrix, "请计算 attn_matrix（对每行 score 做 softmax）。")
    require_shape("TODO-8", attn_matrix, (2, 3), "attn_matrix 的 shape 应为 (2, 3)。")
    require_close("TODO-8", np.sum(attn_matrix, axis=1), np.array([1.0, 1.0]), "attn_matrix 每行和应为 1。")
    require_close(
        "TODO-8",
        attn_matrix,
        np.vstack([reference_softmax(score_matrix[0]), reference_softmax(score_matrix[1])]),
        "attn_matrix 数值与参考 softmax 不一致。",
    )


section("自写版骨架就绪")
print("你可以按 TODO-1 到 TODO-8 逐个补全。")
print("开始自动校验...")
validate_all()
print("校验通过：你当前的实现全部正确。")
