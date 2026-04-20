"""
======================================================
phase0 / 第 4 课（自写版）：线性层反向传播
======================================================

用法：
1. 运行：python3 04_matrix_calculus_self_write.py
2. 按 TODO-N 顺序补全
3. 每补完一个 TODO 就运行一次，依靠数值梯度校验即时纠错

口诀（Y = X @ W + b）：
    dL/dX = dL/dY @ W.T
    dL/dW = X.T  @ dL/dY
    dL/db = sum(dL/dY, axis=0)
形状对得上的就是对的。
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
            f"{name} 数值不对：\nactual={actual}\nexpected={expected}\nmax_diff={np.max(np.abs(np.asarray(actual)-np.asarray(expected)))}"
        )


def numeric_grad(f, x, eps=1e-5):
    """中心差分数值梯度，作为校验金标准。"""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps; fp = f(x)
        x[idx] = orig - eps; fm = f(x)
        x[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


np.random.seed(0)


# ============================================================
section("TODO-1：单层 Linear 的反向传播")
# ============================================================
B, d_in, d_out = 4, 5, 3
X = np.random.randn(B, d_in)
W = np.random.randn(d_in, d_out)
b = np.random.randn(d_out)

Y = X @ W + b
L = (Y ** 2).sum() / 2

dY = Y   # 因为 L = sum(Y^2)/2，所以 dL/dY = Y

# 提示：把 dY 看成上游传下来的"信号"，求 dX / dW / db。
dX = None  # TODO-1-X
dW = None  # TODO-1-W
db = None  # TODO-1-b

require_shape("TODO-1 dX", dX, X.shape)
require_shape("TODO-1 dW", dW, W.shape)
require_shape("TODO-1 db", db, b.shape)

# 数值梯度校验
def loss_W(W_): return ((X @ W_ + b) ** 2).sum() / 2
def loss_X(X_): return ((X_ @ W + b) ** 2).sum() / 2
def loss_b(b_): return ((X @ W + b_) ** 2).sum() / 2

require_close("TODO-1 dW vs num", dW, numeric_grad(loss_W, W.copy()), atol=1e-5)
require_close("TODO-1 dX vs num", dX, numeric_grad(loss_X, X.copy()), atol=1e-5)
require_close("TODO-1 db vs num", db, numeric_grad(loss_b, b.copy()), atol=1e-5)
print("单层 Linear 反向传播 ✓")


# ============================================================
section("TODO-2：两层网络的反向传播（链式法则）")
# ============================================================
# 模型：
#   H = X @ W1 + b1     (B, d_h)
#   Y = H @ W2 + b2     (B, d_out)
#   L = sum(Y^2) / 2
# 已经给你 X, W1, b1, W2, b2，请按链式法则一层一层往前算。

d_h = 6
W1 = np.random.randn(d_in, d_h)
b1 = np.random.randn(d_h)
W2 = np.random.randn(d_h, d_out)
b2 = np.random.randn(d_out)

H = X @ W1 + b1
Y = H @ W2 + b2

# 提示：先算 dY，再算第二层的 (dW2, db2, dH)，再算第一层的 (dW1, db1, dX)。
dY  = Y
dW2 = None  # TODO-2-W2
db2 = None  # TODO-2-b2
dH  = None  # TODO-2-H
dW1 = None  # TODO-2-W1
db1 = None  # TODO-2-b1
dX2 = None  # TODO-2-X

require_shape("TODO-2 dW2", dW2, W2.shape)
require_shape("TODO-2 db2", db2, b2.shape)
require_shape("TODO-2 dH",  dH,  H.shape)
require_shape("TODO-2 dW1", dW1, W1.shape)
require_shape("TODO-2 db1", db1, b1.shape)
require_shape("TODO-2 dX2", dX2, X.shape)

# 数值校验：扰动 W1，看 L 怎么变
def loss_W1_2(W1_):
    H_ = X @ W1_ + b1
    Y_ = H_ @ W2 + b2
    return (Y_ ** 2).sum() / 2

require_close("TODO-2 dW1 vs num", dW1, numeric_grad(loss_W1_2, W1.copy()), atol=1e-4)
print("两层网络反向传播 ✓")


# ============================================================
section("TODO-3：softmax 的雅可比 J = diag(s) - s s.T")
# ============================================================
def softmax_1d(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_jacobian(z):
    """TODO-3：返回 (n, n) 的雅可比矩阵，J[i,j] = ∂s_i/∂z_j。"""
    # s = softmax_1d(z)
    # return np.diag(s) - np.outer(s, s)
    raise NotImplementedError("TODO-3 未完成：请实现 softmax_jacobian")


z = np.array([1.0, 2.0, 3.0])
J = softmax_jacobian(z)
require_shape("TODO-3 J", J, (3, 3))

# 数值校验
def J_numeric(z, eps=1e-5):
    n = z.size
    out = np.zeros((n, n))
    for j in range(n):
        zp = z.copy(); zp[j] += eps
        zm = z.copy(); zm[j] -= eps
        out[:, j] = (softmax_1d(zp) - softmax_1d(zm)) / (2 * eps)
    return out

require_close("TODO-3 J vs num", J, J_numeric(z), atol=1e-6)
print("softmax 雅可比 ✓")


# ============================================================
section("TODO-4：softmax + 交叉熵的捷径 dz = s - y")
# ============================================================
# 这一题最重要：训练 LLM 时每一步反向传播都用到。
# 给定 logits z 和 one-hot 标签 y，求 dL/dz。
# 提示：dz = softmax(z) - y
def grad_softmax_ce(z, y_onehot):
    """TODO-4：返回 dL/dz，shape 与 z 相同。"""
    # s = softmax_1d(z)
    # return s - y_onehot
    raise NotImplementedError("TODO-4 未完成：请实现 grad_softmax_ce")


z = np.array([1.0, 2.0, 3.0])
y = np.array([0.0, 0.0, 1.0])

dz = grad_softmax_ce(z, y)

def loss_z(z_):
    s = softmax_1d(z_)
    return -np.sum(y * np.log(s + 1e-12))

require_shape("TODO-4 dz", dz, z.shape)
require_close("TODO-4 dz vs num", dz, numeric_grad(loss_z, z.copy()), atol=1e-5)
print("softmax+CE 捷径 ✓")


# ============================================================
section("TODO-5：手写一次完整的 batched softmax+CE 反向")
# ============================================================
# 真实训练里 logits 是 (B, V) 形状（B 个样本，V 个类别）。
# 损失 L = -1/B * sum_i log(softmax(z_i)[y_i])
# 求 dL/dz，期望形状 (B, V)，公式 = (softmax(z) - one_hot(y)) / B
B_, V = 4, 5
z = np.random.randn(B_, V)
labels = np.array([0, 2, 4, 1])

# 第 1 步：对最后一维做 softmax，得到 probs (B, V)
def softmax_2d(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

probs = softmax_2d(z)

# 第 2 步：构造 one-hot
one_hot = None  # TODO-5-1   形状 (B, V)
# 提示：np.eye(V)[labels] 一行搞定

require_shape("TODO-5-1 one_hot", one_hot, (B_, V))
require_close("TODO-5-1 [0]", one_hot[0], np.array([1, 0, 0, 0, 0]))
require_close("TODO-5-1 [3]", one_hot[3], np.array([0, 1, 0, 0, 0]))

# 第 3 步：计算 dL/dz
dz_batch = None  # TODO-5-2   形状 (B, V)，公式 (probs - one_hot) / B

require_shape("TODO-5-2 dz_batch", dz_batch, (B_, V))

# 数值梯度验证
def loss_z_batch(z_):
    p = softmax_2d(z_)
    log_p = np.log(p + 1e-12)
    return -np.mean(log_p[np.arange(B_), labels])

require_close("TODO-5-2 dz vs num", dz_batch, numeric_grad(loss_z_batch, z.copy()), atol=1e-5)
print("batched softmax+CE 反向 ✓")

print("\n第 4 课自写练习全部通过！")
print("现在你可以闭着眼把 phase1/03_neural_network.py 的反向传播再写一遍了。")
