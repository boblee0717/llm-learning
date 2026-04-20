"""
phase0 / 第 4 课：矩阵微分入门（线性层反向传播）

学完本课你能回答：
1. 已知 Y = X @ W + b 和 dL/dY，怎么求 dL/dX、dL/dW、dL/db？
2. 这些公式怎么从"形状必须对得上"反推出来？
3. softmax 的雅可比为什么是 diag(s) - s @ s.T？
4. 怎么用"数值梯度"校验你写的"解析梯度"是对的？

跑这个文件：
    python3 04_matrix_calculus.py

只要把这一节练熟，phase1 第 3 课的反向传播就是"套公式"。
"""

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

np.random.seed(0)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
section("1) 一句话掌握线性层求导：'形状对得上' 就是答案")
# ---------------------------------------------------------------------------
# 设：
#   X   形状 (B, d_in)
#   W   形状 (d_in, d_out)
#   b   形状 (d_out,)
#   Y = X @ W + b           形状 (B, d_out)
#   L 是标量损失
#
# 由形状对应（链式法则的"形状直觉"）：
#
#   dL/dY  形状 (B, d_out)        ← 上游传下来的
#   dL/dX  形状 (B, d_in)
#         = dL/dY  @  W.T         （d_out 在中间被消掉）
#
#   dL/dW  形状 (d_in, d_out)
#         = X.T  @  dL/dY         （B 这一维被消掉，沿 batch 求和）
#
#   dL/db  形状 (d_out,)
#         = dL/dY 在 batch 维求和  → np.sum(dL/dY, axis=0)
#
# 不需要硬背：让 (?,?) 的形状凑出来，唯一对得上的就是这种写法。

B, d_in, d_out = 4, 5, 3
X = np.random.randn(B, d_in)
W = np.random.randn(d_in, d_out)
b = np.random.randn(d_out)

Y = X @ W + b
L = (Y ** 2).sum() / 2          # 选个简单的损失：dL/dY = Y

dY = Y                          # ∂L/∂Y = Y（因为 L = sum(Y^2)/2）

dX_analytic = dY @ W.T          # (B, d_out) @ (d_out, d_in) → (B, d_in)
dW_analytic = X.T @ dY          # (d_in, B)  @ (B, d_out)    → (d_in, d_out)
db_analytic = dY.sum(axis=0)    # 沿 batch 求和              → (d_out,)

print("dX 形状:", dX_analytic.shape, " 期望 (B, d_in)   =", (B, d_in))
print("dW 形状:", dW_analytic.shape, " 期望 (d_in,d_out)=", (d_in, d_out))
print("db 形状:", db_analytic.shape, " 期望 (d_out,)    =", (d_out,))


# ---------------------------------------------------------------------------
section("2) 数值梯度校验：把'解析梯度'和'有限差分'对一遍")
# ---------------------------------------------------------------------------
# 数值梯度的定义：
#   ∂L/∂x_i  ≈  ( L(x + ε e_i) - L(x - ε e_i) ) / (2ε)
# 这叫"中心差分"，比 (L(x+ε)-L(x))/ε 精度高一个量级。
#
# 它非常慢（每个参数都要前向算两次），但是检查反向传播代码对错的金标准。

def loss_fn(W_, X_=X, b_=b):
    Y_ = X_ @ W_ + b_
    return (Y_ ** 2).sum() / 2


def numeric_grad(f, W_, eps=1e-5):
    g = np.zeros_like(W_)
    it = np.nditer(W_, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = W_[idx]
        W_[idx] = orig + eps; fp = f(W_)
        W_[idx] = orig - eps; fm = f(W_)
        W_[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


dW_numeric = numeric_grad(loss_fn, W.copy())
print("dW 解析:\n", dW_analytic)
print("dW 数值:\n", dW_numeric)
print("最大绝对误差:", np.max(np.abs(dW_analytic - dW_numeric)))
print("两者一致？", np.allclose(dW_analytic, dW_numeric, atol=1e-6))

# 注意：eps 不是越小越好。
#   太大 → 差分本身不准
#   太小 → 浮点数减法损失精度（catastrophic cancellation）
#   经验值：1e-5 ~ 1e-4 比较稳。


# ---------------------------------------------------------------------------
section("3) 链式法则：两层 Linear 怎么一起反传")
# ---------------------------------------------------------------------------
# 模型：  H = X @ W1 + b1     (B, d_hidden)
#        Y = H @ W2 + b2     (B, d_out)
#        L = sum(Y^2) / 2
#
# 反向：
#   dY = Y
#   dW2 = H.T @ dY
#   db2 = dY.sum(0)
#   dH  = dY @ W2.T
#   dW1 = X.T @ dH
#   db1 = dH.sum(0)
#   dX  = dH @ W1.T
#
# 模式很清晰：每个 Linear 层都是同一组三个公式，逐层往前套。

d_hidden = 6
W1 = np.random.randn(d_in, d_hidden)
b1 = np.random.randn(d_hidden)
W2 = np.random.randn(d_hidden, d_out)
b2 = np.random.randn(d_out)

H = X @ W1 + b1
Y = H @ W2 + b2
L = (Y ** 2).sum() / 2

dY  = Y
dW2 = H.T @ dY
db2 = dY.sum(0)
dH  = dY @ W2.T
dW1 = X.T @ dH
db1 = dH.sum(0)
dX  = dH @ W1.T

print("两层网络梯度形状一览：")
print(f"  dW1 {dW1.shape}, db1 {db1.shape}, dW2 {dW2.shape}, db2 {db2.shape}, dX {dX.shape}")

# 用数值梯度验证 W1：
def loss_W1(W1_):
    H_ = X @ W1_ + b1
    Y_ = H_ @ W2 + b2
    return (Y_ ** 2).sum() / 2

dW1_num = numeric_grad(loss_W1, W1.copy())
print("两层网络 dW1 数值 vs 解析最大误差 =", np.max(np.abs(dW1 - dW1_num)))


# ---------------------------------------------------------------------------
section("4) softmax 的雅可比：diag(s) - s s.T")
# ---------------------------------------------------------------------------
# 输入 z 是 (n,)，s = softmax(z) 也是 (n,)
# 雅可比 J[i, j] = ∂s_i / ∂z_j = s_i (δ_ij - s_j)
# 写成矩阵：J = diag(s) - s ⊗ s

def softmax_1d(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

z = np.array([1.0, 2.0, 3.0])
s = softmax_1d(z)
J_analytic = np.diag(s) - np.outer(s, s)
print("s =", s)
print("J (解析) =\n", J_analytic)

# 用数值梯度验证：扰动每个 z_j，看 s_i 变化多少
def J_numeric(z, eps=1e-5):
    n = z.size
    J = np.zeros((n, n))
    for j in range(n):
        zp = z.copy(); zp[j] += eps
        zm = z.copy(); zm[j] -= eps
        J[:, j] = (softmax_1d(zp) - softmax_1d(zm)) / (2 * eps)
    return J

print("J (数值) =\n", J_numeric(z))
print("最大误差:", np.max(np.abs(J_analytic - J_numeric(z))))


# ---------------------------------------------------------------------------
section("5) 训练时其实不用整个雅可比：softmax + 交叉熵的捷径")
# ---------------------------------------------------------------------------
# 有一个非常有用的简化结论：
#
#   如果 L = CrossEntropy( softmax(z), y_onehot )
#   那么 dL/dz = softmax(z) - y_onehot
#
# 比单独求 softmax 雅可比再乘交叉熵梯度简单太多——这就是 LLM 训练时为什么
# logits 直接减 one-hot 就能反传。

z = np.array([1.0, 2.0, 3.0])
y_onehot = np.array([0.0, 0.0, 1.0])     # 真实类别是 2
s = softmax_1d(z)
L = -np.sum(y_onehot * np.log(s + 1e-12))
dz_analytic = s - y_onehot
print("L =", L)
print("dz (软max+CE 捷径) =", dz_analytic)

def loss_z(z_):
    s_ = softmax_1d(z_)
    return -np.sum(y_onehot * np.log(s_ + 1e-12))

dz_num = numeric_grad(loss_z, z.copy())
print("dz (数值)         =", dz_num)
print("最大误差:", np.max(np.abs(dz_analytic - dz_num)))


print("\n第 4 课结束。建议接着做 04_matrix_calculus_self_write.py")
print("做完之后回去看 phase1/03_neural_network.py，应该会有'啊原来如此'的感觉。")
