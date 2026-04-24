"""
phase0 / 第 3 课：reshape / transpose / split / concat / outer / Hadamard

学完本课你能回答：
1. reshape 和 transpose 有什么本质区别？
2. 多头注意力里 (B, T, d) 怎么变成 (B, H, T, d_h)？为什么这样切？
3. W_qkv 一次算出 Q/K/V 的拼接，怎么 split 回来？
4. 外积 (outer) 和 Hadamard 积 (逐元素乘) 在哪里用？
5. concat / stack 的区别是什么？

跑这个文件：
    python3 03_reshape_transpose_split.py
"""

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
section("1) reshape：只改'怎么看'，不改'数据顺序'")
# ---------------------------------------------------------------------------
x = np.arange(12)                     # [0, 1, 2, ..., 11]
print("x        =", x, "  shape =", x.shape)

a = x.reshape(3, 4)                    # 按行铺开
b = x.reshape(4, 3)
c = x.reshape(2, 2, 3)
print("reshape(3,4)  =\n", a)
print("reshape(4,3)  =\n", b)
print("reshape(2,2,3)=\n", c)

# 关键直觉：reshape 不会改变内存里数据的顺序，只是换一个"窗口"去看它们。
# 必须满足：新 shape 的元素总数 == 旧 shape 的元素总数（这里都是 12）。
# 用 -1 表示"自动算"：x.reshape(-1, 4) 会自动推出第 0 维是 3。
print("reshape(-1, 4):", x.reshape(-1, 4).shape)


# ---------------------------------------------------------------------------
section("2) transpose / swapaxes：交换轴的顺序")
# ---------------------------------------------------------------------------
m = np.array([[1, 2, 3],
              [4, 5, 6]])              # (2, 3)
print("m       =\n", m)
print("m.T     =\n", m.T)              # (3, 2)
print("m.T.shape =", m.T.shape)

# 高维场景：用 transpose / swapaxes，更安全。
y = np.arange(24).reshape(2, 3, 4)     # (B=2, T=3, d=4)
print("y.shape         =", y.shape)
print("swapaxes(-1,-2) =", np.swapaxes(y, -1, -2).shape)   # (2, 4, 3)
print("transpose(0,2,1)=", y.transpose(0, 2, 1).shape)     # (2, 4, 3)

# reshape vs transpose 的本质区别：
#   - reshape：内存顺序不变，"看"的方式变了
#   - transpose：内存顺序"逻辑上"变了（实际是改了 stride），元素的位置关系变了
# 错误用 reshape 替代 transpose 是初学者最常踩的坑！


# ---------------------------------------------------------------------------
section("3) 多头注意力的核心 reshape：(B,T,d) → (B,H,T,d_h)")
# ---------------------------------------------------------------------------
B, T, d = 2, 4, 12
H = 3                                  # 3 个头
d_h = d // H                           # 每个头 4 维
assert d == H * d_h

X = np.arange(B * T * d).reshape(B, T, d).astype(float)
print("X.shape =", X.shape, "（每个 token 是 12 维）")

# 第 1 步：把最后一维 d=12 切成 (H=3, d_h=4)
step1 = X.reshape(B, T, H, d_h)
print("step1: reshape(B,T,H,d_h) =", step1.shape)

# 第 2 步：把 H 这个轴换到 T 前面，方便之后做 batched matmul
step2 = step1.transpose(0, 2, 1, 3)    # (B, H, T, d_h)
print("step2: transpose(0,2,1,3) =", step2.shape)

# 为什么要 transpose？
#   因为后面要算 Q @ K.T，希望 batched matmul 的"批量维"是 (B, H)，
#   "矩阵维"是 (T, d_h)。这样形状自然变成 (B, H, T, T)，每个头有自己的注意力矩阵。

# 反过来：算完 attention 后要拼回去 (B, T, d)
back = step2.transpose(0, 2, 1, 3).reshape(B, T, d)
print("拼回 (B,T,d):", back.shape, "  与 X 等价? ", np.allclose(back, X))


# ---------------------------------------------------------------------------
section("4) split：W_qkv 一次算出，再切回 Q/K/V")
# ---------------------------------------------------------------------------
# GPT 实现里常见的写法：用一个大矩阵 W_qkv (d, 3d) 一次算出 Q/K/V，再 split。
# 好处：一次大 matmul 比三次小 matmul 在 GPU 上更快。

d = 8
B, T = 2, 4
X = np.random.randn(B, T, d)
W_qkv = np.random.randn(d, 3 * d)      # (8, 24)
QKV = X @ W_qkv                         # (2, 4, 24)
print("QKV.shape =", QKV.shape)

Q, K, V = np.split(QKV, 3, axis=-1)     # 各自 (2, 4, 8)
print("Q,K,V 形状 =", Q.shape, K.shape, V.shape)


# ---------------------------------------------------------------------------
section("5) 外积 (outer) 与 Hadamard 积 (逐元素)")
# ---------------------------------------------------------------------------
u = np.array([1, 2, 3])                 # (3,)
v = np.array([10, 20])                  # (2,)
print("外积 u ⊗ v =\n", np.outer(u, v))   # (3, 2)，u_i * v_j

# Hadamard 积 = 逐元素乘 = numpy 里的 *
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([[10, 10, 10],
              [1, 2, 3]])
print("a * b (Hadamard) =\n", a * b)

# 在 LLM 里用得到的地方：
#   - mask: scores * mask 把不想看的位置乘 0 / 加 -inf
#   - GLU / SwiGLU 等门控结构：x * sigmoid(gate(x))
#   - LayerNorm 里的 gamma 缩放：normalized * gamma 也是 Hadamard 积


# ---------------------------------------------------------------------------
section("5.5) np.triu / np.tril：上下三角矩阵构造")
# ---------------------------------------------------------------------------
# 这一族 API 是构造 attention causal mask 的"基础零件"。
# phase2 第 2 课你会看到这一行：
#     mask = np.triu(np.ones((seq_len, seq_len)), k=1)
# 这里先把 np.triu / np.tril 这俩 API 玩熟，等到了 phase2 一眼就懂。

M = np.arange(1, 17).reshape(4, 4)
print("原矩阵 M =\n", M)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]
#  [13 14 15 16]]

# 1) np.tril(M)：保留下三角（含主对角线），上三角置零
print("\nnp.tril(M)        =\n", np.tril(M))
# [[ 1  0  0  0]
#  [ 5  6  0  0]
#  [ 9 10 11  0]
#  [13 14 15 16]]

# 2) np.triu(M)：保留上三角（含主对角线），下三角置零
print("\nnp.triu(M)        =\n", np.triu(M))
# [[ 1  2  3  4]
#  [ 0  6  7  8]
#  [ 0  0 11 12]
#  [ 0  0  0 16]]

# 3) k 参数：对角线偏移量
#    k=0  → 主对角线
#    k=1  → 主对角线"上方"一条副对角线（不含主对角）
#    k=-1 → 主对角线"下方"一条副对角线
print("\nnp.triu(M, k=1)   =\n", np.triu(M, k=1))   # 严格上三角，主对角也被砍
# [[ 0  2  3  4]
#  [ 0  0  7  8]
#  [ 0  0  0 12]
#  [ 0  0  0  0]]

print("\nnp.tril(M, k=-1)  =\n", np.tril(M, k=-1))  # 严格下三角，主对角也被砍
# [[ 0  0  0  0]
#  [ 5  0  0  0]
#  [ 9 10  0  0]
#  [13 14 15  0]]

# 4) 最常用的"全 1 三角矩阵"写法
n = 4
ones_lower = np.tril(np.ones((n, n)))     # 下三角全 1（含主对角）
ones_upper_strict = np.triu(np.ones((n, n)), k=1)  # 严格上三角全 1（不含主对角）

print("\nnp.tril(np.ones((4,4)))        =\n", ones_lower)
# 这就是"允许位置 i 看位置 j（j ≤ i）"的 causal mask：
#   1 = 允许，0 = 屏蔽

print("\nnp.triu(np.ones((4,4)), k=1)   =\n", ones_upper_strict)
# 这就是 phase2 那一行 mask 的真身：
#   1 = 屏蔽（未来位置），0 = 允许
# 后面用 `scores - mask * 1e9` 把 1 的位置打到 -1e9，softmax 后接近 0

# 5) 互补关系：tril(k=0) + triu(k=1) == 全 1 矩阵
assert np.allclose(ones_lower + ones_upper_strict, np.ones((n, n)))
print("\ntril(ones, k=0) + triu(ones, k=1) == ones  ✓")
# 这意味着两种写法等价：
#   "允许集" mask：np.tril(np.ones((n,n)))
#   "屏蔽集" mask：np.triu(np.ones((n,n)), k=1)
# phase2 代码里用了"屏蔽集"写法（更省一次 1-x），但思路是对偶的。

# 在 LLM 里用得到的地方：
#   - causal mask：GPT decoder 的核心 (np.triu(ones, k=1))
#   - 局部窗口注意力：用 np.tril(ones, k=0) - np.tril(ones, k=-w) 切出对角线一条窄带
#   - 上三角 / 下三角分解（线性代数里的 LU 分解，本课不展开）


# ---------------------------------------------------------------------------
section("6) concat vs stack：合并多个张量")
# ---------------------------------------------------------------------------
a = np.zeros((2, 3))
b = np.ones((2, 3))

print("concat axis=0  ->", np.concatenate([a, b], axis=0).shape)   # (4, 3)
print("concat axis=1  ->", np.concatenate([a, b], axis=1).shape)   # (2, 6)
print("stack  axis=0  ->", np.stack([a, b], axis=0).shape)          # (2, 2, 3) 多一个新轴
print("stack  axis=1  ->", np.stack([a, b], axis=1).shape)          # (2, 2, 3)

# 区别一句话：
#   concat：在已有的某个轴上"接长"，不会增加维度数
#   stack ：新增一个轴把它们"摞起来"，维度数会 +1
#
# 多头注意力的输出拼回去用 reshape 或 concat（沿 d 轴）。
# transformer encoder 的多层残差用 stack 一般是不必要的，那是按时间步累加。


print("\n第 3 课结束。建议接着做 03_reshape_transpose_split_self_write.py")
