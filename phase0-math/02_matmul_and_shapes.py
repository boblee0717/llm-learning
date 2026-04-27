"""
phase0 / 第 2 课：矩阵乘法与形状推断

学完本课你能回答：
1. (m, k) @ (k, n) 为什么必须 k 相等？结果是什么形状？
2. "矩阵乘向量"和"向量乘矩阵"在视角上有什么区别？
3. (B, T, d_in) @ (d_in, d_out) 为什么会自动广播？
4. attention 里 Q @ K.T 的 K.T 转置的是哪两个轴？
5. einsum 怎么把 batched matmul 写成一行？

跑这个文件：
    python3 02_matmul_and_shapes.py
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
section("1) 矩阵乘法的形状规则：(m,k) @ (k,n) = (m,n)")
# ---------------------------------------------------------------------------
A = np.array([[1, 2, 3],
              [4, 5, 6]])           # (2, 3)
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])               # (3, 2)

C = A @ B                            # 内层 3 == 3，OK，结果 (2, 2)
print("A.shape =", A.shape, "  B.shape =", B.shape, "  C.shape =", C.shape)
print("C =\n", C)

# 形状口诀：
#   (m, [k]) @ ([k], n)  →  把方括号里的"内层维"消掉，剩下外层 (m, n)
# 这就是为什么 attention 里 K 必须转置：
#   Q.shape  = (T, d_k)
#   K.shape  = (T, d_k)
#   Q @ K   形状不匹配（d_k vs T），必须转置成 (d_k, T) 才能消掉


# ---------------------------------------------------------------------------
section("2) 两种解读：'行视角'与'列视角'")
# ---------------------------------------------------------------------------
# 解读 1（行视角）：C 的第 i 行 = A 的第 i 行 与 B 的每一列做点积
#   适合理解 attention：每个 query 行 与 所有 key 列 做点积 → 一行得分
#
# 解读 2（列视角）：C 的第 j 列 = B 的第 j 列在 A 的"列空间"里的线性组合
#   适合理解 nn.Linear：权重矩阵把输入向量"投影"到新空间

a_row0 = A[0]                # 行向量 (3,) = [1,2,3]
print("A 第 0 行          =", a_row0)
print("A[0] @ B[:, 0]    =", a_row0 @ B[:, 0])   # → C[0,0]
print("A[0] @ B[:, 1]    =", a_row0 @ B[:, 1])   # → C[0,1]
print("和 C[0] 对比     =", C[0])


# ---------------------------------------------------------------------------
section("3) Linear 层视角：(B, T, d_in) @ (d_in, d_out)")
# ---------------------------------------------------------------------------
# 这是 LLM 里最高频的一个形状：把每个 token 的向量从 d_in 维投到 d_out 维。
B_, T, d_in, d_out = 2, 4, 8, 5

X = np.random.randn(B_, T, d_in)        # (2, 4, 8)
W = np.random.randn(d_in, d_out)        # (8, 5)
Y = X @ W                                # (2, 4, 5)
print("X.shape =", X.shape, "  W.shape =", W.shape, "  Y.shape =", Y.shape)

# 注意：这里 W 没有 batch 维，但 numpy 的 @ 会自动广播：
#   把 (2, 4, 8) 看成"批量内每个 (4, 8) 都和 (8, 5) 相乘"，得到 (2, 4, 5)
# 这就是 nn.Linear(d_in, d_out) 在 (B, T, d_in) 输入上的行为。
#
# ---- 高维 @ 怎么读？(看完这 4 个视角再也不用查文档) ----
#
# 视角 1（for 循环）：高维 @ 等价于"只对左侧最左那一维循环"
#       Y[b] = X[b] @ W      # (4,8) @ (8,5) → (4,5)
#   W 在每次循环都被复用，所以它没有 batch 维 —— 它是"一份共享权重"。
#
# 视角 2（机器 + 数据流）：W 是一台不变的机器，X 是 B*T 个 token 排队进去
#   把 X (2,4,8) 看成 8 个 token，每个是 8 维；它们一个个被同一个 W 投到 5 维。
#   等价写法（PyTorch 里很常见）：
#       flat = X.reshape(B*T, d_in)      # (8, 8) 排成一队
#       out  = flat @ W                  # (8, 5) 普通 2D matmul
#       Y    = out.reshape(B, T, d_out)  # (2, 4, 5) 还原回去
#   说明 batch 维和 seq 维对 W 来说没区别，都只是"有多少个 token"。
#
# 视角 3（前缀 + 矩阵）：@ 把最右两维当矩阵，前面的都是 batch 前缀
#       X (2, 4, 8)  → 前缀 (2,)，矩阵 (4, 8)
#       W (   8, 5)  → 前缀 ()  ，矩阵 (8, 5)
#   规则：
#       1) 前缀按广播对齐  → (2,) vs () → 共同前缀 (2,)
#       2) 矩阵部分普通乘  → (4,8) @ (8,5) → (4,5)
#       3) 拼起来           → (2,) + (4,5) = (2, 4, 5)
#   这个套路在 4D / 5D 里同样适用，比如 attention 的 Q @ K.T：
#       (B, H, T, d_h) @ (B, H, d_h, T) → 前缀 (B,H) 相等，矩阵 (T,d_h)@(d_h,T)
#       结果 (B, H, T, T)
#
# 视角 4（盒子比喻）：
#       X 是 2 块 (4×8) 的木板叠在一起
#       W 是一块 (8×5) 的"滤镜"
#       每块板都跟同一块滤镜做矩阵乘，得到 2 块 (4×5) 新板 → (2, 4, 5)


# ---- 用代码亲眼验证上面 4 个视角 ----
#
# 关于 np.allclose(a, b)：
#   逐元素判断 a 和 b "在容差内是否相等"，全部满足才返回 True。
#   等价于（伪码）：(|a - b| <= atol + rtol * |b|).all()
#   默认 rtol=1e-5, atol=1e-8。
#   为什么不直接用 == ？因为浮点运算几乎总有 1e-16 量级的误差，
#   矩阵乘法/reshape 的两条等价路径在数值上不会完全相同，但应"几乎相等"。
#   这是 numpy 里做"两种写法是否等价"验证的标准工具，后面到处会用。

# 验证 1：batch 之间互不干扰（视角 1 的 for 循环）
print("\n[验证] Y[0] ≈ X[0] @ W ?", np.allclose(Y[0], X[0] @ W))   # True
print("[验证] Y[1] ≈ X[1] @ W ?", np.allclose(Y[1], X[1] @ W))     # True

# 验证 2：压平 → 2D matmul → 恢复，结果一致（视角 2 的"排队进机器"）
flat  = X.reshape(B_ * T, d_in)              # (8, 8)
Y_alt = (flat @ W).reshape(B_, T, d_out)     # (2, 4, 5)
print("[验证] 压平再 reshape 回去结果一致 ?", np.allclose(Y, Y_alt))   # True

# 验证 3：单个 token 手算一遍（视角 4 的"每个 token 单独被滤镜处理"）
b, t = 1, 2
print(f"[验证] Y[{b},{t}] ≈ X[{b},{t}] @ W ?",
      np.allclose(Y[b, t], X[b, t] @ W))                              # True


# ---------------------------------------------------------------------------
section("4) Batched matmul：(B, n, k) @ (B, k, m) = (B, n, m)")
# ---------------------------------------------------------------------------
# 真正的"逐 batch 矩阵乘"——常见于 attention 里 Q @ K.T。
B_, H, T, d_h = 2, 3, 4, 5

Q = np.random.randn(B_, H, T, d_h)            # (2, 3, 4, 5)
K = np.random.randn(B_, H, T, d_h)            # (2, 3, 4, 5)

# 想算 attention 分数 = Q @ K.T，其中 K.T 是把"最后两维"转置：
KT = np.swapaxes(K, -1, -2)                   # (2, 3, 5, 4)
scores = Q @ KT                                # (2, 3, 4, 4)  ← 每个头一个 4x4 注意力矩阵
print("Q.shape   =", Q.shape)
print("K.T.shape =", KT.shape)
print("scores.shape =", scores.shape, "  ← (B, H, T_q, T_k)")

# 关键：转置只动最后两维，前面的 (B, H) 全部保留。
# 错误示范：K.T 会把所有维全反过来，结果是 (5, 4, 3, 2)，几乎肯定不是你想要的：
print("K.T 全反维 (错): ", K.T.shape)


# ---------------------------------------------------------------------------
section("5) einsum：把上面这些一行写完")
# ---------------------------------------------------------------------------
# einsum 的口诀：
#   "把出现在两边的字母做点积消掉，把只出现在一边的字母保留"
#
# 例子：
#   矩阵乘  C = A @ B：  'ik,kj->ij'
#   batched 矩阵乘 :     'bik,bkj->bij'
#   attention 分数 :     'bhtd,bhsd->bhts'    ← 不用先转置 K 也能算！

A2 = np.random.randn(2, 3)
B2 = np.random.randn(3, 4)
print("einsum 矩阵乘    :", np.einsum("ik,kj->ij", A2, B2).shape)        # (2, 4)
print("einsum 批量矩阵乘:", np.einsum("bik,bkj->bij", np.random.randn(5,2,3), np.random.randn(5,3,4)).shape)
print("einsum 注意力分数:", np.einsum("bhtd,bhsd->bhts", Q, K).shape)    # (2, 3, 4, 4)

# 三种写法等价，但 einsum 的好处是：
#   - 形状自己看得见（每个字母就是一个维度）
#   - 不需要手动 transpose / swapaxes
#   - 在多头注意力里大大降低出错率


# ---------------------------------------------------------------------------
section("6) 形状推断小测验（看输出对不对得上）")
# ---------------------------------------------------------------------------
checks = [
    ("(2,3) @ (3,4)",        np.random.randn(2,3) @ np.random.randn(3,4)),
    ("(B=4,T=10,d=8) @ (8,16)", np.random.randn(4,10,8) @ np.random.randn(8,16)),
    ("(B=4,H=2,T=10,d_h=8) @ (B=4,H=2,8,16)",
        np.random.randn(4,2,10,8) @ np.random.randn(4,2,8,16)),
]
for name, val in checks:
    print(f"{name:55s} -> {val.shape}")


print("\n第 2 课结束。建议接着做 02_matmul_and_shapes_self_write.py")
