"""
phase0 / 第 1 课：向量、形状与轴（axis / keepdims / broadcasting）

学完本课你能回答：
1. 一个 numpy 数组的"形状"到底是什么？
2. axis=0 / axis=1 / axis=-1 各自指哪一维？
3. keepdims=True 在 softmax 里为什么是必须的？
4. 广播（broadcasting）的两条规则是什么？

跑这个文件：
    python3 01_vectors_and_axes.py
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
section("1) 向量与形状：shape / ndim / 行向量 vs 列向量")
# ---------------------------------------------------------------------------
v = np.array([1, 2, 3])
print("v =", v, "  shape =", v.shape, "  ndim =", v.ndim)

row = v.reshape(1, 3)
col = v.reshape(3, 1)
print("行向量 row.shape =", row.shape)   # (1, 3)
print("列向量 col.shape =", col.shape)   # (3, 1)

# 关键直觉：
#   - 一维数组 (3,) 既不是行也不是列，是"裸向量"
#   - 它在矩阵乘法里会被自动当成行或列，看你乘的另一个东西
#   - 在 LLM 里我们几乎总是用二维以上，避免歧义


# ---------------------------------------------------------------------------
section("2) 点积与范数：dot / @ / norm")
# ---------------------------------------------------------------------------
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print("a · b =", np.dot(a, b))           # 1*4 + 2*5 + 3*6 = 32
print("a @ b =", a @ b)                  # 同上，@ 是矩阵乘法运算符
print("|a|  =", np.linalg.norm(a))       # sqrt(1+4+9)
print("cos(a,b) =", (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# 在 LLM 里：
#   embedding 之间的相似度 = cos(e1, e2) = (e1 @ e2) / (|e1| * |e2|)
#   这也是 RAG 检索最常用的相似度度量。


# ---------------------------------------------------------------------------
section("3) axis 是什么：沿哪一维聚合")
# ---------------------------------------------------------------------------
x = np.array([[1, 2, 3],
              [4, 5, 6]])
print("x.shape =", x.shape)              # (2, 3)

print("sum(x)         =", np.sum(x))            # 标量 21
print("sum(axis=0)    =", np.sum(x, axis=0))    # (3,)  -> [5,7,9]   按列求和
print("sum(axis=1)    =", np.sum(x, axis=1))    # (2,)  -> [6,15]    按行求和
print("sum(axis=-1)   =", np.sum(x, axis=-1))   # 等价 axis=1，"最后一维"

# 记忆口诀：
#   "axis=k 就是把第 k 维压扁掉"
#   x.shape=(2,3)，axis=0 压掉第 0 维 → 剩 (3,)


# ---------------------------------------------------------------------------
section("4) keepdims=True：为广播留一个长度为 1 的占位轴")
# ---------------------------------------------------------------------------
without = np.sum(x, axis=1)                  # (2,)
withkd  = np.sum(x, axis=1, keepdims=True)   # (2, 1)
print("without keepdims:", without.shape)
print("with    keepdims:", withkd.shape)

# 为什么 LLM 里 softmax 一定要加 keepdims？
#   x.shape=(2,3)，row_max.shape=(2,) 时
#   x - row_max 会触发广播错误（或者得到非预期结果）
#   x - row_max(keepdims=True) -> (2,3) - (2,1)，正常广播

scores = np.array([[1.0, 2.0, 3.0],
                   [1.0, 2.0, 3.0]])
row_max = np.max(scores, axis=1, keepdims=True)   # (2, 1)
stable  = np.exp(scores - row_max)
softmax = stable / np.sum(stable, axis=1, keepdims=True)
print("行级 softmax =\n", softmax)
print("每行和 =", softmax.sum(axis=1))   # 全是 1


# ---------------------------------------------------------------------------
section("5) 广播规则：两条就够")
# ---------------------------------------------------------------------------
# 规则 1：从右往左对齐两个 shape，缺失的维自动补 1。
# 规则 2：对应维要么相等，要么其中一个是 1，否则报错。
#
# 例子：
#   (2, 3, 4)  与  (   3, 1)   →  从右对齐  (2, 3, 4) vs (1, 3, 1) → OK，结果 (2, 3, 4)
#   (2, 3, 4)  与  (   2, 4)   →  对齐 (2, 3, 4) vs (1, 2, 4)，第二维 3 vs 2 →  报错

A = np.ones((2, 3, 4))
b1 = np.array([10, 20, 30, 40])         # (4,)        → 自动看作 (1,1,4)
b2 = np.arange(3).reshape(3, 1)          # (3, 1)      → 自动看作 (1,3,1)

print("A + b1 形状 =", (A + b1).shape)   # (2, 3, 4)
print("A + b2 形状 =", (A + b2).shape)   # (2, 3, 4)

# 错误示范（取消注释会报错）：
# bad = np.ones((2, 3, 4)) + np.ones((2, 4))   # (2,4) 对齐成 (1,2,4)，3 vs 2 冲突


# ---------------------------------------------------------------------------
section("6) 在 LLM 里最常见的一个广播：给每个 token 加 bias")
# ---------------------------------------------------------------------------
B, T, d = 2, 4, 5                         # batch=2, seq_len=4, dim=5
X    = np.random.randn(B, T, d)           # (2, 4, 5)
bias = np.random.randn(d)                 # (5,)  ← 每个维度一个偏置

Y = X + bias                              # 自动广播成 (2, 4, 5)
print("X + bias 形状 =", Y.shape)
# 这就是 nn.Linear 的 bias 加法：每个 token、每个 batch 都加同一个 bias 向量。


print("\n第 1 课结束。建议接着做 01_vectors_and_axes_self_write.py")
