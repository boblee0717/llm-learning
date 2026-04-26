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

# 记忆口诀 —— "聚合方向法"：
#   axis=k 是聚合时扫描的方向，扫过的元素被合并成一个数，那一维就消失。
#   axis=0 朝下扫 → 每"列"被合并成 1 个数 → 列还在、行没了 → (3,)
#   axis=1 朝右扫 → 每"行"被合并成 1 个数 → 行还在、列没了 → (2,)


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
print("row_max =", row_max)
broadcast_scores = scores - row_max
print("scores =\n", scores)
print("broadcast_scores =\n", broadcast_scores)
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
#
# ---- 为什么这两条规则是这样设计的？ ----
#
# Q1: 为什么是"从右往左"对齐，而不是从左往右？
#   因为 numpy/张量的约定里，"越靠右的轴 = 越靠近一个数据点的内部结构；
#   越靠左的轴 = 越是 batch / 外层分组"。
#   典型例子（你后面会看一万次）：
#       X    .shape = (batch, seq_len, hidden)
#       bias .shape = (hidden,)
#   你的本意：给每个 token 加同一个 bias。
#   从右对齐 → bias 视作 (1, 1, hidden) → 在 batch、seq 维度上共享 ✅ 正好是你想要的
#   从左对齐 → bias 会被理解成 (hidden, 1, 1) → 完全错了
#   所以"从右对齐"不是随便选的，是配合"右=数据本身、左=批量"的语义。
#
# Q2: 为什么"缺失的维自动补 1"，不是补 0 或别的值？
#   因为 "维度=1" 在 numpy 里有一个明确语义：
#       "这一维上没有变化，需要几份就复制几份"。
#   也就是说，补 1 = "我在这条轴上是共享的"，正好对应"省略 = 默认共享"的直觉。
#   如果补 ≥2 的数，numpy 就得猜你要填什么内容，那就成了黑魔法。
#
# Q3: 为什么"对应维要么相等，要么其中一个是 1"？
#   把 shape 想成两摞盒子，每个轴是一层：
#     - 两边相等 → 一一对应，正常算
#     - 一边是 1 → 那一层只有 1 个盒子，可以"复制 N 份"去配对（这叫 stretching）
#     - 都不是 1 又不相等（比如 3 vs 2）→ numpy 怎么配都不合理，所以报错
#   numpy 的设计哲学："Errors should never pass silently"（错误绝不静默通过）。
#   宁可报错，也不替你猜。
#
# Q4: 广播会不会真的把小数组复制 N 份、占用更多内存？
#   不会。numpy 底层用 stride=0 的小技巧，让同一份数据被反复"读"，
#   不会真去 malloc 大数组。所以广播既快又省内存，这也是它能成为默认行为的前提。
#
# Q5: "两条规则"是不是只够处理 2 维？更高维呢？
#   ⚠️ 注意："两条"指的是规则一共有 2 条，不是 shape 里有 2 个数字。
#   这两条规则对任意维度都成立 —— 1 维 vs 5 维、4 维 vs 2 维都用同一套。
#
#   原因：这两条规则定义的是"每一对维度怎么处理"：
#     规则 1 解决"从哪端开始对" → 答：从右端，左边缺的补 1
#     规则 2 解决"对上之后合不合法" → 答：相等 / 其中之一为 1 / 否则报错
#   有几维就逐位套几次，规则本身不变。
#
#   像加法竖式一样把两个 shape 右端对齐：
#       A.shape = (2, 3, 4, 5, 6)        # 5 维
#       B.shape = (      4, 1, 6)        # 3 维
#       右对齐  → (2, 3, 4, 5, 6)
#               +(1, 1, 4, 1, 6)         # 左边补 1
#       逐位检查：6=6 ✅  5 vs 1 ✅  4=4 ✅  1 vs 3 ✅  1 vs 2 ✅
#       → 结果 (2, 3, 4, 5, 6)
#
#   Transformer 里最常见的高维广播（你后面会写到）：
#       attention_scores: (batch, heads, seq, seq)   # 4 维
#       mask            : (             seq, seq)   # 2 维
#       右对齐后 mask 视作 (1, 1, seq, seq)，在 batch 和 heads 上共享。
#
#   多个数组一起算（A + B + C）也一样：numpy 用同一套规则把所有数组对齐到
#   一个共同的目标 shape，再一起广播。规则不变，只是参与方多了。

A = np.ones((2, 3, 4))
b1 = np.array([10, 20, 30, 40])         # (4,)        → 自动看作 (1,1,4)
b2 = np.arange(3).reshape(3, 1)          # (3, 1)      → 自动看作 (1,3,1)

print("A  形状 =", A.shape, "\nA  =\n", A)
print("b1 形状 =", b1.shape, "  b1 =", b1)
print("b2 形状 =", b2.shape, "\nb2 =\n", b2)

print("A + b1 形状 =", (A + b1).shape, "\nA + b1 =\n", A + b1)   # (2, 3, 4)
print("A + b2 形状 =", (A + b2).shape, "\nA + b2 =\n", A + b2)   # (2, 3, 4)

# 错误示范（取消注释会报错）：
#   (2,4) 从右对齐成 (1,2,4)，与 (2,3,4) 的中间维 3 vs 2 冲突 → 报错
# bad = np.ones((2, 3, 4)) + np.ones((2, 4))

# 正确示范一：把 (2,4) 改成 (3,4)，从右对齐成 (1,3,4)，每一维要么相等要么是 1 ✅
ok1 = np.ones((2, 3, 4)) + np.ones((3, 4))
print("ok1 形状 =", ok1.shape)        # (2, 3, 4)

# 正确示范二：原本就想用 (2,4) 的话，显式 reshape 成 (2,1,4) 表达"在中间维共享"
ok2 = np.ones((2, 3, 4)) + np.ones((2, 4)).reshape(2, 1, 4)
print("ok2 形状 =", ok2.shape)        # (2, 3, 4)



# ---------------------------------------------------------------------------
section("6) 在 LLM 里最常见的一个广播：给每个 token 加 bias")
# ---------------------------------------------------------------------------
np.random.seed(0)                         # 让结果可复现，方便对照
B, T, d = 2, 4, 5                         # batch=2, seq_len=4, dim=5
X    = np.random.randn(B, T, d)           # (2, 4, 5)
bias = np.random.randn(d)                 # (5,)  ← 每个维度一个偏置

Y = X + bias                              # 自动广播成 (2, 4, 5)

print("X.shape    =", X.shape)
print("X =\n", X)
print("bias.shape =", bias.shape, " bias =", bias)
print("Y.shape    =", Y.shape)

# 关键验证：bias 在 batch 和 seq_len 维度上是"共享"的。
# 也就是说 —— 不管哪个 batch、哪个 token，加上的都是同一个 bias 向量。
# 我们直接对比 Y - X 是不是处处等于 bias 就知道了：
diff = Y - X                              # 期望每个 (b, t) 位置都等于 bias
print("\nY - X 在 batch 0、token 0 位置 =", diff[0, 0])
print("Y - X 在 batch 1、token 3 位置 =", diff[1, 3])
print("是否处处等于 bias？", np.allclose(diff, bias))   # True

# 看一个具体位置的"加法过程"：
b_idx, t_idx = 0, 2
print(f"\n手工对照 batch={b_idx}, token={t_idx}：")
print("  X[b,t]    =", X[b_idx, t_idx])
print("  + bias    =", bias)
print("  = Y[b,t]  =", Y[b_idx, t_idx])

# 这就是 nn.Linear 的 bias 加法：每个 token、每个 batch 都加同一个 bias 向量。
# 广播原理：bias (5,) 从右对齐到 X (2,4,5) → 视作 (1,1,5) → 在 batch 和 seq 上共享。


print("\n第 1 课结束。建议接着做 01_vectors_and_axes_self_write.py")
