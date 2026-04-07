"""
======================================================
第 1 课：NumPy 基础 —— 大模型的数据基石
======================================================

为什么学 NumPy？
- 所有深度学习框架（PyTorch、TensorFlow）底层都是张量运算
- 张量就是多维数组，NumPy 是理解张量运算的最佳起点
- 掌握 NumPy 后，切换到 PyTorch 几乎零成本

学习目标：
1. 理解向量、矩阵、张量的概念
2. 掌握广播机制（broadcasting）
3. 理解矩阵乘法 —— Transformer 的核心运算

运行方式：python3 01_numpy_basics.py
"""

import numpy as np

# ============================================================
# 第一部分：从标量到张量
# ============================================================

print("=" * 60)
print("第一部分：从标量到张量")
print("=" * 60)

scalar = np.array(42)          # 0维张量：标量（一个数）
vector = np.array([1, 2, 3])   # 1维张量：向量（一行数）
matrix = np.array([            # 2维张量：矩阵（表格）
    [1, 2, 3],
    [4, 5, 6]
])
tensor_3d = np.random.randn(2, 3, 4)  # 3维张量

print(f"标量        shape: {scalar.shape}     —— 一个数字")
print(f"向量        shape: {vector.shape}      —— 类比：一句话中每个词的编号")
print(f"矩阵        shape: {matrix.shape}    —— 类比：一句话中每个词的向量表示")
print(f"3维张量     shape: {tensor_3d.shape}  —— 类比：一批句子的向量表示")
print()

# 在大模型中的对应：
# - 一个 token 的 embedding → 向量，如 shape (768,)
# - 一句话的所有 token embedding → 矩阵，如 shape (512, 768)
# - 一个 batch 的所有句子 → 3维张量，如 shape (32, 512, 768)
#   其中 32=batch_size, 512=seq_len, 768=hidden_dim

print("【大模型类比】")
batch_size, seq_len, hidden_dim = 2, 5, 4
fake_embeddings = np.random.randn(batch_size, seq_len, hidden_dim)
print(f"模拟 embedding: shape {fake_embeddings.shape}")
print(f"  batch_size={batch_size} (几句话)")
print(f"  seq_len={seq_len} (每句话几个token)")
print(f"  hidden_dim={hidden_dim} (每个token用几维向量表示)")
print(f"\n第一句话、第一个token的向量:\n  {fake_embeddings[0, 0, :]}")
print()


# ============================================================
# 第二部分：核心运算
# ============================================================

print("=" * 60)
print("第二部分：核心运算")
print("=" * 60)

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# 逐元素运算
print(f"逐元素加法: {a} + {b} = {a + b}")
print(f"逐元素乘法: {a} * {b} = {a * b}")

# 点积（dot product）—— 衡量两个向量的"相似度"
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"\n点积: np.dot({a}, {b}) = {dot}")
print("→ 在 Attention 机制中，Query 和 Key 做的就是点积，得分越高越相关")

# 矩阵乘法 —— Transformer 中无处不在
W = np.array([       # 模拟一个权重矩阵 (3, 2)
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])
x = np.array([1.0, 2.0, 3.0])  # 输入向量 (3,)

output = x @ W  # (3,) @ (3, 2) → (2,)
print(f"\n矩阵乘法: x @ W")
print(f"  输入 x: shape {x.shape}")
print(f"  权重 W: shape {W.shape}")
print(f"  输出:   shape {output.shape}, 值 = {output}")
print("→ 这就是神经网络中'线性层'做的事：y = xW + b")
print()


# ============================================================
# 第三部分：广播机制（Broadcasting）
# ============================================================

print("=" * 60)
print("第三部分：广播机制")
print("=" * 60)

# 广播让不同 shape 的数组也能做运算
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])  # shape (2, 3)
bias = np.array([10, 20, 30])    # shape (3,)

result = matrix + bias  # bias 自动"广播"到每一行
print(f"矩阵:\n{matrix}")
print(f"偏置: {bias}")
print(f"矩阵 + 偏置 (广播):\n{result}")
print("→ 神经网络中加 bias 就是这个操作")
print()


# ============================================================
# 第四部分：Softmax —— Attention 的关键函数
# ============================================================

print("=" * 60)
print("第四部分：实现 Softmax")
print("=" * 60)

def softmax(x):
    """将任意实数向量转换为概率分布（所有值在0~1之间，总和为1）"""
    exp_x = np.exp(x - np.max(x))  # 减去最大值防止数值溢出
    return exp_x / np.sum(exp_x)

logits = np.array([2.0, 1.0, 0.5, -1.0])
probs = softmax(logits)

print(f"原始分数 (logits): {logits}")
print(f"Softmax 后 (概率): {np.round(probs, 4)}")
print(f"概率总和: {np.sum(probs):.6f}")
print()
print("→ 在 Attention 中：logits 是 Q·K 的点积分数")
print("  Softmax 后得到注意力权重（概率），表示该关注哪些位置")
print()


# ============================================================
# 第五部分：动手练习
# ============================================================

print("=" * 60)
print("动手练习")
print("=" * 60)
print("""
练习 1：余弦相似度
  给定两个向量，计算它们的余弦相似度：
  cos_sim = dot(a, b) / (norm(a) * norm(b))
  提示：np.linalg.norm() 计算向量的模

练习 2：矩阵批量运算
  创建一个 shape 为 (3, 4) 的随机矩阵（模拟3个token，4维embedding）
  创建一个 shape 为 (4, 2) 的权重矩阵
  做矩阵乘法，观察输出 shape

练习 3：手动实现一个简单的 Attention 分数计算
  query = np.array([1.0, 0.5])
  keys  = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
  计算 query 与每个 key 的点积，然后做 softmax

下面是参考答案（先自己试试！）：
""")

# ---- 练习 1 参考答案 ----
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([2.0, 4.0, 6.0])
v3 = np.array([-1.0, -2.0, -3.0])

cos_sim_12 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
cos_sim_13 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))

print(f"练习1 - 余弦相似度:")
print(f"  v1 和 v2 (同方向): {cos_sim_12:.4f}  → 完全相似")
print(f"  v1 和 v3 (反方向): {cos_sim_13:.4f}  → 完全相反")
print()

# ---- 练习 2 参考答案 ----
tokens = np.random.randn(3, 4)
weights = np.random.randn(4, 2)
output = tokens @ weights
print(f"练习2 - 矩阵乘法:")
print(f"  tokens:  {tokens.shape}")
print(f"  weights: {weights.shape}")
print(f"  output:  {output.shape}  → 3个token，每个从4维映射到2维")
print()

# ---- 练习 3 参考答案 ----
query = np.array([1.0, 0.5])
keys = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

scores = keys @ query  # 每个 key 与 query 做点积
attn_weights = softmax(scores)

print(f"练习3 - 简单 Attention:")
print(f"  Query: {query}")
print(f"  Keys:\n{keys}")
print(f"  点积分数: {scores}")
print(f"  注意力权重: {np.round(attn_weights, 4)}")
print(f"  → 第3个 key [1,1] 与 query 最相似，所以权重最高")
print()

print("=" * 60)
print("恭喜完成第 1 课！")
print("下一课我们将从零实现梯度下降，理解模型如何'学习'")
print("=" * 60)
