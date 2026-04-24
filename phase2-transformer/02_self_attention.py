"""
第 2 课：自注意力机制 (Self-Attention)
=====================================
Transformer 的核心 —— 理解注意力到底在算什么

核心问题：
- "我喜欢猫因为它们很可爱" —— "它们"指的是谁？模型怎么知道？
- 为什么叫 Query、Key、Value？
- Scaled Dot-Product 的每一步在干什么？
- 因果掩码为什么重要？

与大模型的关系：
- 自注意力是 Transformer 的心脏
- GPT 生成每个词时都在做自注意力计算
- 注意力的计算复杂度是 O(n²)，这是长文本的瓶颈

前置知识：
- 矩阵乘法（第一阶段 01 课）
- Softmax（第一阶段 01 课）
"""

import numpy as np

np.random.seed(42)


# ============================================================
# Part 1: 注意力的直觉
# ============================================================

print("=" * 60)
print("Part 1: 注意力的直觉 —— 哪些词对当前词重要？")
print("=" * 60)

# 想象你在读这句话：
# "小猫坐在垫子上，它正在打呼噜"
# 当模型处理 "它" 这个词时，需要知道 "它" 指的是 "小猫"
# 注意力机制就是让模型学会：处理每个词时，该关注哪些其他词

# 最简单的注意力：点积（dot product）
sentence = ["小猫", "坐在", "垫子", "上", "它", "打呼噜"]
seq_len = len(sentence)
d_model = 8

# 假设每个词已经有了嵌入向量
embeddings = np.random.randn(seq_len, d_model) * 0.5
# 让 "小猫" 和 "它" 的向量更相似
embeddings[4] = embeddings[0] + np.random.randn(d_model) * 0.1

print("\n简单点积注意力：")
print("计算 '它' 与每个词的点积（相似度）：")
query = embeddings[4]  # "它" 的向量
for i, word in enumerate(sentence):
    score = np.dot(query, embeddings[i])
    print(f"  它 · {word}: {score:+.4f}")


# ============================================================
# Part 2: Scaled Dot-Product Attention
# ============================================================

print("\n" + "=" * 60)
print("Part 2: Scaled Dot-Product Attention")
print("=" * 60)

# 论文公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# 直觉：
# Q (Query): "我在找什么？"
# K (Key):   "我有什么可以提供的？"
# V (Value): "我实际的内容是什么？"
#
# 类比数据库查询：
# Q = SELECT 条件
# K = 索引/键
# V = 实际数据

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# Step 1: 从输入创建 Q, K, V
d_k = 4  # Q 和 K 的维度（必须相同，否则 Q @ K.T 算不出来）
d_v = 3  # V 的维度（可以独立设定！它决定 attention 输出的最后一维）
         # 论文 §3.2.1 中 d_k 和 d_v 就是两个独立符号，
         # 只是 §3.2.2 多头注意力里为了简化才取 d_k = d_v = d_model/h

W_Q = np.random.randn(d_model, d_k) * 0.3
W_K = np.random.randn(d_model, d_k) * 0.3
W_V = np.random.randn(d_model, d_v) * 0.3

Q = embeddings @ W_Q  # (seq_len, d_k)  6 × 4
K = embeddings @ W_K  # (seq_len, d_k)  6 × 4，必须和 Q 对齐
V = embeddings @ W_V  # (seq_len, d_v)  6 × 3，独立维度

print(f"\n输入形状: {embeddings.shape}  (seq_len, d_model)")
print(f"W_Q 形状: {W_Q.shape}  (d_model, d_k)")
print(f"W_K 形状: {W_K.shape}  (d_model, d_k)")
print(f"W_V 形状: {W_V.shape}  (d_model, d_v)  ← 注意 d_v 可以和 d_k 不同")
print(f"Q 形状:   {Q.shape}  (seq_len, d_k)")
print(f"K 形状:   {K.shape}  (seq_len, d_k)")
print(f"V 形状:   {V.shape}  (seq_len, d_v)  ← V 用 d_v")

# Step 2: 计算注意力分数
scores = Q @ K.T  # (seq_len, seq_len)
print(f"\n注意力分数矩阵 (Q @ K^T) 形状: {scores.shape}")
print(f"scores[i][j] = 第 i 个词对第 j 个词的关注程度")

# Step 3: 缩放 (Scale)
# 为什么要除以 sqrt(d_k)？
# 当 d_k 很大时，点积的值会很大，导致 softmax 变得非常尖锐
# 除以 sqrt(d_k) 可以保持梯度稳定
scale = np.sqrt(d_k)
scaled_scores = scores / scale

print(f"\n为什么要缩放？")
print(f"  d_k = {d_k}, sqrt(d_k) = {scale:.2f}")
print(f"  缩放前分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"  缩放后分数范围: [{scaled_scores.min():.4f}, {scaled_scores.max():.4f}]")

# Step 4: Softmax → 注意力权重
attention_weights = softmax(scaled_scores, axis=-1)

print(f"\n注意力权重矩阵（Softmax 后）:")
print(f"  每行之和 = 1（概率分布）")
for i, word in enumerate(sentence):
    weights_str = ", ".join(f"{w:.3f}" for w in attention_weights[i])
    print(f"  {word:4s}: [{weights_str}]")

# Step 5: 加权求和
output = attention_weights @ V  # (seq_len, d_v) ← 输出维度由 d_v 决定，不是 d_k！
print(f"\n输出 = 注意力权重 @ V, 形状: {output.shape}")
print(f"  每个词的新表示 = 它关注的所有词的 V 的加权平均")
print(f"  → 输出的最后一维 = d_v ({d_v})，与 d_k ({d_k}) 解耦")
print(f"  → score 的方差由 d_k 决定（所以除以 sqrt(d_k)），")
print(f"    输出的'宽度'由 d_v 决定，两者职责不同")


# ============================================================
# Part 3: 完整的自注意力函数
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 完整的 Self-Attention 实现")
print("=" * 60)


def self_attention(X, W_Q, W_K, W_V):
    """
    Scaled Dot-Product Self-Attention

    X: (seq_len, d_model)  输入序列
    W_Q, W_K: (d_model, d_k)  必须共享 d_k，否则 Q @ K.T 算不出来
    W_V:      (d_model, d_v)  d_v 可以独立，决定输出维度

    返回:
    - output:  (seq_len, d_v)     注意力输出（最后一维 = d_v）
    - weights: (seq_len, seq_len) 注意力权重（与 d_k / d_v 都无关）
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = weights @ V

    return output, weights


output, weights = self_attention(embeddings, W_Q, W_K, W_V)
print(f"输入形状:  {embeddings.shape}")
print(f"输出形状:  {output.shape}")
print(f"权重形状:  {weights.shape}")


# ============================================================
# Part 4: 因果掩码 (Causal Mask)
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 因果掩码 —— GPT 的关键")
print("=" * 60)

# GPT 在生成文本时，第 i 个词只能看到位置 0 到 i-1 的词
# 不能偷看未来的词！
# 通过 mask 实现：把未来位置的分数设为 -∞，softmax 后变成 0

def causal_self_attention(X, W_Q, W_K, W_V):
    """带因果掩码的自注意力（GPT 使用的版本）"""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    seq_len = X.shape[0]
    # np.triu(np.ones(...), k=1)：构造严格上三角全 1 矩阵（不含主对角线）
    # 1 = 未来位置 = 要屏蔽；0 = 当前及过去位置 = 允许看
    # （这一族 API 在 phase0-math/03_reshape_transpose_split.py §5.5 详细讲过）
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    scores = scores - mask * 1e9  # 未来位置 → -∞（softmax 后接近 0）

    weights = softmax(scores, axis=-1)
    output = weights @ V

    return output, weights


output_causal, weights_causal = causal_self_attention(embeddings, W_Q, W_K, W_V)

print("\n无掩码的注意力权重（BERT 风格）:")
for i, word in enumerate(sentence):
    weights_str = ", ".join(f"{w:.3f}" for w in weights[i])
    print(f"  {word:4s}: [{weights_str}]")

print("\n有因果掩码的注意力权重（GPT 风格）:")
for i, word in enumerate(sentence):
    weights_str = ", ".join(f"{w:.3f}" for w in weights_causal[i])
    print(f"  {word:4s}: [{weights_str}]")

print("\n→ 注意：每行只有当前位置及之前的权重非零！")
print("→ 这就是为什么 GPT 是'自回归'的 —— 只能从左到右生成")


# ============================================================
# Part 5: 注意力的计算复杂度
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 计算复杂度分析")
print("=" * 60)

print("""
Q @ K^T 的复杂度: O(n² · d_k)
  - n = 序列长度
  - d_k = Q/K 的维度

这意味着：
  - 序列长度翻倍 → 计算量翻 4 倍！
  - GPT-3 的上下文长度 2048，注意力矩阵是 2048 × 2048
  - GPT-4 的上下文长度 128K，注意力矩阵是 128K × 128K！

这就是为什么长上下文模型需要各种优化技巧：
  - Flash Attention（优化内存访问）
  - Sparse Attention（只关注部分位置）
  - Sliding Window Attention（只看附近的词）
""")

for seq_len in [128, 512, 2048, 8192, 131072]:
    ops = seq_len * seq_len
    print(f"  序列长度 {seq_len:>6d} → 注意力矩阵大小: {ops:>15,}")


# ============================================================
# 练习
# ============================================================

print("\n" + "=" * 60)
print("练习")
print("=" * 60)
print("""
1. 移除缩放因子（不除以 sqrt(d_k)），对比 softmax 输出的变化
   - 你会发现分布变得更"尖锐"，梯度可能消失

2. 修改 d_k 的大小（比如改成 64），观察注意力权重的变化
   - 更大的 d_k = 更丰富的查询能力

3. 手动构造两个句子 "猫坐在垫子上" 和 "狗坐在垫子上"
   - 让"猫"/"狗"的嵌入相似，观察注意力模式的差异

4. (进阶) 实现交叉注意力 (Cross-Attention)：
   - Q 来自一个序列，K 和 V 来自另一个序列
   - 这是机器翻译中 Encoder-Decoder 注意力的核心

5. 修改 d_v（比如 d_v = 16，d_k 保持 4），观察：
   - output 形状如何变化？(变成 (seq_len, 16))
   - attention_weights 形状是否变化？(不会，仍然 (seq_len, seq_len))
   - 这说明：d_k 决定"匹配空间"，d_v 决定"输出宽度"，两者解耦
""")
