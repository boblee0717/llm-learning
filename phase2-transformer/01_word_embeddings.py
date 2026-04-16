"""
第 1 课：词嵌入与位置编码
=========================
理解文本如何变成模型能处理的数字

核心问题：
- 计算机只能处理数字，文字怎么办？
- 为什么 one-hot 编码不够用？
- "国王" 和 "王后" 的关系怎么用数字表达？
- 模型怎么知道词的顺序？

与大模型的关系：
- GPT 的第一层就是 Embedding 层
- GPT-3 的 Embedding 维度是 12288
- 位置编码让 Transformer 知道词的先后顺序
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'Noto Sans CJK SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part 1: 为什么需要词嵌入？
# ============================================================

print("=" * 60)
print("Part 1: 从 One-Hot 到 Word Embedding")
print("=" * 60)

vocab = ["我", "喜欢", "猫", "狗", "爱", "你"]
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}

# One-Hot 编码：每个词是一个稀疏向量
print("\n--- One-Hot 编码 ---")
for word in vocab:
    one_hot = np.zeros(vocab_size)
    one_hot[word_to_idx[word]] = 1
    print(f"  {word}: {one_hot}")

# One-Hot 的问题：
# 1. 维度 = 词汇表大小（GPT 用 50257 个 token → 50257 维！）
# 2. 所有词之间的距离都一样（"猫" 和 "狗" 跟 "猫" 和 "你" 一样远）
# 3. 无法表达语义关系

print("\n计算 One-Hot 之间的余弦相似度：")
cat_vec = np.zeros(vocab_size); cat_vec[word_to_idx["猫"]] = 1
dog_vec = np.zeros(vocab_size); dog_vec[word_to_idx["狗"]] = 1
love_vec = np.zeros(vocab_size); love_vec[word_to_idx["爱"]] = 1

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0

print(f"  猫 vs 狗: {cosine_similarity(cat_vec, dog_vec):.4f}")  # 0，但它们应该相似！
print(f"  猫 vs 爱: {cosine_similarity(cat_vec, love_vec):.4f}")  # 也是 0

print("\n→ One-Hot 无法表达语义关系，所有词都互相正交")


# ============================================================
# Part 2: 词嵌入 (Word Embedding)
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 词嵌入 —— 用密集向量表达语义")
print("=" * 60)

# 词嵌入：把每个词映射到一个低维密集向量
# 在训练中学习，使得语义相似的词向量也相似

embedding_dim = 4  # 实际中 GPT-3 用 12288 维

# 模拟一个 Embedding 矩阵（实际中会通过训练学到）
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.1

print(f"\nEmbedding 矩阵形状: {embedding_matrix.shape}")
print(f"  (词汇表大小 {vocab_size}) × (嵌入维度 {embedding_dim})")

# 查找一个词的嵌入向量 = 用 index 从矩阵中取一行
print("\n--- 查找词的嵌入向量 ---")
for word in vocab:
    idx = word_to_idx[word]
    vec = embedding_matrix[idx]
    print(f"  {word} (idx={idx}): {vec}")

# Embedding 查找本质上等价于 one-hot × embedding_matrix
print("\n--- 验证：one-hot 乘矩阵 = 直接查找 ---")
word = "猫"
idx = word_to_idx[word]
one_hot = np.zeros(vocab_size)
one_hot[idx] = 1
result_matmul = one_hot @ embedding_matrix
result_lookup = embedding_matrix[idx]
print(f"  矩阵乘法结果: {result_matmul}")
print(f"  直接查找结果: {result_lookup}")
print(f"  两者相同: {np.allclose(result_matmul, result_lookup)}")


# ============================================================
# Part 3: 手动设置有语义关系的嵌入
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 语义嵌入的直觉")
print("=" * 60)

# 手动设置一些有意义的嵌入向量来演示
#           [动物性, 情感性, 大小, 亲密度]
semantic_embeddings = {
    "猫":   np.array([0.9, 0.1, -0.3, 0.7]),
    "狗":   np.array([0.9, 0.2, 0.3, 0.8]),
    "喜欢": np.array([0.0, 0.8, 0.0, 0.5]),
    "爱":   np.array([0.0, 0.9, 0.0, 0.9]),
    "我":   np.array([-0.5, 0.0, 0.0, 0.0]),
    "你":   np.array([-0.5, 0.0, 0.0, 0.3]),
}

print("\n语义嵌入的余弦相似度：")
pairs = [("猫", "狗"), ("猫", "爱"), ("喜欢", "爱"), ("我", "你")]
for w1, w2 in pairs:
    sim = cosine_similarity(semantic_embeddings[w1], semantic_embeddings[w2])
    print(f"  {w1} vs {w2}: {sim:.4f}")

print("\n→ 语义相近的词（猫/狗、喜欢/爱）相似度更高！")


# ============================================================
# Part 4: 位置编码 (Positional Encoding)
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 位置编码 —— 让模型知道词的顺序")
print("=" * 60)

# Transformer 没有递归结构（不像 RNN），所以不知道词的先后顺序
# 需要额外加上位置信息
# 原论文使用正弦/余弦位置编码

def sinusoidal_position_encoding(max_len, d_model):
    """
    原始 Transformer 论文的位置编码公式:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]       # (max_len, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )                                                    # (d_model/2,)

    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度用 sin
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度用 cos
    return pe


max_len = 50
d_model = 16
pe = sinusoidal_position_encoding(max_len, d_model)

print(f"\n位置编码矩阵形状: {pe.shape}")
print(f"  (最大序列长度 {max_len}) × (模型维度 {d_model})")

print("\n前 3 个位置的编码（前 8 维）：")
for pos in range(3):
    print(f"  位置 {pos}: {pe[pos, :8]}")

# 可视化位置编码
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
im = ax.imshow(pe, cmap='RdBu', aspect='auto')
ax.set_xlabel('Embedding 维度')
ax.set_ylabel('位置')
ax.set_title('正弦位置编码热力图')
plt.colorbar(im, ax=ax)

ax = axes[1]
for dim in [0, 1, 2, 3, 4, 5]:
    ax.plot(pe[:, dim], label=f'dim {dim}')
ax.set_xlabel('位置')
ax.set_ylabel('编码值')
ax.set_title('不同维度的位置编码曲线')
ax.legend()

plt.tight_layout()
plt.savefig('01_positional_encoding.png', dpi=100, bbox_inches='tight')
print("\n→ 已保存位置编码可视化: 01_positional_encoding.png")


# ============================================================
# Part 5: 完整的嵌入流程
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 完整嵌入流程 Token Embedding + Position Encoding")
print("=" * 60)

# 模拟一句话的嵌入过程
sentence = ["我", "喜欢", "猫"]
d_model_demo = 8

np.random.seed(42)
token_embedding = np.random.randn(vocab_size, d_model_demo) * 0.1
pos_encoding = sinusoidal_position_encoding(10, d_model_demo)

print(f"\n输入句子: {sentence}")
print(f"嵌入维度: {d_model_demo}")

token_indices = [word_to_idx[w] for w in sentence]
print(f"Token 索引: {token_indices}")

token_vecs = token_embedding[token_indices]
print(f"\nToken Embedding (形状 {token_vecs.shape}):")
for i, word in enumerate(sentence):
    print(f"  {word}: {token_vecs[i, :4]}...")

pos_vecs = pos_encoding[:len(sentence)]
print(f"\nPosition Encoding (形状 {pos_vecs.shape}):")
for i in range(len(sentence)):
    print(f"  位置 {i}: {pos_vecs[i, :4]}...")

# 最终嵌入 = Token Embedding + Position Encoding
final_embedding = token_vecs + pos_vecs
print(f"\n最终嵌入 = Token + Position (形状 {final_embedding.shape}):")
for i, word in enumerate(sentence):
    print(f"  {word} @ pos {i}: {final_embedding[i, :4]}...")

print("\n→ 这个矩阵就是 Transformer 第一层的输入！")
print("→ 每行代表一个词，包含了语义信息和位置信息")


# ============================================================
# 练习
# ============================================================

print("\n" + "=" * 60)
print("练习")
print("=" * 60)
print("""
1. 修改 embedding_dim，观察嵌入向量的变化
   - 维度越高，能表达的语义越丰富，但计算量也越大

2. 计算位置编码中，位置 0 和位置 1 的余弦相似度，与位置 0 和位置 49 的对比
   - 你会发现相邻位置的编码更相似 —— 这就是位置信息

3. 思考：为什么用 sin/cos 而不是简单的 [0, 1, 2, ...] 作为位置编码？
   - 提示：考虑数值范围和泛化到训练时没见过的长度

4. (进阶) 实现可学习的位置编码，用一个随机矩阵代替 sin/cos
   - GPT 系列实际上用的是可学习的位置编码
""")
