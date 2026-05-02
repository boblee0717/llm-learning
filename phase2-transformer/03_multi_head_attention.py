"""
第 3 课：多头注意力与残差连接
==============================
为什么多个注意力头比一个好？如何稳定深层网络训练？

核心问题：
- 一个注意力头只能关注一种模式，怎么办？
- 为什么深层网络容易训练失败？残差连接怎么解决？
- LayerNorm 在做什么？为什么很重要？

与大模型的关系：
- GPT-3 有 96 个注意力头，每个头的维度是 128
- 残差连接让训练 96 层的深层网络成为可能
- LayerNorm 是训练稳定性的关键

前置知识：
- 上一课的自注意力机制
"""

import numpy as np

np.random.seed(42)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ============================================================
# Part 1: 为什么需要多头？
# ============================================================

print("=" * 60)
print("Part 1: 单头注意力的局限")
print("=" * 60)

# 考虑句子: "The animal didn't cross the street because it was too tired"
# "it" 指代 "animal"（语义关系）
# "it" 紧挨着 "because"（句法关系）
# 一个注意力头只能学到一种关系模式！
# 多头 = 多个头各自学不同的关系

print("""
单头注意力只能学一种模式，比如：
  - Head A 可能学到: 语义指代关系 ("it" → "animal")
  - Head B 可能学到: 句法结构关系 ("it" → "because")
  - Head C 可能学到: 位置邻近关系 ("it" → "was")

多头注意力 = 同时使用 A + B + C，获得更丰富的表示

注意：上面只是直觉示例。实际每个 head 学到什么完全由数据 + 训练决定，
并不能预设。Anthropic 的可解释性研究发现过真实的 induction heads
（专门做 in-context 复制）等模式，但大多数 head 的功能仍是黑箱。
""")


# ============================================================
# Part 2: 多头注意力实现
# ============================================================

print("=" * 60)
print("Part 2: Multi-Head Attention 实现")
print("=" * 60)

seq_len = 6
d_model = 16
n_heads = 4
d_head = d_model // n_heads  # 每个头的维度（和第 2 课的 d_k 是同一个东西）

print(f"模型维度: {d_model}")
print(f"注意力头数: {n_heads}")
print(f"每头维度: {d_head}")
print(f"验证: {n_heads} x {d_head} = {n_heads * d_head} = d_model (OK)")

X = np.random.randn(seq_len, d_model) * 0.5


def single_head_attention(Q, K, V, mask=None):
    """
    单头注意力（和第 2 课的 self_attention 等价）。

    mask 约定（与第 2 课 02_self_attention.py 完全一致）：
      - 形状: (seq_len, seq_len)
      - 1 表示"屏蔽"（不允许看），0 表示"可见"
      - 因果掩码用 np.triu(np.ones((n, n)), k=1) 生成
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores - mask * 1e9  # 屏蔽位置打成极小值，softmax 后≈0
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    """
    多头注意力

    核心思想：
    1. 把 Q, K, V 分成 n_heads 份
    2. 每份独立做注意力
    3. 拼接结果，再做一次线性变换

    mask 形状约定: (seq_len, seq_len)，所有 head 共用同一份 mask
    （因果约束和 head 无关，每个 head 都不能看未来）
    """
    seq_len, d_model = X.shape
    d_head = d_model // n_heads

    # 线性投影
    Q = X @ W_Q  # (seq_len, d_model)
    K = X @ W_K
    V = X @ W_V

    # 分成多个头: (seq_len, d_model) → (n_heads, seq_len, d_head)
    #
    # 可视化理解（以本例 d_model=16, n_heads=4, d_head=4 为例）:
    #
    #   原来每个 token 是一整条 16 维向量:
    #     token_0: [ q0 q1 q2 q3 | q4 q5 q6 q7 | q8 q9 q10 q11 | q12 q13 q14 q15 ]
    #
    #   reshape(seq_len, n_heads, d_head) 后:
    #     token_0:
    #       head_0: [ q0  q1  q2  q3  ]
    #       head_1: [ q4  q5  q6  q7  ]
    #       head_2: [ q8  q9  q10 q11 ]
    #       head_3: [ q12 q13 q14 q15 ]
    #
    #   transpose(1, 0, 2) 后把 head 维挪到最前面:
    #     head_0: [ token_0 的前 4 维, token_1 的前 4 维, ... ]
    #     head_1: [ token_0 的第 2 段, token_1 的第 2 段, ... ]
    #
    #   transpose 参数的含义:
    #     reshape 后的维度编号是 (0, 1, 2):
    #       0 = seq_len   表示第几个 token
    #       1 = n_heads   表示第几个 head
    #       2 = d_head    表示 head 内部的特征维度
    #
    #     transpose(1, 0, 2) 就是把维度顺序从:
    #       (seq_len, n_heads, d_head)
    #     改成:
    #       (n_heads, seq_len, d_head)
    #
    #     用索引看就是:
    #       before[token_i, head_j, dim_k] -> after[head_j, token_i, dim_k]
    #
    #   所以后面 Q[h], K[h], V[h] 就表示:
    #     第 h 个 head 看到的完整序列，形状是 (seq_len, d_head)
    Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    # 每个头独立做注意力
    all_head_outputs = []
    all_head_weights = []
    for h in range(n_heads):
        head_output, head_weights = single_head_attention(
            Q[h], K[h], V[h], mask
        )
        all_head_outputs.append(head_output)
        all_head_weights.append(head_weights)

    # 拼接所有头: (n_heads, seq_len, d_head) → (seq_len, d_model)
    concat = np.concatenate(all_head_outputs, axis=-1)

    # 最终线性变换
    output = concat @ W_O  # (seq_len, d_model)

    return output, all_head_weights


# 初始化权重
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1
W_O = np.random.randn(d_model, d_model) * 0.1

output, head_weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads)

print(f"\n输入形状:  {X.shape}")
print(f"输出形状:  {output.shape}")
print(f"注意力头数: {len(head_weights)}")

print("\n各头的注意力模式（第一个词关注其他词的权重）:")
for h, weights in enumerate(head_weights):
    weights_str = ", ".join(f"{w:.3f}" for w in weights[0])
    print(f"  Head {h}: [{weights_str}]")
print("→ 不同的头学到不同的注意力分布！")


# ============================================================
# Part 3: 残差连接 (Residual Connection)
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 残差连接 —— 让信息流动起来")
print("=" * 60)

# 残差连接: output = X + SubLayer(X)
# 没有残差连接时，信息必须全部通过注意力层
# 有残差连接后，原始信息可以直接跳到输出

print("""
没有残差连接:
  X → [Attention] → output
  信息必须全部通过 Attention 层，深层网络容易退化

有残差连接:
  X → [Attention] → output + X
  原始信息可以直接"跳过"这一层
  即使 Attention 输出很差，至少还有原始 X 保底
""")

attn_output, _ = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads)
residual_output = X + attn_output

print(f"原始输入 X[0, :4]:           {X[0, :4]}")
print(f"注意力输出 Attn[0, :4]:      {attn_output[0, :4]}")
print(f"残差连接后 (X + Attn)[0, :4]: {residual_output[0, :4]}")

# 残差连接的梯度优势
print("""
梯度角度的理解:
  正常: dL/dX = dL/dF(X) · dF(X)/dX
        → 链式相乘，每层 dF/dX 若 < 1，越深梯度越接近 0（梯度消失）

  残差: dL/dX = dL/d(X+F(X)) · (1 + dF(X)/dX)
                                ^^^
                                这个 +1 提供一条"恒等高速公路"

  精确说法（避免常见误解）：
    "+1" 不是保证梯度 ≥ 1，而是给梯度留了一条不必完全经过子层的直路。
    即使子层本身把梯度变得很小（比如子层还没学会），梯度仍能传到前面去；
    这正是 ResNet 让 100+ 层、Transformer 让 96 层成为可能的根本原因。
""")


# ============================================================
# Part 4: Layer Normalization
# ============================================================

print("=" * 60)
print("Part 4: Layer Normalization —— 稳定训练")
print("=" * 60)


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """
    Layer Normalization

    对每个样本的特征维度（最后一维）做归一化:
    1. 计算均值和方差
    2. 归一化到均值 0、方差 1
    3. 缩放和偏移（可学习参数 gamma, beta；不传则只做归一化）

    形状:
      x:     (..., d_model)
      gamma: (d_model,)  —— 缩放，初始化为 1
      beta:  (d_model,)  —— 偏移，初始化为 0
    """
    # mean/var 都是沿最后一维 d_model 计算的。
    # 如果 x 的形状是 (seq_len, d_model)，就是对每个 token 自己的特征向量
    # 分别计算均值和方差，而不是跨 token 计算。
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # eps 是一个很小的数，用来防止 var=0 时除以 0。
    # 即使 var 非常接近 0，sqrt(var) 也会让分母过小，导致归一化结果爆大；
    # 加上 eps 后分母至少有一个稳定的下限。
    x_norm = (x - mean) / np.sqrt(var + eps)

    if gamma is not None and beta is not None:
        x_norm = gamma * x_norm + beta

    return x_norm


print("\n归一化前后对比:")
x = np.random.randn(3, d_model) * 10 + 5  # 均值偏离 0，方差很大

x_normed = layer_norm(x)

for i in range(3):
    print(f"\n  位置 {i}:")
    print(f"    归一化前: mean={x[i].mean():+.4f}, std={x[i].std():.4f}, "
          f"range=[{x[i].min():.2f}, {x[i].max():.2f}]")
    print(f"    归一化后: mean={x_normed[i].mean():+.4f}, std={x_normed[i].std():.4f}, "
          f"range=[{x_normed[i].min():.2f}, {x_normed[i].max():.2f}]")

# 演示 gamma/beta 的作用：初始化为 (1, 0) 时和不传等价（恒等仿射）
gamma = np.ones(d_model)
beta = np.zeros(d_model)
x_normed_affine = layer_norm(x, gamma=gamma, beta=beta)
print(f"\n传入 gamma=1, beta=0 的结果与不传等价: "
      f"{np.allclose(x_normed, x_normed_affine)}")
print("→ 训练中 gamma/beta 会被学习，让模型自己决定每一维的缩放和偏移")

print("""
为什么需要 LayerNorm？
  - 不同层的输出范围可能差异很大
  - 归一化后数值稳定，训练更容易收敛
  - Batch Norm 在序列任务中效果差（不同序列长度不同）
  - Layer Norm 对每个样本独立归一化，不依赖 batch
""")


# ============================================================
# Part 5: Pre-Norm vs Post-Norm
# ============================================================

print("=" * 60)
print("Part 5: Pre-Norm vs Post-Norm")
print("=" * 60)


def post_norm_block(x, W_Q, W_K, W_V, W_O, n_heads):
    """Post-Norm: 原始 Transformer 论文的方式"""
    # X → Attention → Add → LayerNorm
    attn_out, _ = multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads)
    return layer_norm(x + attn_out)


def pre_norm_block(x, W_Q, W_K, W_V, W_O, n_heads):
    """Pre-Norm: GPT-2/3 使用的方式"""
    # X → LayerNorm → Attention → Add
    x_norm = layer_norm(x)
    attn_out, _ = multi_head_attention(x_norm, W_Q, W_K, W_V, W_O, n_heads)
    return x + attn_out


post_out = post_norm_block(X, W_Q, W_K, W_V, W_O, n_heads)
pre_out = pre_norm_block(X, W_Q, W_K, W_V, W_O, n_heads)

print(f"\nPost-Norm 输出 [0, :4]: {post_out[0, :4]}")
print(f"Pre-Norm 输出 [0, :4]:  {pre_out[0, :4]}")

print("""
区别：
  Post-Norm (原始论文): LayerNorm(X + Attention(X))
  Pre-Norm  (GPT-2/3):  X + Attention(LayerNorm(X))

Pre-Norm 的优势：
  - 残差通路上没有 LayerNorm，从输出一路直通到输入 → 梯度高速公路
  - 训练更稳定，不需要复杂的 learning rate warmup
  - 大模型（GPT-2/3、LLaMA、Gemma）几乎都用 Pre-Norm

Pre-Norm 的代价（重要！第 5 课会遇到）：
  - 残差不断累加而不被归一化 → 输出方差随层数指数增长
  - 因此工业实现会在所有 block 之后再补一个 final LayerNorm
    （nanoGPT 里的 `ln_f`，GPT-2 论文 2.3 节明确提到）
  - 公式: out = LayerNorm_final( X_L )，其中 X_L 是最后一层的输出
""")


# ============================================================
# 练习
# ============================================================

print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 修改 n_heads（比如 1, 2, 8, 16），观察注意力模式的变化
   - 更多头 = 更丰富的关注模式，但每头维度更小（d_model 总量不变）
   - 注意 d_model 必须能被 n_heads 整除，否则 reshape 会报错

2. 去掉残差连接，堆叠 10 层多头注意力，观察输出值的变化
   - 你会发现值会爆炸或趋近于 0
   - 加回残差再跑一次，对比稳定性

3. 对比 LayerNorm 前后的梯度大小
   - LayerNorm 帮助梯度保持在合理范围

4. (进阶) 实现带因果掩码的多头注意力 —— 完整提示：
   - 用第 2 课的方式生成 mask:
       mask = np.triu(np.ones((seq_len, seq_len)), k=1)
   - 调用时传进去:
       output, _ = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=mask)
   - 验证: 第 0 个位置的 attention weights 应该只在自己身上为 1，其他全 0
""")
