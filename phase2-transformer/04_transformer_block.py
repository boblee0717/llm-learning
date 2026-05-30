"""
第 4 课：完整的 Transformer Block
=================================
把所有组件拼在一起 —— 注意力 + FFN + 残差 + LayerNorm

核心问题：
- 一个 Transformer Block 内部长什么样？
- Feed-Forward Network 的作用是什么？
- 为什么 FFN 的隐藏层要比模型维度大 4 倍？
- 堆叠多个 Block 会怎样？

与大模型的关系：
- GPT-3 = 96 个 Transformer Block 堆叠
- 每个 Block 结构完全相同，只是参数不同
- FFN 的参数量占了模型的 2/3

前置知识：
- 多头注意力（第 3 课）
- 残差连接和 LayerNorm（第 3 课）
"""

import os

import numpy as np

np.random.seed(42)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None and beta is not None:
        x_norm = gamma * x_norm + beta
    return x_norm


def gelu(x):
    r"""
    GELU 激活函数 —— GPT 使用的激活函数
    比 ReLU 更平滑，在 0 附近有非零梯度

    精确定义（标准正态 CDF Φ）：
        GELU(x) = x · Φ(x) = x · 0.5 · [1 + erf(x / √2)]

    下面用的是 tanh 多项式近似（erf 计算慢，GPT-2 等用此近似，误差 < 0.001）：
        GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ============================================================
# Part 1: Feed-Forward Network (FFN)
# ============================================================

print("=" * 60)
print("Part 1: Feed-Forward Network")
print("=" * 60)

# FFN 是一个简单的两层 MLP：
# FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
#
# 关键：中间层维度 d_ff = 4 × d_model
# 为什么？FFN 是模型"记忆知识"的主要地方

d_model = 16
d_ff = d_model * 4  # 通常是 4 倍

print(f"模型维度 d_model: {d_model}")
print(f"FFN 隐藏维度 d_ff: {d_ff} (= 4 × d_model)")
print(f"\nFFN 参数量: {d_model * d_ff + d_ff + d_ff * d_model + d_model}")
print(f"  W1: {d_model} × {d_ff} = {d_model * d_ff}")
print(f"  b1: {d_ff}")
print(f"  W2: {d_ff} × {d_model} = {d_ff * d_model}")
print(f"  b2: {d_model}")


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = gelu(x @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        return output


# 测试 FFN
seq_len = 6
x = np.random.randn(seq_len, d_model)
ffn = FeedForward(d_model, d_ff)
ffn_out = ffn.forward(x)

print(f"\nFFN 输入形状:  {x.shape}")
print(f"FFN 输出形状:  {ffn_out.shape}")
print(f"中间隐藏层形状: ({seq_len}, {d_ff})")

# GELU vs ReLU 对比
print("\n--- GELU vs ReLU ---")
test_x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
relu = np.maximum(0, test_x)
gelu_out = gelu(test_x)
for i in range(len(test_x)):
    print(f"  x={test_x[i]:+.1f}  ReLU={relu[i]:.4f}  GELU={gelu_out[i]:.4f}")
print("→ GELU 在负数区域不是完全为 0，保留了一些信息")


def plot_relu_vs_gelu(save_path="relu_vs_gelu.png"):
    """
    画一张 ReLU vs GELU 的对比图（函数值 + 梯度），帮助直观理解二者区别。
    左图：负数区 ReLU 直接砍成 0，GELU 保留少量负值（平滑过渡）。
    右图：ReLU 梯度在 x=0 处 0→1 突变，GELU 梯度处处连续。
    需要 matplotlib；未安装时静默跳过，不影响主流程。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面环境也能保存图片
        import matplotlib.pyplot as plt
    except ImportError:
        print("（未安装 matplotlib，跳过画图。可执行 pip install matplotlib 后重试）")
        return

    matplotlib.rcParams["axes.unicode_minus"] = False

    xs = np.linspace(-4, 4, 400)
    relu_y = np.maximum(0, xs)
    gelu_y = gelu(xs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：函数值
    ax = axes[0]
    ax.plot(xs, relu_y, label="ReLU", color="#d62728", lw=2.5)
    ax.plot(xs, gelu_y, label="GELU", color="#1f77b4", lw=2.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.fill_between(xs, relu_y, gelu_y, where=(xs < 0), color="#1f77b4", alpha=0.15)
    ax.set_title("ReLU vs GELU (output value)", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # 右图：导数 / 梯度（GELU 用数值微分近似）
    relu_grad = (xs > 0).astype(float)
    eps = 1e-4
    gelu_grad = (gelu(xs + eps) - gelu(xs - eps)) / (2 * eps)

    ax = axes[1]
    ax.plot(xs, relu_grad, label="ReLU' (gradient)", color="#d62728", lw=2.5)
    ax.plot(xs, gelu_grad, label="GELU' (gradient)", color="#1f77b4", lw=2.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("Gradient: ReLU vs GELU", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("f'(x)")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"→ 已保存 ReLU vs GELU 对比图: {save_path}")


# 在脚本所在目录生成对比图，方便配合 README 查看
_here = os.path.dirname(os.path.abspath(__file__))
plot_relu_vs_gelu(os.path.join(_here, "relu_vs_gelu.png"))


# ============================================================
# 辅助：单头/多头注意力（复用上一课，本段不打印，仅定义类）
# ============================================================

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 形状是 (d_model, d_model) 而不是 (d_model, d_k)：
        # 输入输出都是 d_model 维，这一个大矩阵把 n_heads 个头的投影打包在一起算，
        # 算完再 reshape 切成各个头。所以单看维度像"没分头"，
        # 其实分头是在后面 forward() 里的 reshape 那一步做的。
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x, mask=None):
        seq_len = x.shape[0]
        d_k = self.d_k

        Q = (x @ self.W_Q).reshape(seq_len, self.n_heads, d_k).transpose(1, 0, 2)
        K = (x @ self.W_K).reshape(seq_len, self.n_heads, d_k).transpose(1, 0, 2)
        V = (x @ self.W_V).reshape(seq_len, self.n_heads, d_k).transpose(1, 0, 2)

        # K.transpose(0, 2, 1): 保留"头"这一维，只把每个头内部的
        #   (seq_len, d_k) 转成 (d_k, seq_len)，相当于在每个头里做 K^T。
        # @ 作用在 >2 维数组时是"批量矩阵乘法"：最后两维做矩阵乘法，
        #   前面的"头"维当作 batch，对所有头并行各算各的（省掉 for 循环）。
        #   Q (n_heads, seq_len, d_k) @ Kᵀ (n_heads, d_k, seq_len)
        #     -> scores (n_heads, seq_len, seq_len)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

        if mask is not None:
            scores = scores - mask * 1e9

        weights = softmax(scores, axis=-1)
        head_outputs = weights @ V  # (n_heads, seq_len, d_k)

        concat = head_outputs.transpose(1, 0, 2).reshape(seq_len, -1)
        return concat @ self.W_O


# ============================================================
# Part 2: 完整的 Transformer Block
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 完整的 Transformer Block")
print("=" * 60)


class TransformerBlock:
    """
    一个完整的 Transformer Block (Pre-Norm 风格):

    x → LayerNorm → MultiHeadAttention → + (残差) → LayerNorm → FFN → + (残差) → output
    |_____________________________________|         |_____________________|
                  残差连接                                残差连接

    残差连接（Residual Connection）怎么理解：
      一句话：输出 = 输入 + 子层的变换，即代码里的 `x = x + 子层(x)`。
      类比：老师改作文时不重写，只在你原文上"打补丁"（标注修改意见），
            最终 = 原文 + 修改。哪怕这次没什么可改的（子层输出≈0），原文也原样保留，绝不变差。
      作用：
        1. 信息不丢失：每层只在原基础上微调，原始信息有一条"直通车"一路传到最后。
        2. 梯度修高速公路（最重要）：求导时这个 `x` 贡献一个 +1，给梯度留了一条
           不必经过子层的直路；即使子层还没学会东西，梯度也能稳稳传回前面，
           避免深层网络梯度消失。
      前提：x 和子层输出形状必须完全一样（都是 (seq_len, d_model)）才能逐元素相加，
            这也是为什么全程要保证输入输出同形状。
    """
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.d_model = d_model

    def forward(self, x, mask=None):
        # Sub-layer 1: LayerNorm → Attention → Residual
        x_norm = layer_norm(x)
        attn_out = self.attn.forward(x_norm, mask)
        x = x + attn_out  # 残差：原信息保底 + 给梯度留直路（详见类 docstring）

        # Sub-layer 2: LayerNorm → FFN → Residual
        x_norm = layer_norm(x)
        ffn_out = self.ffn.forward(x_norm)
        x = x + ffn_out  # 残差：同上，子层只做"打补丁"式的微调

        return x


# 测试单个 Block
n_heads = 4
d_ff = d_model * 4
block = TransformerBlock(d_model, n_heads, d_ff)

x = np.random.randn(seq_len, d_model) * 0.5
causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)

output = block.forward(x, mask=causal_mask)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"→ 输入和输出形状完全一样！所以可以堆叠任意多层")

# 参数量统计
attn_params = 4 * d_model * d_model  # W_Q, W_K, W_V, W_O
ffn_params = 2 * d_model * d_ff + d_ff + d_model  # W1, W2, b1, b2
total = attn_params + ffn_params

print(f"\n--- 单个 Block 的参数量 ---")
print(f"  注意力层: {attn_params:,} ({attn_params/total*100:.0f}%)")
print(f"  FFN 层:   {ffn_params:,} ({ffn_params/total*100:.0f}%)")
print(f"  总计:     {total:,}")
print(f"→ FFN 占了大部分参数！FFN 是模型'存储知识'的地方")


# ============================================================
# Part 3: 堆叠多个 Block
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 堆叠多个 Block —— 更深的模型")
print("=" * 60)

n_layers = 6
blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

x = np.random.randn(seq_len, d_model) * 0.5

print(f"堆叠 {n_layers} 个 Transformer Block")
print(f"\n各层输出的统计信息:")
print(f"  {'层':>4s}  {'均值':>10s}  {'标准差':>10s}  {'最大值':>10s}")

h = x
for i, block in enumerate(blocks):
    h = block.forward(h, mask=causal_mask)
    print(f"  {i:>4d}  {h.mean():>+10.4f}  {h.std():>10.4f}  {np.abs(h).max():>10.4f}")

print(f"\n→ 即使堆叠了 {n_layers} 层，数值依然稳定")
print("→ 这要归功于残差连接和 LayerNorm！")


# ============================================================
# Part 4: 对比真实模型的规模
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 真实模型的规模对比")
print("=" * 60)

models = {
    "我们的玩具模型": {"layers": n_layers, "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff},
    "GPT-2 Small":   {"layers": 12, "d_model": 768, "n_heads": 12, "d_ff": 3072},
    "GPT-2 Large":   {"layers": 36, "d_model": 1280, "n_heads": 20, "d_ff": 5120},
    "GPT-3 175B":    {"layers": 96, "d_model": 12288, "n_heads": 96, "d_ff": 49152},
    "LLaMA-2 70B":   {"layers": 80, "d_model": 8192, "n_heads": 64, "d_ff": 28672},
}

print(f"\n{'模型':<18s} {'层数':>6s} {'d_model':>8s} {'n_heads':>8s} {'d_ff':>8s} {'~参数量':>12s}")
print("-" * 70)
for name, cfg in models.items():
    n = cfg["layers"]
    d = cfg["d_model"]
    h = cfg["n_heads"]
    ff = cfg["d_ff"]
    params = n * (4 * d * d + 2 * d * ff + ff + d)
    if params > 1e9:
        param_str = f"{params/1e9:.1f}B"
    elif params > 1e6:
        param_str = f"{params/1e6:.1f}M"
    else:
        param_str = f"{params/1e3:.1f}K"
    print(f"{name:<18s} {n:>6d} {d:>8d} {h:>8d} {ff:>8d} {param_str:>12s}")

print("""
关键观察:
  - 结构完全一样，只是数字不同
  - GPT-3 和我们的模型用的是同样的 Transformer Block
  - 规模差异: 我们 ~几千参数 vs GPT-3 ~175B 参数
  - 大力出奇迹！但架构的本质不变
""")


# ============================================================
# Part 5: Dropout（正则化）
# ============================================================

print("=" * 60)
print("Part 5: Dropout —— 防止过拟合")
print("=" * 60)


def dropout(x, rate=0.1, training=True):
    """
    Dropout（反向 dropout / inverted dropout）：防止过拟合的正则化手段。

    - 训练时：随机把约 rate 比例的值置 0（"丢弃"），存活的值放大 1/(1-rate)。
              逼模型别过度依赖某几个神经元，学到更鲁棒的特征。
    - 推理时：什么都不做，原样返回 x（既不丢弃，也不缩放）。

    为什么训练时要 /(1-rate) 缩放：
      训练丢掉一部分后数值总量会变小，而推理时全部保留、总量更大，两者会对不上。
      训练时把存活的值放大 1/(1-rate)，使其期望值与推理时一致 —— 这样推理时就无需再做任何处理。
      （这种"训练时缩放、推理时不动"的写法就叫 inverted dropout，是目前的标准实现。）
    """
    # 推理阶段（或 rate=0）：直接返回，不丢弃也不缩放
    if not training or rate == 0:
        return x
    # 训练阶段：生成 0/1 掩码，约 rate 比例为 0（丢弃），其余为 1（保留）
    mask = (np.random.rand(*x.shape) > rate).astype(float)
    # 应用掩码并把存活值放大 1/(1-rate)，让期望值与推理时对齐
    return x * mask / (1 - rate)


x = np.ones((3, 8))
x_dropped = dropout(x, rate=0.3)
print(f"\n原始值:     {x[0]}")
print(f"Dropout 后: {x_dropped[0]}")
print(f"→ 部分值被置 0，其余值被放大 (÷ 0.7)")
print(f"→ 训练时随机'丢弃'一些神经元，防止过度依赖某些特征")

print("""
Transformer 中 Dropout 的位置:
  1. 注意力权重上: 随机忽略一些注意力连接
  2. FFN 输出上: 随机忽略一些特征
  3. 残差连接前: 在加到残差之前 dropout
  
GPT-3 的 dropout rate = 0.1
""")


# ============================================================
# 练习
# ============================================================

print("=" * 60)
print("练习")
print("=" * 60)
print("""
1. 把 d_ff 改成 2 × d_model 和 8 × d_model，观察输出变化
   - 思考：为什么通常用 4 倍？

2. 去掉 LayerNorm，堆叠 20 层，观察数值是否爆炸
   - 你会看到 NaN 或极大的数字

3. 去掉残差连接，堆叠 10 层，观察输出是否退化
   - 所有位置的输出可能变得几乎一样

4. 用 ReLU 替换 GELU，对比输出差异
   - 小模型差异不大，但 GELU 的平滑性在大规模训练时很重要

5. (进阶) 为 TransformerBlock 添加可学习的 LayerNorm 参数 (gamma, beta)
""")
