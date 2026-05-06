# Attention Is All You Need 精读：3.3 Position-wise Feed-Forward Networks

本文聚焦论文一个小节：

- `3.3 Position-wise Feed-Forward Networks`

目标不是逐字翻译，而是把每一句背后的设计动机讲清楚，并和第 4 课 `04_transformer_block.py` 的实现逐项对齐。

---

## 阅读导航

如果你只想先抓主线，可以先看这 5 个结论：

- Transformer 的每一层不只有注意力，还有一个逐位置的两层 MLP（FFN）。
- FFN 对每个位置独立计算，但参数在所有位置共享，形式是 `max(0, xW1 + b1)W2 + b2`（原论文写法）。
- 从维度看，常见是 `d_model -> d_ff -> d_model`，原论文取 `d_model=512, d_ff=2048`，即 `4x` 扩展。
- 可以把它理解为“每个 token 位置上的小网络”，本质是跨通道变换，不做位置间信息混合。
- 在一个标准 Transformer block 里，attention 负责“位置间交互”，FFN 负责“位置内非线性变换”，两者缺一不可。

---

## 3.3 Position-wise Feed-Forward Networks

这一节非常短，但信息密度很高。下面按句拆开。

### 句 1：每层都包含一个逐位置 FFN

**原文**

> In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.

**解读**

除了注意力子层外，encoder 和 decoder 的每一层都包含一个全连接前馈网络（FFN）。后半句强调：该网络对每个位置分别施加，且在各位置上的运算形式相同（同一套变换、共享参数）。

**补充理解**

这句话先回答“为什么 block 里不只写 attention”。Transformer 不是“纯注意力模型”，而是：

`Attention sub-layer + FFN sub-layer` 的组合。

第 4 课 `04_transformer_block.py` 里 `TransformerBlock.forward()` 也是两段式：

1. `LayerNorm -> MultiHeadAttention -> Residual`
2. `LayerNorm -> FFN -> Residual`

这里的 “position-wise” 很关键：FFN 不负责让第 1 个 token 和第 5 个 token 交流。位置之间的信息混合已经由 attention 做过了；FFN 做的是对每个位置自己的向量再加工。

---

### 句 2：FFN 是两层线性变换 + ReLU

**原文**

> This consists of two linear transformations with a ReLU activation in between.

紧随其后的公式 **(2)**（论文排版中单列为一条）：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2 \quad \text{(2)}
$$

（PDF 中为 `W1, W2` 与 `b1, b2` 的下标形式；语义相同。）

**解读**

FFN 的结构是两个线性变换，中间夹一个 ReLU 激活。

**补充理解**

数学上就是论文里的 equation (2)：

`FFN(x) = max(0, xW1 + b1)W2 + b2`

也就是常说的两层 MLP。你的课程代码里用的是 GELU（更现代，GPT 常用），本质仍是同一模式：`Linear -> Activation -> Linear`。

这里的“两层”指两个 Linear / 全连接层：

1. 先把 `d_model` 维向量映射到更宽的 `d_ff` 维：`xW1 + b1`
2. 经过 ReLU 非线性
3. 再映射回 `d_model` 维：`(...)W2 + b2`

为什么中间要有 ReLU？如果没有激活函数，两次线性变换可以合并成一次线性变换，模型表达能力会弱很多。

---

### 句 3：同一变换作用于各位置；层与层之间参数不同

**原文**

> While the linear transformations are the same across different positions, they use different parameters from layer to layer.

**解读**

前半句：在各位置上施加的是同一种（同一套参数的）线性变换；后半句：不同网络层之间的 FFN 使用彼此不同的参数。与下面「逐位置共享权重」的表述是一致的：共享发生在同一层内的各个位置上，不发生在不同层之间。

**补充理解**

这是 “position-wise” 的关键：
- **跨位置共享参数**：第 1 个词和第 20 个词都用同一个 `W1/W2`。
- **跨层不共享参数**：第 3 层 FFN 和第 7 层 FFN 各有各的 `W1/W2`。

所以它像“在每个位置并行跑同一个小网络”。这也解释了为什么 FFN 不改变序列长度维：它不在位置轴上做卷积或注意力。

---

### 句 4：FFN 等价于两个 kernel=1 的卷积

**原文**

> Another way of describing this is as two convolutions with kernel size 1.

**解读**

FFN 也可看作两个 `kernel=1` 的一维卷积。

**补充理解**

`kernel=1` 这句是在帮你建立工程直觉：

- `kernel=1` 不看相邻位置，只在通道维做线性混合；
- 与逐位置 MLP 完全等价。

如果输入形状是 `(seq_len, d_model)`，那么这个 FFN 相当于对每个位置的 `d_model` 维向量做同一套变换：

```text
(seq_len, d_model)
  -> Linear / kernel=1 conv
(seq_len, d_ff)
  -> ReLU 或 GELU
(seq_len, d_model)
```

注意：它不会把相邻 token 的表示卷在一起，所以它不是在做“看前后文”。看前后文是 attention 的职责。

---

### 句 5：输入输出维度是 d_model，中间层是 d_ff

**原文**

> The dimensionality of input and output is $d_{\text{model}} = 512$, and the inner-layer has dimensionality $d_{\text{ff}} = 2048$.

（NeurIPS PDF 正文中排版为 $d_{\mathrm{model}}$、$d_{\mathrm{ff}}$，此处与论文记号一致。）

**解读**

FFN 的输入和输出都是 `d_model=512` 维，中间层是 `d_ff=2048` 维。

**补充理解**

维度变化可以记成：

```text
d_model -> d_ff -> d_model
512     -> 2048 -> 512
```

最后必须回到 `d_model`，因为 FFN 子层外面还有残差连接。残差连接要做的是：

```text
x + FFN(x)
```

这要求 `x` 和 `FFN(x)` 形状完全一样。你的第 4 课代码也保持了这个约束：`FeedForward.forward()` 输入是 `(seq_len, d_model)`，输出仍是 `(seq_len, d_model)`。

`d_ff = 4 * d_model` 是经典配置，与你第 4 课代码里的：

`d_ff = d_model * 4`

完全一致。工业模型里这个比例不一定固定 4（也有 3.5、8/3 等变体），但 4 是最常见基线。

---

## 为什么 FFN 这么重要？

只看 attention，会做“位置之间的信息路由”；但如果没有 FFN，每层对单个位置的非线性表征能力会明显不足。

可以用一句话记：

- **Attention**：让 token 彼此看见，做“信息交换”。
- **FFN**：让每个位置做更强的“特征变换”。

两者叠加后，block 才既能建模依赖关系，又能提升每个位置的表示质量。

---

## 与 `04_transformer_block.py` 对照

| 论文 3.3 描述 | 代码中的对应 |
|---|---|
| 两层线性 + 激活 | `FeedForward.forward()` 里的 `x @ W1 + b1 -> gelu -> hidden @ W2 + b2` |
| 逐位置应用 | 输入 `(seq_len, d_model)`，输出仍是 `(seq_len, d_model)` |
| 常见 `d_ff=4*d_model` | `d_ff = d_model * 4` |
| 每层独立参数 | 每个 `TransformerBlock` 都持有独立 `FeedForward` 实例 |

你可以直接用脚本里的参数量打印验证这句直觉：

- Attention 参数约 `4 * d_model * d_model`
- FFN 参数约 `2 * d_model * d_ff`（主项）

当 `d_ff=4*d_model` 时，FFN 参数通常占大头。

---

## 建议配套阅读

- `papers/notes/attention_is_all_you_need_reading_3.1_3.2.2.md`：encoder/decoder 栈与多头注意力主干。
- `papers/notes/notes_gpt2_input_and_model.md`：GPT-2 §2.3（Pre-Norm、上下文窗口、词表等工程配置）。
- `papers/notes/gpt3_reading_2.1_model_and_architectures.md`：同构 block 如何沿着 scale law 放大。

---

## 文献

Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
