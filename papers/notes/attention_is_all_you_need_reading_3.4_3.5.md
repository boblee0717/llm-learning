# Attention Is All You Need 精读

本文聚焦论文两个小节：

- `3.4 Embeddings and Softmax`
- `3.5 Positional Encoding`

目标不是逐字翻译，而是把每一句背后的设计动机讲清楚，方便后续对照 Transformer 代码来理解。

---

## 阅读导航

如果你只想先抓主线，可以先看这 4 个结论：

- 输入 token 和输出 token 都先被映射成 `d_model` 维向量。
- 解码器最后还要经过 `线性层 + Softmax`，才能变成“下一个 token 的概率”。
- Transformer 把输入嵌入、输出嵌入和输出层前的线性变换做了权重共享，减少参数量。
- 因为 Transformer 没有 RNN/CNN，所以必须额外加入位置编码，否则模型不知道词序。

---

## 3.4 Embeddings and Softmax

这一节回答的是两个问题：

- token 进入 Transformer 之前，先变成什么？
- 解码器输出之后，怎么变成最终的词概率？

### 核心结论

- 输入 token 和输出 token 都用可学习的 embedding 表示。
- decoder 的输出先做线性映射，再经过 Softmax 得到词表概率。
- 两个 embedding 层和 pre-softmax 线性层共享同一套权重矩阵。
- embedding 层的输出额外乘上 `sqrt(d_model)`，避免和位置编码相加时量级过小。

### 句 1

> Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model.

**解读**

和其他序列到序列模型一样，Transformer 不会直接处理离散 token，而是先通过可学习的嵌入层，把输入 token、输出 token 都映射成 `d_model` 维向量。论文里的默认值是 `d_model = 512`。

**补充理解**

这里做的是 NLP 里最基础的一步：把“词表索引”变成“连续向量”。只有这样，模型后面的线性层、注意力层、前馈网络才能进行数值计算。embedding 学到的，本质上是“词在语义空间中的位置”。

### 句 2

> We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

**解读**

解码器最后输出的不是词，而是一个隐藏状态向量。模型还需要再经过一层可学习的线性变换，把这个向量映射到“词表大小”维度，然后用 Softmax 把它变成一个概率分布，用来预测下一个 token。

**补充理解**

这一步可以理解成：

`隐藏状态 -> 每个词的打分(logits) -> 概率分布`

其中：

- 线性层负责“给词表中每个词打分”
- Softmax 负责“把分数归一化成概率”

### 句 3

> In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30].

**解读**

论文在这里用了一个很重要的技巧：输入 embedding、输出 embedding，以及 Softmax 之前的线性层，共享同一个权重矩阵。

**补充理解**

这通常叫 `weight tying`。它的好处主要有两个：

- 减少参数量，模型更轻，不容易过拟合
- 让“词的输入表示”和“词的输出打分”共享同一套语义空间，训练更一致

直觉上可以把它理解成：模型“读入一个词”和“预测一个词”时，使用的是同一套词语表示系统。

### 句 4

> In the embedding layers, we multiply those weights by sqrt(d_model).

**解读**

在 embedding 层输出后，论文额外乘了一个缩放因子 `sqrt(d_model)`。

**补充理解**

这一步常常容易被忽略，但它和后面的 `positional encoding` 有直接关系。

因为 embedding 会和位置编码直接相加，而位置编码来自正弦余弦函数，数值范围大致在 `[-1, 1]`。如果 embedding 的量级偏小，那么词本身的语义信息就可能被位置编码“压过去”。

所以这里乘上 `sqrt(d_model)`，本质上是在放大 embedding 的尺度，让两者相加时更平衡。对论文默认的 `d_model = 512` 来说，这个因子约等于 `22.6`。

### 小结

这一节的主线可以压缩成一句话：

`token -> embedding -> 加入位置信息 -> 进入 Transformer`

而在 decoder 的输出端，则是：

`隐藏状态 -> 线性映射 -> Softmax -> 下一个 token 的概率`

---

## 3.5 Positional Encoding

这一节回答的是一个更关键的问题：

Transformer 既没有 RNN 的“按顺序递推”，也没有 CNN 的“局部卷积”，那它怎么知道一句话里的顺序？

答案就是：显式加入位置编码。

### 核心结论

- self-attention 本身不天然理解顺序。
- 所以必须把“位置信息”人工注入到 embedding 里。
- 论文使用的是固定的正弦/余弦位置编码，而不是纯可学习参数。
- 这种设计的一个重要优势，是更容易泛化到比训练时更长的序列。

### 第 1 段：为什么必须加位置编码

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

**解读**

因为 Transformer 没有循环结构，也没有卷积结构，所以模型本身并不知道谁在前、谁在后。为了让它利用序列顺序，必须显式注入“绝对位置”或“相对位置”的信息。论文的做法是：在 encoder 和 decoder 的最底层，把位置编码直接加到 token embedding 上。

**补充理解**

self-attention 对输入的顺序并不敏感。换句话说，如果你不提供位置信息，那么模型只会把输入看成“一组 token 的集合”，而不是“有先后顺序的句子”。

这也是为什么位置编码不是可选项，而是 Transformer 能处理文本序列的前提条件。

### 第 2 段：为什么位置编码能和 embedding 直接相加

> The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].

**解读**

位置编码的维度被设计成和 embedding 一样，都是 `d_model`，这样两者就可以直接逐元素相加。

**补充理解**

这里论文选择“相加”，而不是“拼接”，有一个很实际的工程原因：不改变隐藏维度。

如果改成拼接，后续层的输入维度就会翻倍，计算量和参数量都会明显增大。相加则更轻量，也更容易保持网络结构简洁。

另外，位置编码并不只有一种做法。大体可以分成两类：

- `learned positional embedding`：把每个位置当成参数学出来
- `fixed positional encoding`：用固定公式直接算出来

Transformer 论文最终选的是第二类。

### 第 3 段：正余弦位置编码的公式是什么意思

**原文**

> In this work, we use sine and cosine functions of different frequencies:

**论文里的 LaTeX 写法（在 GitHub 网页上通常会渲染成分式；Cursor 内置 Markdown 预览常常不会渲染，下面给出同一含义的纯文本。）**

```text
PE(pos, 2i)    = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1)  = cos( pos / 10000^(2i/d_model) )
```

与分式记号完全等价：`sin( pos / 10000^(2i/d_model) )` 即论文中的 `sin( pos / (10000 的 (2i/d_model) 次方) )`；`cos` 同理。

其中 **pos** 为序列中的位置下标（0, 1, 2, …），**i** 为维度对索引：偶数维 `2i` 与奇数维 `2i+1` 共用同一角频率 `pos / 10000^(2i/d_model)`，对应嵌入向量中相邻两列（见下文「这里的维度到底是什么」）。

**解读**

论文把位置编码设计成一组不同频率的正弦波和余弦波：

- 偶数维用 `sin`
- 奇数维用 `cos`
- `pos` 表示位置
- `i` 表示当前维度编号

**补充理解**

这个公式的核心不是“好看”，而是“不同维度对应不同尺度的位置变化”。

可以把它理解成这样：

- 前面的维度变化快，擅长区分相邻位置
- 后面的维度变化慢，擅长表示更大范围的位置关系

因此，同一个位置会被编码成一个由多种频率共同组成的向量。这个向量既保留了局部位置信息，也保留了较长距离的位置模式。

#### 公式拆解：频率是怎么变化的

公式中最关键的部分是分母 `10000^(2i/d_model)`。先看最简单的情况——假设 `d_model = 2`，只有两个维度（i=0）：

```
维度 0: sin(pos / 10000^0) = sin(pos)     ← 除以 1，频率最高，变化最快
维度 1: cos(pos / 10000^0) = cos(pos)
```

再看 i=1（如果 d_model 更大）：

```
维度 2: sin(pos / 10000^(2/d_model))      ← 除以了一个大数，频率变低
维度 3: cos(pos / 10000^(2/d_model))
```

**随着维度 i 增大，`10000^(2i/d_model)` 越来越大，除法结果越来越小，波的频率越来越低**：

```
i=0:    sin(pos / 1)        秒针 ⟳⟳⟳⟳  变化很快，区分相邻位置
i=1:    sin(pos / 31.6)     分针 ⟳⟳      变化较慢，编码中等范围
i=2:    sin(pos / 1000)     时针 ⟳        变化很慢，编码远距离关系
...
i=max:  sin(pos / 10000)    日历          几乎不变，编码全局位置
```

#### 多指针时钟的类比

把位置编码想象成一个多指针时钟：

```
位置 0:  🕛  所有指针在起点
位置 1:  🕐  秒针动了，分针几乎没动，时针更没动
位置 2:  🕑  秒针又动了，分针微微动了一点
...
位置 50: 🕐  秒针转了好多圈，分针转了一点，时针几乎没动
```

每个时刻（位置），所有指针的组合都是唯一的。快指针区分相近位置，慢指针区分远距离位置。模型读取这些"指针读数"就知道每个词在哪了。

#### 为什么每个位置都有唯一编码

和二进制编码的思路类似：`0=000, 1=001, 2=010, 3=011...`，不同位（bit）的翻转频率不同。位置编码用连续的 sin/cos 做了同样的事——多个不同频率的波组合起来，每个位置的编码向量都是唯一的。

#### 为什么值域是有界的

不管位置 pos 多大，sin/cos 的值域永远在 `[-1, 1]`。如果用简单的 `[0, 1, 2, ...]` 作为位置，训练时最长 512 个词，遇到第 513 个词就是一个没见过的数字。而 sin/cos 编码不存在这个问题，数值范围始终可控。

### 第 4 段：不同维度为什么要用不同波长

> That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π.

**解读**

每个维度都对应一条不同频率的波。随着维度增加，波长按等比数列逐渐变长，从 `2π` 一直到 `10000 x 2π`。

**补充理解**

这意味着：

- 低维度的位置编码变化很快，能区分非常近的位置
- 高维度的位置编码变化很慢，能表达更大跨度的位置趋势

组合起来以后，模型就像同时拿到了“短焦镜头”和“长焦镜头”：

- 短焦负责近距离差异
- 长焦负责远距离结构

所以位置编码不是只告诉模型“这是第 7 个词”，而是在多个尺度上共同描述“它处在什么位置”。

#### 这里的“维度”到底是什么

“维度”指的是 **embedding 向量中每一个元素的位置（列索引）**。例如 `d_model = 16`，每个词被表示为长度 16 的向量：

```
“猫” 的嵌入向量 = [v₀, v₁, v₂, v₃, v₄, v₅, ... , v₁₅]
                  dim0 dim1 dim2 dim3 dim4 dim5      dim15
```

每个分量 `v_n` 就是一个“维度”。对应到位置编码矩阵（形状 `max_len × d_model`），**行是位置，列是维度**：

```
              dim0   dim1   dim2   dim3   dim4   dim5  ...  dim15
              (i=0)  (i=0)  (i=1)  (i=1)  (i=2)  (i=2)
              sin    cos    sin    cos    sin    cos
position 0  [ 0.00,  1.00,  0.00,  1.00,  0.00,  1.00, ... ]
position 1  [ 0.84,  0.54,  0.10,  0.99,  0.01,  1.00, ... ]
position 2  [ 0.91, -0.42,  0.20,  0.98,  0.02,  1.00, ... ]
...
              ↑              ↑              ↑
              频率最高        频率中等        频率最低
              变化最快        变化较慢        几乎不变
```

可以看到：靠前的列（低维度）在不同位置间变化剧烈，靠后的列（高维度）几乎不变。每个位置的整行组合起来，就构成了该位置的唯一编码。

### 第 5 段：为什么这种设计有助于学习相对位置

> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE(pos + k) can be represented as a linear function of PE(pos).

**解读**

作者选择正余弦编码，不只是为了表达绝对位置，还因为它可能有助于模型学习“相对位置”。原因是：对于任意固定偏移量 `k`，位置 `pos + k` 的编码可以由位置 `pos` 的编码通过线性关系表示出来。

**补充理解**

这背后依赖的是三角函数的和角公式，例如：

`sin(a + b) = sin(a)cos(b) + cos(a)sin(b)`

这意味着，如果两个词之间相隔固定距离，那么它们的位置编码之间存在稳定的线性关系。对注意力机制来说，这很有价值，因为模型更容易学到：

- “前一个词”
- “后两个词”
- “距离当前词 5 个位置的词”

这种相对距离模式，而不是只记住绝对编号。

### 第 6 段：为什么最后选固定编码，而不是学习出来

> We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)).

**解读**

作者也试过可学习的位置嵌入，实验结果发现，它和固定正余弦编码的效果几乎一样。

### 第 7 段：固定编码真正的优势是什么

> We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

**解读**

最终论文选择正余弦编码，不是因为它在训练集上显著更强，而是因为它更有可能泛化到比训练时更长的序列。

**补充理解**

可学习位置编码通常只能覆盖训练时见过的位置范围。例如，如果训练时最长只见过 512 个位置，那么模型对更长输入的泛化能力往往有限。

而正余弦位置编码是一个固定公式：

- 只要给定位置 `pos`
- 就总能算出对应编码

这让模型在推理时更容易处理“训练时没见过的更长句子”。

### 小结

这一节最重要的理解不是记住公式，而是记住这条逻辑链：

1. Transformer 没有顺序结构。
2. self-attention 本身不知道词序。
3. 所以要把位置信息提前注入 embedding。
4. 论文用的是固定的正余弦编码。
5. 这种编码既能表达绝对位置，也更容易帮助模型学习相对位置，还具备更好的长度外推能力。

---

## 最后用一句话串起来

`3.4` 解决的是“token 怎么变成模型能处理的向量，以及输出怎么变成词概率”。

`3.5` 解决的是“模型怎么知道词序”。

两节合起来，就是 Transformer 输入端最基础的一层设计：

`token embedding + positional encoding -> 送入后续注意力层`
