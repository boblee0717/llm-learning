# Attention Is All You Need 精读

本文聚焦论文三段内容：

- `§3 Model Architecture` 开头对 encoder-decoder 的铺垫
- `3.1 Encoder and Decoder Stacks`
- `3.2.2 Multi-Head Attention`

目标不是逐字翻译，而是把每一句背后的设计动机讲清楚，方便后续对照 Transformer 代码来理解。

---

## 阅读导航

如果你只想先抓主线，可以先看这 5 个结论：

- Encoder / Decoder 都是 `N = 6` 个相同结构的层堆叠，不是单层 attention。
- 每个 sub-layer 外面都包着残差连接和 LayerNorm，原始 Transformer 用的是 Post-Norm：`LayerNorm(x + Sublayer(x))`。
- 残差要做加法，所以**所有 sub-layer 与 embedding 的输出维度都是 `d_model`**。
- Decoder 比 Encoder 多一个 sub-layer：cross-attention，用来回头看 encoder 的输出。
- Multi-Head Attention 是把 Q/K/V 投影到多个较小子空间并行做 attention，再 concat 后用 `W^O` 投回 `d_model`。

---

## §3 开头铺垫

论文在进入 §3.1 前，先用几句话把 encoder-decoder 这个大框架和 auto-regressive 的性质铺好。先读懂它，后面 §3.1 才不会突兀。

### 核心结论

- 大多数 sequence transduction 模型都是 encoder-decoder 结构。
- Encoder 把输入符号序列变成连续向量表示。
- Decoder 一次生成一个符号，每一步都把之前生成的符号作为输入（auto-regressive）。
- Transformer 沿用这套框架，但内部换成了 self-attention + 全连接前馈层。

### 句 1

> Most competitive neural sequence transduction models have an encoder-decoder structure.

**解读**

大多数有竞争力的「序列到序列」模型，都采用 encoder-decoder 这种两段式结构：先理解，再生成。

**补充理解**

`sequence transduction` 可以先粗略理解成「把一个序列转换成另一个序列」，最经典的例子是机器翻译（英文句子 → 中文句子）。

这一句先把读者拉回 2017 年的时代背景：encoder-decoder 不是 Transformer 发明的，而是当时的「主流共识」。Transformer 的创新不是抛弃这个框架，而是替换里面的核心计算单元。

### 句 2

> Here, the encoder maps an input sequence of symbol representations (x_1, ..., x_n) to a sequence of continuous representations z = (z_1, ..., z_n).

**解读**

Encoder 的任务是：把输入的「符号序列」（每个 `x_i` 是一个 token）映射成一串「连续向量表示」`z_i`。注意输入长度 `n` 和输出长度一致。

**补充理解**

这里有两个细节值得留意：

- **「符号 → 连续向量」**：`x_i` 是离散 token id，`z_i` 是稠密向量。只有变成连续向量，后面的矩阵乘、注意力、归一化才有意义。
- **长度不变**：encoder 是 `n → n` 的映射，每个输入位置都对应一个输出向量，并不像有些 RNN 模型那样压成一个固定长度的 context vector。这一点是后续 attention 能成立的前提：decoder 之后会逐位置查询 `z_1 ... z_n`。

维度变化可以记成：

```text
token ids:   (seq_len,)
embedding:   (seq_len, d_model)
encoder out: (seq_len, d_model)   ← 这就是 z
```

### 句 3

> Given z, the decoder then generates an output sequence (y_1, ..., y_m) of symbols one element at a time.

**解读**

Decoder 拿到 encoder 输出的 `z` 之后，开始生成目标序列 `y_1, ..., y_m`，**一次生成一个符号**。

**补充理解**

注意这里输入长度是 `n`，输出长度是 `m`，两者**不要求相等**。例如英文 5 个词翻译到中文可能是 4 个或 6 个字。

「一次一个」是这一段的关键词。它意味着 decoder 不是一口气把整句话生成出来，而是一个 token 一个 token 往后蹦。这也是后面 causal mask 出现的根本原因。

### 句 4

> At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

**解读**

每一步生成时，模型都是自回归的（auto-regressive）：把已经生成的 token 拼回输入，再去预测下一个。

**补充理解**

举个例子，生成「我 喜欢 猫」的过程是：

```text
看到 <BOS>           -> 预测 我
看到 <BOS> 我        -> 预测 喜欢
看到 <BOS> 我 喜欢   -> 预测 猫
看到 <BOS> 我 喜欢 猫 -> 预测 <EOS>
```

`auto-regressive` 拆开看：`auto` = 自己，`regressive` = 回归到前面。也就是「拿自己之前的输出，回头当作输入」。

这条规则有一个硬性约束：**预测第 i 个位置时不能看到第 i 个之后的内容**，否则就是抄答案。这正是后面 §3.1 decoder 段里 causal mask 的动机。

### 句 5

> The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

**解读**

Transformer 沿用上述「encoder-decoder + 自回归生成」的整体框架，但把内部计算换成了：**堆叠的 self-attention** + **逐位置（point-wise）的全连接层**。Figure 1 的左半边是 encoder，右半边是 decoder。

**补充理解**

这一句是整篇论文的架构宣言。可以记成一张对照表：

| 维度 | RNN/CNN seq2seq | Transformer |
|------|-----------------|-------------|
| 外壳 | encoder-decoder | encoder-decoder |
| 序列建模方式 | 循环 / 卷积 | self-attention |
| 位置上的非线性 | RNN cell / 卷积 | point-wise FFN |
| 顺序信息来源 | 天然按时间步 | 位置编码（§3.5） |

`point-wise` 的意思是：FFN 对每个 token 位置独立施加同一套 MLP，**不在位置之间混合信息**。位置之间的混合完全交给 self-attention 完成。这是 Transformer 「分工清晰」的关键：

- self-attention 负责「token 之间」的信息流动
- FFN 负责「单个 token 内部」的非线性加工

---

## 3.1 Encoder and Decoder Stacks

这一节回答的是两个问题：

- Encoder 和 Decoder 各自长什么样？内部有几个子层？
- 用了哪些技巧来保证深层网络的训练稳定性？

### 核心结论

- Encoder 和 Decoder 都是 `N = 6` 个相同结构的 layer 堆叠（结构相同，参数不共享）。
- Encoder 每层有 2 个 sub-layer：multi-head self-attention + position-wise FFN。
- Decoder 每层多一个 sub-layer：cross-attention，对 encoder 输出做 attention。
- 每个 sub-layer 外面都是 `LayerNorm(x + Sublayer(x))`，即残差 + LayerNorm（Post-Norm）。
- 为了让残差加法成立，所有 sub-layer 和 embedding 都输出 `d_model = 512` 维。
- Decoder 的 self-attention 加上 causal mask，再配合输出右移一位，保证自回归性质。

---

#### Encoder 段

### 句 6

> The encoder is composed of a stack of N = 6 identical layers.

**解读**

Encoder 由 `N = 6` 个**结构相同**的 layer 堆叠组成。

**补充理解**

这里有两个词需要拆开理解：

- `stack`：堆叠。Transformer 不是只做一次 attention 然后就完事，而是把同样的结构反复套很多遍。
- `identical layers`：**结构相同**，不是「参数共享」。第 1 层和第 2 层都长成「attention + FFN + 两次 Add & Norm」的样子，但里面的权重矩阵 `W_Q, W_K, W_V, W_O, W_1, W_2` 各自独立学习。

可以画成：

```text
X0
 │
 ▼
EncoderLayer 1   (一套独立参数)
 │
 ▼
EncoderLayer 2   (另一套独立参数)
 │
 ▼
...
 │
 ▼
EncoderLayer 6
 │
 ▼
encoder output / memory
```

为什么要堆 6 层？直觉上，每一层负责把表示「再精炼一次」：浅层可能学到局部特征（搭配、词性），深层可能学到更抽象的关系（指代、语法结构）。论文 §6.2 的消融实验里也试过 `N = 2, 4, 8`，6 是 base model 的折中选择。

### 句 7

> Each layer has two sub-layers.

**解读**

每一个 encoder layer 内部由两个子层（sub-layer）组成。

**补充理解**

这里要把「层（layer）」和「子层（sub-layer）」分清楚：

- **layer**：一个完整的 encoder block，外部看是一个黑盒。
- **sub-layer**：layer 内部的功能模块，是 attention 或 FFN。

```text
EncoderLayer
├── sub-layer 1: Multi-Head Self-Attention
└── sub-layer 2: Position-wise FFN
```

为什么要强调这个区分？因为下一句开始，论文会说「每个 sub-layer 外面都包 residual + LayerNorm」——是包在每个 sub-layer 外面，**不是只在整个 EncoderLayer 外面包一次**。一个 encoder layer 有两次 Add & Norm，不是一次。

### 句 8

> The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.

**解读**

第一个 sub-layer 是多头自注意力（multi-head self-attention），第二个 sub-layer 是「逐位置（position-wise）」的全连接前馈网络（FFN）。

**补充理解**

这句话里藏了三个关键词：

- **self-attention**：Q、K、V 都来自**同一个序列**。在 encoder 里就是上一层 encoder 的输出，每个 token 都能看见同一句话里的所有 token（包括未来的，因为 encoder 不需要 mask）。
- **multi-head**：不是只做一组 Q/K/V attention，而是切成多个 head 并行做。详细机制在 §3.2.2 讲。
- **position-wise**：FFN 在每个 token 位置上**独立**做同一个 MLP：

```text
位置 0:  x_0 ──FFN──> y_0
位置 1:  x_1 ──FFN──> y_1
位置 2:  x_2 ──FFN──> y_2
...
```

所有位置共用**同一套** FFN 参数，但每个位置的计算彼此独立。位置之间的信息已经由 self-attention 混合过了，FFN 不再做位置间交互，它只负责给每个位置的表示做一次非线性变换。

论文 §3.3 给出的 FFN 公式是：`FFN(x) = max(0, xW1 + b1)W2 + b2`，是一个两层 MLP 加 ReLU，内部先升维再降维：`d_model -> d_ff -> d_model`，即 `512 -> 2048 -> 512`。最后必须回到 `d_model`，因为后面要和残差相加。

### 句 9

> We employ a residual connection around each of the two sub-layers, followed by layer normalization.

**解读**

论文在两个 sub-layer 外面都加上残差连接（residual connection），然后再做 LayerNorm。

**补充理解**

这句对应 Figure 1 里的 `Add & Norm` 方框。顺序很重要：

```text
1. 先算 Sublayer(x)
2. 再做 Add：x + Sublayer(x)
3. 最后做 LayerNorm
```

**为什么要残差？** 残差给原始输入开了一条「直通车」：

```text
x ──┬──────────────► +
    │                ▲
    └─► Sublayer ────┘
```

即使 `Sublayer(x)` 学得很糟糕（甚至完全输出 0），最终结果至少还是 `x` 本身，不会比原来更差。这让深层网络更容易优化，梯度也能直接回传到浅层。这是 ResNet 思想的延伸，2015 年就被证明对深网很关键。

**为什么紧跟 LayerNorm？** 残差相加之后数值范围可能变大，LayerNorm 把每个位置的向量沿特征维做归一化，让训练稳定。注意：LayerNorm 是**对每个 token 单独做归一化**（沿 `d_model` 维），不是对整个 batch 做，这点和 BatchNorm 不同。

### 句 10

> That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.

**解读**

论文给出统一公式：每个 sub-layer 的最终输出是 `LayerNorm(x + Sublayer(x))`。这里的 `Sublayer` 是一个**占位符**，代表当前 sub-layer 自己实现的那个函数。

**补充理解**

`Sublayer` 在不同位置代表不同的具体函数：

| 位置 | `Sublayer(x)` 是什么 |
|------|----------------------|
| Encoder 第 1 个子层 | `MultiHeadSelfAttention(x)` |
| Encoder 第 2 个子层 | `FFN(x)` |
| Decoder 第 1 个子层 | `MaskedMultiHeadSelfAttention(x)` |
| Decoder 第 2 个子层 | `CrossAttention(x, encoder_output)` |
| Decoder 第 3 个子层 | `FFN(x)` |

论文用一个简短的公式涵盖了 5 种不同的 Add & Norm 用法。这种「外壳一致、内部可替换」的设计让代码也很优雅。

**这里可以理解为「后 Norm（Post-Norm）」吗？** 可以。原始 Transformer 是典型的 Post-Norm：

```text
Post-Norm: y = LayerNorm(x + Sublayer(x))   ← 论文写法
Pre-Norm:  y = x + Sublayer(LayerNorm(x))   ← 现代 GPT 常用
```

两者训练稳定性差异很大：Post-Norm 在深网下需要更小心的学习率调度（warmup），Pre-Norm 通常更稳定。这也是后续 GPT/LLaMA 系列改用 Pre-Norm 的原因之一。

### 句 11

> To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512.

**解读**

为了让残差连接成立，模型里所有 sub-layer 以及 embedding 层的输出维度都是 `d_model = 512`。

**补充理解**

这是 Transformer 里非常重要的一条「形状纪律」。

残差要做加法 `x + Sublayer(x)`，加法要求两边形状一致：

```text
x:           (seq_len, d_model)
Sublayer(x): (seq_len, ???)   ← ??? 必须等于 d_model
```

所以无论 sub-layer 内部怎么折腾——multi-head 拆成多个小头、FFN 升到 2048 维再降回来——**输出端必须回到 `d_model`**。

可以记成这样的口诀：

```text
模块内部可以分头、升维、降维；
模块输出必须回到 d_model。
```

为什么连 embedding 也要 `d_model`？因为 embedding 是 Transformer 的入口，紧接着就要和 positional encoding 相加（§3.5），然后送入第一个 encoder layer 做残差，这些步骤都要求维度对齐。

---

#### Decoder 段

### 句 12

> The decoder is also composed of a stack of N = 6 identical layers.

**解读**

Decoder 也是由 `N = 6` 个结构相同的 layer 堆叠组成。

**补充理解**

和 encoder 完全对称：6 层、结构相同、参数不共享。

也要注意：encoder 和 decoder 的 6 层是**两套独立**的层，不是同一份参数。整个 base model 加起来其实是 12 个 block（encoder 6 + decoder 6）。

### 句 13

> In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

**解读**

在 encoder layer 已有的两个子层之上，decoder 还插入了**第三个子层**，它对 encoder stack 的输出做 multi-head attention。

**补充理解**

把两边对照一下：

```text
Encoder layer：
1. Self-Attention
2. FFN

Decoder layer：
1. Masked Self-Attention
2. Encoder-Decoder Attention   ← 新增的第三个 sub-layer
3. FFN
```

这个新增的子层通常叫 **cross-attention** 或 **encoder-decoder attention**。它是机器翻译能跑通的关键：decoder 在生成目标语言时，需要不断回头看 encoder 对源语言的理解。

它和 self-attention 的区别在于 Q/K/V 来源：

```text
self-attention:  Q, K, V 都来自当前 decoder 的隐藏状态
cross-attention: Q 来自 decoder，K 和 V 来自 encoder 的输出
```

形象地说：decoder 拿着一个「问题」（Q）去 encoder 那一摞向量里「检索」（K 算相关度），然后取出相关的「内容」（V）。

### 句 14

> Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.

**解读**

和 encoder 一样，decoder 的每个 sub-layer 外面也是残差连接 + LayerNorm。

**补充理解**

Decoder 每层有 3 个 sub-layer，所以有 3 次 Add & Norm。完整写出来是：

```text
x1 = LayerNorm(x0 + MaskedSelfAttention(x0))
x2 = LayerNorm(x1 + CrossAttention(x1, encoder_output))
x3 = LayerNorm(x2 + FFN(x2))
```

注意 cross-attention 那一行：残差用的是 `x1`（decoder 自己的当前状态），不是 encoder 的输出。也就是说，残差始终走「decoder 这一侧的主干道」，encoder 输出只在 K/V 这两条支路出现。

cross-attention 的输出长度必须和 `x1` 一致，才能做加法：

```text
Q length      = target_seq_len  ← 决定输出长度
K/V length    = source_seq_len  ← 决定能检索的范围
output length = target_seq_len  ← 与残差对齐
```

### 句 15

> We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.

**解读**

论文修改了 decoder 里的 self-attention：让某个位置**不能**关注它后面的位置。这就是 causal mask（因果掩码）。

**补充理解**

为什么需要这个 mask？回到 §3 的铺垫——decoder 是 auto-regressive 的，生成第 `i` 个 token 时只能看 `i` 之前的 token。

如果训练时不加 mask，self-attention 会让所有位置互相看见，模型相当于在「抄答案」：训练 loss 会很低，但推理时根本用不了，因为推理时第 `i` 步根本没有 `i+1, i+2, ...` 的输入。

形象地说：**训练时的能见范围必须和推理时一致**，否则模型学到的能力在部署时根本派不上用场。

具体实现方式（§3.2.3 里讲）：在 softmax 之前，把「不允许看」的位置的 score 设成 `-inf`，softmax 后那些位置的权重就是 0。mask 矩阵长这样（`1` 表示要屏蔽）：

```text
0 1 1 1
0 0 1 1
0 0 0 1
0 0 0 0
```

第 `i` 行表示「位置 `i` 能看见谁」：上三角全是 1，意味着位置 `i` 看不见 `i+1, i+2, ...`。

### 句 16

> This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

**解读**

这个 mask，再加上「输出 embedding 偏移一位」（output shift right），就保证了：预测位置 `i` 时，模型只能依赖 `i` 之前已知的输出。

**补充理解**

「mask」和「右移一位」是**两件不同的事**，缺一不可。

**右移一位（shift right）** 解决的是「decoder 的输入到底是什么」：

```text
真实目标:     我    喜欢   猫    <EOS>
decoder 输入: <BOS> 我     喜欢  猫
预测目标:     我    喜欢   猫    <EOS>
```

可以看到，decoder 的输入是「目标序列右移一位、开头补 `<BOS>`」。这样位置 `i` 的输入就是「真实第 `i-1` 个 token」，预测目标是「真实第 `i` 个 token」——也就是「拿前一个词预测下一个词」。

**mask** 解决的是「能看见谁」：在 self-attention 里，位置 `i` 不能往后看到 `i+1, i+2, ...`。

两者一起，保证了一条非常硬的规则：

```text
预测位置 i 时，模型能用的信息 = 真实输出在位置 < i 的内容
```

这就是 GPT 等 decoder-only 模型训练的基本玩法：一次 forward 算所有位置的预测，每个位置都满足「只用前面信息」。这种「教师强制（teacher forcing）+ 因果 mask」的组合让 decoder 训练能完全并行化，是 Transformer 比 RNN 训练快的关键之一。

### 小结

这一节的主线可以压缩成这样：

**Encoder**：6 层堆叠，每层有两个 sub-layer（self-attention + FFN），每个 sub-layer 外面包 `Add & Norm`。

**Decoder**：也是 6 层，但每层比 encoder 多一个 cross-attention sub-layer，用来回头看 encoder 的输出。decoder 的 self-attention 还加了 causal mask，确保自回归生成时不会偷看未来 token。

而整个结构有一条贯穿始终的形状纪律：**所有模块输出都是 `d_model` 维，这是残差连接成立的前提**。

---

## 3.2.2 Multi-Head Attention

这一节回答的是一个关键设计问题：

> 为什么不做一次大的 attention，而要切成多个小的并行做？

### 核心结论

- 不做一次 `d_model` 维 attention，而是把 Q/K/V 投影 `h` 次，每次到更低维度（`d_k`、`d_v`）。
- 每组投影后的 Q/K/V 各自并行做 attention，得到 `h` 个 `d_v` 维输出。
- 把 `h` 个输出 concat 起来，再乘 `W^O` 投回 `d_model`。
- 多头让模型同时关注「不同表示子空间」和「不同位置」的信息。
- 单头 attention 只做一次加权平均，会把这些信息混成一份，丢失多视角能力。
- 由于每个 head 的维度被降低（`d_k = d_v = d_model / h`），多头总计算量与单头 full-dim 接近。
- 论文 base model：`h = 8`，`d_k = d_v = 64`。

### 句 17：不要只做一次 attention，而是投影 h 次

> Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions, respectively.

**解读**

不是只做一次「Q/K/V 都是 `d_model` 维」的 attention，而是用不同的可学习线性投影，把 Q、K、V **各投影 `h` 次**，分别投到 `d_k`、`d_k`、`d_v` 维度。

**补充理解**

把这句拆成两层意思：

**第一层：从单头到多头。** 单头做法是：

```text
Q (d_model), K (d_model), V (d_model) ─► Attention ─► output
```

多头做法是：

```text
QW_1^Q, KW_1^K, VW_1^V ─► head_1
QW_2^Q, KW_2^K, VW_2^V ─► head_2
...
QW_h^Q, KW_h^K, VW_h^V ─► head_h
```

每个 head 的投影矩阵不同，所以同一个 token 在不同 head 里会被看成不同的特征切面。

**第二层：为什么 Q 和 K 都是 `d_k`，V 是 `d_v`？** 这是 attention 公式的内在约束：

- Q 和 K 要做点积 `QK^T`，**点积要求两边维度相同**，所以 Q 和 K 都是 `d_k`。
- V 不参与 QK 点积，它只是被加权求和，所以可以是另一个维度 `d_v`。

实践中论文为了简化，让 `d_k = d_v = d_model / h = 64`，但理论上它们可以不同。

### 句 18：对每组投影后的 Q/K/V 并行做 attention

> On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding d_v-dimensional output values.

**解读**

在每一组投影后的 Q/K/V 上，模型**并行**地执行 attention 函数，每个 head 输出 `d_v` 维结果。

**补充理解**

「并行」是关键词：`h` 个 head **互不依赖**，可以同时计算。在 GPU 上这意味着可以打包成一个大矩阵乘法，效率几乎没有惩罚。

每个 head 都完整执行 §3.2.1 的 scaled dot-product attention：

```text
head_i = softmax( (QW_i^Q)(KW_i^K)^T / sqrt(d_k) ) (VW_i^V)
```

每个 head 都有自己一套独立的 scores / weights / output：

```text
head_i 的 scores:  (seq_len, seq_len)
head_i 的 weights: (seq_len, seq_len)
head_i 的 output:  (seq_len, d_v)
```

不同 head 的注意力分布通常不一样——这正是后面会强调的「不同位置、不同表示子空间」的来源。

### 句 19：把所有 head concat，再投影，得到最终输出

> These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

**解读**

把 `h` 个 head 的输出 **concat** 起来，再做**一次线性投影**（乘 `W^O`），就得到 multi-head attention 的最终输出。

**补充理解**

`concat` 是沿特征维拼接：

```text
head_1: (seq_len, d_v)
head_2: (seq_len, d_v)
...
head_h: (seq_len, d_v)

concat 后: (seq_len, h * d_v)
```

接下来乘 `W^O`：

```text
W^O: (h * d_v, d_model)
output = concat @ W^O   ─►   (seq_len, d_model)
```

`W^O` 有两个作用：

1. **混合 head 间信息**：每个 head 只在自己的子空间里看世界，`W^O` 让它们可以互相借用对方的发现。
2. **维度对齐**：把 `h * d_v` 投回 `d_model`，这样输出才能和残差相加。

注意：base model 里 `h * d_v = 8 * 64 = 512 = d_model`，所以 `W^O` 是 `(512, 512)` 的方阵——concat 后维度刚好对得上，但仍然需要这次投影来「拌匀」。

### 句 20：multi-head 让模型同时关注不同子空间、不同位置

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

**解读**

Multi-head attention 让模型可以**同时**从「不同表示子空间」、「不同位置」上汇集信息。

**补充理解**

这句话是 multi-head 的核心动机，可以拆成两部分：

| 关键词 | 含义 | 来源 |
|--------|------|------|
| different representation subspaces | 不同 head 的 `W_i^Q, W_i^K, W_i^V` 不同，把同一个 token 投影到不同的特征切面 | 投影矩阵不同 |
| different positions | 不同 head 的 attention 权重分布不同，可以关注序列里不同的 token | 权重学到的注意力模式不同 |

举个直觉例子（注意：是直觉，不一定真这么分工）：处理代词 `it` 时，

```text
某个 head 可能更关注前文的 the animal      ← 找指代对象
某个 head 可能更关注后面的 was tired       ← 找谓语
某个 head 可能更关注上一个修饰它的形容词    ← 找局部搭配
```

如果只有一个 head，它就只能学到一种关注模式，要么找指代要么找谓语，难以同时兼顾。多个 head 让模型有「分工」的可能。

### 句 21：单头 attention 的平均会抑制这种能力

> With a single attention head, averaging inhibits this.

**解读**

如果只有一个 attention head，「加权平均」会**抑制**上面那种多视角能力。

**补充理解**

为什么单头会有问题？因为 attention 本质上是一次加权求和：

```text
output = w_1 V_1 + w_2 V_2 + ... + w_n V_n
```

权重 `w_i` 只有一组分布。如果一个位置同时需要「指代信息 + 句法结构 + 局部搭配」，单头只能学一种妥协后的混合权重——**任何想关注多种关系的需求，都会被压成一个平均**。

多头的关键是：每个 head **独立**做一次加权平均，得到 `h` 份「不同视角的混合结果」，最后由 `W^O` 学习如何组合这些视角，而不是在 attention 阶段就把它们搅在一起。

可以类比拍照：

- 单头 = 用一个镜头拍一张照片，所有信息融合到一张画面
- 多头 = 用 `h` 个不同焦段的镜头各拍一张，最后在后期合成

后期合成显然比一次曝光保留的信息更丰富。

### 句 22：论文给出正式公式

> MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O, where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

**解读**

论文给出 multi-head attention 的正式公式：每个 head 是 Q/K/V 各自被 `W_i` 投影后做一次 attention；`h` 个 head concat 后再乘 `W^O` 得到最终输出。

**补充理解**

把这条公式和前面四句对照，正好是一一对应：

- 句 17 → `Q W_i^Q`、`K W_i^K`、`V W_i^V`（投影 `h` 次）
- 句 18 → `head_i = Attention(...)`（并行做 attention）
- 句 19 → `Concat(...) W^O`（拼接 + 投影）
- 句 20 / 21 → 公式背后的动机

整个流程其实就是「split → attend → concat → mix」四步：

```text
split:   X ─► Q_i, K_i, V_i  (h 份)
attend:  对每份独立做 attention  ─► head_i
concat:  把 h 个 head 拼起来
mix:     乘 W^O，让不同 head 的发现互相交融
```

理解了这四步，就抓住了 multi-head 的全部。

### 句 23：投影矩阵的形状

> Where the projections are parameter matrices W_i^Q ∈ R^{d_model × d_k}, W_i^K ∈ R^{d_model × d_k}, W_i^V ∈ R^{d_model × d_v} and W^O ∈ R^{h·d_v × d_model}.

**解读**

论文明确每个投影矩阵的形状：

```text
W_i^Q: (d_model, d_k)
W_i^K: (d_model, d_k)
W_i^V: (d_model, d_v)
W^O:   (h * d_v, d_model)
```

**补充理解**

这些都是模型参数，随机初始化、由训练学习。形状决定了维度怎么变：

```text
X        : (seq_len, d_model)
X W_i^Q  : (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)   ✓
```

`W^O` 的形状特别值得注意：输入维度是 `h * d_v`（concat 后的维度），输出是 `d_model`。base model 里两者都是 512，刚好是方阵，但意义上仍然是「从 `h` 个 head 的拼接空间投回主表示空间」。

**为什么不直接让每个 head 输出就是 `d_model`？** 因为那样会让总参数量和计算量乘以 `h` 倍。论文的设计是：每个 head 的维度降为 `d_model / h`，`h` 个加起来还是 `d_model`，**总宽度不变**。

在代码里通常把 `h` 个 head 的投影矩阵合成一个大矩阵：`W_Q = [W_1^Q | W_2^Q | ... | W_h^Q]`，形状 `(d_model, d_model)`，因为 `d_model = h * d_k`，两种写法完全等价。

### 句 24：使用 8 个 head

> In this work we employ h = 8 parallel attention layers, or heads.

**解读**

论文 base model 用了 `h = 8` 个并行 attention layer，也就是 8 个 head。

**补充理解**

这里 `parallel attention layers` 和 `heads` 是同义词——「层」在这里指「一个独立的 attention 计算」，而不是「encoder layer」。术语容易混，注意区分。

`8` 不是什么神圣数字。论文 §6.2 Table 3 row (A) 的消融实验里试过 `h = 1, 4, 8, 16, 32`，发现：

- `h = 1`（单头）效果明显差（BLEU −0.9）
- 中间几档差异不大
- `h = 32` 反而轻微下降

也就是说「太少不行，太多也不一定好」。背后的直觉是：`h` 增大时每个 head 的 `d_k` 必须减小，太小的 `d_k` 会让单个 head 的表达能力下降，最终得不偿失。

### 句 25：每个 head 使用 d_k = d_v = 64

> For each of these we use d_k = d_v = d_model / h = 64.

**解读**

每个 head 的 `d_k` 和 `d_v` 都设成 `d_model / h`，base model 里就是 `512 / 8 = 64`。

**补充理解**

这一句把 `d_model`、`h`、`d_k`、`d_v` 四个超参绑定到一起：

```text
d_model = 512
h = 8
d_k = d_v = d_model / h = 64
```

满足 `h * d_v = d_model`，所以 concat 后维度刚好回到 `d_model = 512`。

为什么这么绑？两个原因：

1. **形状对齐**：concat 后正好是 `d_model`，`W^O` 也成方阵，工程上最干净。
2. **参数和计算量守恒**：相比单头 full-dim attention，多头总参数量和 FLOPs 几乎不变（下一句正式说）。

理论上 `d_k` 和 `d_v` 可以解耦、可以不等于 `d_model / h`，但那样要么参数变多，要么形状对不齐，需要额外补一次投影。论文选了最经济的方案。

### 句 26：多头总计算量和单头 full-dim 接近

> Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

**解读**

因为每个 head 的维度被降低了，多头 attention 的**总计算量**和「单头 full-dim attention」差不多。

**补充理解**

这句话经常被误读成「多头免费」，其实它说的是：**和「单头但用 `d_model` 维度做 attention」相比，多头并没有显著变贵**。直觉验算：

- 单头 full-dim：1 个 attention，`d_model = 512`
- 多头：`h = 8` 个 attention，每个 `d_k = 64`

对比 attention 主要成本（点积部分）：

```text
单头：seq_len × seq_len × d_model = n² × 512
多头：8 × (n² × 64) = n² × 512   ← 总和相同
```

也就是说，把「一个胖 attention」切成「`h` 个瘦 attention 并行」，总宽度仍是 `d_model`。多头**不是免费**，但**没有比单头 full-dim 贵**。

需要警惕的常见误解：

- ❌ 「8 个 head 比 1 个 head 慢 8 倍」——错，每个 head 维度也降到 `1/h`
- ❌ 「多头是免费提升」——错，它和「单头 full-dim」总量相当
- ✅ 「在不显著增加成本的前提下，多头换来了多视角能力」

### 小结

这一节最重要的理解不是记住公式，而是记住这条逻辑链：

1. 不做一次大的 attention，而是切成 `h` 个小的并行做。
2. 每个 head 用自己的投影矩阵，看到不同的特征切面。
3. concat 后再用 `W^O` 把信息混合回来，投回 `d_model`。
4. 这么做的收益是多视角能力，而代价几乎没有增加（总宽度不变）。

---

## 最后用一句话串起来

`§3.1` 解决的是「Transformer 的 encoder / decoder 分别长什么样，用了哪些训练稳定技巧（残差 + LayerNorm），以及怎么保证自回归（causal mask + 右移）」。

`§3.2.2` 解决的是「为什么不做一次大 attention，而要拆成多个小 attention 并行做再合并」。

两节合起来，你就能画出 Transformer 的完整 block 结构：

`Multi-Head Attention → Add & Norm → FFN → Add & Norm`（encoder 每层重复这个；decoder 每层在中间再插一个 cross-attention + Add & Norm）。
