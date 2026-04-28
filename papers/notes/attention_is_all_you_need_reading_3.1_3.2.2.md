# Attention Is All You Need 逐句精读：§3.1 与 §3.2.2

> 配合第二阶段第 3 课 `phase2-transformer/03_multi_head_attention.py` 一起读。
>
> 论文：Vaswani et al., 2017, *Attention Is All You Need*  
> 章节：`3.1 Encoder and Decoder Stacks`、`3.2.2 Multi-Head Attention`  
> 本地 PDF：[`../core-transformers/Attention_Is_All_You_Need_2017.pdf`](../core-transformers/Attention_Is_All_You_Need_2017.pdf)  
> arXiv：<https://arxiv.org/abs/1706.03762>

这篇只做一件事：**按论文原句顺序逐句理解**。

为了避免变成机械全文翻译，下面每一句都用这个格式：

```text
论文句意
解读
代码 / 维度回扣
```

其中“论文句意”是对原句的中文转述，不是逐字翻译。必要时只摘很短的英文关键词，比如 `stack of identical layers`、`Add & Norm`、`heads`。

---

## 先抓主线

这两节只要读懂 4 件事：

1. Encoder / Decoder 都是很多层堆起来的，不是一层 attention。
2. 每个 sub-layer 外面都有残差连接和 LayerNorm。
3. 原始 Transformer 用的是 Post-Norm：`LayerNorm(x + Sublayer(x))`。
4. Multi-Head Attention 是多个 attention head 并行，concat 后再乘 `W^O`。

---

## 0. 前置：§3 Model Architecture 的两句铺垫

论文在进入 §3.1 前，先用两句话定义了 encoder-decoder 的整体任务。读 §3.1 之前最好先理解它。

### 句 0.1：大多数 sequence transduction 模型都有 encoder-decoder 结构

**论文句意**

很多序列转换模型都会用 encoder-decoder：encoder 把输入符号序列变成连续表示，decoder 再基于这些表示生成输出符号序列。

**解读**

这里的 `sequence transduction` 可以先理解成“把一个序列变成另一个序列”。

典型例子是机器翻译：

```text
英文句子 -> 中文句子
```

Encoder 做的是“读入并理解源句子”：

```text
input tokens -> continuous representations
```

Decoder 做的是“根据理解结果生成目标句子”：

```text
encoder representations -> output tokens
```

这里的 `continuous representations` 就是连续向量，不再是离散 token id。

**代码 / 维度回扣**

进入 Transformer 前，token 已经先变成 embedding：

```text
token ids:  (seq_len,)
embedding:  (seq_len, d_model)
```

Encoder 接收的是 embedding 后的向量序列。

---

### 句 0.2：Decoder 是 auto-regressive 的

**论文句意**

Decoder 每一步生成一个符号，并且生成当前位置时，会把之前已经生成的符号作为额外输入。

**解读**

`auto-regressive` 的意思是：一个一个往后生成。

比如生成：

```text
我 喜欢 猫
```

过程是：

```text
看到 <BOS>        -> 预测 我
看到 <BOS> 我     -> 预测 喜欢
看到 <BOS> 我 喜欢 -> 预测 猫
```

所以 decoder 不能提前看到未来词。否则它不是在“预测”，而是在“抄答案”。

**代码 / 维度回扣**

这就是后面 causal mask 的动机：

```python
mask = np.triu(np.ones((seq_len, seq_len)), k=1)
```

上三角代表未来位置，要屏蔽。

---

### 句 0.3：Transformer 沿用 encoder-decoder 结构，但内部换成 self-attention 和 FFN

**论文句意**

Transformer 仍然采用 encoder-decoder 大框架，只是 encoder 和 decoder 里面使用堆叠的 self-attention 和逐位置全连接层。

**解读**

这句话是整篇论文的架构宣言：

```text
外壳：还是 encoder-decoder
内部：不用 RNN / CNN，改成 attention + FFN
```

所以 Transformer 不是把所有旧概念全推翻。它保留了“encoder 读输入、decoder 生成输出”的经典框架，但把每层的核心计算换成了 attention。

**代码 / 维度回扣**

第 3 课写的是 attention 部分：

```python
output, head_weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads)
```

第 4 课会把 attention 和 FFN 拼成完整 block。

---

## 1. §3.1 Encoder and Decoder Stacks 逐句精读

这一节分成两段：先讲 encoder，再讲 decoder。

---

## 1.1 Encoder 段

### 句 1：Encoder 由一叠相同结构的层组成

**论文句意**

Encoder 是由 `N = 6` 个相同结构的 layer 堆叠起来的。

**解读**

这里的关键词是：

```text
stack of identical layers
```

`stack` 表示“堆叠”。Transformer 不是只做一次 attention，而是把类似结构重复很多次。

`identical layers` 表示“结构相同”，不是“参数共享”。第 1 层和第 2 层都有 attention + FFN，但它们的权重参数各自独立。

可以画成：

```text
X0
 │
 ▼
EncoderLayer 1
 │
 ▼
EncoderLayer 2
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

**代码 / 维度回扣**

第 3 课目前只是在实现单个 attention 子层。真正堆叠 block 的形态会更像：

```python
for layer in layers:
    x = layer(x)
```

每层输入输出都保持：

```text
(seq_len, d_model)
```

---

### 句 2：每个 encoder layer 有两个 sub-layer

**论文句意**

Encoder 的每一层里面有两个子层。

**解读**

这里的 `sub-layer` 可以理解成“Transformer layer 内部的一个功能模块”。

Encoder layer 不是一个整体黑盒，它里面有两个模块：

```text
EncoderLayer
├── sub-layer 1: Multi-Head Self-Attention
└── sub-layer 2: Feed-Forward Network
```

这一点很重要，因为后面论文说“每个 sub-layer 外面都有 residual + LayerNorm”，不是只在整个 EncoderLayer 外面包一次。

**代码 / 维度回扣**

一个 encoder layer 的抽象结构是：

```text
x -> attention sub-layer -> FFN sub-layer -> output
```

其中每个 sub-layer 都要保证输入输出形状一致：

```text
(seq_len, d_model) -> (seq_len, d_model)
```

---

### 句 3：第一个 sub-layer 是 multi-head self-attention

**论文句意**

第一个子层是多头自注意力机制。

**解读**

这句话里有两个概念：

```text
multi-head
self-attention
```

`self-attention` 表示 Q、K、V 都来自同一个序列。

如果当前 encoder 输入是：

```text
X: (seq_len, d_model)
```

那么：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

每个 token 都可以看同一句子里的所有 token。

`multi-head` 表示不是只做一组 Q/K/V attention，而是分成多个 head 并行做。这个在 §3.2.2 详细讲。

**代码 / 维度回扣**

第 3 课代码：

```python
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
```

因为 Q/K/V 都来自 `X`，所以这是 self-attention。

---

### 句 4：第二个 sub-layer 是逐位置的全连接前馈网络

**论文句意**

第二个子层是一个简单的、逐位置应用的全连接前馈网络。

**解读**

`position-wise` 的意思是：对每个 token 位置单独做同一个 MLP。

假设序列长度是 4：

```text
x0 -> FFN -> y0
x1 -> FFN -> y1
x2 -> FFN -> y2
x3 -> FFN -> y3
```

它不在不同位置之间混合信息。位置之间的信息交流已经由 self-attention 完成；FFN 负责对每个位置的表示做非线性加工。

论文 §3.3 里的 FFN 公式是：

```text
FFN(x) = max(0, xW1 + b1)W2 + b2
```

**代码 / 维度回扣**

FFN 内部会升维再降维：

```text
d_model -> d_ff -> d_model
512     -> 2048 -> 512
```

最后仍要回到 `d_model`，因为后面要接残差。

---

### 句 5：每个 sub-layer 外面都有残差连接，后面接 LayerNorm

**论文句意**

论文在两个子层外面都使用 residual connection，然后再做 layer normalization。

**解读**

这句就是 Figure 1 里的 `Add & Norm`。

它的顺序是：

```text
先 Sublayer
再 Add residual
再 LayerNorm
```

公式：

```text
output = LayerNorm(x + Sublayer(x))
```

所以 encoder layer 更完整地写成：

```text
x1 = LayerNorm(x0 + MultiHeadSelfAttention(x0))
x2 = LayerNorm(x1 + FFN(x1))
```

这里的 `x + ...` 是残差连接。它让原始输入有一条直接通路，不必所有信息都穿过 attention 或 FFN。

**这里可以理解为“后 Norm”吗？**

可以。原始 Transformer 是 **Post-Norm / 后 Norm**：

```text
Norm 放在 Sublayer 和 Add 后面
```

对比现代 GPT 常用的 **Pre-Norm / 前 Norm**：

```text
Post-Norm: y = LayerNorm(x + Sublayer(x))
Pre-Norm:  y = x + Sublayer(LayerNorm(x))
```

**代码 / 维度回扣**

第 3 课里的 Post-Norm：

```python
def post_norm_block(x, W_Q, W_K, W_V, W_O, n_heads):
    attn_out, _ = multi_head_attention(x, W_Q, W_K, W_V, W_O, n_heads)
    return layer_norm(x + attn_out)
```

这就是论文这句话的代码形态。

---

### 句 6：论文把 sub-layer 输出写成统一公式

**论文句意**

每个子层的输出都可以写成 `LayerNorm(x + Sublayer(x))`，其中 `Sublayer` 是当前子层自己实现的函数。

**解读**

`Sublayer` 不是一个具体模块名，而是一个占位符。

在不同位置，它代表不同函数：

| 位置 | `Sublayer(x)` 代表什么 |
|------|------------------------|
| Encoder 第一个子层 | `MultiHeadSelfAttention(x)` |
| Encoder 第二个子层 | `FFN(x)` |
| Decoder 第一个子层 | `MaskedMultiHeadSelfAttention(x)` |
| Decoder 第二个子层 | `CrossAttention(x, encoder_output)` |
| Decoder 第三个子层 | `FFN(x)` |

所以论文用一个公式统一描述所有 Add & Norm：

```text
LayerNorm(x + Sublayer(x))
```

**代码 / 维度回扣**

如果当前 sub-layer 是 attention：

```python
out = layer_norm(x + attention(x))
```

如果当前 sub-layer 是 FFN：

```python
out = layer_norm(x + ffn(x))
```

外壳一样，里面的函数不同。

---

### 句 7：为了能做残差连接，所有 sub-layer 和 embedding 都输出 `d_model` 维

**论文句意**

为了让这些残差连接成立，模型里的所有子层以及 embedding 层都输出 `d_model` 维。

**解读**

残差连接要做加法：

```text
x + Sublayer(x)
```

加法要求两边形状相同。

如果：

```text
x:           (seq_len, d_model)
Sublayer(x): (seq_len, ???)
```

那么 `???` 必须等于 `d_model`。

这就是 Transformer 里很重要的形状纪律：

```text
模块内部可以分头、升维、降维；
模块输出必须回到 d_model。
```

**代码 / 维度回扣**

第 3 课里：

```python
seq_len = 6
d_model = 16
n_heads = 4
d_head = d_model // n_heads
```

多头内部拆成：

```text
4 个 head，每个 4 维
```

concat 后回到：

```text
4 * 4 = 16 = d_model
```

最后才能：

```python
residual_output = X + attn_output
```

---

## 1.2 Decoder 段

### 句 8：Decoder 也由一叠相同结构的层组成

**论文句意**

Decoder 和 encoder 一样，也是由 `N = 6` 个相同结构的 layer 堆叠起来。

**解读**

Decoder 也不是一层，而是一串层：

```text
Y0
 │
 ▼
DecoderLayer 1
 │
 ▼
DecoderLayer 2
 │
 ▼
...
 │
 ▼
DecoderLayer 6
```

每一层结构相同，但参数不共享。

**代码 / 维度回扣**

Decoder 每层输入输出也保持：

```text
(target_seq_len, d_model)
```

这样才能连续堆叠多层。

---

### 句 9：Decoder 在 encoder 的两个 sub-layer 基础上，插入第三个 sub-layer

**论文句意**

除了 encoder layer 里的两个子层，decoder 还额外插入第三个子层。

**解读**

Encoder layer：

```text
1. Self-Attention
2. FFN
```

Decoder layer：

```text
1. Masked Self-Attention
2. Encoder-Decoder Attention
3. FFN
```

多出来的是 `Encoder-Decoder Attention`，也常叫 `Cross-Attention`。

为什么需要它？

因为 decoder 生成目标句子时，需要回头查看 encoder 对源句子的表示。

**代码 / 维度回扣**

如果是翻译：

```text
encoder_output: 英文句子的表示
decoder_state:  当前已生成中文前缀的表示
```

Cross-attention 让 decoder_state 去查询 encoder_output。

---

### 句 10：第三个 sub-layer 对 encoder stack 的输出做 multi-head attention

**论文句意**

Decoder 新增的这个子层，会对 encoder stack 的输出执行 multi-head attention。

**解读**

这是 cross-attention 的关键：

```text
Q 来自 decoder
K 来自 encoder
V 来自 encoder
```

也就是：

```text
Q = decoder_hidden W_Q
K = encoder_output W_K
V = encoder_output W_V
```

Self-attention 是“自己看自己”：

```text
Q/K/V 都来自同一个 X
```

Cross-attention 是“decoder 看 encoder”：

```text
Q 来自 decoder，K/V 来自 encoder
```

**代码 / 维度回扣**

第 3 课代码只实现了 self-attention 风格：

```python
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
```

如果改成 cross-attention，会变成类似：

```python
Q = decoder_x @ W_Q
K = encoder_out @ W_K
V = encoder_out @ W_V
```

---

### 句 11：Decoder 每个 sub-layer 也使用 residual connection 和 LayerNorm

**论文句意**

和 encoder 一样，decoder 的每个子层外面也有残差连接，然后接 LayerNorm。

**解读**

Decoder layer 可以写成：

```text
x1 = LayerNorm(x0 + MaskedSelfAttention(x0))
x2 = LayerNorm(x1 + CrossAttention(x1, encoder_output))
x3 = LayerNorm(x2 + FFN(x2))
```

Encoder 每层有 2 次 Add & Norm；Decoder 每层有 3 次 Add & Norm。

**代码 / 维度回扣**

残差要求每个子层输出都和输入同形：

```text
(target_seq_len, d_model)
```

Cross-attention 虽然看的是 encoder output，但输出长度要跟 decoder query 长度一致：

```text
Q length = target_seq_len
K/V length = source_seq_len
output length = target_seq_len
```

---

### 句 12：Decoder 的 self-attention 被修改，防止当前位置关注后续位置

**论文句意**

论文修改了 decoder stack 里的 self-attention，让某个位置不能关注它后面的 token。

**解读**

这就是 causal mask。

如果目标序列是：

```text
<BOS> 我 喜欢 猫
```

那么位置 2 预测时只能看：

```text
<BOS>, 我, 喜欢
```

不能看后面的：

```text
猫
```

否则训练时就泄露答案。

**代码 / 维度回扣**

mask 矩阵：

```text
0 1 1 1
0 0 1 1
0 0 0 1
0 0 0 0
```

在第 3 课代码里，`1` 表示屏蔽：

```python
mask = np.triu(np.ones((seq_len, seq_len)), k=1)
```

---

### 句 13：mask 加上输出 embedding 右移，保证预测只能依赖已知输出

**论文句意**

这种 mask，再加上输出 embedding 偏移一位，可以保证预测位置 `i` 时，只依赖 `i` 之前的已知输出。

**解读**

训练 decoder 时，输入和目标通常这样错开：

```text
目标句子: 我    喜欢   猫    <EOS>
输入序列: <BOS> 我     喜欢  猫
预测目标: 我    喜欢   猫    <EOS>
```

这叫右移，也就是 `shift right`。

右移解决“输入是什么”的问题：

```text
用前一个 token 预测下一个 token
```

mask 解决“能看见谁”的问题：

```text
不能看到当前位置之后的 token
```

两者结合，保证自回归生成规则成立。

**代码 / 维度回扣**

第 3 课只练 mask，没有实现完整 decoder 输入右移。但理解 GPT 时会反复遇到这个逻辑：

```text
input_ids[:, :-1] -> logits
labels[:, 1:]     -> loss target
```

---

## 2. §3.2.2 Multi-Head Attention 逐句精读

这一节开始解释 Multi-Head Attention 的公式和动机。

---

### 句 14：Figure 2 左边是 scaled dot-product attention，右边是 multi-head attention

**论文句意**

论文用 Figure 2 展示：左边是单个 scaled dot-product attention，右边是多个 attention layer 并行组成的 multi-head attention。

**解读**

Multi-head 不是串行：

```text
head1 -> head2 -> head3
```

而是并行：

```text
head1 ┐
head2 ├── concat -> W^O
head3 ┘
```

每个 head 都独立算 attention，最后把结果合并。

**代码 / 维度回扣**

第 3 课用循环写并行逻辑：

```python
for h in range(n_heads):
    head_output, head_weights = single_head_attention(Q[h], K[h], V[h], mask)
```

真实深度学习框架里通常会把所有 head 放进一个张量批量计算。

---

### 句 15：不要只用一次 `d_model` 维 attention，而是把 Q/K/V 线性投影 `h` 次

**论文句意**

相比只做一次完整维度的 attention，论文发现更好的做法是：用不同的可学习线性投影，把 queries、keys、values 分别投影 `h` 次。

**解读**

单头做法：

```text
Q, K, V -> Attention -> output
```

多头做法：

```text
QW_1^Q, KW_1^K, VW_1^V -> head_1
QW_2^Q, KW_2^K, VW_2^V -> head_2
...
QW_h^Q, KW_h^K, VW_h^V -> head_h
```

每个 head 的投影矩阵不同，所以同一个 token 在不同 head 里会被看成不同的特征切面。

**代码 / 维度回扣**

代码里一次性算出所有 head 的投影：

```python
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
```

然后再 reshape 切成多个 head：

```python
Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
```

---

### 句 16：Q/K 被投影到 `d_k` 维，V 被投影到 `d_v` 维

**论文句意**

每个 head 里，query 和 key 的投影维度是 `d_k`，value 的投影维度是 `d_v`。

**解读**

Q 和 K 必须同维，因为要做点积：

```text
QK^T
```

如果 Q 是 64 维，K 也必须是 64 维。

V 可以是另一个维度 `d_v`，因为 V 不参与 QK 点积。它只在最后被 attention weights 加权求和：

```text
weights: (seq_len, seq_len)
V:       (seq_len, d_v)
output:  (seq_len, d_v)
```

**代码 / 维度回扣**

第 3 课代码为了简化，令：

```text
d_k = d_v = d_head
```

代码变量：

```python
d_head = d_model // n_heads
```

---

### 句 17：对每一组投影后的 Q/K/V 并行执行 attention

**论文句意**

在每个投影版本的 Q/K/V 上，模型并行执行 attention 函数。

**解读**

每个 head 都完整执行第 2 课公式：

```text
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

展开就是：

```text
head_i = softmax((QW_i^Q)(KW_i^K)^T / sqrt(d_k)) (VW_i^V)
```

所以每个 head 都有自己的：

```text
scores
weights
output
```

**代码 / 维度回扣**

```python
head_output, head_weights = single_head_attention(Q[h], K[h], V[h], mask)
```

`Q[h]`、`K[h]`、`V[h]` 是第 `h` 个 head 的 Q/K/V。

---

### 句 18：每个 head 产生 `d_v` 维输出

**论文句意**

每个 attention head 会输出 `d_v` 维的结果。

**解读**

单个 head：

```text
head_i: (seq_len, d_v)
```

如果有 `h` 个 head：

```text
head_1: (seq_len, d_v)
head_2: (seq_len, d_v)
...
head_h: (seq_len, d_v)
```

这些 head 还不是最终输出，后面要 concat。

**代码 / 维度回扣**

第 3 课：

```text
seq_len = 6
d_model = 16
n_heads = 4
d_head = 4
```

所以每个 head 输出：

```text
(6, 4)
```

---

### 句 19：把所有 head concat，然后再次投影，得到最终输出

**论文句意**

这些 head 的输出会被拼接起来，然后再做一次线性投影，得到最终的 multi-head attention 输出。

**解读**

`concat` 是 `concatenate` 的缩写，意思是拼接。

如果有 4 个 head，每个 head 输出 4 维：

```text
head_1 = [a1, a2, a3, a4]
head_2 = [b1, b2, b3, b4]
head_3 = [c1, c2, c3, c4]
head_4 = [d1, d2, d3, d4]
```

concat 后：

```text
[a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4]
```

形状变化：

```text
4 个 (seq_len, 4)
-> 1 个 (seq_len, 16)
```

然后再乘输出矩阵 `W^O`：

```text
Concat(head_1, ..., head_h) W^O
```

`W^O` 的作用是混合不同 head 的信息，并把输出保持在 `d_model` 维。

**代码 / 维度回扣**

```python
concat = np.concatenate(all_head_outputs, axis=-1)
output = concat @ W_O
```

`axis=-1` 表示沿最后一维拼接，也就是沿特征维拼接。

---

### 句 20：Multi-head 让模型同时关注不同表示子空间、不同位置的信息

**论文句意**

Multi-head attention 让模型可以同时从不同表示子空间、不同位置上关注信息。

**解读**

这句话是 multi-head 的核心动机。

拆成两部分：

| 说法 | 意思 |
|------|------|
| different representation subspaces | 不同 head 的投影矩阵不同，看到的是不同特征切面 |
| different positions | 不同 head 的 attention weights 可以关注不同 token |

比如处理 `it` 时：

```text
某个 head 可能更关注 animal
某个 head 可能更关注 because
某个 head 可能更关注 was / tired
```

注意：这只是帮助理解的例子，不代表训练后一定按人类命名的语法功能分工。

**代码 / 维度回扣**

第 3 课会打印每个 head 的注意力模式：

```python
for h, weights in enumerate(head_weights):
    print(f"Head {h}: ...")
```

如果不同 head 的权重分布不同，就说明它们在关注不同位置。

---

### 句 21：单头 attention 的平均会抑制这种能力

**论文句意**

如果只有一个 attention head，所有信息会被一次加权平均混在一起，这会限制模型同时捕捉多种关系的能力。

**解读**

单头 attention 对每个 token 只有一个注意力分布：

```text
[0.1, 0.2, 0.6, 0.1]
```

最后只能得到一个混合结果：

```text
0.1V1 + 0.2V2 + 0.6V3 + 0.1V4
```

如果一个 token 同时需要语义指代、局部搭配、句法结构等信息，一次平均可能会把这些关系搅在一起。

多头则是多次独立平均：

```text
head_1: 一套注意力分布
head_2: 另一套注意力分布
head_3: 另一套注意力分布
...
```

最后再由 `W^O` 学习如何融合。

**代码 / 维度回扣**

这就是为什么 `all_head_outputs` 是一个列表：

```python
all_head_outputs.append(head_output)
```

每个 head 保留自己的输出，先不混在一起。

---

### 句 22：论文给出投影矩阵的形状

**论文句意**

论文说明，每个 head 的投影矩阵是 `W_i^Q`、`W_i^K`、`W_i^V`，最后还有一个输出投影矩阵 `W^O`。

**解读**

形状是：

```text
W_i^Q: (d_model, d_k)
W_i^K: (d_model, d_k)
W_i^V: (d_model, d_v)
W^O:   (h * d_v, d_model)
```

这些矩阵都是模型参数，随机初始化，然后通过训练学出来。

**代码 / 维度回扣**

代码里没有显式写 `W_1^Q`、`W_2^Q`，而是把所有 head 的投影矩阵合成一个大矩阵：

```text
W_Q = [W_1^Q | W_2^Q | ... | W_h^Q]
```

所以：

```python
W_Q = np.random.randn(d_model, d_model) * 0.1
```

因为：

```text
d_model = h * d_head
```

---

### 句 23：原始 Transformer 使用 8 个 head

**论文句意**

在这项工作中，论文使用 `h = 8` 个并行 attention layer，也就是 8 个 head。

**解读**

这是原始 Transformer base model 的配置：

```text
h = 8
```

不是说 8 是永远最优，而是论文 base model 采用这个设置。

论文后面的消融实验也比较过不同 head 数，结果不是简单“越多越好”。

**代码 / 维度回扣**

第 3 课默认：

```python
n_heads = 4
```

你可以改成：

```python
n_heads = 1
n_heads = 2
n_heads = 8
```

但要保证：

```text
d_model % n_heads == 0
```

---

### 句 24：每个 head 使用 `d_k = d_v = d_model / h = 64`

**论文句意**

每个 head 的 key/query 维度和值维度都设为 64，也就是 `d_model / h`。

**解读**

原始论文 base model：

```text
d_model = 512
h = 8
d_k = d_v = 64
```

因为：

```text
512 / 8 = 64
```

这样 concat 后刚好回到：

```text
8 * 64 = 512
```

**代码 / 维度回扣**

第 3 课是缩小版：

```text
d_model = 16
n_heads = 4
d_head = 16 / 4 = 4
```

逻辑完全一样。

---

### 句 25：由于每个 head 维度降低，总计算成本接近单头 full-dimensional attention

**论文句意**

因为每个 head 的维度被降低了，所以多头 attention 的总计算成本，和一个完整维度的单头 attention 差不多。

**解读**

一个常见误解是：

```text
8 个 head = 计算量变 8 倍
```

论文这里说的正好相反：每个 head 变窄，所以总成本不会简单乘 8。

对 base model：

```text
单头：1 个 512 维 attention
多头：8 个 64 维 attention
```

总宽度仍然是 512。

**代码 / 维度回扣**

第 3 课：

```text
单头 full dim: 1 * 16
多头:          4 * 4 = 16
```

宽度总量没变，只是切成 4 份并行处理。

---

## 3. 把两节连成一条数据流

### Encoder layer

```text
x
│
├─ MultiHeadSelfAttention(x)
│
├─ Add: x + attention_output
│
├─ LayerNorm
│
├─ FFN
│
├─ Add
│
└─ LayerNorm
```

公式：

```text
x1 = LayerNorm(x0 + MultiHeadSelfAttention(x0))
x2 = LayerNorm(x1 + FFN(x1))
```

### Decoder layer

```text
x
│
├─ MaskedSelfAttention
├─ Add & Norm
├─ CrossAttention over encoder output
├─ Add & Norm
├─ FFN
└─ Add & Norm
```

公式：

```text
x1 = LayerNorm(x0 + MaskedSelfAttention(x0))
x2 = LayerNorm(x1 + CrossAttention(x1, encoder_output))
x3 = LayerNorm(x2 + FFN(x2))
```

### Multi-Head Attention

```text
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

代码对应：

```python
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

for h in range(n_heads):
    head_output, head_weights = single_head_attention(Q[h], K[h], V[h], mask)
    all_head_outputs.append(head_output)

concat = np.concatenate(all_head_outputs, axis=-1)
output = concat @ W_O
```

---

## 4. 读完必须能回答

1. `stack of identical layers` 里的 identical 是结构相同还是参数共享？
2. Encoder 每层两个 sub-layer 分别是什么？
3. Decoder 为什么比 encoder 多一个 sub-layer？
4. `LayerNorm(x + Sublayer(x))` 为什么叫 Post-Norm？
5. 为什么残差连接要求 sub-layer 输出 `d_model`？
6. Self-attention 和 cross-attention 的 Q/K/V 来源有什么区别？
7. Multi-head 里的 `concat` 是沿哪个维度拼接？
8. 为什么多头不是让计算量简单乘以 head 数？

---

## 5. 一句话总结

§3.1 告诉你：Transformer 的 encoder / decoder 是多层堆叠结构，每个子层都包着 residual + LayerNorm，原始论文采用 Post-Norm。

§3.2.2 告诉你：Multi-Head Attention 把 Q/K/V 投影到多个较小子空间并行做 attention，再 concat 并用 `W^O` 投回 `d_model`，这样既能看不同位置和不同表示子空间，又能保持残差连接需要的形状。
