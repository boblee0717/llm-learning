# Attention Is All You Need 论文精读

本篇主要针对论文的 **3.4 Embeddings and Softmax** 和 **3.5 Positional Encoding** 两个核心章节进行逐句/逐段的深度解析。

---

## 3.4 Embeddings and Softmax (词嵌入与 Softmax)

这一章节主要介绍了模型如何处理输入、输出数据，以及如何生成最终的概率预测。

### 逐句解析

**原文 1：**
> Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$.

**解读：**
与其他序列转换模型类似，我们使用学习到的词嵌入（Learned Embeddings）将输入词元和输出词元转换为维度为 $d_{model}$（在本文中默认值为512）的向量。
*注：这是 NLP 任务的标准操作，通过嵌入层将离散的单词（词索引）映射到连续的稠密向量空间中，使得模型可以进行后续的数值计算，并捕捉词汇之间的语义相似性。*

**原文 2：**
> We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

**解读：**
我们同样使用了常见的可学习的线性变换（Linear Transformation）和 Softmax 函数，将解码器最终的输出转化为预测“下一个词元”的概率分布。
*注：解码器最后输出的是隐状态向量。线性层的作用是把这个 $d_{model}$ 维度的向量映射到“词表大小（Vocabulary Size）”的维度，随后 Softmax 将其转换为和为 1 的概率分布，从而挑选出概率最大的词作为输出。*

**原文 3：**
> In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30].

**解读：**
在我们的模型中，输入嵌入层、输出嵌入层以及 Softmax 前的线性变换层这三者**共享了同一个权重矩阵**（参考了论文[30]）。
*注：这是一个非常关键的工程技巧（Weight Tying）。因为这三个层都涉及到在“词表维度”和“$d_{model}$ 维度”之间的转换，共享权重不仅可以大幅度减少模型的参数量，防止过拟合，还能让词嵌入在解码预测时得到更好的更新和训练。*

**原文 4：**
> In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.

**解读：**
在嵌入层中，我们将这些权重乘以 $\sqrt{d_{model}}$。
*注：为什么要乘以这个缩放因子？因为在后续步骤中（3.5节），词嵌入向量需要与“位置编码（Positional Encoding）”直接相加。位置编码是基于正余弦函数生成的，其数值范围是 $[-1, 1]$。如果不放大词嵌入向量，它本身的信息可能会被位置编码掩盖。通过乘以 $\sqrt{d_{model}}$（对于512维度来说大约是22.6），可以放大嵌入向量的尺度（方差），平衡两者在相加时的权重。*

---

## 3.5 Positional Encoding (位置编码)

由于 Transformer 完全抛弃了 RNN 和 CNN，它本质上失去了感知序列顺序的能力。这一节解释了如何巧妙地把“位置信息”重新注入给模型。

### 第一段：为什么需要位置编码以及如何注入

**原文：**
> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

**解读：**
由于我们的模型既不包含循环神经网络（RNN）也没有卷积神经网络（CNN），为了让模型能够利用序列的顺序信息，我们必须向序列的词元中注入关于“相对或绝对位置”的信息。为此，我们在编码器和解码器栈底部的输入嵌入中加入了“位置编码（Positional Encodings）”。
*注：自注意力机制（Self-Attention）本身是“位置无关（Permutation Invariant）”的。打乱一句话的语序，输出的特征聚合结果其实是一样的。因此，必须在数据进入第一层自注意力之前，就显式地告诉模型“谁在前面，谁在后面”。*

**原文：**
> The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].

**解读：**
位置编码的维度与词嵌入的维度 $d_{model}$ 相同，因此这两者可以直接相加（Summed）。位置编码有许多种选择，包括可学习的位置编码和固定的位置编码。
*注：这里选择“相加”而不是“拼接（Concat）”，主要是为了不增加模型的特征维度，从而节省计算量。由于高维空间极其稀疏，相加能在不丢失太多原本语义信息的情况下将位置信息混合进去。*

### 第二段：正余弦位置编码的数学公式

**原文：**
> In this work, we use sine and cosine functions of different frequencies:
> $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
> $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$
> where $pos$ is the position and $i$ is the dimension. 

**解读：**
在这项工作中，我们使用了不同频率的正弦和余弦函数：
- 对于偶数维度索引（$2i$），使用正弦函数。
- 对于奇数维度索引（$2i+1$），使用余弦函数。
其中，$pos$ 代表词在序列中的绝对位置，$i$ 代表向量中的特征维度索引。

**原文：**
> That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.

**解读：**
也就是说，位置编码的每一个维度都对应一条正弦或余弦曲线。这些波的波长形成了一个从 $2\pi$ 到 $10000 \cdot 2\pi$ 的等比数列。
*注：这意味着什么？靠前的特征维度（$i$ 较小）频率极高、波长极短，位置每变化一点，它的值就剧烈变化；而靠后的特征维度（$i$ 较大）频率极低，波长极长。这使得模型既能区分极其接近的“细粒度”位置，也能感知跨度很大的“粗粒度”全局位置。*

**原文：**
> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

**解读：**
我们之所以选择这个三角函数，是因为我们假设它能让模型轻易学习到“基于相对位置的注意力”。原因在于，对于任意固定的位置偏移量 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数。
*注：这是核心数学原理。借助三角函数的和差公式： $\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$。模型在计算注意力内积时，能够通过线性组合感知到词与词之间相距 $k$ 的相对距离，而不仅仅是死记硬背绝对位置 $pos$。*

### 第三段：固定编码 vs. 可学习编码的实验对比

**原文：**
> We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). 

**解读：**
我们也尝试了使用“可学习的位置嵌入”来替代，结果发现这两种版本产生了几乎相同的翻译效果（参见论文表3的E行）。

**原文：**
> We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

**解读：**
最终我们之所以选择正弦曲线固定版本的编码，是因为它可以让模型更好地“外推（Extrapolate）”到比训练时遇到的序列更长的文本。
*注：“可学习位置编码”必须在训练集中见过该长度才能学到对应的参数；而基于正余弦公式的固定编码，即便推理时遇到前所未有的超长句子，也能依靠周期函数的连续性自然地计算出新位置的编码，因此具有更好的泛化和外推能力。*
