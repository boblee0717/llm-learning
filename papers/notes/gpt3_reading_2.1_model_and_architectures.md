# GPT-3 精读：2.1 Model and Architectures

本文聚焦 GPT-3 论文（*Language Models are Few-Shot Learners*, 2020）的一个小节：

- `2.1 Model and Architectures`

目标不是逐字翻译，而是把每一句背后的设计动机讲清楚：GPT-3 到底在架构上做了什么改动、8 个尺寸是怎么选的、那张参数表该怎么读。

---

## 阅读导航

如果你只想先抓主线，可以先看这 5 个结论：

- GPT-3 的架构**几乎和 GPT-2 一模一样**，核心改动只有一个：注意力从纯 dense 换成 dense 和局部带状稀疏（locally banded sparse）交替。
- 一共训练了 **8 个尺寸**的模型，参数量从 1.25 亿到 1750 亿，跨越三个数量级，目的是验证 Scaling Law。
- 所有模型都沿用 GPT-2 的三项改进：**modified initialization、pre-normalization（Pre-Norm）、reversible tokenization（BPE）**。
- FFN 内部维度始终是 `d_ff = 4 * d_model`，所有模型上下文窗口都是 `n_ctx = 2048`。
- 具体超参数不是理论推导的，而是根据 **GPU 并行效率和负载均衡**实验调出来的；Scaling Law 论文指出在合理范围内验证损失对这些参数不太敏感。

---

## 2.1 Model and Architectures

这一节回答的是三个问题：

- GPT-3 相比 GPT-2，架构上到底改了什么？
- 为什么要训练 8 个不同大小的模型？
- Table 2.1 里的每一列是什么意思？

### 核心结论

- GPT-3 本质上是 GPT-2 架构的直接放大版，不是全新设计。
- 唯一的结构性改动是引入了 dense 和 sparse attention 交替的模式。
- 训练 8 个模型是为了验证 Scaling Law：验证损失是否随模型规模呈平滑幂律下降。
- 超参数（层数、头数、宽度）的选择主要受 GPU 并行效率驱动，而非理论推导。

---

### 句 1：GPT-3 沿用 GPT-2 架构，只改了注意力模式

> We use the same model and architecture as GPT-2 [RWC+19], including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer [CGRS19].

**解读**

GPT-3 的模型架构和 GPT-2 完全相同，保留了 GPT-2 引入的三项改进（modified initialization、pre-normalization、reversible tokenization），唯一的区别是：在 Transformer 的层中使用了 dense 和局部带状稀疏注意力交替排列的方式，类似 Sparse Transformer。

**补充理解**

这一句信息密度非常高，需要拆成几块来理解：

**（1）"same model and architecture as GPT-2"** — 这是全文最关键的一句话。它意味着 GPT-3 **不是一次架构创新**，而是一次**规模实验**。核心贡献在于证明「把已有架构放大到 1750 亿参数会发生什么」，而不是设计一个更好的 Transformer 变体。

**（2）三项从 GPT-2 继承的改进：**

| 改进 | 含义 | 为什么重要 |
|------|------|-----------|
| modified initialization | 残差路径上的输出投影权重（如 attention 的 `W_O` 和 FFN 的 `W_2`）初始化时额外乘以 `1/√(2N)`（N 为 Transformer 层数），其余权重正常初始化 | 层数越深，残差累加越多，只缩放残差出口的权重就能控制深层输出量级 |
| pre-normalization | LayerNorm 放在 sub-layer **之前**而不是之后（Pre-Norm） | 比原始 Transformer 的 Post-Norm 训练更稳定，减少对 warmup 的依赖 |
| reversible tokenization | 使用字节级 BPE（Byte Pair Encoding），可以无损还原原始文本 | 避免 `<UNK>` token，任何 Unicode 文本都能编码 |

对比原始 Transformer 的 Post-Norm：

```text
原始 Transformer (Post-Norm): y = LayerNorm(x + Sublayer(x))
GPT-2 / GPT-3   (Pre-Norm):  y = x + Sublayer(LayerNorm(x))
```

Pre-Norm 让梯度更容易直通回浅层，对深网（GPT-3 最深 96 层）尤其重要。

**（3）"alternating dense and locally banded sparse attention"** — 这是 GPT-3 唯一的结构性改动。具体来说：

- **dense attention**：标准的全注意力，每个 token 可以 attend 到上下文窗口内所有前面的 token。复杂度 `O(n²)`。
- **locally banded sparse attention**：每个 token 只 attend 到自己附近一个固定带宽内的 token。复杂度接近 `O(n)`。
- **alternating**：两种注意力在不同层之间交替排列，比如奇数层做 dense、偶数层做 sparse（具体模式参考 Sparse Transformer 论文）。

```text
Layer 1:  Dense Attention       ← 全局视野
Layer 2:  Sparse Attention      ← 局部视野
Layer 3:  Dense Attention       ← 全局视野
Layer 4:  Sparse Attention      ← 局部视野
...
```

这样做的好处是：dense 层负责捕捉长距离依赖，sparse 层降低计算量。交替后整体计算成本比全 dense 低，同时仍保持全局信息流。

这里不要把 sparse attention 理解成 GPT-3 最核心的创新。GPT-3 的主线仍然是 **scale up GPT-2-style decoder-only Transformer**。从后来的公开开源模型和工程实践看，主流路线更多转向 dense attention 配合 FlashAttention 等高效 kernel；但闭源 GPT 系列的内部细节不能仅凭公开论文完全确认，所以这里更稳妥的理解是：GPT-3 在当时用 sparse attention 作为工程折中，而不是把论文贡献建立在一种全新的 attention 架构上。

---

### 句 2：训练 8 个不同大小的模型，横跨三个数量级

> To study the dependence of ML performance on model size, we train 8 different sizes of model, ranging over three orders of magnitude from 125 million parameters to 175 billion parameters, with the last being the model we call GPT-3.

**解读**

为了研究模型性能和模型规模之间的关系，作者训练了 8 个不同大小的模型，参数量从 1.25 亿到 1750 亿，跨越三个数量级。其中最大的那个就是 GPT-3。

**补充理解**

这一句揭示了 GPT-3 论文的核心研究方法：**不是只训练一个最大模型然后报告结果，而是训练一整条「模型尺寸梯度」**，让你看到性能如何随规模变化。

「三个数量级」的意思是：`1.25 × 10⁸` 到 `1.75 × 10¹¹`，实际约差 1400 倍（论文取整说"三个数量级"即 10³）。这种跨度足够画出清晰的 Scaling 曲线。

为什么要叫 GPT-3？因为只有最大的那个（175B）才被冠名 `GPT-3`，其余 7 个是实验对照组。论文中提到 "GPT-3" 时，默认指的就是 175B 版本。

---

### 句 3：Scaling Law — 验证损失应随模型规模呈平滑幂律下降

> Previous work [KMH+20] suggests that with enough training data, scaling of validation loss should be approximately a smooth power law as a function of size; training models of many different sizes allows us to test this hypothesis both for validation loss and for downstream language tasks.

**解读**

之前的工作（Kaplan 等人的 Scaling Laws 论文）提出：只要训练数据足够，验证损失会近似随模型规模呈平滑的幂律（power law）下降。训练多种不同规模的模型，可以同时在验证损失和下游任务上检验这个假说。

**补充理解**

这一句引用的 `[KMH+20]` 就是 OpenAI 2020 年初发表的 *Scaling Laws for Neural Language Models*（Kaplan et al.），它可以说是 GPT-3 存在的理论基础。那篇论文的核心发现是：

```text
L(N) ≈ (N_c / N)^α
```

其中 `L` 是验证损失，`N` 是模型参数量，`α` 是幂指数（约 0.076）。画在 log-log 图上就是一条直线：模型参数翻 10 倍，损失按固定比例下降。

这里的 loss 是验证集上的 **next-token prediction 误差**。loss 越低，表示模型对"下一个 token 应该是什么"的概率分布预测得越准。Scaling Law 说的不是"模型突然理解了一切"，而是一个更朴素的统计规律：

```text
更多参数 + 更多数据 + 更多训练 compute
        ↓
next-token prediction loss 下降
        ↓
模型对语言、事实、格式、任务模式的统计建模更准
        ↓
很多下游任务和 in-context learning 能力也随之变强
```

所以「loss 下降」和「暴力出奇迹」不是两件矛盾的事。**loss 下降是可观测的训练指标，"暴力出奇迹"是这些指标下降后在能力层面表现出来的现象**。小的平均 loss 改善也可能很重要，因为语言模型的 loss 是按 token 平均的；每个 token 都预测得稍微准一点，累积到长文本、复杂任务和多步推理里，差异就会被放大。

GPT-3 论文的目标之一就是：**把这条直线从 15 亿参数推到 1750 亿参数，看它是不是继续成立**。结果是——基本成立。这给了后续「暴力出奇迹」的 scaling 路线强有力的实验支撑。

但这里还有一个微妙的地方：Scaling Law 论文只关注了**验证损失**，而 GPT-3 还要检验下游任务（如翻译、问答、常识推理）的性能是否也呈类似的规律。结论是：大方向吻合，但下游任务的曲线没有验证损失那么平滑，不同任务之间差异较大。

---

### Table 2.1：8 个模型的完整参数表

在进入第二段逐句分析之前，先把这张核心表格列出来：

| Model Name | n_params | n_layers | d_model | n_heads | d_head | Batch Size | Learning Rate |
|---|---|---|---|---|---|---|---|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | 6.0×10⁻⁴ |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | 3.0×10⁻⁴ |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | 2.5×10⁻⁴ |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | 2.0×10⁻⁴ |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | 1.6×10⁻⁴ |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | 1.2×10⁻⁴ |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 | 2M | 1.0×10⁻⁴ |
| GPT-3 175B ("GPT-3") | 175.0B | 96 | 12288 | 96 | 128 | 3.2M | 0.6×10⁻⁴ |

这张表按 GPT-3 论文 PDF 的 Table 2.1 原值抄录。注意里面有两处常被讨论的可疑值：`GPT-3 XL` 的 `2048 != 24 × 128`，`GPT-3 13B` 的 `5140` 也不能被 `40` 整除。很多二手资料会把它们修正为 `XL n_heads = 16`、`13B d_model = 5120`。但做论文精读时，最好先保留原表，再单独标注疑点，不要静默改表。

所有模型都在 **300B tokens** 上训练完成。

**表格速读：几条一眼能看出来的规律**

```text
1. 模型越大，层数越深：12 → 24 → 32 → 40 → 96
2. 模型越大，宽度越宽：768 → 1024 → ... → 12288
3. 模型越大，学习率越小：6.0e-4 → 0.6e-4（差 10 倍）
4. 模型越大，batch size 越大：0.5M → 3.2M（差 6.4 倍）
5. d_head 基本落在 64-128，不随模型变大而大幅增长
```

---

### 句 4：Table 2.1 展示了 8 个模型的尺寸和架构

> Table 2.1 shows the sizes and architectures of our 8 models.

**解读**

这一句只是引导读者看表，本身没有新信息。

---

### 句 5：表格每一列的含义 + d_ff = 4 * d_model 的硬规则

> Here n_params is the total number of trainable parameters, n_layers is the total number of layers, d_model is the number of units in each bottleneck layer (we always have the feedforward layer four times the size of the bottleneck layer, d_ff = 4 * d_model), and d_head is the dimension of each attention head.

**解读**

论文逐一解释了表格中的列名：`n_params` 是可训练参数总数，`n_layers` 是 Transformer 层数，`d_model` 是每层"瓶颈"的宽度，`d_head` 是每个注意力头的维度。同时给出一条硬规则：FFN 内部维度始终是 `d_ff = 4 * d_model`。

**补充理解**

这一句需要拆成几个要点来理解：

**（1）`d_model` 被叫做 "bottleneck layer"**

为什么叫「瓶颈」？因为在每一层 Transformer 中，FFN 内部先把维度从 `d_model` 升到 `d_ff = 4 * d_model`（扩展），再降回 `d_model`（压缩）。`d_model` 是层间传递信息的"最窄处"，所以是瓶颈：

```text
d_model ──[W1]──> d_ff = 4 * d_model ──[ReLU/GELU]──[W2]──> d_model
  窄           宽（展开计算）                         窄（压缩回来）
```

以 GPT-3 175B 为例：`d_model = 12288`，`d_ff = 49152`。每个 token 在 FFN 里先被映射到 49152 维做非线性加工，再压回 12288 维。

**（2）`d_ff = 4 * d_model` 是固定倍数，不是可调超参**

原始 Transformer 论文里 `d_model = 512, d_ff = 2048` 也是 4 倍。GPT-3 延续了这个比例。这个 4 倍是经验性的工程选择：

- 太小（如 2 倍）→ FFN 表达能力不够，模型容量受限
- 太大（如 8 倍）→ FFN 参数暴增，但收益递减

4 倍在各种实验中被证明是一个不错的折中。后来的 LLaMA 系列改用了 `8/3` 倍的 SwiGLU FFN，但总参数量基本等价。

**（3）`d_head` 是每个注意力头的维度**

常见的多头注意力实现里，会把隐藏维度 `d_model` 平均切成 `n_heads` 份，因此通常有：

```text
d_model = n_heads × d_head
```

以 GPT-3 175B 为例：`12288 = 96 × 128`，完美整除。这符合你在第 3 课代码里看到的 reshape 逻辑：`d_model` 必须能被 `n_heads` 整除，多头注意力才能把隐藏维度均匀切成多个 head。

但 Table 2.1 里不要机械套这个公式去“改表”。按论文 PDF 原表抄录时，有两行会对不上：

```text
Small:  768   = 12 × 64    ✓
Medium: 1024  = 16 × 64    ✓
Large:  1536  = 16 × 96    ✓
XL:     2048  ≠ 24 × 128   ← 原表可疑；若 n_heads=16 才对齐
2.7B:   2560  = 32 × 80    ✓
6.7B:   4096  = 32 × 128   ✓
13B:    5140  ≠ 40 × 128   ← 原表可疑；若 d_model=5120 才对齐
175B:  12288  = 96 × 128   ✓
```

> **读表原则**：原论文表格怎么写，笔记表格就先怎么抄；和实现直觉冲突的地方再用注释标出来。这样既尊重原始资料，又不会把可能的排版 typo 当成确定事实。

`d_head` 并不是所有模型都一样——小模型用 64，中间出现过 80 和 96，大模型端多为 128。这背后的直觉是：每个注意力头需要足够的维度来表达 Q/K 匹配模式，但继续增大单头维度会增加计算成本，收益未必线性增长。后续很多公开模型也把单头维度保持在 128 左右，更多通过增加层数、宽度或 head 数来扩容。

**（4）表格里没有的隐含列：`d_ff`**

虽然表格没有显式列出 `d_ff`，但根据 `d_ff = 4 * d_model` 可以算出来：

| Model | d_model | d_ff |
|---|---|---|
| Small | 768 | 3,072 |
| Medium | 1,024 | 4,096 |
| Large | 1,536 | 6,144 |
| XL | 2,048 | 8,192 |
| 2.7B | 2,560 | 10,240 |
| 6.7B | 4,096 | 16,384 |
| 13B | 5,140（论文原表） | 20,560（按原表计算） |
| 175B | 12,288 | 49,152 |

如果采用常见勘误口径，把 13B 的 `d_model` 读作 `5120`，那对应的 `d_ff` 就是 `20480`。这一点不影响本节主线：GPT-3 统一使用 `d_ff = 4 * d_model`。

---

### 句 6：所有模型的上下文窗口都是 2048 tokens

> All models use a context window of n_ctx = 2048 tokens.

**解读**

不管模型多大，上下文窗口一律是 2048 个 token。

**补充理解**

`n_ctx = 2048` 意味着模型在做 next token prediction 时，最多只能"回头看"前面 2048 个 token。超出这个范围的内容对当前预测完全不可见。

这个数字和 GPT-2 一样，没有增长。这也说明 GPT-3 的重点完全在**参数规模**，而非**上下文长度**。

后续模型的上下文窗口后来越做越长，但那已经是另一个技术主题：位置编码方式、attention kernel、训练数据长度分布、推理显存管理都会一起影响长上下文能力。读 GPT-3 2.1 时先抓住一点就够了：**GPT-3 没有把重点放在延长上下文窗口，而是把参数规模放大到 175B**。在 2020 年，`2048 tokens` 已经是主流水平。

---

### 句 7：模型在 GPU 上沿深度和宽度两个方向并行

> We partition the model across GPUs along both the depth and width dimension in order to minimize data-transfer between nodes.

**解读**

为了最小化节点之间的数据传输，模型被沿深度（层数）和宽度（每层的矩阵维度）两个方向切分到多块 GPU 上。

**补充理解**

这一句描述的是 **模型并行（model parallelism）** 的策略。GPT-3 175B 的参数量约 700GB（FP32），单张 V100（32GB 显存）根本放不下，所以必须把模型拆开：

```text
深度方向切分（Pipeline Parallelism）:
  GPU 0: Layer 1-24
  GPU 1: Layer 25-48
  GPU 2: Layer 49-72
  GPU 3: Layer 73-96

宽度方向切分（Tensor Parallelism）:
  同一层的矩阵乘法被拆成多份，分布在多张 GPU 上
  例如 W_Q 矩阵 [12288, 12288] 被切成 8 份，每张 GPU 计算 [12288, 1536]
```

两种并行可以组合使用。宽度切分（Tensor Parallelism）适合同一台机器内的高速 NVLink 连接，深度切分（Pipeline Parallelism）适合跨机器的通信。

这也解释了为什么 GPT-3 论文在选架构参数时要考虑 GPU 布局——不是纯粹的数学最优，而是要让矩阵的形状恰好能均匀切分到硬件上。

---

### 句 8：超参数的选择由 GPU 效率和负载均衡驱动

> The precise architectural parameters for each model are chosen based on computational efficiency and load-balancing in the layout of models across GPU's.

**解读**

每个模型的具体架构参数（层数、宽度、头数）不是某个理论公式推导出来的，而是根据 GPU 上的计算效率和负载均衡来选择的。

**补充理解**

这一句非常实诚地承认了：**超参数选择主要是工程驱动，不是理论驱动**。

举个例子，为什么 GPT-3 175B 选了 `n_layers = 96, d_model = 12288, n_heads = 96`？

- `12288 = 96 × 128`，恰好能被 96 个 head 整除
- `12288` 是 2 的高次幂附近的数（`12288 = 12 × 1024 = 3 × 4096`），做 Tensor Parallelism 时容易均匀切分到 8 / 16 / 32 张 GPU 上
- `96` 层也便于在 Pipeline Parallelism 中均匀分配（96 = 2 × 48 = 3 × 32 = 4 × 24 = 8 × 12 …）

这也解释了为什么 13B 这一行的 `5140` 常被怀疑是排版 typo：如果读成 `5120 = 5 × 1024`，它就能被常见的 Tensor Parallelism 切分数（8、16、32）整除；同时 `40 × 128 = 5120` 也恰好和 `n_heads × d_head` 对齐。这种形状更符合大模型训练里的硬件布局直觉。

> **值得记住的直觉**：对于超大模型，架构设计不再是纯粹的"什么参数组合学得最好"，而是"什么参数组合能在给定硬件上最高效地训练"。这是从 GPT-3 开始越来越明显的趋势。

---

### 句 9：验证损失对这些参数不太敏感

> Previous work [KMH+20] suggests that validation loss is not strongly sensitive to these parameters within a reasonably broad range.

**解读**

再次引用 Scaling Law 论文的结论：在合理的范围内，验证损失对具体的架构参数（层数/宽度/头数的比例）不太敏感。

**补充理解**

这一句是为前面「超参数按 GPU 效率选」提供理论背书：反正验证损失主要由**总参数量**决定（加上数据量和计算量），具体怎么分配到层数和宽度上，影响不大。

Scaling Law 论文的核心发现可以总结成：

```text
性能 ≈ f(总参数量 N, 训练数据量 D, 总计算量 C)
```

在总参数量 N 固定的前提下，把 N 拆成 `深×宽`（更深更窄 vs 更浅更宽）的方式只要别太极端，对最终效果影响不大。这让工程师有了很大的自由度：可以根据硬件拓扑来选择最高效的组合，而不用担心损失精度。

不过这个结论有一个前提：**"在合理的范围内（within a reasonably broad range）"**。如果走极端（比如只有 1 层但宽度极大，或者 1000 层但宽度极小），性能肯定会崩。

---

## 综合分析：从 Table 2.1 看 Scaling 的规律

把 8 个模型排成一条线，可以看到几条有意思的 scaling 趋势：

### 1. 宽度的增长幅度远大于深度

```text
深度（n_layers）:  12 → 96     增长 8 倍
宽度（d_model）:  768 → 12288  增长 16 倍
```

参数量和 `d_model²` 大致成正比（因为注意力和 FFN 的矩阵都是 `d_model × d_model` 量级），所以要增加参数量，加宽比加深更"划算"。这和原始 Transformer 论文的做法一致：big model 也是优先加宽而非加深。

### 2. 学习率随模型变大而缩小

```text
Small (125M):  6.0 × 10⁻⁴
175B:          0.6 × 10⁻⁴
```

模型变大以后，训练稳定性和优化超参通常会更敏感。学习率太大时，单步参数更新更容易破坏已经学到的表示；所以表里可以看到一个清晰趋势：模型越大，学习率越小。这是工程上的稳定训练选择，不需要把它理解成某个严格理论公式。

### 3. Batch size 随模型变大而增加

```text
Small (125M):  0.5M tokens
175B:          3.2M tokens
```

大模型每一步的计算更贵，但大 batch 可以让每一步的梯度估计更准确、GPU 利用率更高。Scaling Law 论文也指出，gradient noise scale 越大的模型（通常是更大的模型），能从更大的 batch 中受益更多。

### 4. 所有模型都训练 300B tokens

不管模型大小，训练 token 总量统一为 300B。论文 Table 2.2 给出了训练数据的混合比例：Common Crawl（60%）、WebText2（22%）、Books1/2（16%）、Wikipedia（3%）。由于采样权重并不按数据集大小等比分配，不同数据集被重复看到的次数差异很大——Wikipedia 被看了约 3.4 遍，而 Common Crawl 不到 0.5 遍。这个重复次数对**所有模型都一样**，因为它取决于数据集大小和采样权重，与模型参数量无关。

后来 Chinchilla 论文（2022）指出 GPT-3 其实是 **under-trained** 的：以 175B 参数量来说，最优的训练 token 数应该接近 3.5T（而不是 300B）。这为后来更注重数据效率的训练策略奠定了基础。

---

## 小结

Section 2.1 虽然只有两段话加一张表，但信息量很大。可以记住这条主线：

1. **GPT-3 不是新架构**——它就是 GPT-2 直接放大，唯一改动是加了 sparse attention。
2. **核心贡献是 Scaling 实验**——训练 8 个模型，验证 Scaling Law 在跨越三个数量级的规模上是否成立。
3. **Table 2.1 是整篇论文最常被引用的表格**——后续无数工作都参考了这张表来设计自己的模型规模梯度。
4. **超参数选择是工程驱动的**——GPU 并行效率比理论最优更重要，反正验证损失对此不太敏感。
5. **训练策略（学习率、batch size）也随模型变大而系统调整**——大模型用更小的学习率和更大的 batch。
