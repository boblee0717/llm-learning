# BERT 精读：Model Architecture（及 §3 开篇）

本文聚焦 BERT 论文（*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, Devlin et al., 2018）中与 **phase2-transformer 第 3 课**阅读清单对应的一段：

- `§3 BERT` 开篇（两阶段框架、统一架构）
- **`Model Architecture`**（多层双向 Transformer encoder、记号 $L,H,A$、与 GPT 的对照）
- **`Input/Output Representations`** 中与架构衔接的首段（可选扩展：WordPiece 30k、`[CLS]`、`[SEP]`、三段式 embedding 之和）

目标不是逐字翻译，而是把每一句背后的设计动机讲清楚，并与 **《Attention Is All You Need》§3.1 encoder stack + §3.2.2 多头注意力**、`GPT`/`GPT-3` 的 **decoder-only** 约束对照起来读。

---

## 阅读导航

如果你只想先抓主线，可以先看这 5 个结论：

- BERT 的主体是 **Vaswani et al. (2017) 原版 Transformer 的 encoder 堆叠**，实现上与 tensor2tensor 近乎一致；**不是** invent 一套全新 block。
- 论文用 **$L$** 表示 Transformer block（层）数，**$H$** 表示 hidden size（即常说 `d_model`），**$A$** 表示 **self-attention head 数**；FFN 中间层固定为 **$4H$**。
- **$\mathrm{BERT}_{\mathrm{BASE}}$** 与 OpenAI GPT 体量对齐便于比较；关键差异是 **BERT 用双向 self-attention**，GPT 用 **只允许看左侧上下文** 的受限 attention。
- 文献里常把 **双向版**叫 **Transformer encoder**，把 **只看左上下文**、可用于生成的版本叫 **Transformer decoder**（脚注 4）。
- 下游任务里输入一般是 **WordPiece token + segment + position** 三种 embedding **逐元素相加**；这与 encoder 内部 $\text{seq\_len} \times H$ 的张量流是自然衔接的（详见下文「输入表示」首段）。

---

## §3 开篇：两阶段框架与「统一架构」

这一小段交代 BERT 的使用方式：**先预训练、再微调**，且强调预训练与下游结构差异极小——后面读 `Model Architecture` 时，要记住「你微调时改的主要是 **输出头**，不是把 Transformer 推翻重写」。

### 句 1：本节交代 BERT 及实现细节

> We introduce BERT and its detailed implementation in this section.

**解读**

本节正式开始介绍 BERT 及其 **实现层面的细节**（不只停留在抽象idea）。

**补充理解**

phase2 第 3 课在读 GPT-3「规模表」、在读 Transformer「encoder/decoder stack」时，已经把「论文记号 $\leftrightarrow$ 代码维度」当作基本功；BERT 这一节同样会把 $L,H,A$ 与具体模型尺寸钉死，便于你做表格对照。

---

### 句 2：框架两步——预训练与微调

> There are two steps in our framework: pre-training and fine-tuning.

**解读**

BERT 的使用范式固定为两阶段：**预训练**（无标注数据、多个自监督任务）与 **微调**（有标注数据、端到端更新参数）。

**补充理解**

这和 GPT 系列的叙事一致（都是「通用语言模型 $\rightarrow$ 任务适配」），差别主要在 **预训练目标能不能双向融合上下文**（BERT 用 MLM 等，见 §3.1）。架构段落本身不关心 loss，但你要知道：**同样的 encoder stack，先在 MLM 上训，再在具体任务上训**。

---

### 句 3：预训练在无标注数据、多任务上进行

> During pre-training, the model is trained on unlabeled data over different pre-training tasks.

**解读**

预训练阶段只用 **无标注文本**，并且同时在 **多个预训练子任务**（BERT 里是 MLM + NSP 等）上优化。

**补充理解**

这里的 **MLM = Masked Language Modeling**，即 **掩码语言模型**：从原句里随机选一部分 token 作为预测目标，把输入中的这些位置大多替换成 `[MASK]`，再让模型根据 **左右上下文** 预测原来的 token。

例如：

```text
原句：The dog is chasing the ball.
输入：The [MASK] is chasing the ball.
目标：预测 [MASK] = dog
```

关键差异在于：BERT 做 MLM 时，预测位置可以同时看左边和右边，所以它适合训练 **双向 encoder 表征**；GPT 这类 left-to-right LM 则训练 $p(x_t \mid x_{<t})$，只能用左侧上下文预测下一个 token。BERT 原论文里，被选中的 token 也不是 100% 都换成 `[MASK]`，而是大致按 **80% `[MASK]` / 10% 随机 token / 10% 保持原 token** 处理，用来缓解「预训练总看到 `[MASK]`、微调时没有 `[MASK]`」的分布差异。

「different pre-training tasks」提醒你：**表征学习 ≠ 单一 LM likelihood**。这和只做强 left-to-right LM 的 GPT 预训练形成对照——后面的架构仍然是 Transformer，但 **训练信号**不同导致了「双向 vs 单向」能否实现的分岔。

---

### 句 4：微调时初始化全部参数、用下游标注数据更新

> For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks.

**解读**

微调阶段：用预训练权重初始化 **全部**（不仅仅是顶层）参数，并在下游 **标注数据**上继续训练。

**补充理解**

这是典型的 **full fine-tuning**。实践中也会有冻结部分层、adapter、LoRA 等变体，但论文原型句强调的是：**表征与浅层句法统计等都加载自预训练**，再通过任务数据整体校准。

---

### 句 5：每个下游任务各有微调模型，尽管初始化相同

> Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters.

**解读**

不同下游任务会各自保存一套 **微调后的权重**；它们的起点是同一份预训练 checkpoint。

**补充理解**

这与「一个 GPT base model + 不同 prompting」的生态不同：BERT 年代更典型的是 **任务专属微调**。读架构时记住：**encoder 主体架构相同、初始化 checkpoint 相同**；但每个任务 fine-tune 后会得到各自不同的 encoder 权重，不再共享同一份参数。任务之间通常只是在输出头形式上不同（分类头、span 预测头等）。

---

### 句 6：Figure 1 的 QA 例子贯穿本节

> The question-answering example in Figure 1 will serve as a running example for this section.

**解读**

后面会用图中的 **问答（QA）**管线当作贯穿例子，把输入构造与读出方式钉直观。

**补充理解**

若你手边有 PDF，建议对照 Figure 1：`[CLS]`、`[SEP]`、句子对打包，以及输出端的 $C$ 和 $T_i$ 分别怎么用。注意：$C$ 主要作为 **分类任务**的整句表示；抽取式 QA 更依赖每个 token 的 $T_i$ 去预测答案 span 的 start / end。这里先建立输入输出符号，后文任务头再展开。

---

### 句 7：BERT 的鲜明特征是跨任务的统一架构

> A distinctive feature of BERT is its unified architecture across different tasks.

**解读**

BERT 强调：**不同 NLP 任务共享同一套主干网络**，而不是每个任务一套手工订制架构。

**补充理解**

这句话与 ELMo「特征抽取 + 重型任务架构」形成对照；也和 GPT「decoder stack + LM head」在同一哲学谱系里：**先做大模型，再 Minimal task-specific head**。第 3 课关心多头与残差时，可以把它理解成：**多头注意力 + FFN + Add & Norm 的重复模块**就是这套「统一主干」。

---

### 句 8：预训练与最终下游架构差异极小

> There is minimal difference between the pre-trained architecture and the final downstream architecture.

**解读**

微调几乎不改变 encoder 的结构，只在外围加 **很轻**的任务头（以及可能的输入拼接方式）。

**补充理解**

读 `Model Architecture` 时不妨自问：哪些东西在 **所有任务**里不变？通常是：**$L$ 层 Transformer encoder block、hidden size $H$、head 数 $A$、FFN 宽度 $4H$**。变的多半是：**输出如何从 $T_i$ 或 $C$ 接到 softmax / span / etc.**。

---

## Model Architecture

本节正面回答：**BERT 的神经网络长什么样**，以及与同期 GPT 对照时 **哪些超参一致、哪些约束不同**。

### 句 1：主体是多层双向 Transformer encoder

> BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library.

**解读**

BERT = **多层** + **双向** + **Transformer encoder**；实现贴 Vaswani et al. (2017) 原版与 tensor2tensor。

**补充理解**

三个关键词拆开：

- **multi-layer**：对应本段后面定义的记号 $L$（blocks 堆叠），与 phase2 第 4 课「Block 重复 $N$ 次」同一意象。
- **bidirectional**：**同一序列内每个 token 的自注意力可以看到两侧上下文**（实现上不施加 GPT 那种 causal mask）。这正是「encoder 式」self-attention。
- **Transformer encoder**：术语上与脚注 4 对齐——区别于只允许左上下文的「decoder」注意力。

---

### 句 2：实现近乎原版，故不详述背景

> Because the use of Transformers has become common and our implementation is almost identical to the original, we will omit an exhaustive background description of the model architecture and refer readers to Vaswani et al. (2017) as well as excellent guides such as "The Annotated Transformer."

**解读**

架构细节不再赘述，请读者直接回溯 Vaswani et al. (2017) 或 Annotated Transformer。

**补充理解**

phase2 的定位正好是：**你用 `attention_is_all_you_need_reading_3.1_3.2.2.md` 把 encoder stack、Multi-Head Attention 读厚**，BERT 这一篇只做 **差异与记号**。遇到「Add & Norm 在前在后」之类细节，要以 **2017 原版 encoder（Post-Norm）**为准；BERT 沿用的是那一套的工程实现习惯（具体变体以代码/checkpoint 为准，论文此句不写死 Pre/Post）。

---

### 句 3：记号 $L$、$H$、$A$

> In this work, we denote the number of layers (i.e., Transformer blocks) as $L$, the hidden size as $H$, and the number of self-attention heads as $A$.

**解读**

$L$：层数；$H$：隐藏维度；$A$：**self-attention 的头数**。

**补充理解**

与代码对照时常写成：

```text
L   ↔ n_layers
H   ↔ d_model
A   ↔ n_heads
```

多头做法与 Transformer §3.2.2 一致：**先把 Q/K/V 投影再分头**，每头维度约为 $H/A$（需整除）。phase2 的 `03_multi_head_attention.py` 里 `d_head = d_model // n_heads` 就是在实现这一关系。

---

### 句 4：两类尺寸 $\mathrm{BERT}_{\mathrm{BASE}}$ 与 $\mathrm{BERT}_{\mathrm{LARGE}}$

> We primarily report results on two model sizes: $\mathrm{BERT}_{\mathrm{BASE}}$ (L=12, H=768, A=12, Total Parameters=110M) and $\mathrm{BERT}_{\mathrm{LARGE}}$ (L=24, H=1024, A=16, Total Parameters=340M).

**解读**

报告重点是两个规格：**Base** 与 **Large**，并给出 $L,H,A$ 与总参数量。

**补充理解**

一张快速核对表：

| 模型 | $L$ | $H$ | $A$ | 近似参数量 |
|------|-------|-------|-------|--------------|
| $\mathrm{BERT}_{\mathrm{BASE}}$ | 12 | 768 | 12 | 110M |
| $\mathrm{BERT}_{\mathrm{LARGE}}$ | 24 | 1024 | 16 | 340M |

注意 $H/A$：Base 为 $768/12=64$，Large 为 $1024/16=64$。**每头维度相同**，便于在不同宽度间做直觉类比。

---

### 句 5：$\mathrm{BERT}_{\mathrm{BASE}}$ 体量对齐 OpenAI GPT

> $\mathrm{BERT}_{\mathrm{BASE}}$ was chosen to have the same model size as OpenAI GPT for comparison purposes.

**解读**

Base 规格是为了与当时 OpenAI GPT **公平对比**而选的。

**补充理解**

读论文表格时：**相同量级 $L,H,A$**，差别主要来自 **注意力是否双向** 与 **预训练目标**，而不是「BERT 偷偷用了更深的网络」。这是在控制变量。

---

### 句 6：关键差异——双向 vs 受限（左向）self-attention

> Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.

**解读**

**决定性差异**：BERT 的自注意力是 **双向**；GPT 是 **只能看左边** 的 masked / causal self-attention。

**补充理解**

- 与 phase2 **第 2 课因果掩码**直接对应：GPT 类生成模型要保证 $p(x_t \mid x_{<t})$，必须在 softmax 前屏蔽「未来位置」。
- BERT encoder **不做 causal mask**（同一序列内不屏蔽右侧 token），所以每个位置表示融合左右上下文——但代价是 **不能直接当自回归 LM 用**（除非另加约束或改造）。实际实现里仍然会有 **padding mask**，用于避免 `[PAD]` 这种补齐 token 参与 attention；这和 GPT 的三角因果 mask 是两回事。
- 脚注进一步约定术语：**双向栈 $\approx$ encoder**；**左上下文版 $\approx$ decoder（可用于生成）**。

---

### 脚注 4：文献里 encoder / decoder 的惯用叫法

> We note that in the literature the bidirectional Transformer is often referred to as a "Transformer encoder" while the left-context-only version is referred to as a "Transformer decoder" since it can be used for text generation.

**解读**

双向 Transformer 常被称作 **Transformer encoder**；**只能看左侧上下文**的版本常被称作 **Transformer decoder**，因为它适合接在自回归生成上。

**补充理解**

这条脚注是在消歧 **术语**，不是在改 Vaswani 论文里「encoder stack / decoder stack」的图示分工：
- **BERT** 用的是 **encoder** 这一支（全序列双向 self-attention）。
- **GPT** 用的是带 **causal mask** 的自注意力栈，文献里常类比为 **decoder**（与带 cross-attention 的翻译 decoder 不是同一回事，但 **「左向可见」**这一点一致）。

---

### 脚注与工程细节（feed-forward 维度）

> In all cases we set the feed-forward/filter size to be $4H$, i.e., 3072 for $H = 768$ and 4096 for $H = 1024$.

**解读**

FFN 隐层宽度取 **$4H$**，与原版 Transformer 一致。

**补充理解**

这与 phase2 第 4 课「FFN 一般是 $d_{\mathrm{model}}\to 4d_{\mathrm{model}}\to d_{\mathrm{model}}$」同一惯例；读 GPT-3 / GPT-2 表格时也会反复看到 **$4\times$** 这一列。

---

## Input/Output Representations（架构衔接首段，可选）

README 若只要求 `Model Architecture`，你可以略读本段；若要理解 **「encoder 的输入张量从哪来」**，建议读完下列句子。

### 句 1：输入要能表达单句或句对

> To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., $\langle\mathrm{Question}, \mathrm{Answer}\rangle$) in one token sequence.

**解读**

同一套 tokenizer + embedding 方案，既要能编码 **单个句子**，也要能编码 **两个句子组成的对**，且不产生歧义。

**补充理解**

这是 **NSP / 句间关系任务**的工程前提：后面会用 `[SEP]`、`sentence A/B embedding` 把两段文本pack进一条序列。

---

### 句 2：「句子」与「序列」在文中的含义

> Throughout this work, a "sentence" can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

**解读**

论文里的 **sentence** 不等于语言学句子，只是连续片段；**sequence** 指送进模型的整条 token 序列。

**补充理解**

避免在读论文时纠结标点：**本质是 chunk / span**，适配维基百科、BookCorpus 等预训练语料的切段方式。

---

### 句 3：WordPiece 词表

> We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary.

**解读**

BERT 使用 **WordPiece** tokenizer，词表大小是 **30,000**。

**补充理解**

这里先把输入 token 的来源说清楚：原始文本不会直接进入 Transformer，而是先被切成 WordPiece token，再查 embedding 表变成向量。

这和 GPT-2 / GPT-3 的 byte-level BPE 不一样：

| 模型 | tokenizer | 词表规模 | 直觉 |
|------|-----------|----------|------|
| BERT | WordPiece | 30,000 | 面向理解任务，常见词/子词优先；多数未登录词可继续拆成子词，极端情况会落到 `[UNK]` |
| GPT-2 / GPT-3 | byte-level BPE | 50,257 | 从 byte 出发，几乎天然覆盖任意 Unicode 字符串 |

所以 BERT 的输入表示重点不是「任意字符串都能无损还原」，而是：用一个固定的 WordPiece 词表，把文本切成适合 encoder 理解的 token 序列。

---

### 句 4：`[CLS]` 与整句表示 $C$

> The first token of every sequence is always a special classification token (`[CLS]`).

**解读**

每条输入序列的第一个 token 固定是 **`[CLS]`**，它是专门为分类等任务准备的特殊 token。

**补充理解**

`[CLS]` 不对应原文里的某个词，而是一个人为加进去的「聚合位」。因为 BERT encoder 的 self-attention 是双向的，最后一层 `[CLS]` 位置的向量可以吸收整条序列的信息。

---

### 句 5：`[CLS]` 的最终 hidden state 用于分类

> The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.

**解读**

`[CLS]` 位置最后一层的隐向量会作为 **整条序列的聚合表示**，用于分类任务。

**补充理解**

论文后面把这个向量记作 $C\in\mathbb{R}^H$。注意它主要服务 **sequence-level classification**，例如情感分类、句子关系判断等。抽取式 QA 不主要靠 $C$ 输出答案，而是看每个 token 的最终向量 $T_i$，再预测 start / end 位置。

---

### 句 6：句对打包成一条序列

> Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token (`[SEP]`). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B.

**解读**

句对会被拼成一条序列。BERT 用两种方式区分两段文本：一是用 **`[SEP]`** 分隔，二是给每个 token 加一个 **sentence A/B segment embedding**。

**补充理解**

典型形状是：

```text
[CLS] sentence A tokens [SEP] sentence B tokens [SEP]
```

例如 QA 里，sentence A 可以是 question，sentence B 可以是 passage。这样做的好处是：两段文本进入同一个 encoder，self-attention 可以直接在 question 和 passage 之间做双向交互，而不需要先分别编码再额外做 cross-attention。

---

### 句 7：$E$、$C$、$T_i$ 三个符号

> We denote input embedding as E, the final hidden vector of the special [CLS] token as C, and the final hidden vector for the ith input token as Ti.

**解读**

论文约定三个符号：

```text
E    = 输入 embedding
C    = [CLS] 位置的最终 hidden vector
T_i  = 第 i 个输入 token 的最终 hidden vector
```

**补充理解**

三者维度都和 hidden size $H$ 对齐：

```text
E_i ∈ R^H
C   ∈ R^H
T_i ∈ R^H
```

区别在于阶段不同：$E_i$ 是进入第 1 层 encoder 前的输入向量，$C$ 和 $T_i$ 是经过 $L$ 层 Transformer encoder 后的输出向量。分类任务常读 $C$，token-level 任务和 span 任务常读各位置的 $T_i$。

---

### 句 8：词项、片段、位置三类 embedding 之和

> For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings.

**解读**

每个位置的输入向量 = **token embedding + segment embedding + position embedding**（逐元素相加）。

**补充理解**

与 Transformer 原版「token + position」相比，BERT **多了一块 segment**，服务句对任务；**position** 可与 §3.5 sinusoidal 对照。BERT 原始实现使用的是 **可学习 position embedding**，最大长度常见设置为 512。

可以把单个位置的输入写成：

```text
E_i = token_embedding_i + segment_embedding_i + position_embedding_i
```

这三项必须同为 $H$ 维，才能逐元素相加。相加之后形状仍然是 `(seq_len, H)`，正好接到后面的 Transformer encoder stack。

---

## 对照 phase2：`03_multi_head_attention.py` 的五个 Part

`phase2-transformer/README.md` 第 3 课「按顺序做」第 ③ 步要求运行 `phase2-transformer/03_multi_head_attention.py` 并关注 **5 个 Part**。下面的映射帮你把 **BERT Model Architecture（encoder + 多头）**与脚本输出对齐——论文不写 Python，但 **张量形状与「分头—并行注意力—拼接—$W^O$」**与此脚本一致。

| Part | 脚本标题 | 你在输出里应抓住什么 | 与 BERT / Transformer 的呼应 |
|------|----------|----------------------|--------------------------------|
| **Part 1** | 单头注意力的局限 | 直觉：一头往往偏向一种「关系模式」；多头并行才可能覆盖多种模式 | BERT 中 $A=12$ 或 $16$ 个头，是在 **同一 $H$ 维表示空间**里购买 **多组并行 QKV 子空间**（论文用语：multi-head 扩展注意力「表示子空间」的能力，见 Vaswani et al. §3.2.2） |
| **Part 2** | Multi-Head Attention 实现 | `reshape` 成 `(n_heads, seq_len, d_head)`，逐头 $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$，再 `concat` 与 `@ W_O` | 对应 BERT 每一层里的 **MultiHead(SelfAttn)**；$H$ 对应 `d_model`，$A$ 对应 `n_heads` |
| **Part 3** | 残差连接 | `X + Attn(X)` 数值对比；文中关于 **$+1$ 恒等通路**的梯度解释 | BERT encoder stack **每一 sub-layer 外都有残差**（与 Fig.1 / Vaswani encoder 一致）；深层可达 $L=24$ 依赖这条通路 |
| **Part 4** | Layer Normalization | 对每个位置的最后维做均值方差归一化；与 BatchNorm 对比 | 原版 Transformer encoder 使用 **LayerNorm**（在序列长度可变任务上比 BN 更自然）；你在脚本里看到的是 **与序列 batch 轴无关**的归一化 |
| **Part 5** | Pre-Norm vs Post-Norm | `LayerNorm(x + Attn)` vs `x + Attn(LayerNorm(x))` 输出差异 | **BERT 源自 2017 encoder 栈习惯（多为 Post-Norm）**；GPT-2/3 则典型为 **Pre-Norm**。读 BERT 时不要默认它和 GPT 系列的 Norm 位置相同——以具体开源实现为准，phase2 刻意并用两种写法帮你建立直觉 |

**掩码提示**：脚本注释强调与第 2 课一致：**`1` = 屏蔽，`0` = 可见**。BERT encoder **不做 GPT 式因果三角掩码**；若你在练习里给多头注意力加 `np.triu`，那是在模拟 **decoder-only LM**，用于对照实验而非 BERT 原版推理形态。BERT 仍可能使用 padding mask 来屏蔽 `[PAD]`，但那不是因果约束。

---

## 与其它 Notes 的交叉索引

- Encoder stack、Post-Norm、decoder 多出来的 cross-attention：`papers/notes/attention_is_all_you_need_reading_3.1_3.2.2.md`（§3.1 与 §3.2.2）。
- GPT-3 的规模表、Pre-Norm、sparse/dense attention：`papers/notes/gpt3_reading_2.1_model_and_architectures.md`。
- 第 2 课自注意力与 mask：`papers/notes/notes_attention_qkv.md`（若你有）及 `02_self_attention.py` 注释约定。

---

## 文献

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805.

Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
