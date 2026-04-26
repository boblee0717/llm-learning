# GPT-2 论文 §2.2 Input Representation 逐段精读

> 论文：Radford et al., 2019, *Language Models are Unsupervised Multitask Learners*  
> PDF：[GPT2_Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf](../core-transformers/GPT2_Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf)  
> 目标章节：Section **2.2 Input Representation**（原文约 p.3–p.4）

本节配合第二阶段第 1 课《词嵌入与位置编码》一起学习。
重点回答一个问题：**"一段任意的文本，是怎么变成模型能吃的一串整数 token 的？"**

---

## 0. 先建立直觉：Input Representation 到底在解决什么？

一个"通用语言模型（general LM）"理论上应该能对**任何字符串**计算概率。
但现实中各家 LM 都对输入做了不同程度的"预处理"：

- 小写化（lowercasing）
- 分词（tokenization）
- 未登录词（out-of-vocabulary, OOV）用 `<UNK>` 替换

这些预处理都会让模型**只能建模一部分字符串**——比如大小写信息丢了、罕见词全变成 `<UNK>`，你就没法对"带这些字符的原始字符串"正确打分。

GPT-2 想走得更"通用"：**给我任何 Unicode 字符串，我都能给你一个概率。**
要做到这一点，就必须在"输入表示"这一层想清楚：**用什么作为最小单位（symbol）？**

候选方案有三档：

| 粒度 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| **word-level**（词级） | 传统 LM | 序列短，语义紧凑 | 词表巨大、OOV 严重，无法表示任意字符串 |
| **character-level**（字符级） | char-LM | 无 OOV | 序列太长，性能差 |
| **byte-level**（字节级） | Gillick et al. 2015 | 真·无 OOV，词表只有 256 | 大规模数据上打不过 word-level |
| **subword-level**（子词级） | BPE / WordPiece | 介于词和字符之间 | 需要训练一个 tokenizer |

GPT-2 选择的答案是：**Byte-level BPE**（字节级 BPE）——subword 的一种，但最小单位是 byte 而不是 Unicode code point。下面逐段拆。

---

## 1. 第一段逐句精读（对应论文 p.3 开头那一段）

> A general language model (LM) should be able to compute the probability of (and also generate) any string.

一个通用 LM 应当能对**任何**字符串打分、生成。注意关键词 **any string** —— 这是 GPT-2 设计 tokenizer 的第一性原则。

> Current large scale LMs include pre-processing steps such as lowercasing, tokenization, and out-of-vocabulary tokens which restrict the space of model-able strings.

当前大规模 LM 的预处理（小写化、分词、`<UNK>`）**人为缩小了"可建模字符串"的集合**。
例如：训练时把所有文本小写化，那模型永远分不清 `Apple`（公司）和 `apple`（水果）。

> While processing Unicode strings as a sequence of UTF-8 bytes elegantly fulfills this requirement as exemplified in work such as Gillick et al. (2015), current byte-level LMs are not competitive with word-level LMs on large scale datasets such as the One Billion Word Benchmark.

**把 Unicode 字符串按 UTF-8 编码成字节序列**，理论上很优雅：

- 词表固定只有 256（每个 byte 只有 256 种可能）
- 不存在 OOV，任何字符串都能编码

但实测：**纯 byte-level LM 在大数据集上打不过 word-level LM**。
原因很朴素：序列太长了。一句 20 词的英文在 word 级是 20 步，在 byte 级可能是 100+ 步，注意力的二次复杂度让它很吃亏。

> We observed a similar performance gap in our own attempts to train standard byte-level LMs on WebText.

GPT-2 团队自己实验也印证了这个 gap。所以**纯 byte-level 不行，纯 word-level 丢失通用性**——需要折中方案。

---

## 2. 第二段逐句精读：BPE 的登场与坑

> Byte Pair Encoding (BPE) (Sennrich et al., 2015) is a practical middle ground between character and word level language modeling which effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences.

**BPE（Byte Pair Encoding，字节对编码）** 就是那个"折中方案"。
它的核心思想一句话：

> **高频组合合并成"词级 token"，低频组合保持"字符级 token"。**

直觉例子（假设词表上限 5000）：

- `the`、`ing`、`tion` 很常见 → 合成整体 token
- `schizophrenia` 罕见 → 拆成 `schi + zo + phre + nia` 这样的碎片

这样既控制了序列长度（常见词只占一个 token），又保留了罕见词的可表示性（拆成已知碎片拼回去）。

> Despite its name, reference BPE implementations often operate on Unicode code points and not byte sequences.

**重点踩坑点 1**：虽然名字叫 "Byte" Pair Encoding，但**标准实现其实是在 Unicode code point 上操作的，不是在 byte 上**。

这意味着什么？Unicode 有 10 万+ 字符（中日韩、emoji、阿拉伯文……）。

> These implementations would require including the full space of Unicode symbols in order to model all Unicode strings. This would result in a base vocabulary of over 130,000 before any multi-symbol tokens are added. This is prohibitively large compared to the 32,000 to 64,000 token vocabularies often used with BPE.

如果想让这种 code-point 级 BPE 能表示"所有 Unicode"，**基础词表就得装下 13 万+ 个 Unicode 符号**——然后再在上面加合并出来的 subword。
但实际 BPE 词表通常只有 **32k–64k**。13 万的 base 根本装不下，大部分 Unicode 字符只能被迫 `<UNK>`，通用性又丢了。

> In contrast, a byte-level version of BPE only requires a base vocabulary of size 256.

对比之下，**byte-level BPE 的 base 词表只需要 256**（0x00–0xFF）。
再叠加合并出来的 subword，最终 GPT-2 用的是 **50,257** 的词表（见 §2.3）——干净、完备、通用。

> However, directly applying BPE to the byte sequence results in sub-optimal merges due to BPE using a greedy frequency based heuristic for building the token vocabulary.

但是！**直接在 byte 上跑 BPE 会出问题**。原因是 BPE 是**贪心按频率合并**：谁共同出现得多，就把谁合并成新 token。

> We observed BPE including many versions of common words like dog since they occur in many variations such as dog. dog! dog?. This results in a sub-optimal allocation of limited vocabulary slots and model capacity.

**具体病症**：合并出现大量冗余 token。
比如 `dog` 这个词本身很常见，而 `dog.`、`dog!`、`dog?`、`dog,` 也都很常见。
贪心 BPE 会把这些都单独合并成 token：

```
dog      → id 1234
dog.     → id 2345
dog!     → id 3456
dog?     → id 4567
...
```

结果：

1. 词表里塞满了 **"同一个词 + 不同标点"** 的变体，浪费宝贵的词表槽位
2. 模型需要分别学 `dog`/`dog.`/`dog?` 的表示，能力被稀释

> To avoid this, we prevent BPE from merging across character categories for any byte sequence.

**GPT-2 的解决方案**：在训练 BPE 时**禁止跨字符类别合并**。
"字符类别" 指字母 / 数字 / 标点 / 空格 / 控制符 等大类。这样 `dog` 和 `.` 分属字母类和标点类，BPE 就不会把 `dog.` 合并成一个 token，强制拆成 `dog` + `.`。

> We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.

**唯一例外：空格**。
允许空格和前/后字符合并。这就是 GPT-2 tokenizer 里 **`Ġ`（G 加上面一点，其实是 `\u0120`，代表"前导空格"）** 的由来。你看 tokenizer 输出经常有：

```
"hello world" → ["hello", "Ġworld"]
                           ↑ 这个 Ġ 就代表前面原本有个空格
```

这样做既大幅提高了压缩率（`Ġthe`、`Ġof`、`Ġand` 这种高频"空格+词"能合并），又不会让同一个单词在不同位置被切成不同碎片。

---

## 3. 第三段：这套设计带来的好处

> This input representation allows us to combine the empirical benefits of word-level LMs with the generality of byte-level approaches.

一句话总结 byte-level BPE：**word-level 的效果 + byte-level 的通用性**。

> Since our approach can assign a probability to any Unicode string, this allows us to evaluate our LMs on any dataset regardless of pre-processing, tokenization, or vocab size.

最实在的工程收益：**可以直接在任何预处理过的评测集上评价 GPT-2**，不用针对每个数据集做 tokenization 兼容。
这也是为什么后来 GPT-2 能在 PTB / WikiText / 1BW 等各种奇葩预处理的 benchmark 上做 zero-shot 评估（§3.1 的 Table 3）。

---

## 4. 整体流程图：一个字符串是怎么变成 token id 序列的

```
原始字符串（任意 Unicode）
        │
        │  ① UTF-8 编码
        ▼
字节序列（byte sequence, 每字节 0–255）
        │
        │  ② Byte-level BPE 合并（已预训练好的合并规则表）
        │      - 不跨字符类别合并（字母↔标点 禁止）
        │      - 空格例外（允许 " x" 合并 → Ġx）
        ▼
subword token 序列（每个 token 是若干 byte 的组合）
        │
        │  ③ 查词表（大小 50,257）
        ▼
token id 序列（整数列表，喂给 embedding 层）
```

对比 BERT 的 Input Representation（论文 §3）：

|  | GPT-2 | BERT |
|---|---|---|
| tokenizer | **Byte-level BPE** | WordPiece |
| 词表大小 | 50,257 | 30,000 |
| 特殊 token | 几乎没有（`<\|endoftext\|>` 一个） | `[CLS]`、`[SEP]`、`[MASK]` |
| 句对建模 | 靠纯文本拼接 | Segment embedding（A/B 句） |
| 覆盖能力 | 任意 Unicode | 限于词表内 WordPiece，否则 `[UNK]` |
| 方向 | 单向（causal） | 双向 |

---

## 5. 和第 1 课代码的对齐

第 1 课 `01_word_embeddings.py` 里我们做的事情（简化版）：

```python
vocab = {"hello": 0, "world": 1, ...}          # 词表
ids = [vocab[w] for w in text.split()]          # 简单按空格切，再查表
embedding = nn.Embedding(vocab_size, d_model)   # token id → 向量
x = embedding(ids) + positional_encoding        # 加位置信息
```

GPT-2 在生产系统里做的升级：

| 教学代码 | GPT-2 生产 |
|---|---|
| `text.split()` 按空格切 | **Byte-level BPE** 切（处理任意 Unicode） |
| 词表 = 训练集见过的单词 | 词表 = 预训练好的 50,257 个 subword |
| OOV 用 `<UNK>` | 无 OOV（最坏也能拆到单 byte） |
| 正余弦位置编码 | 可学习的位置嵌入（§2.3 "Model"） |

自己写 tokenizer 不现实，实际项目里用：

```python
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello, world!")
# → [15496, 11, 995, 0]
print(tok.convert_ids_to_tokens(ids))
# → ['Hello', ',', 'Ġworld', '!']   ← 注意 Ġ 的出现
```

可以亲手跑一下看 `Ġ` 是怎么冒出来的，加深印象。

---

## 6. 复盘（回答这 3 个问题就过关）

1. **GPT-2 为什么选 byte-level BPE 而不是 word-level 或 character-level？**
   - word-level：OOV 无法避免，遇到没见过的字符串就废
   - char/byte-level：词表小、无 OOV，但序列太长，大规模数据上效果不如 word
   - byte-level BPE：base 词表只 256，完全覆盖 Unicode；再用 BPE 合并常见 byte 组合成 subword，既短又通用

2. **原始 BPE 直接用在 byte 上会出什么问题？GPT-2 怎么改的？**
   - 问题：贪心合并让词表里充斥 `dog`、`dog.`、`dog!`、`dog?` 这种重复变体，浪费词表槽、稀释模型能力
   - 改法：**禁止跨字符类别合并**（字母/数字/标点/空格等大类之间不能合）；**空格作为唯一例外**，允许和后续字符合并，这就产生了 GPT-2 tokenizer 里标志性的 `Ġ` 前缀

3. **这套输入表示对下游评测有什么好处？**
   - 任何 Unicode 字符串都能打分 → 可以**不做任何预处理适配**，直接在 PTB、WikiText、1BW 等各种 benchmark 上做 zero-shot 评估，这是 §3 实验能全面铺开的前提

---

## 7. 延伸阅读（可选）

- Sennrich et al., 2015, *Neural Machine Translation of Rare Words with Subword Units* —— BPE 原始论文
- [HuggingFace `tokenizers` 库文档](https://huggingface.co/docs/tokenizers/) —— 想亲手训一个 byte-level BPE 可以用
- GPT-2 源码 `encoder.py` —— 官方的 byte-level BPE 实现，只有 ~120 行，值得读一遍
- 后续课程阅读：BERT §3 "Input/Output Representations" —— 对比 WordPiece + segment embedding 的另一种设计

---

# GPT-2 论文 §2.3 Model 逐段精读

> 目标章节：Section **2.3 Model**（原文 p.4）

§2.3 全文非常短（正文仅一段 + Table 2），但每一句都对应一个具体的"相对 GPT-1 / 原版 Transformer 的改动"。
逐句拆完，你就能画出 GPT-2 的完整架构图。

## 0. 一句话定位

> **GPT-2 的架构 ≈ GPT-1 的架构 + 4 项小改动 + 规模扩大**，仍然是 decoder-only Transformer。

如果你已经学完第 3、4 课（多头注意力 + Transformer Block），这一节要做的就是：**把 GPT-1 的 block 拿过来，改 4 个细节，然后从 12 层一路堆到 48 层**。

---

## 1. 第一句：用什么基座？

> We use a Transformer (Vaswani et al., 2017) based architecture for our LMs.

**基座就是 Transformer**。具体是哪种 Transformer？

- 原版 Transformer（Vaswani 2017）有 **Encoder + Decoder**，为机器翻译设计
- BERT 用 **Encoder-only**（双向注意力）
- GPT 系列用 **Decoder-only**（因果掩码，只能看左边）

GPT-2 继承 GPT-1 的选型：**decoder-only，带 causal mask**。这意味着训练目标只有一个——**next token prediction**（用前面所有 token 预测下一个）。

> The model largely follows the details of the OpenAI GPT model (Radford et al., 2018) with a few modifications.

**"largely follows GPT-1 + a few modifications"** —— 接下来就是列这 "a few modifications"。

---

## 补充：先搞懂 "sub-block"（子模块）是什么

论文这一段反复出现 **sub-block** 一词。它**不是一个官方术语**，只是描述 Block 内部结构的惯用说法：

**一个 Transformer Block 内部 = 两个 sub-block 串起来**：

```
Transformer Block
│
├─ Sub-block 1：Multi-Head Self-Attention
│    ┌─────────────────────────┐
│    │ LayerNorm               │
│    │ Multi-Head Attention    │  ← 这一整坨叫一个 sub-block
│    │ + 残差连接               │
│    └─────────────────────────┘
│
└─ Sub-block 2：Feed-Forward Network (FFN)
     ┌─────────────────────────┐
     │ LayerNorm               │
     │ Linear → GELU → Linear  │
     │ + 残差连接               │
     └─────────────────────────┘
```

所以：

- GPT（decoder-only）的一个 Block = **2 个 sub-block**，2 条残差通路
- 原版 Transformer 的 Decoder Block 有 **3 个 sub-block**（Self-Attn + Cross-Attn + FFN），GPT 砍掉了 Cross-Attn
- 论文原句 "LayerNorm was moved to the input of each sub-block" 就是说：**Attention 前放一个 LN，FFN 前再放一个 LN**，一个 Block 里一共有 2 个 LayerNorm

这块会在**第 4 课《Transformer Block》里亲手用代码搭出来**，届时对应 nanoGPT 风格的 `Block` 类里 `ln_1` / `ln_2` 两个 LayerNorm。现在先有个印象，往下读 §2.3 就不会卡住。

---

## 2. 修改点 1：LayerNorm 的位置（Pre-Norm）

> Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a pre-activation residual network (He et al., 2016)

这是 GPT-2 最重要的一个改动。对比：

### Post-Norm（原版 Transformer / GPT-1）

```
x ──► Sub-block (Attention or FFN) ──► Add ──► LayerNorm ──► 下一层
 │                                      ▲
 └──────────────────────────────────────┘
```

数学上：`y = LayerNorm(x + SubBlock(x))`

### Pre-Norm（GPT-2 开始，后来所有主流 LLM 都这么干）

```
x ──► LayerNorm ──► Sub-block ──► Add ──► 下一层
 │                                 ▲
 └─────────────────────────────────┘
```

数学上：`y = x + SubBlock(LayerNorm(x))`

**为什么这一改这么关键？**
关键在 **"残差通路（residual path）是否干净"**：

- Post-Norm：残差加完后又过一次 LayerNorm，等于每层都"扰动"一下残差流，深层训练时梯度会被反复缩放，容易不稳定
- Pre-Norm：**残差是完全干净的 identity 通路，从输入一路贯通到输出**，只有 sub-block 的分支会被 LayerNorm 规范化。梯度可以沿着残差通路直接回流，深层也能稳定训练

类比：

- Post-Norm 像每上一层楼都要过一次安检（信号每层被规范化一次）
- Pre-Norm 像修了一条从 1 楼到 48 楼的直梯（残差通路），每层的工作室在电梯旁边（sub-block 分支）

> 这也是为什么 GPT-1（12 层）用 Post-Norm 还能训，到了 GPT-2（最大 48 层）就必须换 Pre-Norm —— 深了以后 Post-Norm 基本训不动。

这就是"similar to a pre-activation residual network (He et al., 2016)"的含义 —— 借鉴了 ResNet v2 的思想。

## 3. 修改点 2：最后多加一个 LayerNorm

> and an additional layer normalization was added after the final self-attention block.

因为用了 Pre-Norm，**最后一层 block 的输出没有经过任何 LayerNorm**（残差通路是裸的）。
如果直接把它送进 LM Head（线性层 → softmax），各个维度的量级会很不均衡，训练不稳。
所以 GPT-2 在**最后一个 block 之后、LM Head 之前**再补一个 LayerNorm：

```
Embedding
   │
   ▼
[Block 1] ─┐
[Block 2]  │  (每个 block 内部用 Pre-Norm)
   ...     │
[Block N] ─┘
   │
   ▼
LayerNorm    ← 这就是 2.3 说的"additional layer normalization"
   │
   ▼
LM Head (Linear → Softmax over vocab)
```

在 HuggingFace 的 GPT-2 实现里，这个就叫 `ln_f`（final LayerNorm），代码里能直接找到。

## 4. 修改点 3：残差层缩放初始化（1/√N）

> A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers.

这是一个**初始化 trick**，解决 "深层 Pre-Norm 模型里残差累加导致激活值爆炸" 的问题。

### 直觉推导

假设每个 sub-block 的输出方差是 `σ²`，残差通路累加 N 次：

```
x_N = x_0 + f_1(x_0) + f_2(x_1) + ... + f_N(x_{N-1})
```

如果每个 `f_i` 的输出独立且方差为 `σ²`，那么 `x_N` 的方差近似是：

```
Var(x_N) ≈ Var(x_0) + N · σ²
```

**层数 N 越大，方差线性增长**。GPT-2 最大模型 N = 48 个残差层（实际是 48×2=96 个 sub-block 残差），不加控制，前向时激活值会越来越大，训练一开始就炸。

### 解法

**把残差分支里的权重初始化再乘一个 1/√N 的缩放因子**。
这样每个 sub-block 输出的方差变成 `σ²/N`，累加 N 次后：

```
Var(x_N) ≈ Var(x_0) + N · (σ²/N) = Var(x_0) + σ²
```

**总方差不随深度膨胀**，深层也能稳定训练。

注意：**只缩放"残差分支"里的权重**（Attention 的输出投影 `W_O`、FFN 的第二层 `W_2`），不影响 attention 本身的 Q/K/V 计算。

这个 trick 后来在 GPT-3、Llama 等模型里继续沿用，是"可稳定训练深层 Transformer" 的三件套之一（另外两件是 Pre-Norm 和 warmup LR schedule）。

## 5. 修改点 4：词表、上下文长度、batch 一起放大

> The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.

| 超参 | GPT-1 | GPT-2 | 含义 |
|---|---|---|---|
| 词表大小 | 40,478 | **50,257** | Byte-level BPE 新训的词表（见 §2.2） |
| 上下文长度 | 512 | **1024** | 一次能看 1024 个 token 的历史，长文理解更强 |
| Batch size | 64 | **512** | 大 batch 更稳，也更吃显存 |

这里的"词表 50,257"和 §2.2 的"byte-level BPE"是**完全一致的配套**：

- 256（基础 byte）
- \+ 50,000（BPE 合并出的 subword）
- \+ 1（`<|endoftext|>` 特殊 token）
- = 50,257

上下文从 512 → 1024 看似只是 ×2，但自注意力是 **O(n²)** 复杂度，计算量其实是 ×4，所以这个改动也不"便宜"。

## 6. Table 2：四档规模

论文正文里给的 Table 2：

| Parameters | Layers (N) | d_model |
|---|---|---|
| 117M | 12 | 768 |
| 345M | 24 | 1024 |
| 762M | 36 | 1280 |
| **1542M (GPT-2)** | **48** | **1600** |

几个要点：

1. **"GPT-2" 严格来说指的只是最大那档 1.5B 模型**，其他三档只是对照用的缩放梯度
2. 最小档（117M, 12 层, 768）**与 GPT-1 规模对齐** —— 论文里明说 "The smallest model is equivalent to the original GPT"
3. 第二档（345M, 24 层, 1024）**与 BERT-Large 规模对齐** —— "the second smallest equivalent to the largest model from BERT"
4. 头数（A）、FFN 大小论文正文没给，但惯例是：
   - `A = d_model / 64`（每个头 64 维）→ 12, 16, 20, 25 个头
   - `FFN hidden = 4 × d_model`（原版 Transformer 就是这个比例）

这个"**按 log 均匀分布采四档尺寸**"的做法，其实就是 GPT-3 "规模定律（scaling law）" 论文的前身—— GPT-2 已经在暗示：**同一架构下，模型越大，效果越好，能力越涌现**（后面 §3 的所有实验就是在证明这一点）。

---

## 7. 整体架构图（GPT-2 完整 forward）

```
输入 token ids  (batch, seq_len)
        │
        ▼
┌───────────────────────┐
│ Token Embedding       │  查表：id → d_model 维向量
│ (50257 × d_model)     │
└───────────┬───────────┘
            │
            + Positional Embedding (1024 × d_model, 可学习)
            │
            ▼
   ┌──────────────────────────┐
   │ Block 1                  │
   │  ┌────────────────────┐  │
   │  │ LayerNorm (Pre)    │  │ ← Pre-Norm（改动 1）
   │  │ Masked Multi-Head  │  │
   │  │   Self-Attention   │  │ ← causal mask
   │  │ + Residual         │  │ ← W_O 初始化 ×1/√N（改动 3）
   │  └────────────────────┘  │
   │  ┌────────────────────┐  │
   │  │ LayerNorm (Pre)    │  │
   │  │ FFN (4·d_model)    │  │
   │  │ + Residual         │  │ ← W_2 初始化 ×1/√N
   │  └────────────────────┘  │
   └──────────────┬───────────┘
                  │
                 ... (重复 N 次，N=12/24/36/48)
                  │
                  ▼
         LayerNorm (ln_f)     ← 最后再来一次（改动 2）
                  │
                  ▼
         LM Head (Linear, 权重与 Token Embedding 共享)
                  │
                  ▼
     logits  (batch, seq_len, 50257)
                  │
                  ▼
         softmax → 下一个 token 的概率分布
```

其中 **LM Head 与 Token Embedding 权重共享**（weight tying）是 GPT 系列的惯例，虽然 §2.3 没明说，但 GPT-1 和 GPT-2 代码里都是这么做的：输入 embedding 矩阵 `E ∈ R^{V×d}` 转置后当 LM Head 用，省一份大参数。

---

## 8. 和第 3/4/5 课代码的对齐

| §2.3 描述 | 第几课 | 代码对应 |
|---|---|---|
| "Transformer-based architecture" | 第 3、4 课 | `MultiHeadAttention`、`TransformerBlock` |
| "LayerNorm moved to the input of each sub-block" | 第 4 课 | Block 内部 `x + Attn(LN(x))`、`x + FFN(LN(x))` |
| "additional LayerNorm after the final self-attention block" | 第 5 课 | `model.ln_f`，在最后一个 block 之后、LM Head 之前 |
| "scale residual weights by 1/√N" | 第 5 课 | 初始化时对 `attn.c_proj.weight`、`mlp.c_proj.weight` 乘 `1/sqrt(2*n_layer)` |
| "vocab = 50,257, ctx = 1024" | 第 5 课 | 模型配置 `vocab_size`、`block_size` |

如果你在第 5 课照着 nanoGPT 写迷你 GPT，这 4 个改动全都会在代码里一一出现，到时候对照回来读，会比现在理解更深。

---

## 9. 复盘（回答这 4 个问题就过关）

1. **GPT-2 相对 GPT-1 的 4 个架构改动分别是什么？为什么？**
   - Pre-Norm：让残差通路干净，深层训练稳定
   - 最后加一个 LayerNorm：补偿 Pre-Norm 留给 LM Head 的"未归一化"输出
   - 残差权重 ×1/√N 初始化：防止残差累加导致激活爆炸
   - 放大 vocab / ctx / batch：50257 / 1024 / 512

2. **Pre-Norm 和 Post-Norm 的区别？为什么深层 LLM 都选 Pre-Norm？**
   - Post-Norm：`y = LN(x + f(x))`，残差被 LN 扰动
   - Pre-Norm：`y = x + f(LN(x))`，残差是干净 identity
   - 深层（几十到上百层）时 Pre-Norm 梯度传播更稳，训练不炸

3. **1/√N 初始化是在缩放什么？解决什么问题？**
   - 缩放的是**残差分支里的输出投影权重**（`W_O` in attention、`W_2` in FFN），不是所有权重
   - 解决的是：N 个残差层累加时，激活值方差随 N 线性增长、深层爆炸的问题；缩放后总方差与深度解耦

4. **GPT-2 的最大模型是多大？和 GPT-1、BERT-Large 怎么对标？**
   - GPT-2 = 1542M 参数，48 层，d_model=1600
   - 最小档 117M 对标 GPT-1
   - 第二小档 345M 对标 BERT-Large
   - 这是"同架构 + 不同 scale" 的第一次系统化展示，为 GPT-3 的 scaling law 铺路

---

## 10. 延伸阅读（可选）

- Xiong et al., 2020, *On Layer Normalization in the Transformer Architecture* —— **Pre-Norm vs Post-Norm 的理论分析**，强烈推荐配合这一节读
- He et al., 2016, *Identity Mappings in Deep Residual Networks* —— 论文里引用的 "pre-activation ResNet"，Pre-Norm 思想的源头
- nanoGPT `model.py` —— 100 行代码实现本节全部 4 个改动，是看 GPT-2 架构落地的最佳参考
- 后续课程阅读：GPT-3 §2.1 *Model and Architectures* —— 同样的架构继续往上堆到 175B，看 scaling 的极致
