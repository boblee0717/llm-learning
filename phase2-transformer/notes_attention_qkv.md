# 第 2 课配套精读：Attention 三论文中英对照

> 配合第二阶段第 2 课《自注意力机制》一起学习。
> 三篇论文同时拆解，目的只有一个：**理解 Q/K/V、scaled dot-product、causal mask 这三个关键词，到底在论文里长什么样**。

涉及章节：

| 论文 | 章节 | PDF 路径 |
|------|------|----------|
| Attention Is All You Need (Vaswani et al., 2017) | §3.2 Attention、§3.2.1 Scaled Dot-Product Attention | `papers/Attention_Is_All_You_Need_2017.pdf` |
| GPT-2 (Radford et al., 2019) | §2.3 Model（聚焦 masked self-attention / 自回归约束） | `papers/GPT2_Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf` |
| GPT-3 (Brown et al., 2020) | §2.1 Model and Architectures（看 decoder 堆叠中 attention 的使用） | `papers/GPT3_Language_Models_are_Few-Shot_Learners_2020.pdf` |

阅读顺序建议：**先 Vaswani §3.2 打地基 → 再 GPT-2 §2.3 看自回归约束如何落到 attention 上 → 最后 GPT-3 §2.1 看同一架构如何被堆到极致**。

---

# Part 1：Attention Is All You Need §3.2 / §3.2.1 中英对照精读

## 0. 先建立直觉：Attention 到底在解决什么？

在 RNN / LSTM 时代，处理序列的方式是"一步一步往后走"，第 t 步只能基于第 t-1 步的隐藏状态。这带来两个致命问题：

1. **长距离依赖丢失**：第 100 个词想"看到"第 1 个词，信息要经过 99 步传递，早被冲淡了
2. **无法并行**：必须按顺序算，GPU 浪费

Attention 的核心思想一句话：

> **每个位置直接和序列里所有位置算"相关度"，然后按相关度加权求和。**

不再"传话"，而是"全员开会"。这就同时解决了长距离依赖（直连）和并行（矩阵乘法一次算完）两个问题。

而 Q / K / V 三个矩阵，就是把这个"全员开会"的过程数学化的工具。

---

## 1. §3.2 Attention 引言段

> **EN**: An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

**中**：注意力函数可以描述为：把一个 **query**（查询）和一组 **key-value 对**（键-值对）映射到一个 **output**（输出）；其中 query、keys、values、output 全都是向量。

**精读**：

这一句是整个 Transformer 的基石，把它拆开：

- **Query (Q)**：当前位置"想问什么"。比如"我是动词，请告诉我我的主语在哪"
- **Key (K)**：每个位置"提供什么样的索引"。类似图书馆每本书背后的标签
- **Value (V)**：每个位置"实际携带的内容"。类似图书馆里书的真实内容
- **Output**：用 Q 去匹配所有 K，按匹配程度对 V 加权求和

**生活类比**：你（Q）走进图书馆找"机器学习"相关的书。每本书的书脊（K）都有标签，你拿你的需求去对每个标签算"相关度"，相关度高的书（V）你就多翻几页，相关度低的就略过。最后你脑子里记下的"机器学习的认知"= 所有书的内容按相关度加权求和。

> **EN**: The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

**中**：输出是 values 的**加权求和**，每个 value 的权重由 query 与对应 key 的"相容性函数（compatibility function）"算出。

**精读**：

- "weighted sum of values" → 这就是为什么最后输出维度和 V 一样
- "compatibility function" → 用 Q 和 K 算相似度的函数。可以是点积、加性、余弦……Vaswani 选了**点积**（最快、最适合 GPU）

到这里你已经知道 Attention 的"形状"了：**输入一堆 Q/K/V，输出一组按相关度加权后的向量**。下一节 §3.2.1 就要回答："具体的 compatibility function 长什么样？"

---

## 1.5 补充：Q/K/V 到底是怎么"算"出来的？（第 2 课最常见的疑问）

论文 §3.2 里只说"query / keys / values are all vectors"，但**没明说这三个向量是怎么得来的**——这是初学者最容易卡的点。下面分三层把它讲透。

### 1.5.1 数学上：三次矩阵乘法，就这么简单

设输入是一段 token 的 embedding（来自第 1 课的"词嵌入 + 位置编码"输出），形状 $(n, d_{model})$，记作 $X$：

- $n$ = 序列长度（比如 5 个词）
- $d_{model}$ = 每个词的向量维度（比如 512）

模型里有**三个可学习的权重矩阵** $W_Q, W_K, W_V$，形状都是 $(d_{model}, d_k)$（V 的最后一维可以不同记作 $d_v$，但通常一样，先按一样讲）：

$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
$$

代码就是三行：

```python
Q = X @ W_Q   # (n, d_model) @ (d_model, d_k) → (n, d_k)
K = X @ W_K   # (n, d_k)
V = X @ W_V   # (n, d_k)
```

**就这样。** Q/K/V 就是把同一份输入 $X$，**用三个不同的"投影矩阵"投影到三个不同空间**得到的。

> 📌 **代码回扣（推荐做完一遍再回来看）**
>
> `02_self_attention.py` 第 84-85 行**故意**把 $d_v$ 设成和 $d_k$ 不同（`d_k = 4`、`d_v = 3`），就是为了让你在跑代码时**亲眼看到**两件事：
>
> 1. **$d_k$ 和 $d_v$ 承担的职责不同**
>    - $d_k$ → 决定 Q/K 的"匹配空间"大小（影响 score 的方差，所以才要除 $\sqrt{d_k}$）
>    - $d_v$ → 决定 V 的"内容携带能力"，也直接决定 **attention 输出的最后一维**
>    - 注意力权重矩阵 `(n, n)` 跟 $d_k$ / $d_v$ **都无关**
> 2. **形状链路一目了然**
>    - `Q, K`: $(n, d_k)$ —— 必须共享 $d_k$，否则 `Q @ K.T` 算不出来
>    - `V`: $(n, d_v)$ —— 独立维度
>    - `output = weights @ V`: $(n, n) @ (n, d_v) = (n, d_v)$ ← **输出宽度由 $d_v$ 决定**
>
> 大多数论文/教材为了简化都直接写 $d_k = d_v$，导致初学者很难分清"哪个维度是哪个负责的"。**做完代码再回来读论文公式**，会有"原来这俩符号在 Vaswani §3.2.1 里就是分开写的"的恍然感——这正是 §3.2.2 多头注意力把 $h$ 个 $d_v$ 维 head concat 起来再投回 $d_{model}$ 的基础。

### 1.5.2 $W_Q / W_K / W_V$ 这三个矩阵从哪来？

**随机初始化 + 训练时反向传播自动学出来**。

- 模型刚出生时，$W_Q, W_K, W_V$ 是随机数，没有任何意义
- 训练时，每次前向算 attention → 算 loss → 反向传播会更新这三个矩阵
- 训练几百万步之后，它们慢慢"学会"：
  - $W_Q$ 学到"怎么把一个词映射成有意义的查询信号"
  - $W_K$ 学到"怎么把一个词映射成可以被查询的索引信号"
  - $W_V$ 学到"怎么把一个词映射成可以被搬运的内容信号"

> 你只需要知道：**三个矩阵都是模型参数，跟你训练 MLP 时的权重矩阵 $W$ 完全一样，都是 SGD/Adam 学出来的**，不是人工设计的。

### 1.5.3 为什么"同一个 X"要算三遍、得到三个不同的东西？

最直观的类比 —— **图书馆查书**：

| 角色 | 数学对应 | 图书馆角色 |
|------|---------|----------|
| Q（query）| $X W_Q$ | 你写的"查询条"："我想找机器学习的书" |
| K（key）| $X W_K$ | 每本书贴的标签："这本是 ML 教材" |
| V（value）| $X W_V$ | 书的实际内容（你最终要带走的东西）|

**为什么不能用同一个表示？** 因为"我想找什么"（Q）和"我自己是什么"（K）不是同一件事。

举个语言上的例子，句子 `The cat sat on the mat`，处理 `sat` 这个词：

- `sat` 作为**动词**，它的 **Q** 应该表达："我需要找到我的主语"
- `cat` 作为**名词**，它的 **K** 应该表达："我是一个名词，可以当主语"
- `cat` 作为**名词**，它的 **V** 应该表达："我携带的语义是'猫'这个动物"

如果只用一个矩阵，模型没法同时表达这三个角色。**用三个不同的投影矩阵，让同一个词在不同角色里有不同表达**——这就是 Q/K/V 分离的本质。

### 1.5.4 第 2 课你需要做到什么程度

按"必须懂 → 推荐懂 → 不必懂"分三档：

| 程度 | 内容 | 第 2 课要求 |
|------|------|-----------|
| **必须懂** | Q/K/V 就是 `X @ W_Q/K/V` 三个矩阵乘法 | ✓ |
| **必须懂** | 三个 W 矩阵是模型参数，训练时学出来 | ✓ |
| **必须懂** | 算出来后接着做 $\text{softmax}(QK^T/\sqrt{d_k}) V$ | ✓ |
| **必须懂** | 最终输出形状还是 $(n, d_v)$，和输入 $X$ 同形（方便残差连接） | ✓ |
| **推荐懂** | 为什么要分成三个角色，而不是一个矩阵搞定 | 有个直觉就行 |
| **推荐懂** | 多头注意力里 W_Q/K/V 是怎么切成多份的 | 第 3 课讲 |
| **不必懂** | W 在训练中具体学到了什么模式（比如某个头专门看主谓关系）| 看可视化论文，第 5 课之后再读 |

### 1.5.5 对应到 `02_self_attention.py` 看一眼最清楚

第 2 课代码里大概率会有这样的片段：

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        Q = self.W_q(x)      # 这就是 X @ W_Q
        K = self.W_k(x)      # 这就是 X @ W_K
        V = self.W_v(x)      # 这就是 X @ W_V

        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return out
```

- 上面三个 `nn.Linear(d_model, d_k, bias=False)` 内部就是一个 $(d_{model}, d_k)$ 的权重矩阵
- 这三个 `Linear` 的权重，就是 $W_Q, W_K, W_V$
- 它们在 `__init__` 里被 PyTorch 自动随机初始化，在训练时被自动更新

**跑代码时盯紧这三件事**：

1. 输入 `x` 形状是什么 → 比如 `(2, 5, 64)`（batch=2, seq=5, d_model=64）
2. `Q, K, V` 形状是什么 → `(2, 5, d_k)`
3. 最终 `out` 形状是什么 → 通常和 `x` 一样 `(2, 5, 64)`，方便往下接残差

### 1.5.6 一句话总结

**Q/K/V = 同一个输入 $X$ 分别乘以三个可学习的权重矩阵 $W_Q, W_K, W_V$**。三个矩阵随机初始化、靠反向传播自动学出来，目的是让同一个词在"查询者 / 被查询者 / 携带内容者"三个角色里有不同的表达。读懂这一节，再回头看 §3.2.1 的公式 $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ 就只剩"形状变化"这一件事了。

---

### 1.5.7 深挖："$XW_Q = Q$ 里每个 token 代表一种查询还是一组查询？"

这是精读 §3.2 时最容易被字面意思误导的一个点，单独拎出来讲透。

#### （1）结论先行

- **单头注意力（本节）**：每个 token **只产出一个查询向量 $q_i \in \mathbb{R}^{d_k}$**，在数学上就是"一次查询"（one singular query）。
- **多头注意力（第 3 课）**：每个 token 会产出 $h$ 个查询向量 $q_i^{(1)}, \dots, q_i^{(h)}$，这才是真正意义上的"一组查询"。
- 所以"一组查询"这个直觉是对的，但它的**正确归宿是 Multi-Head，不是 $XW_Q$ 本身**。

#### （2）从形状反推

$$
Q = XW_Q \in \mathbb{R}^{n \times d_k},\quad q_i = x_i W_Q \in \mathbb{R}^{d_k}
$$

- 第 $i$ 个 token 对应 $Q$ 的第 $i$ 行，就一行
- 这一行整体作为一个查询，去和每一个 $k_j$ 做点积 $q_i \cdot k_j$，得到**一个**标量相关度
- 所以"一个 token → 一个 $q_i$ → 一次查询"，这条链在单头里是刚性的

#### （3）那 $d_k$ 维里装的是什么？

$q_i$ 这个 $d_k$ 维向量，是一个**"把多种语义特征压缩在同一条向量里"的复合查询**。类比搜索框：

> 你在搜索框输入「北京 2026 春天 便宜 机票」——这是**一条查询**，但内部编码了「地点 / 时间 / 价格 / 品类」多个子意图。机器不会拆成 5 个独立查询去并行搜，而是把整串 embedding 成一个向量整体匹配。

$q_i$ 也是如此：它的某些维度可能隐式编码"我是不是动词"、"我在找主语吗"、"我的时态是什么"……但这些子意图是**纠缠在一起的、隐式的**，不是正交分明的 $d_k$ 条独立查询。

| 说法 | 对不对 |
|------|--------|
| 每个 token 一个查询，查询是一个**标量** | ❌ |
| 每个 token 一个查询，查询是一个**向量** | ✅ 本节 |
| 每个 token 一个查询，向量内部**编码了多种语义特征** | ✅ 正确直觉 |
| 每个 token **同时发出多个独立查询** | 单头 ❌ / 多头 ✅ |

#### （4）判断公式（通用）

以后看任何 attention 变体，一条铁律：

> **一次 $q \cdot k$ 点积 = 一次查询；一个 token 产出几个 $q$ 向量，就代表它发出几次查询。**

- 单头：每 token 1 个 $q$ → 1 次查询
- $h$ 头多头：每 token $h$ 个 $q$ → $h$ 次并行查询
- MQA / GQA：$q$ 头数和 $k$ 头数可以不对称，但数查询次数仍看 $q$ 的个数

#### （5）关于 $d_k$：它是查询的"特征维度"，但不是随便选

你问得对 —— **$d_k$ 描述的就是"一个查询向量 $q_i$ 能携带多少维特征"**，本质上是个超参数 / 经验值。但有几条**硬约束**不能违反：

| 约束 | 说明 | 为什么 |
|------|------|--------|
| **硬约束 1**：$Q$ 和 $K$ 的维度必须都是 $d_k$ | $W_Q, W_K \in \mathbb{R}^{d_{model} \times d_k}$ | 因为要做 $QK^T$ 点积，两边列数必须对齐 |
| **硬约束 2**：$V$ 的维度 $d_v$ 可以和 $d_k$ 不同 | $W_V \in \mathbb{R}^{d_{model} \times d_v}$ | $V$ 只参与加权求和，不参与点积 |
| **硬约束 3（多头里）**：$h \times d_k = d_{model}$ | 比如 $d_{model}=512, h=8 \Rightarrow d_k=64$ | 这样拼接后维度能对齐，方便残差连接 |
| **软约束（经验）**：$d_k$ 通常取 64 | Vaswani 原论文 base 模型 $d_{model}=512, h=8, d_k=64$ | 太小 → 查询表达力不足；太大 → 点积方差爆炸、$\sqrt{d_k}$ 救不回来、算力浪费 |

所以可以说：

> **$d_k$ 是经验值，但不是"任意值"——它由 $d_{model}$ 和头数 $h$ 共同约束，典型取 64。**

换句话说，$d_k$ 是个**在架构设计里被"一起拍板"的超参数**，不是你能单独调的自由变量。第 3 课讲多头时会看到，真正的自由度其实是 $d_{model}$ 和 $h$ 这两个，$d_k$ 跟着它俩定。

#### （6）一句话收束

> **$XW_Q = Q$ 里，每个 token 对应一个 $q_i$，是"一个复合查询向量"而非"多个独立查询"；真正的"一组查询"要等到多头注意力里才出现。$d_k$ 是 $q_i$ 的特征维度，属于经验超参，但受 $d_{model}$ 和头数 $h$ 联合约束，不是任选。**

---

## 2. §3.2.1 Scaled Dot-Product Attention 逐句精读

### 2.1 命名与公式

> **EN**: We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$.

**中**：我们把这种特定的 attention 称为 **"Scaled Dot-Product Attention"（缩放点积注意力）**。输入由维度为 $d_k$ 的 queries 和 keys、维度为 $d_v$ 的 values 组成。

**精读**：

- 名字三个词分别是：**Scaled**（缩放）+ **Dot-Product**（点积）+ **Attention**
- Q 和 K 的维度必须一样（都是 $d_k$），因为要做点积
- V 的维度可以不一样（$d_v$），最终 output 维度跟 V 走

> **EN**: We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.

**中**：用 query 与所有 key 计算点积，然后**每个都除以 $\sqrt{d_k}$**，再用 softmax 得到 values 上的权重。

**精读**：

这一句把整个流程一次性讲完，对应公式 (1)：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

> 📌 **公式里"藏"了一个 mask 没写出来**
>
> 这个公式来自 Vaswani §3.2.1，是"通用版" attention——**默认不带 mask**。但实际用在 decoder（GPT）里时，softmax 的输入要先加一个 mask 矩阵 $M$：
>
> $$
> \text{Attention}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
> $$
>
> 其中：
>
> $$
> M_{ij} = \begin{cases} 0 & \text{允许位置 } i \text{ 看位置 } j \\ -\infty & \text{禁止位置 } i \text{ 看位置 } j \end{cases}
> $$
>
> 对应代码就是 `02_self_attention.py` 第 201-204 行的 `scores - mask * 1e9`（用 `1e9` 近似 $\infty$）。
>
> **关键：mask 必须加在 softmax 之前，不是之后**——
>
> | 时机 | 后果 |
> |------|------|
> | softmax **之前** mask（`scores + M`） | $e^{-\infty}=0$，分母里压根不出现被屏蔽位置的 logit → softmax 自然归一（行和=1）+ 无信息泄露 ✓ |
> | softmax **之后** mask（`weights * 0`） | 分母里**已经包含了未来位置的 $e^{x_j}$** → 概率分布破了（行和 < 1）+ 信息已泄露 ✗ |
>
> 论文 §3.2.3 也明确说："masking out (setting to −∞) all values in the **input of the softmax** which correspond to illegal connections"——**input of the softmax** 这五个字就是在强调"必须在 softmax 之前"。
>
> **为什么 §3.2.1 不显式把 $M$ 写进公式？** 因为 mask 是**任务相关的**，不是 attention 本身的属性——
>
> - BERT encoder：**没有 mask**（双向）
> - GPT decoder：**causal mask**
> - 翻译里 cross-attention：**没有 causal mask**，但可能有 padding mask
>
> 所以论文把通用公式留在 §3.2.1，到 §3.2.3 介绍 decoder 时再补一句"对 decoder self-attention 要加 mask"。Part 2 §3 那段 `masked_fill(... -inf)` 代码就是这句话的真身。

**逐项拆开**：

| 步骤 | 形状变化 | 含义 |
|------|----------|------|
| $QK^T$ | (n, d_k) × (d_k, m) → (n, m) | 第 i 个 query 和第 j 个 key 的"相关度" |
| $\div \sqrt{d_k}$ | 形状不变 | **缩放**（下一段会讲为什么） |
| softmax | 行方向归一化 | 每一行（每个 query）得到一个总和为 1 的"权重分布" |
| × V | (n, m) × (m, d_v) → (n, d_v) | 用权重对 V 加权求和 |

n = query 的个数，m = key/value 的个数。在自注意力（self-attention）里 n = m = 序列长度。

### 2.2 矩阵化的形式

> **EN**: In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V.

**中**：实际计算时，我们把一组 queries 打包成矩阵 Q，同时计算所有 query 的注意力。Keys 和 values 也分别打包成矩阵 K、V。

**精读**：

这就是 Transformer 能并行化的关键——**整个序列的 attention 一次矩阵乘法搞定**，不用像 RNN 那样一步一步算。这一句话翻译成 PyTorch 大概就是：

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (..., n, m)
weights = scores.softmax(dim=-1)                    # (..., n, m)
out = weights @ V                                   # (..., n, d_v)
```

四行代码，对应一篇 transformer 论文的核心。

### 2.3 为什么要除以 √d_k —— 整篇论文最难也最关键的一段

> **EN**: The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $1/\sqrt{d_k}$.

**中**：最常用的两种 attention 函数是 **加性注意力（additive attention）** 和 **点积（乘性）注意力（dot-product attention）**。点积注意力与我们的算法基本相同，**唯一区别就是多了 $1/\sqrt{d_k}$ 这个缩放因子**。

> **EN**: Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.

**中**：加性注意力用一个带单隐层的前馈网络来算相容性。即 $\text{score}(q,k) = w^T \tanh(W_q q + W_k k)$，参数更多、计算更慢。

> **EN**: While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

**中**：两者理论复杂度相近，但**点积注意力在实践中快得多、内存占用更少**，因为它能用高度优化的矩阵乘法实现（CUDA / cuBLAS）。

> **EN**: While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$.

**中**：当 $d_k$ 较小时两者效果相近；但 **$d_k$ 较大时，没有缩放的点积注意力会输给加性注意力**。

> **EN**: We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $1/\sqrt{d_k}$.

**中**：我们怀疑：当 $d_k$ 较大时，**点积的数值会变得很大，使 softmax 进入梯度极小的饱和区**。为抵消这个效应，我们用 $1/\sqrt{d_k}$ 缩放点积。

**精读 —— 这是整篇论文的"灵魂段落"**：

为什么"d_k 大 → 点积大"？做一个简单的方差计算：

假设 $q$ 和 $k$ 都是独立的、均值 0、方差 1 的随机向量，那么：

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

每一项 $q_i k_i$ 的方差是 1，独立的 $d_k$ 项相加，**总方差 = $d_k$**，标准差 = $\sqrt{d_k}$。

也就是说，**$d_k$ 越大，点积的数值波动范围就越大**（量级随 $\sqrt{d_k}$ 增长）。

**为什么这会害死 softmax？** 看一个直观例子：

| 输入 | softmax 输出 | 梯度 |
|------|--------------|------|
| `[1, 2, 3]` | `[0.09, 0.24, 0.67]` | 健康，梯度均匀 |
| `[10, 20, 30]` | `[2e-9, 4.5e-5, ~1.0]` | **几乎全压在 1 上，其他位置梯度 ≈ 0** |

softmax 的梯度形式是 $p_i(1 - p_i)$，当 $p_i \to 1$ 或 $p_i \to 0$ 时梯度都 ≈ 0，**反向传播信号无法回流**，模型卡住。

**解决办法**：在送进 softmax 之前，把点积除以 $\sqrt{d_k}$，让方差重新拉回到 1，softmax 就工作在"健康区域"。

> 一句话总结：**$\sqrt{d_k}$ 不是数学美感，是实打实为了让 softmax 不要饱和**。

### 2.4 Mask（论文里只有一句，但极其重要）

> **EN**: We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by **masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections**. (See Figure 2)

**中**：在 decoder 中，我们需要防止"信息向左流动（leftward information flow）"，以保持**自回归（auto-regressive）性质**。具体做法是在 scaled dot-product attention 内部，**把 softmax 输入中所有"非法连接"位置的值设为 −∞**。

**精读 —— 这就是 GPT 的 causal mask**：

"Auto-regressive"是 GPT 系列的灵魂。意思是：**生成第 t 个 token 时，只能基于第 1 ~ t-1 个 token，不能偷看后面的**。

**为什么不能偷看后面的？** 因为训练目标是"用前面所有 token 预测下一个 token"。如果让模型在算第 t 步时看见了第 t+1 步的真实答案，那训练就成了作弊：模型只需"复制下一个位置的真值"就 100% 准确，但推理时根本没有未来 token，模型立刻崩溃。

**怎么实现"只看左边"？** 在 attention 分数矩阵上加一个**下三角的 mask**：

```
score (4×4)：     mask 后：           softmax 后：
[. . . .]        [. -∞ -∞ -∞]       [1.0  0   0   0 ]
[. . . .]   →    [. .  -∞ -∞]   →   [.   .   0   0 ]
[. . . .]        [. .  .  -∞]       [.   .   .   0 ]
[. . . .]        [. .  .  . ]       [.   .   .   . ]
```

- 第 1 行（第 1 个 token）：只能看自己 → 后面 3 个位置全 −∞，softmax 后权重为 0
- 第 2 行（第 2 个 token）：能看自己 + 前 1 个 → 后 2 个 −∞
- 以此类推

**为什么用 −∞ 而不是 0？** 因为 mask 加在 **softmax 之前**：$e^{-\infty} = 0$，softmax 后这些位置自然变成 0；而如果直接把 score 设为 0，softmax 后还会有一个非零权重 $e^0 = 1$，达不到屏蔽效果。

> 这一句"masking out (setting to −∞)"就是**第 2 课代码 `02_self_attention.py` 里 causal mask 的全部理论依据**。后面 GPT-2 §2.3 没再展开讲 mask，是因为已经在 Vaswani 这里讲完了。

---

## 3. §3.2.1 整段公式 + 维度对齐总览

```
        Q (n, d_k)        K (m, d_k)        V (m, d_v)
            │                 │                 │
            └───── @ ─────────┘                 │
                  ▼                             │
            QK^T (n, m)  ← 每个 query 与每个 key 的相关度
                  │
                ÷ √d_k       ← 缩放（防 softmax 饱和）
                  │
            + mask           ← (可选) 把非法位置设 −∞，强制因果
                  │
              softmax (按行) ← 归一化成权重分布
                  │
                  └───────── @ ────────────────┐
                                               ▼
                                      Output (n, d_v)
```

**自注意力（self-attention）特例**：Q、K、V 全部从同一个输入 $X \in \mathbb{R}^{n \times d_{model}}$ 通过三个线性投影得到：

$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
$$

此时 n = m = 序列长度，每个 token 用自己的 Q 去和**所有 token 的 K**算相关度，再用这些权重加权**所有 token 的 V**——这就是"自"注意力的"自"。

#### 补充：这里的"序列长度"到底是哪个长度？

这里的 **"序列长度"** 指的是：**当前这一次前向计算输入到注意力层的 token 个数**，也就是输入张量 $X \in \mathbb{R}^{n \times d_{model}}$ 里的那个 $n$。

**1）它具体是什么**

把一段文本喂进模型时会经历：

```
原始文本 → 分词器 (tokenizer) → token id 序列 → embedding → X
"我爱学习"  →  ["我","爱","学","习"]  →  [12, 88, 41, 77]  →  (4, d_model)
```

最后那个矩阵 $X$ 的 **行数 = token 数 = 序列长度 $n$**。

所以在自注意力里：

- $Q = X W_Q \in \mathbb{R}^{n \times d_k}$ → 共 $n$ 个 query
- $K = X W_K \in \mathbb{R}^{n \times d_k}$ → 共 $n$ 个 key
- $V = X W_V \in \mathbb{R}^{n \times d_v}$ → 共 $n$ 个 value

每个 token 既贡献一个 query，也贡献一个 key 和 value，所以自然 $n = m$。

**2）这个 $n$ 在不同场景下是多少**

| 场景 | 序列长度 $n$ 是什么 |
|------|---------------------|
| 训练时（batch 内一条样本） | 这条样本被截断/填充后的 token 数，比如 GPT-2 里常见的 `1024` |
| 推理时第 1 步 | prompt 的 token 数，比如你输入 `"今天天气"` 分词后是 4，那 $n=4$ |
| 推理时逐步生成 | 已生成的 token 总数（含 prompt），每生成一个新 token，$n$ 就 +1 |
| 批处理（batch） | 形状变成 `(batch, n, d_model)`，但 attention 还是在 $n$ 这一维上做 |

**3）和"上下文窗口"的区别**

容易混的两个概念：

- **序列长度 $n$**：这一次实际算的 token 数（动态，每次 forward 都可能不同）
- **上下文窗口 / max sequence length**：模型**最大能处理**的 $n$（静态上限，由位置编码、训练设置决定，比如 GPT-2 的 1024、Llama-2 的 4096）

也就是：**$n \le$ 上下文窗口**。

**4）为什么 cross-attention 里 $n \ne m$**

对照看一下交叉注意力（比如机器翻译里 decoder 看 encoder）：

- $Q$ 来自 decoder 当前的目标序列，长度 $n$（比如要生成的中文 5 个 token）
- $K, V$ 来自 encoder 输出的源序列，长度 $m$（比如英文原句 7 个 token）

此时 $n=5, m=7$，**两个序列的长度不一样**，所以得分矩阵 $QK^T \in \mathbb{R}^{5 \times 7}$。

而自注意力中 Q/K/V 全部来自同一个 $X$，所以两个长度被锁成同一个数 → $n=m=$ 这条输入序列的 token 数。

**一句话总结**：$n$ 就是这次输入到 attention 层那条 token 序列的长度——训练时是样本截断后的固定长度，推理时是 prompt 加上已生成 token 的当前长度。

---

# Part 2：GPT-2 §2.3 中聚焦 masked self-attention / 自回归约束

> §2.3 全文已经在 `notes_gpt2_input_and_model.md` 第二部分逐句精读过。这里**只重新聚焦"masked self-attention 和自回归约束"在这一节里的体现**，不重复讲 4 个改动。

## 1. 第一句就锁死了"causal"基调

> **EN**: We use a Transformer (Vaswani et al., 2017) based architecture for our LMs.

**中**：我们使用基于 Transformer（Vaswani et al., 2017）的架构作为语言模型。

**精读 —— "based architecture" 四个字背后的取舍**：

Vaswani 2017 同时提供了 encoder（双向）和 decoder（带 causal mask）。GPT 系列**只取 decoder 的那一半**，理由是"语言模型 = next token prediction"，天然要求 causal mask。

| 流派 | 用什么部分 | mask | 训练目标 |
|------|------------|------|----------|
| BERT | Encoder | 双向（无 mask） | MLM（完形填空） |
| 原版 NMT | Encoder + Decoder | Decoder 端 causal | 翻译 |
| **GPT 系列** | **Decoder-only** | **Causal** | **Next token prediction** |

> 一旦选定"decoder-only + next token"，§3.2.1 里那个 "−∞ mask" 就**强制**进入了每一个 attention 层——这是 GPT 系列与 BERT 系列的根本分水岭。

## 2. 4 个改动里哪些与 attention 相关？

> ⚠️ **提前透题预警（第 2 课读者请先看）**
>
> 这一小节会出现 **$W_O$**（多头输出投影）、**$W_2$**（FFN 第二层）、**残差连接**、**Pre-Norm / LayerNorm** 这几个概念。它们**严格意义上属于第 3、4 课**：
>
> - **$W_O$、残差、LayerNorm** → 第 3 课《多头注意力与残差》正式登场
> - **$W_2$、FFN、Pre-Norm vs Post-Norm** → 第 4 课《Transformer Block》正式登场
>
> 之所以在第 2 课的笔记里提前出现，是因为 GPT-2 §2.3 一次性把 4 个改动全讲完了，论文跳不开。
>
> **读到这里你只要做到一件事**：先记住下面表格里"是什么 + 在 sub-block 哪个位置"，**不必纠结细节**——等第 3、4 课正式学到时，回来回扣这一节，会有"啊原来这个就是当时那个"的感觉。

回顾 §2.3 的 4 个改动（详见 `notes_gpt2_input_and_model.md`）：

| 改动 | 是否与 attention 直接相关 | 概念正式登场 |
|------|--------------------------|--------------|
| Pre-Norm（LayerNorm 前置） | ✓ 直接影响 attention 子模块的输入归一化 | 📌 LayerNorm 在第 3 课、Pre-Norm vs Post-Norm 在第 4 课 |
| 末尾 LayerNorm | ✗ 在 LM Head 前，与 attention 无关 | 📌 第 3 课 |
| 残差权重 ×1/√N 初始化 | ✓ **专门缩放 attention 的输出投影 $W_O$** 和 FFN 的 $W_2$ | 📌 $W_O$ 在第 3 课、$W_2$（FFN）在第 4 课 |
| 词表 / 上下文 / batch 放大 | ✓ **上下文从 512 → 1024 直接让 attention 矩阵从 512² → 1024²，计算量 ×4** | 本课就够用 |

**重点 1：Pre-Norm 改的是什么？**

```
Pre-Norm 后的 attention 子模块：
  x ──► LayerNorm ──► [Q,K,V projection → Scaled Dot-Product Attention → W_O] ──► Add ──► 下一层
   │                                                                              ▲
   └──────────────────────────────────────────────────────────────────────────────┘
```

attention 内部的"Scaled Dot-Product"完全没变，**变的是：进入 attention 之前先 LN，残差通路不再被 LN 扰动**。深层下能稳。

**重点 2：1/√N 缩放专门作用在哪？**

> 📌 **$W_O$ / $W_2$ 是什么？**（第 2 课只需建立"位置感"，细节留给第 3、4 课）
>
> 一个 Transformer 层 = **2 个带残差的 sub-block**：Attention 子模块 + FFN 子模块。每个 sub-block 都有一个"汇入残差主路前的最后一层投影"：
>
> | 符号 | 所属子模块 | 在子模块里的位置 | 形状 | 正式登场 |
> |------|-----------|------------------|------|---------|
> | **$W_O$** | Multi-Head Attention | 多头 concat 之后的**输出投影**（汇入残差前最后一步） | $(d_{model}, d_{model})$ | 第 3 课 |
> | **$W_2$** | FFN / MLP | 两层 MLP 的**第二层**（降维回 $d_{model}$，汇入残差前最后一步） | $(4 d_{model}, d_{model})$ | 第 4 课 |
>
> 直观一句话：**它们是各自子模块"最后一锤子"的输出投影**，输出值会被直接 Add 到残差主路上。

只缩放 **$W_O$**（attention 的输出投影）和 **$W_2$**（FFN 的第二层），**不动 $W_Q / W_K / W_V$**。

为什么？因为这两个矩阵处在"残差分支汇入主路之前的最后一步"，缩放它们 = 直接让 sub-block 的输出方差变小 = 累加 N 次后总方差不爆。$W_Q / W_K / W_V$ 影响的是 attention 内部的 score 计算，与残差累加无关。

**重点 3：上下文从 512 → 1024 对 attention 意味着什么？**

self-attention 的复杂度是 $O(n^2 \cdot d)$。n 翻倍 → 计算量 ×4，显存 ×4。这就是为什么后来出现了一堆"长上下文优化"工作（FlashAttention、稀疏注意力、ALiBi、RoPE 等）——**attention 的二次复杂度是真·瓶颈**。

> 📌 **"稀疏注意力 (sparse attention)"先记一句话**：让每个位置**只看附近一小段**而不是所有左侧位置，把 $O(n^2)$ 砍到 $O(n \cdot w)$（w 是窗口大小）。GPT-3 就用了这个套路，本笔记 **Part 3 §1.1 / §1.2 / §1.3** 会详细展开 dense / sparse 的区别和"交替堆叠"的玩法。

## 3. GPT-2 §2.3 没有但你必须脑补的事

§2.3 全节没有再次提"causal mask"或"masked self-attention"——但**这并不代表它不在**。它是被 §2.3 第一句"Transformer based + GPT-1 follow"**默认继承下来的**。在 nanoGPT 风格代码里你会看到：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        ...
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = ...                                           # (B, nh, T, hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)) # § Vaswani 3.2.1
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))  # § Vaswani 3.2.1 mask
        att = att.softmax(dim=-1)
        y = att @ v
        ...
```

**论文 §2.3 一个字没讲 mask，但代码里 `masked_fill(... -inf)` 这一行不能省**——这一行就是 Vaswani §3.2.1 那句"setting to −∞"在 GPT-2 里的真身。

---

# Part 3：GPT-3 §2.1 中 attention 在 decoder 堆叠中的使用

> 论文：Brown et al., 2020, *Language Models are Few-Shot Learners*
> 目标章节：Section **2.1 Model and Architectures**

§2.1 同样很短，但一句话能拆出 GPT-3 全部 175B 参数的架构骨架。

## 1. 一句话定位：GPT-3 = GPT-2 架构 + 一个 attention 改动 + 极致 scaling

> **EN**: We use the same model and architecture as GPT-2 [RWC+19], including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use **alternating dense and locally banded sparse attention patterns** in the layers of the transformer, similar to the Sparse Transformer [CGRS19].

**中**：我们使用与 GPT-2 [RWC+19] **完全相同的模型和架构**，包括其中描述的改进初始化、Pre-Norm、可逆 tokenization；**唯一的例外**是我们在 transformer 的各层中使用**交替的稠密注意力（dense attention）和局部带状稀疏注意力（locally banded sparse attention）模式**，类似 Sparse Transformer [CGRS19]。

**精读**：

这一句把架构层面的工作量降到了**最小**——"和 GPT-2 一样"。然后在 attention 上做了**唯一一处改动**：

**先建立直觉：dense / sparse 这两个词在 attention 里到底指什么？**

这两个词描述的是**注意力矩阵 $QK^T$ 长什么样**——也就是"哪些位置之间允许互相算相关度"。因为是 causal（只能看左边），**有效区域只占下三角**。在这个下三角里：

- **dense（稠密）** = "下三角**全部填满**" → 每个位置都和**所有左邻**算相关度 → 第 i 行有 i+1 个非零项，复杂度 $O(n^2)$
- **sparse（稀疏）** = "下三角里**只在对角线附近一条窄带里填**" → 每个位置只和**附近 w 个左邻**算相关度 → 每行至多 w 个非零项，复杂度 $O(n \cdot w)$

直接画一张 $n=8$ 的注意力矩阵示意（`■` = 真的算相关度，`·` = 被 mask 掉、不算）：

```
dense（标准 causal）:           sparse, 窗口 w=3:

  K→  0 1 2 3 4 5 6 7            K→  0 1 2 3 4 5 6 7
Q↓ 0  ■ · · · · · · ·          Q↓ 0  ■ · · · · · · ·
   1  ■ ■ · · · · · ·             1  ■ ■ · · · · · ·
   2  ■ ■ ■ · · · · ·             2  ■ ■ ■ · · · · ·
   3  ■ ■ ■ ■ · · · ·             3  · ■ ■ ■ · · · ·
   4  ■ ■ ■ ■ ■ · · ·             4  · · ■ ■ ■ · · ·
   5  ■ ■ ■ ■ ■ ■ · ·             5  · · · ■ ■ ■ · ·
   6  ■ ■ ■ ■ ■ ■ ■ ·             6  · · · · ■ ■ ■ ·
   7  ■ ■ ■ ■ ■ ■ ■ ■             7  · · · · · ■ ■ ■

非零数 ≈ n²/2，O(n²)              非零数 ≈ n·w，O(n·w)
```

三个词义对照：

- **dense**（稠密）= 矩阵里有效格子**很多** → "信息密集"
- **sparse**（稀疏）= 矩阵里有效格子**很少** → "信息稀疏"
- **locally banded**（局部带状）= 砍法是"只保留对角线附近一条**带子**" → 所以叫"**带状 (banded) + 局部 (local)** 稀疏注意力"

**一句话区分**：dense 看所有左邻、sparse 只看附近左邻。下面 §1.1 / §1.2 就分别把这两个词拆成"看哪些位置 + 复杂度"两件事来讲。

### 1.1 什么是 dense attention？

就是 §3.2.1 那个标准的 scaled dot-product attention：**每个位置都和所有左边的位置算相关度**。复杂度 $O(n^2)$。

```
Layer 1 (dense): 每个位置看所有左边位置
  pos 5  ◄── pos 0,1,2,3,4
  pos 100 ◄── pos 0,1,2,...,99
```

### 1.2 什么是 locally banded sparse attention？

只让每个位置看**附近一个固定窗口**内的左边位置，比如窗口 256：

```
Layer 2 (sparse, window=256): 每个位置只看附近 256 个左邻
  pos 5    ◄── pos 0,1,2,3,4
  pos 1000 ◄── pos 744,...,999  (只看左边 256 个)
```

复杂度从 $O(n^2)$ 降到 $O(n \cdot w)$（w 是窗口大小），上下文越长省得越多。

### 1.3 "交替"是什么意思？

**奇数层用 dense，偶数层用 sparse**（或类似的交替模式）。这样：

- dense 层负责"全局视野"（让信息能跨长距离传播）
- sparse 层负责"局部精修"（高效处理近邻关系，省算力）

两种层堆叠起来，**既保留全局表达力，又把总算力压下来**。GPT-3 上下文是 2048，96 层 Transformer，如果每层都用 dense，attention 部分的算力是天文数字；交替之后能省一大半。

## 2. 为什么 GPT-3 不大改架构？

> 这是一个隐藏的设计哲学。

GPT-3 团队的核心信息是：**"在 GPT-2 架构基本不变的前提下，把规模拉到 175B，看会涌现出什么能力"**。如果他们同时改动架构 + 放大规模，就**没法分清"涌现来自规模还是来自架构改动"**。

所以 §2.1 极简：

- **保持** GPT-2 的 4 个 trick（Pre-Norm、末尾 LN、1/√N 初始化、weight tying）
- **只动一处** attention 模式（dense / sparse 交替），是为了让长上下文（2048）训得动
- **真正的大头**全在规模上（下面 Table 2.1）

## 3. Table 2.1：8 档规模

论文 Table 2.1 给出 8 档模型（GPT-3 family）：

| 模型 | $n_{params}$ | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $d_{head}$ | Batch | LR |
|------|--------------|--------------|-------------|-------------|------------|-------|-----|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | 6e-4 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | 3e-4 |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | 2.5e-4 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | 2e-4 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | 1.6e-4 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | 1.2e-4 |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 128 | 2M | 1e-4 |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** | **128** | **3.2M** | **0.6e-4** |

聚焦 attention 相关的列：

- **$n_{layers}$ 从 12 → 96**：attention 层数翻 8 倍
- **$n_{heads}$ 从 12 → 96**：每层注意力头数翻 8 倍
- **$d_{head} = 128$（最大几档统一）**：每个头的维度固定，不变
- **$d_{model} = n_{heads} \times d_{head}$**：12288 = 96 × 128，完美整除

> "每个 head 64 ~ 128 维"是 Vaswani 以来的惯例，GPT-3 没破这个常识，只是**把头数堆起来**。

## 4. 关键观察：scaling 才是主角

把 GPT-1 → GPT-2 → GPT-3 最大档放在一起：

| 项目 | GPT-1 | GPT-2 | GPT-3 |
|------|-------|-------|-------|
| 层数 | 12 | 48 | **96** |
| $d_{model}$ | 768 | 1600 | **12288** |
| 头数 | 12 | 25 | **96** |
| 上下文 | 512 | 1024 | **2048** |
| 参数量 | 117M | 1.5B | **175B**（×116） |
| 架构改动 | base | +4 个 trick | +sparse attention 交替 |

可以看到：**架构每代只小改 1-4 处，参数量却涨了 100 倍以上**。这就是 GPT-3 想用实验证明的事：**Transformer 这套架构本身已经够好，剩下的 90% 收益靠堆规模**。这个观察直接催生了后来的 Chinchilla scaling law、Llama、GPT-4 等一系列工作。

## 5. attention 在 decoder 堆叠中的具体位置

放回完整流程图（与 GPT-2 几乎一致，只在 attention 模式上不同）：

```
输入 token ids  (batch, seq_len ≤ 2048)
        │
        ▼
Token Embedding + Positional Embedding
        │
        ▼
┌─────────────────────────────────────────┐
│ Block 1 (dense attention)               │
│   LN → Multi-Head Self-Attn → +Residual │
│   LN → FFN → +Residual                  │
├─────────────────────────────────────────┤
│ Block 2 (sparse attention, 局部带状)     │
│   LN → Multi-Head Self-Attn → +Residual │
│   LN → FFN → +Residual                  │
├─────────────────────────────────────────┤
│ Block 3 (dense)                         │
├─────────────────────────────────────────┤
│ Block 4 (sparse)                        │
│   ... (交替直至 96 层)                   │
└─────────────────────────────────────────┘
        │
        ▼
LayerNorm (ln_f)
        │
        ▼
LM Head (与 Token Embedding 共享权重)
        │
        ▼
logits → softmax → next token
```

**对照第 2 课要做的事**：你在 `02_self_attention.py` 里会自己实现一个 dense + causal 的 self-attention，**这就是 GPT-3 中"奇数层"的样子**。sparse 模式属于工程优化，第 2 课不必实现，但要知道它存在的目的：**让长上下文训得动**。

---

# Part 4：三论文 attention 关键概念串讲

| 关键词 | Vaswani §3.2.1 | GPT-2 §2.3 | GPT-3 §2.1 |
|--------|----------------|------------|------------|
| **Q / K / V** | 首次定义并写出公式 | 隐式继承（"based on Transformer"） | 隐式继承（"same as GPT-2"） |
| **Scaled Dot-Product** | 详细推导 √d_k 由来 | 不再讨论，默认使用 | 不再讨论，默认使用 |
| **Causal Mask** | "−∞ mask" 一句定调 | 隐式继承（auto-regressive LM 必备） | 隐式继承 |
| **Multi-Head**（第 3 课主题） | §3.2.2 引入 | 不再讨论 | 头数 12 → 96 |
| **Pre-Norm**（第 4 课主题） | 无（原版是 Post-Norm） | 首次提出 | 沿用 |
| **位置编码**（第 1 课主题） | 正余弦 | 改为可学习 | 沿用可学习 |
| **Attention 模式** | 全 dense | 全 dense | **dense / sparse 交替** |
| **上下文长度** | 不固定 | 1024 | 2048 |

**一图看懂 attention 在三代论文里的演化**：

```
Vaswani 2017                GPT-2 2019                   GPT-3 2020
───────────                 ──────────                   ──────────
Encoder + Decoder      →    只留 Decoder            →    Decoder 不变
Post-Norm              →    Pre-Norm                →    沿用
全 dense attention      →    全 dense                 →    dense / sparse 交替
ctx 不固定              →    ctx = 1024               →    ctx = 2048
n_layer = 6            →    n_layer ≤ 48             →    n_layer ≤ 96
正余弦位置编码           →    可学习位置编码            →    沿用
```

---

# Part 5：和第 2 课代码的对齐

第 2 课 `02_self_attention.py` 里的关键代码片段（按顺序）：

| 代码模块 | 对应论文位置 |
|----------|--------------|
| `q = x @ W_q; k = x @ W_k; v = x @ W_v` | Vaswani §3.2 Q/K/V 三个线性投影 |
| `scores = q @ k.T` | Vaswani §3.2.1 公式 (1) 的 $QK^T$ |
| `scores = scores / np.sqrt(d_k)` | §3.2.1 缩放因子，整段方差推导的落点 |
| `scores = scores + causal_mask`（mask 里非法位为 -inf） | §3.2.1 "masking out (setting to −∞)" |
| `weights = softmax(scores, axis=-1)` | §3.2.1 softmax 步骤 |
| `out = weights @ v` | §3.2.1 公式 (1) 最后一步 |

**第 2 课动手写部分（⑤）的两个修改建议**与论文的对应关系：

| 修改 | 你预期会观察到 | 对应论文段落 |
|------|----------------|--------------|
| 去掉 `/ √d_k` | softmax 输出更"尖锐"（接近 one-hot），梯度变小，训练困难 | §3.2.1 关于"softmax 饱和区"的整段 |
| 修改 mask（去掉因果约束） | 模型在训练时能"偷看未来"，loss 异常低，但生成时崩溃 | §3.2.1 关于"auto-regressive property"的那一句 + GPT-2 §2.3 隐式继承 |

---

# Part 6：复盘（回答这 5 个问题就过关）

1. **Q/K/V 各是什么？为什么必须分三个矩阵？**
   - Q：当前位置发出的"查询"
   - K：每个位置提供的"索引标签"
   - V：每个位置实际携带的"内容"
   - 分三个矩阵是为了**让"查询表达"、"被查询的标签"、"实际内容"解耦**，模型才能学到不同的角色（动词的 Q 找主语、名词的 K 标自己是名词、V 携带具体语义）

2. **Scaled Dot-Product Attention 的完整公式是什么？每一步形状是什么？**
   - 公式：$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
   - 形状：(n,d_k) × (d_k,m) → (n,m) → softmax → (n,m) × (m,d_v) → (n,d_v)

3. **为什么要除以 √d_k？不除会怎样？**
   - 不除：$QK^T$ 方差为 $d_k$，$d_k$ 大时点积量级大 → softmax 饱和到 one-hot → **梯度消失，训练卡死**
   - 除以 $\sqrt{d_k}$：把方差拉回 1，softmax 工作在健康区域，梯度均匀

4. **Causal mask 怎么实现？为什么用 −∞ 而不是 0？**
   - 实现：在 score 矩阵上叠加一个下三角 mask，非法位（自己右边的位置）设为 −∞
   - 用 −∞：因为 mask 加在 softmax **之前**，$e^{-\infty}=0$，softmax 后权重正好是 0；如果设 0，$e^0=1$，达不到屏蔽效果
   - 目的：保持 **auto-regressive 性质**——训练时只能看左边，与推理（一个 token 一个 token 生成）的条件保持一致，否则推理立刻崩

5. **从 Vaswani → GPT-2 → GPT-3，attention 的演化是什么？**
   - Vaswani：定义 Q/K/V、Scaled Dot-Product、causal mask；架构是 Encoder + Decoder
   - GPT-2：扔掉 Encoder，只用带 causal mask 的 Decoder；Pre-Norm + 1/√N 初始化让深层稳定；上下文 1024
   - GPT-3：架构与 GPT-2 几乎一致，**唯一改动是 dense 与 sparse attention 交替**，让 2048 上下文 × 96 层训得动；其余 99% 工作量都在 scaling

---

# Part 7：延伸阅读（可选）

- **Vaswani et al., 2017** 全文 §3.2.2 *Multi-Head Attention* —— 第 3 课会精读，先建立印象
- **Child et al., 2019**, *Generating Long Sequences with Sparse Transformers* —— GPT-3 §2.1 引用的 Sparse Transformer 原始论文
- **Dao et al., 2022**, *FlashAttention* —— 后续工业界对 attention 二次复杂度的硬件级优化，Llama / GPT-4 训练标配
- **Xiong et al., 2020**, *On Layer Normalization in the Transformer Architecture* —— Pre-Norm 在数学上为何更稳，配合 GPT-2 §2.3 一起读
- **The Illustrated Transformer**（Jay Alammar）—— 把 Q/K/V 全可视化，看完这一篇再回头看公式更清晰
- **后续课程阅读**：第 3 课会精读 Vaswani §3.2.2 Multi-Head Attention 与 GPT-3 §2.1 中的 head 数量配置，本节是它的前置。
