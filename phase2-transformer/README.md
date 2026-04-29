# 第二阶段：Transformer 架构

> 从注意力机制到 GPT —— 理解大模型的核心

## 前置要求

完成第一阶段的全部课程，理解：
- 矩阵运算与 Softmax
- 梯度下降与反向传播
- 前馈神经网络的完整训练流程

## 环境准备

```bash
# 在项目根目录激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 课程结构

按顺序学习，每课约 90-120 分钟：

| 课程 | 主课文件 | 自写练习 | 核心内容 | 关键概念 |
|------|----------|----------|----------|----------|
| 第 1 课 | `01_word_embeddings.py` | `01_word_embeddings_self_write.py` | 词嵌入、位置编码 | 文本如何变成数字向量 |
| 第 2 课 | `02_self_attention.py` | `02_self_attention_self_write.py` | Q/K/V、注意力分数、掩码 | Transformer 的核心机制 |
| 第 3 课 | `03_multi_head_attention.py` | — | 多头注意力、残差连接、LayerNorm | 为什么多头比单头好 |
| 第 4 课 | `04_transformer_block.py` | — | 完整 Transformer Block、FFN | 把所有组件拼起来 |
| 第 5 课 | `05_gpt_from_scratch.py` | — | 完整 GPT 模型、文本生成 | 从零搭建一个能生成文本的模型 |

### 自写练习与重置（当前进度）

- ✅ 第 1 课：`01_word_embeddings_self_write.py`（重置脚本：`reset_exercises_01.py`）
- ✅ 第 2 课：`02_self_attention_self_write.py`（8 个 TODO，覆盖 softmax / Q/K/V / scaled dot-product / 因果掩码；含内置 `require_*` 校验）
- 🚧 第 3 课：进行中 —— 已补充中文学习资料（B 站 / 知乎），按"看视频 → 读论文 → 跑 `03_multi_head_attention.py`"的节奏推进
- ⏳ 第 4-5 课：自写练习与重置脚本待补

```bash
# 在项目根目录执行
python3 phase2-transformer/reset_exercises_01.py   # 重置第二阶段第 1 课练习
python3 phase2-transformer/02_self_attention_self_write.py   # 跑第 2 课练习并自动校验
```

## 必读论文

在 `papers/core-transformers/` 目录下提供了核心论文的 PDF，延伸论文见 `papers/README.md`：

| 论文 | 年份 | 为什么必读 |
|------|------|-----------|
| Attention Is All You Need | 2017 | Transformer 的开山之作 |
| GPT-1 | 2018 | 生成式预训练 + 判别式微调，GPT 系列起点 |
| BERT | 2018 | 双向编码器，理解 Encoder 的威力 |
| GPT-2 | 2019 | 纯 Decoder 架构，语言建模范式 |
| GPT-3 | 2020 | 规模定律、In-context Learning |
| InstructGPT | 2022 | RLHF，对齐技术的起点 |

### 论文阅读建议

1. **不用逐字精读** —— 重点看 Architecture 和 Results 部分
2. **带着问题读** —— 为什么用 Scaled Dot-Product？为什么要多头？
3. **跳过数学证明** —— 先建立直觉，数学后面再补
4. **每课只读指定章节** —— 下面每课都标注了该读哪几节，不要贪多

---

## 每课统一节奏

每一课都按以下 6 步执行，不要跳步：

| 步骤 | 做什么 | 时间 | 要点 |
|------|--------|------|------|
| **① 看视频** | 看当课指定视频 | 20-40 min | 只记 3 个关键词，建立直觉，不求全懂 |
| **② 读论文** | 读当课指定的论文章节 | 15-20 min | 只读指定小节，带着视频中的印象去对照 |
| **③ 跑代码** | 运行当课 `.py`，逐段读输出 | 10-15 min | 先跑通看效果，不急着改 |
| **④ 对照理解** | 把代码和论文公式逐行对齐 | 15-20 min | 重点：维度变化、公式中每个符号对应代码哪一行 |
| **⑤ 动手写** | 完成自写练习（如有）或改超参数 | 20-30 min | 自写练习 > 改超参数 > 加注释，优先级递减 |
| **⑥ 复盘** | 回答 3 个问题，记录到笔记 | 5-10 min | 输入是什么？核心计算是什么？输出是什么？ |

> **卡住时**：直接回看视频中对应概念的片段，不要整段重看。

---

## 每课详细大纲

### 第 1 课：词嵌入与位置编码

**核心概念**
- 为什么不能直接用 one-hot？维度灾难
- Word2Vec 的核心思想：相似的词有相似的向量
- 位置编码：正弦/余弦编码 vs 可学习编码
- **与 LLM 的关系**：GPT 的第一层就是嵌入层

**按顺序做**

1. **① 看视频**（20-40 min）：[3Blue1Brown - Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M)
   - 关键词：embedding、positional encoding、token
2. **② 读论文**（15-20 min）：
   - `Attention Is All You Need` → `3.4 Embeddings and Softmax`、`3.5 Positional Encoding`
   - `BERT` → `Input/Output Representations`（看 token / segment / position embedding 的输入构成）
   - `GPT-2` → `2.2 Input Representation`、`2.3 Model`（关注 decoder-only 的输入嵌入处理方式）
3. **③ 跑代码**：运行 `01_word_embeddings.py`，观察嵌入向量的维度和可视化输出
4. **④ 对照理解**：把论文中正弦位置编码公式和代码实现逐行对齐
5. **⑤ 动手写**：完成 `01_word_embeddings_self_write.py`（重置用 `reset_exercises_01.py`）
6. **⑥ 复盘**：能口述 `token → embedding → positional encoding` 的完整流程

**常见疑问：位置编码里的 `div_term` 为什么写成 `exp(log(...))`？**

`01_word_embeddings.py` 第 153-155 行：

```python
div_term = np.exp(
    np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
)  # (d_model/2,)
```

它对应论文公式里的分母倒数 $1/10000^{2i/d_{model}}$。利用恒等变换：

$$
\frac{1}{10000^{2i/d_{model}}} = e^{-\,2i \cdot \ln(10000)/d_{model}}
$$

- `np.arange(0, d_model, 2)` → 公式里的 `2i`，形状 `(d_model/2,)`
- `-(np.log(10000.0) / d_model)` → 标量 `-ln(10000)/d_model`
- 两者相乘再 `exp`，就是整条频率向量

为什么不直接写 `10000 ** (...)`：

1. **数值稳定**：`d_model` 较大时，幂运算的浮点误差比 `exp(log(...))` 更大
2. **计算高效**：`exp` 对向量是并行的，底层有 SIMD 优化
3. **工程惯例**：PyTorch 官方教程、HuggingFace 的实现都是这样写的

配合后两行，`position * div_term` 通过广播 `(max_len, 1) * (d_model/2,)` 得到 `(max_len, d_model/2)`，再分别填入 `pe[:, 0::2]`（sin，偶数维）和 `pe[:, 1::2]`（cos，奇数维）。

**关键领悟：位置编码的直觉理解**

跑完代码、看完热力图后，需要在脑子里建立起下面这套心智模型：

**1. 两个矩阵的形状，行列分别是什么？**

| 矩阵 | 形状 | 行 | 列 | 来源 |
|------|------|----|----|----|
| Token Embedding | `(vocab_size, d_model)` | 词表里每个词 | 该词的语义特征 | 随机初始化 + 训练更新 |
| Position Encoding | `(max_len, d_model)` | 每个位置 | 该位置的位置特征 | sin/cos 公式算出（固定）|

- `vocab_size`（词典容量，万级）和 `seq_len`（当前句子长度，百级）是两回事，别混淆
- 两个矩阵的**列数必须都是 `d_model`**，这样才能逐元素相加

**2. `max_len` 和 `d_model` 到底是什么？**

- `max_len` = 上下文长度上限（GPT-2 是 1024、GPT-3 是 2048）
- `d_model` = 每个 token / 每个位置用多长的向量表示（GPT-2 是 768、GPT-3 是 12288）

位置编码和词嵌入共享同一个 `d_model`，**不是把维度"一半给语义、一半给位置"**，而是两张同形状的表**逐元素相加**，叠加后的每一维都同时承载语义 + 位置信号。模型从高维空间里自己学会区分两种信息。

**3. 为什么 embedding 和位置编码相加不会白白丢信息？**

相加不是让位置编码“增加语义”，而是把**词是什么**和**词在哪**放进同一个 `d_model` 维输入里：

```text
x_i = token_embedding_i + position_encoding_i
```

后面的 Q/K/V 线性层看到的是这个混合向量，但线性变换会把相加拆成两部分的贡献：

```text
x_i W = token_embedding_i W + position_encoding_i W
```

所以模型不需要先把原始 token embedding 和 position encoding 完整还原出来，只需要学习哪些方向主要像“语义信号”，哪些方向主要像“位置信号”，以及两者组合后对任务有什么用。

这确实不像拼接 `concat(token, position)` 那样显式保留两块独立信息；从严格信息论角度看，相加可能存在混叠。但在高维空间里，位置编码有稳定的结构，token embedding 是训练学出来的分布，后续层也会适应这种输入形式。相加的工程收益是 `d_model` 不变，attention/MLP 的参数量和计算量不翻倍；实践上已经足够好。

一句话：**相加不是无损压缩，也不是语义变多，而是用低成本把顺序信息注入到语义向量里，让后续 Transformer 层学会使用这个混合信号。**

**4. 为什么要用多频率 sin/cos：放大镜 + 望远镜**

`div_term` 从 `1.0` 递减到 `~1/10000`，对应 16 维（或 768 维）里从高频到低频的一组信号：

| 维度区 | 频率 | 相邻位置差 | 擅长 | 类比 |
|-------|------|----------|------|------|
| **小维度**（dim 0, 1, 2, 3） | 高频（波长 ≈ 6） | ≈ 0.68（强信号）| **看近处**（分辨相邻词）| 时钟秒针 / 二进制低位 |
| 中维度（dim 6, 7, 8, 9） | 中频 | 中等 | 看中等距离 | 时钟分针 |
| **大维度**（dim 14, 15） | 低频（波长 ≈ 20000）| ≈ 0.0003（淹没）| **看远处**（分辨段落级）| 时钟时针 / 二进制高位 |

所以**低维维度负责"近距离精度"，高维维度负责"远距离区分度"**，合起来就是一把同时带毫米、厘米、米刻度的"多尺度尺子"。

**5. "看不清近的"指什么？**

不是 `pos` 整数不精确，而是**低频维度对相邻位置的响应差 ≈ 0.0003**——这个差会被词向量（±1 量级）和 fp16 浮点精度淹没，模型根本学不到"这两个词挨着"的信号。高频维度响应差 ≈ 0.68，才够"显眼"。

反过来，高频维度是周期函数（波长 ≈ 6），`pos=5` 和 `pos=11` 在 `dim 0` 上几乎相同 → 高频维度区分远距离会"混淆"，这时候靠波长极长的低频维度做"线性尺子"。

**6. 回头再看热力图就全通了**

- 左边条纹密 = 低维高频 = 看近处的放大镜
- 右边整列一个颜色 = 高维低频 = 看远处的望远镜
- 每一行 = 一个位置的完整 `d_model` 维"指纹"
- 成对列（偶数 sin / 奇数 cos）共享同一个 `div_term`，只差 90° 相位 → 三角恒等式让 `PE(pos+k)` 能从 `PE(pos)` 线性推导出来，模型天然学得到相对位置

> 能口述出以上 6 点，位置编码就彻底过关了。

### 第 2 课：自注意力机制

**核心概念**
- 注意力的直觉：哪些词对当前词重要？
- Q (Query)、K (Key)、V (Value) 的含义
- Scaled Dot-Product Attention 的完整推导
- 因果掩码 (Causal Mask)：防止看到未来的词
- **与 LLM 的关系**：这就是 Transformer 的心脏

**按顺序做**

1. **① 看视频**（20-40 min）：[3Blue1Brown - Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)
   - 关键词：Q/K/V、scaled dot-product、mask
   - **额外学习内容**（视频结尾推荐，可选，建立更全面的直觉）：
     - [VCubingX - What is a Language Model? A visual explanation](https://www.youtube.com/watch?v=1il-s4mgNdI)：从可视化角度讲清"语言模型到底在预测什么"，用 `the / blue / sky / green / falling` 的概率分布直观演示 next-token prediction
     - [Art of the Problem - ChatGPT: 30 Year History | How AI Learned to Talk](https://www.youtube.com/watch?v=OFS90-FX6pg)：从 30 年的语言模型发展史切入，串起统计语言模型 → RNN → Transformer → ChatGPT 的完整脉络
2. **② 读论文**（15-20 min）：
   - `Attention Is All You Need` → `3.2 Attention`、`3.2.1 Scaled Dot-Product Attention`
   - `GPT-2` → `2.3 Model`（重点看 masked self-attention / 自回归约束）
   - `GPT-3` → `2.1 Model and Architectures`（看 decoder 堆叠中 attention 的使用）
3. **③ 跑代码**：运行 `02_self_attention.py`，观察注意力分数矩阵的热力图
4. **④ 对照理解**：把 Attention(Q,K,V) = softmax(QK^T / √d_k) V 和代码逐行对齐
5. **⑤ 动手写**：完成 `02_self_attention_self_write.py`（8 个 TODO，每填一个就跑一次依靠 `require_*` 校验即时纠错）；做完可继续手动修改 mask 或去掉 scale 观察效果
6. **⑥ 复盘**：能画出 Q/K/V 的维度变化图，并说明 mask 的作用

### 第 3 课：多头注意力与残差

**核心概念**
- 单头 → 多头：不同的头关注不同的模式
- 残差连接：解决深层网络的退化问题
- Layer Normalization：稳定训练过程
- Pre-Norm vs Post-Norm：为什么现代大模型几乎都用 Pre-Norm
- **与 LLM 的关系**：GPT-3 有 96 个注意力头

**按顺序做**

1. **① 看视频**（45 min）：[李宏毅【機器學習2021】自注意力機制 Self-attention（下）](https://www.youtube.com/watch?v=gmsMY5kc-zw)
   - 关键词：multi-head、residual、LayerNorm
   - 把"分头 → 各自注意力 → 拼接 → 投影"讲得最透
   - 配合阅读：[1010Code 课程笔记（基于本集整理）](https://andy6804tw.github.io/2021/05/03/ntu-multi-head-self-attention/)，把 multi-head 的图示按文字版重写了一遍，复习时用
   - **强力补充**（如果只看一个英文视频，就看这个）：[3Blue1Brown - Attention in transformers, step-by-step (Chapter 6)](https://www.youtube.com/watch?v=eMlx5fFNoYc)（26 min，4M 播放）
     - 视频 19:19-23:19 用 4 分钟可视化 96 个头如何并行产生 96 个 ΔE 加回 embedding，是目前我看过对 multi-head **几何直觉**最好的解释
     - 配套图文版（更适合复习）：[Visualizing Attention - Grant Sanderson](https://3blue1brown.substack.com/p/visualizing-attention)，举了"glass ball 砸碎 steel table"为什么需要堆多层 attention block 的反直觉例子
2. **② 读论文**（15-20 min）：
   - `Attention Is All You Need` → `3.1 Encoder and Decoder Stacks`（残差 + LayerNorm）、`3.2.2 Multi-Head Attention`
   - `GPT-3` → `2.1 Model and Architectures`（关注层数、头数、宽度这些规模配置）
   - `BERT` → `Model Architecture`（对照 encoder 结构中的多头注意力）
3. **③ 跑代码**：运行 `03_multi_head_attention.py`，重点看 5 个 Part 的输出
   - mask 约定与第 2 课完全一致（1=屏蔽，0=可见），可以直接复用 `np.triu(...)` 那套
4. **④ 对照理解**：把 MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O 与代码实现对齐
   - 关键维度变换：`(seq_len, d_model)` → reshape → `(n_heads, seq_len, d_head)` → concat 回 `(seq_len, d_model)`
5. **⑤ 动手写**（至少做 2 个）：
   - 修改 `n_heads`（1/2/8/16），对比输出差异（注意 d_model 必须能被整除）
   - 完成代码末尾"练习 4"——给 multi_head_attention 加因果掩码（提示已写在代码里）
   - 把 `pre_norm_block` 堆叠 10 层，对比加不加 final LayerNorm 时输出方差的变化
6. **⑥ 复盘**：能说清以下 4 点
   - **单头 vs 多头**：为什么"分头"在参数量不变的情况下能学到更丰富的模式
   - **残差连接的本质**：那个 +1 不是"保证梯度 ≥ 1"（这个说法不准确），而是提供一条**不被 sublayer 雅可比衰减的恒等高速公路**——即使 sublayer 还没学会（dF/dX ≈ 0），梯度仍能传到前面去
   - **LayerNorm vs BatchNorm**：归一化维度不同 → 在变长序列上谁更友好
   - **Pre-Norm 的代价**：残差通路绕过 LN 让训练稳定，但输出方差会随层数累积，所以工业实现末尾必须再加一个 final LayerNorm（nanoGPT 的 `ln_f`，第 5 课会再遇到）
   - 卡住时看这篇：[Pre-Norm vs Post-Norm（含交互动图）](https://mbrenndoerfer.com/writing/pre-norm-vs-post-norm) —— 把"+1 梯度高速公路"和 LayerNorm Jacobian 画成图，是我找到的解释 Pre-Norm 最清楚的一篇

### 第 4 课：Transformer Block

**核心概念**
- 完整的 Transformer Block 结构
- Feed-Forward Network (FFN)：两层 MLP + 激活函数
- Pre-Norm vs Post-Norm 的区别
- 堆叠多个 Block：从 1 层到 N 层
- **与 LLM 的关系**：GPT-3 就是 96 个这样的 Block 堆叠

**按顺序做**

1. **① 看视频**（20-40 min）：[Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)（**前半段**，到 Block 搭建完为止）
   - 关键词：FFN、Pre-Norm、Block 堆叠
2. **② 读论文**（15-20 min）：
   - `Attention Is All You Need` → `3.1 Encoder and Decoder Stacks`、`3.3 Position-wise Feed-Forward Networks`
   - `GPT-2` → `2.3 Model`（把论文中的 block 结构与你代码逐项对齐）
   - `GPT-3` → `2.1 Model and Architectures`（理解"同构 block 堆叠 + 扩大规模"的主线）
3. **③ 跑代码**：运行 `04_transformer_block.py`，跑通前向传播，观察中间张量维度
4. **④ 对照理解**：画一张 Block 内部流程图（Attention → Add & Norm → FFN → Add & Norm），对齐代码
5. **⑤ 动手写**：修改 FFN 隐藏层维度、堆叠层数，观察参数量和输出变化；尝试切换 Pre-Norm / Post-Norm
6. **⑥ 复盘**：能口述 Block 内部的完整数据流，说清残差 + LayerNorm 在 Block 内的位置

### 第 5 课：从零构建 GPT

**核心概念**
- 完整模型：Embedding → N × Block → Linear → Softmax
- 训练循环：用小数据集训练一个迷你 GPT
- 文本生成：贪心搜索、Temperature、Top-K 采样
- **与 LLM 的关系**：你写的就是 GPT 的完整架构，只是小了 1000 倍

**按顺序做**

1. **① 看视频**（20-40 min）：[Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)（**后半段**，从训练循环到文本生成）
   - 关键词：training loop、temperature、top-k
   - **强烈推荐配套看**：[Jay Alammar - The Narrated Transformer](https://www.youtube.com/watch?v=-QH8fRhqFHM)（30 min）—— 这个视频把 GPT 的完整推理流水线（tokenize → embed → blocks → output projection → softmax → 采样）讲得最清楚，正好对应你 `05_gpt_from_scratch.py` 的代码结构
2. **② 读论文**（15-20 min）：
   - `GPT-2` → `2.3 Model`、`3. Experiments`（先看模型结构，再看训练与生成设置）
   - `GPT-3` → `2. Training Dataset and Model`、`3. Methodology`（理解数据规模、训练范式和 few-shot 评估）
3. **③ 跑代码**：运行 `05_gpt_from_scratch.py`，完成一次完整训练（观察 loss 下降曲线）
4. **④ 对照理解**：把 GPT-2 论文的模型描述和你的代码逐模块对齐（Embedding → Blocks → LM Head）
5. **⑤ 动手写**：调整 temperature 和 top-k 参数生成文本，感受不同采样策略的效果差异
6. **⑥ 复盘**：能口述完整 GPT 的前向流程，并解释训练目标（next token prediction）
7. **⑦ 延伸阅读**（可选）：`InstructGPT` → `3. Methods`（预习"预训练模型如何通过 RLHF 对齐"）

---

## 复习与补充资源

建议在完成第 1-5 课后，用 2 天做集中复盘：
- `Day 6`：[B 站 - Transformer 从零详细解读](https://www.bilibili.com/video/BV1Di4y1c7Zm/) + 回看 `02_self_attention.py`、`03_multi_head_attention.py`
- `Day 7`：[B 站 - Transformer 理论到实战系列](https://www.bilibili.com/video/BV12bfPY1E1S/) + 回看 `04_transformer_block.py`、`05_gpt_from_scratch.py`
- 复盘目标：补齐注释、修正命名、能口述完整 GPT 前向流程

**最重要的 3 个补充资源**（其他都可以不看）：

1. 🎬 [跟李沐学 AI - Transformer 论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE/) —— 中文，1.5 小时逐句过《Attention Is All You Need》，配合 phase2 论文阅读用
2. 🎬 [karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) —— 英文（已在第 4/5 课用），从零手撕一个 GPT，看完整个 phase2 就通了
3. 📝 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ｜ [中译版](https://blog.csdn.net/longxinchen_ml/article/details/86533005) —— 图解经典，卡概念时回来扫一眼

## 下一步

完成第二阶段后，进入第三阶段：**训练与微调** —— 学习 LoRA、量化、RLHF 等实用技术。
