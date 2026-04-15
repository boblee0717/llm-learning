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

按顺序学习，每课约 60-90 分钟：

| 课程 | 文件 | 核心内容 | 关键概念 |
|------|------|----------|----------|
| 第 1 课 | `01_word_embeddings.py` | 词嵌入、位置编码 | 文本如何变成数字向量 |
| 第 2 课 | `02_self_attention.py` | Q/K/V、注意力分数、掩码 | Transformer 的核心机制 |
| 第 3 课 | `03_multi_head_attention.py` | 多头注意力、残差连接、LayerNorm | 为什么多头比单头好 |
| 第 4 课 | `04_transformer_block.py` | 完整 Transformer Block、FFN | 把所有组件拼起来 |
| 第 5 课 | `05_gpt_from_scratch.py` | 完整 GPT 模型、文本生成 | 从零搭建一个能生成文本的模型 |

### 自写练习与重置（当前进度）

- 第 1 课已提供自写练习：`01_word_embeddings_self_write.py`
- 对应重置脚本：`reset_exercises_01.py`

```bash
# 在项目根目录执行
python3 phase2-transformer/reset_exercises_01.py   # 重置第二阶段第 1 课练习
```

## 必读论文

在 `papers/` 目录下提供了核心论文的 PDF：

| 论文 | 年份 | 为什么必读 |
|------|------|-----------|
| Attention Is All You Need | 2017 | Transformer 的开山之作 |
| BERT | 2018 | 双向编码器，理解 Encoder 的威力 |
| GPT-2 | 2019 | 纯 Decoder 架构，语言建模范式 |
| GPT-3 | 2020 | 规模定律、In-context Learning |
| InstructGPT | 2022 | RLHF，对齐技术的起点 |

### 论文阅读建议

1. **先读 Attention Is All You Need** —— 配合第 2-4 课的代码理解
2. **不用逐字精读** —— 重点看 Architecture 和 Results 部分
3. **带着问题读** —— 为什么用 Scaled Dot-Product？为什么要多头？
4. **跳过数学证明** —— 先建立直觉，数学后面再补

## 每课详细大纲

### 第 1 课：词嵌入与位置编码

- 为什么不能直接用 one-hot？维度灾难
- Word2Vec 的核心思想：相似的词有相似的向量
- 位置编码：正弦/余弦编码 vs 可学习编码
- 动手实现：构建一个简单的嵌入层
- **与 LLM 的关系**：GPT 的第一层就是嵌入层

**配套资源与实践**
- 视频：`Day 1` [3Blue1Brown - Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M)
- 代码：`01_word_embeddings.py`
- 完成标准：能解释 `token -> embedding -> positional encoding`

**对应论文与章节（先读这些）**
- `Attention Is All You Need`：`3.4 Embeddings and Softmax`、`3.5 Positional Encoding`
- `BERT`：`3.2 Input/Output Representations`（看 token / segment / position embedding 的输入构成）
- `GPT-2`：`2. Model`（关注 decoder-only 的输入嵌入处理方式）

### 第 2 课：自注意力机制

- 注意力的直觉：哪些词对当前词重要？
- Q (Query)、K (Key)、V (Value) 的含义
- Scaled Dot-Product Attention 的完整推导
- 因果掩码 (Causal Mask)：防止看到未来的词
- 动手实现：纯 NumPy 实现自注意力
- **与 LLM 的关系**：这就是 Transformer 的心脏

**配套资源与实践**
- 视频：`Day 2` [3Blue1Brown - Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- 代码：`02_self_attention.py`
- 完成标准：能画出 Q/K/V 维度并说明 mask 的作用

**对应论文与章节（先读这些）**
- `Attention Is All You Need`：`3.2 Attention`、`3.2.1 Scaled Dot-Product Attention`
- `GPT-2`：`2. Model`（重点看 masked self-attention / 自回归约束）
- `GPT-3`：`2.1 Model and Architectures`（看 decoder 堆叠中 attention 的使用）

### 第 3 课：多头注意力与残差

- 单头 → 多头：不同的头关注不同的模式
- 残差连接：解决深层网络的退化问题
- Layer Normalization：稳定训练过程
- 动手实现：用 PyTorch 实现多头注意力
- **与 LLM 的关系**：GPT-3 有 96 个注意力头

**配套资源与实践**
- 视频：`Day 3` [Jay Alammar - The Narrated Transformer](https://www.youtube.com/watch?v=-QH8fRhqFHM)
- 代码：`03_multi_head_attention.py`
- 完成标准：能说明单头 vs 多头，并跑通代码

**对应论文与章节（先读这些）**
- `Attention Is All You Need`：`3.1 Encoder and Decoder Stacks`（残差 + LayerNorm）、`3.2.2 Multi-Head Attention`
- `GPT-3`：`2.1 Model and Architectures`（关注层数、头数、宽度这些规模配置）
- `BERT`：`3.1 BERT Model Architecture`（对照 encoder 结构中的多头注意力）

### 第 4 课：Transformer Block

- 完整的 Transformer Block 结构
- Feed-Forward Network (FFN)：两层 MLP + 激活函数
- Pre-Norm vs Post-Norm 的区别
- 堆叠多个 Block：从 1 层到 N 层
- 动手实现：完整的 Transformer Block
- **与 LLM 的关系**：GPT-3 就是 96 个这样的 Block 堆叠

**配套资源与实践**
- 视频：`Day 4` [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)（前半段）
- 代码：`04_transformer_block.py`
- 完成标准：能跑通 block 前向，理解残差 + LayerNorm 位置

**对应论文与章节（先读这些）**
- `Attention Is All You Need`：`3.1 Encoder and Decoder Stacks`、`3.3 Position-wise Feed-Forward Networks`
- `GPT-2`：`2. Model`（把论文中的 block 结构与你代码逐项对齐）
- `GPT-3`：`2.1 Model and Architectures`（理解“同构 block 堆叠 + 扩大规模”的主线）

### 第 5 课：从零构建 GPT

- 完整模型：Embedding → N × Block → Linear → Softmax
- 训练循环：用小数据集训练一个迷你 GPT
- 文本生成：贪心搜索、Temperature、Top-K 采样
- 动手实现：一个能生成 Shakespeare 风格文本的小模型
- **与 LLM 的关系**：你写的就是 GPT 的完整架构，只是小了 1000 倍

**配套资源与实践**
- 视频：`Day 5` [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)（后半段）
- 代码：`05_gpt_from_scratch.py`
- 完成标准：完成一次小训练并生成可读文本

**对应论文与章节（先读这些）**
- `GPT-2`：`2. Model`、`3. Experiments`（先看模型，再看训练与生成设置）
- `GPT-3`：`2. Training Dataset and Model`、`3. Methodology`（理解数据规模、训练范式和 few-shot 评估）
- `InstructGPT`：`3. Methods`（在你完成基础 GPT 后，预习“预训练模型如何对齐”）

## 学习方式

1. **先读论文的对应章节**：比如学注意力前，先读 Attention Is All You Need 的 3.2 节
2. **跑代码看效果**：每课都有可视化输出
3. **对照论文的公式和代码**：确认自己真正理解了
4. **修改超参数**：改 head 数、layer 数、embedding 维度，观察效果

## 完成后你将理解

- Transformer 为什么能取代 RNN/LSTM
- 自注意力机制到底在计算什么
- GPT 和 BERT 的本质区别（Decoder vs Encoder）
- 为什么模型越大效果越好（Scaling Law 的直觉）
- 为什么训练大模型需要那么多数据和算力
- RLHF 是如何让模型变得"听话"的

## 复习与补充资源

建议在完成第 1-5 课后，用 2 天做集中复盘：
- `Day 6`：[B 站 - Transformer 从零详细解读](https://www.bilibili.com/video/BV1Di4y1c7Zm/) + 回看 `02_self_attention.py`、`03_multi_head_attention.py`
- `Day 7`：[B 站 - Transformer 理论到实战系列](https://www.bilibili.com/video/BV12bfPY1E1S/) + 回看 `04_transformer_block.py`、`05_gpt_from_scratch.py`
- 复盘目标：补齐注释、修正命名、能口述完整 GPT 前向流程

### 每次学习的固定节奏（建议）

1. **先看 20-40 分钟视频**：只记 3 个关键词（例如 Q/K/V、mask、residual）
2. **再写 45-90 分钟代码**：只实现当天一个文件，不跨天
3. **最后 10 分钟复盘**：回答 3 个问题：输入是什么？核心计算是什么？输出是什么？
4. **卡住就回看视频对应片段**：不要整段重看，直接定位到该概念章节

可选补充：
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（图解）
- [nanoGPT](https://github.com/karpathy/nanoGPT)（最小 GPT 实现）
- [李宏毅 - Transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA)（中文课程）

## 下一步

完成第二阶段后，进入第三阶段：**训练与微调** —— 学习 LoRA、量化、RLHF 等实用技术。
