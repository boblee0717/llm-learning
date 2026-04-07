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

### 第 2 课：自注意力机制

- 注意力的直觉：哪些词对当前词重要？
- Q (Query)、K (Key)、V (Value) 的含义
- Scaled Dot-Product Attention 的完整推导
- 因果掩码 (Causal Mask)：防止看到未来的词
- 动手实现：纯 NumPy 实现自注意力
- **与 LLM 的关系**：这就是 Transformer 的心脏

### 第 3 课：多头注意力与残差

- 单头 → 多头：不同的头关注不同的模式
- 残差连接：解决深层网络的退化问题
- Layer Normalization：稳定训练过程
- 动手实现：用 PyTorch 实现多头注意力
- **与 LLM 的关系**：GPT-3 有 96 个注意力头

### 第 4 课：Transformer Block

- 完整的 Transformer Block 结构
- Feed-Forward Network (FFN)：两层 MLP + 激活函数
- Pre-Norm vs Post-Norm 的区别
- 堆叠多个 Block：从 1 层到 N 层
- 动手实现：完整的 Transformer Block
- **与 LLM 的关系**：GPT-3 就是 96 个这样的 Block 堆叠

### 第 5 课：从零构建 GPT

- 完整模型：Embedding → N × Block → Linear → Softmax
- 训练循环：用小数据集训练一个迷你 GPT
- 文本生成：贪心搜索、Temperature、Top-K 采样
- 动手实现：一个能生成 Shakespeare 风格文本的小模型
- **与 LLM 的关系**：你写的就是 GPT 的完整架构，只是小了 1000 倍

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

## 推荐配套资源

- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - 从零实现 GPT，最好的教程
- [3Blue1Brown - Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) - 注意力机制的可视化
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 图解 Transformer
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy 的最小 GPT 实现
- [李宏毅 - Transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA) - 中文讲解 Transformer

## 下一步

完成第二阶段后，进入第三阶段：**训练与微调** —— 学习 LoRA、量化、RLHF 等实用技术。
