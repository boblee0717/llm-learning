# LLM Learning — 从零理解大模型

> 面向后端开发者的大模型学习路径，用代码理解原理。

## 我的学习进度

当前状态：第 0 阶段第 1 课进行中，第一阶段已完成，第二阶段第 3 课进行中（Attention 论文精读 §3.1 / §3.2.2 已推进完成，GPT-3 §2.1 与 BERT Model Architecture 精读笔记已新增，代码实验继续）。

详细推进记录见 [learning-progress.md](learning-progress.md)。

## 项目结构

```
llm-learning/
├── phase0-math/                     # 第 0 阶段：矩阵运算补强（按需复习）
│   ├── 01_vectors_and_axes.py            # 向量、形状、axis、广播
│   ├── 02_matmul_and_shapes.py           # 矩阵乘法、@ / einsum、batched matmul
│   ├── 03_reshape_transpose_split.py     # reshape / transpose / split / 多头切分
│   ├── 04_matrix_calculus.py             # 线性层求导、softmax 雅可比、数值梯度校验
│   ├── 0X_*_self_write.py                # 每课配套自写练习
│   ├── reset_exercises_0X.py             # 每课配套重置脚本
│   └── README.md
│
├── phase1-foundations/              # 第一阶段：深度学习基础
│   ├── 01_numpy_basics.py               # 张量运算、矩阵乘法、Softmax
│   ├── 01_numpy_basics_self_write.py     # ↳ 自写练习（8 个 TODO）
│   ├── 02_gradient_descent.py            # 损失函数、梯度、参数更新
│   ├── 02_gradient_descent_self_write.py # ↳ 自写练习（6 个 TODO）
│   ├── 03_neural_network.py              # 前向/反向传播、激活函数
│   ├── 03_neural_network_self_write.py   # ↳ 自写练习（7 个 TODO）
│   ├── reset_exercises_01.py             # 重置第 1 课练习
│   ├── reset_exercises_02.py             # 重置第 2 课练习
│   ├── reset_exercises_03.py             # 重置第 3 课练习
│   └── README.md                         # 第一阶段详细指南
│
├── phase2-transformer/              # 第二阶段：Transformer 架构
│   ├── 01_word_embeddings.py             # 词嵌入、位置编码
│   ├── 01_word_embeddings_self_write.py  # ↳ 自写练习（第 1 课）
│   ├── 02_self_attention.py              # Q/K/V、注意力分数、掩码
│   ├── 02_self_attention_self_write.py   # ↳ 自写练习（第 2 课，8 个 TODO）
│   ├── 03_multi_head_attention.py        # 多头注意力、残差连接、LayerNorm
│   ├── 04_transformer_block.py           # 完整 Transformer Block
│   ├── 05_gpt_from_scratch.py            # 从零搭建 GPT，文本生成
│   ├── reset_exercises_01.py             # 重置第 1 课练习
│   └── README.md
│
├── phase3-training/                 # 第三阶段：训练与微调
│   ├── 01_training_pipeline.py           # DataLoader、AMP、梯度累积
│   ├── 02_lora.py                        # LoRA 低秩微调
│   ├── 03_quantization.py                # 模型量化 (INT8/INT4)
│   ├── 04_rlhf.py                        # RLHF / DPO 人类偏好对齐
│   ├── 05_inference_optimization.py      # KV Cache、采样策略、投机解码
│   └── README.md
│
├── papers/                          # 论文库
│   ├── core-transformers/                # Transformer / GPT / BERT / InstructGPT 主线论文
│   ├── attention-extensions/             # 位置编码、Self-Attention、线性注意力延伸论文
│   ├── efficient-transformers/           # 高效 Transformer 与长上下文论文
│   ├── scaling-laws/                     # 规模定律与 compute-optimal 训练论文
│   ├── vision-transformers/              # Vision Transformer 论文
│   ├── notes/                            # 论文精读笔记
│   └── README.md                         # 论文阅读顺序与建议
│
├── karpathy-best-resources.md       # Karpathy 精选文章/视频与学习路径
├── github-copilot-claude-code.md    # GitHub Copilot + Claude Code 配置指南
├── llm-interview-questions.md       # 大模型面试题整理（含精修参考答案）
├── harness-engineering-li-hongyi.md # 李宏毅《Harness Engineering》视频总结
├── learning-progress.md             # 个人学习进度记录
│
├── attention_paper_prerequisites.md # 读论文的前置知识清单
├── requirements.txt
└── README.md                        # ← 你正在看的文件
```

## 学习路线

```
第 0 阶段 (4课, 按需)        第一阶段 (3课)            第二阶段 (5课)             第三阶段 (5课)
矩阵运算补强            →  NumPy/梯度/神经网络    →  Attention/Transformer/GPT  →  LoRA/量化/RLHF/推理优化
   形状与反向                  基础数学                   核心架构                     工业实践
   🚧 第 1 课进行中            ✅ 已完成三课               🚧 第 3 课进行中             待学习
                                                        GPT-3 §2.1 / BERT 架构精读已新增
```

> **phase0-math 不是必经环节**：当你在 phase1/phase2 遇到形状或求导卡壳时回来跑对应那节即可。

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/boblee0717/llm-learning.git
cd llm-learning

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 4. 从第一课开始
python3 phase1-foundations/01_numpy_basics.py
```

## 每课的学习方式

每课包含 **主课文件** 和 **自写练习** 两部分（第一阶段已全部配备，第二阶段已开始配备）：

| 步骤 | 做什么 | 目的 |
|------|--------|------|
| 1 | 运行主课脚本，看输出 | 建立整体印象 |
| 2 | 逐段阅读代码，理解原理 | 搞懂"为什么" |
| 3 | 打开自写练习，按 TODO 从前往后填 | 亲手实现加深理解 |
| 4 | 每完成一个 TODO 就运行一次 | 利用校验系统即时纠错 |
| 5 | 全部通过后做 5 分钟复盘 | 写下"本课 3 个关键结论" |

### 练习重置

做完想再练一遍？每课都有重置脚本：

```bash
python3 phase1-foundations/reset_exercises_01.py   # 重置第 1 课
python3 phase1-foundations/reset_exercises_02.py   # 重置第 2 课
python3 phase1-foundations/reset_exercises_03.py   # 重置第 3 课
```

## 课程总览

### 第 0 阶段：矩阵运算补强（按需复习，4 课）

| 课程 | 主课文件 | 自写练习 | 核心内容 |
|------|----------|----------|----------|
| 第 1 课 | `01_vectors_and_axes.py` | `01_vectors_and_axes_self_write.py` | 向量、形状、`axis` / `keepdims`、广播规则 |
| 第 2 课 | `02_matmul_and_shapes.py` | `02_matmul_and_shapes_self_write.py` | 矩阵乘法两种解读、`@` / `einsum`、batched matmul |
| 第 3 课 | `03_reshape_transpose_split.py` | `03_reshape_transpose_split_self_write.py` | reshape / transpose / split、多头注意力切分 |
| 第 4 课 | `04_matrix_calculus.py` | `04_matrix_calculus_self_write.py` | 线性层反向传播、softmax 雅可比、数值梯度校验 |

> 详见 [phase0-math/README.md](phase0-math/README.md)

### 第一阶段：深度学习基础（已完成搭建）

| 课程 | 主课文件 | 自写练习 | 核心内容 |
|------|----------|----------|----------|
| 第 1 课 | `01_numpy_basics.py` | `01_numpy_basics_self_write.py` | 张量运算、矩阵乘法、Softmax、广播 |
| 第 2 课 | `02_gradient_descent.py` | `02_gradient_descent_self_write.py` | MSE 损失、梯度计算、参数更新、学习率实验 |
| 第 3 课 | `03_neural_network.py` | `03_neural_network_self_write.py` | ReLU/Sigmoid、前向传播、反向传播、XOR 分类 |

> 详见 [phase1-foundations/README.md](phase1-foundations/README.md)

### 第二阶段：Transformer 架构（第 1-2 课已配自写练习，第 3 课进行中）

| 课程 | 主课文件 | 自写练习 | 核心内容 |
|------|----------|----------|----------|
| 第 1 课 | `01_word_embeddings.py` | `01_word_embeddings_self_write.py` | 词嵌入、位置编码 |
| 第 2 课 | `02_self_attention.py` | `02_self_attention_self_write.py` | Q/K/V、注意力分数、因果掩码 |
| 第 3 课 | `03_multi_head_attention.py` | — | 多头注意力、残差连接、LayerNorm |
| 第 4 课 | `04_transformer_block.py` | — | 完整 Transformer Block、FFN |
| 第 5 课 | `05_gpt_from_scratch.py` | — | 完整 GPT 模型、文本生成 |

> 详见 [phase2-transformer/README.md](phase2-transformer/README.md)

### 第三阶段：训练与微调

| 课程 | 文件 | 核心内容 |
|------|------|----------|
| 第 1 课 | `01_training_pipeline.py` | DataLoader、AMP、梯度累积、Checkpoint |
| 第 2 课 | `02_lora.py` | LoRA 低秩微调 |
| 第 3 课 | `03_quantization.py` | 模型量化 (INT8/INT4) |
| 第 4 课 | `04_rlhf.py` | RLHF / DPO 人类偏好对齐 |
| 第 5 课 | `05_inference_optimization.py` | KV Cache、采样策略、投机解码 |

> 详见 [phase3-training/README.md](phase3-training/README.md)

### 核心论文

配合第二阶段一起阅读：

| 论文 | 年份 | 一句话概括 |
|------|------|-----------|
| Attention Is All You Need | 2017 | Transformer 的开山之作 |
| GPT-1（*Improving Language Understanding by Generative Pre-Training*） | 2018 | 生成式预训练 + 判别式微调，Decoder 迁移学习 |
| BERT | 2018 | 双向编码器，预训练+微调范式 |
| GPT-2 | 2019 | 纯 Decoder 语言模型，无监督多任务 |
| GPT-3 | 2020 | 175B 参数，In-context Learning |
| InstructGPT | 2022 | RLHF 落地，让模型遵循人类指令 |
| Harness engineering（OpenAI） | 2026 | Agent 为先：人设计环境与反馈回路，Codex 产出代码与工程资产 |
| Harness Engineering Is Cybernetics（George Zhang，X 原文） | 2026 | 将 Harness 置于控制论史：反馈回路在架构层闭合时的工程含义 |

> 详见 [papers/README.md](papers/README.md)（Harness 章节附 Anthropic *Emotion Concepts* 长文延伸阅读）

### 推荐外部资源

如果你希望配合本项目学习一条高质量主线，建议从 Karpathy 的内容开始：

- [karpathy-best-resources.md](karpathy-best-resources.md)（项目内整理：优先级、推荐理由、建议顺序）
- [github-copilot-claude-code.md](github-copilot-claude-code.md)（用 Copilot 订阅跑 Claude Code 的配置步骤）
- [harness-engineering-li-hongyi.md](harness-engineering-li-hongyi.md)（李宏毅 2026 Spring《Harness Engineering：有时候语言模型不是不够聪明，只是没有人类好好引导》视频精华笔记，含 Gemma 小实验、agents.md 地图论、Ralph Loop、Lifelong Agent 等案例）

## 面试复习

- [llm-interview-questions.md](llm-interview-questions.md)：大模型面试题整理（含精修参考答案），覆盖**应用开发一/二面、LLM 基础面、进阶面、微调与领域训练面**共 5 大板块 80+ 题，配有显存估算、解码策略速查表和「一页纸」速记。

## 依赖

- Python 3.8+
- NumPy + Matplotlib（第一阶段）
- PyTorch + tiktoken（第二、三阶段）

## 学完后你将理解

- 为什么大模型训练需要大量 GPU（本质是大规模矩阵运算）
- 梯度下降、前向传播、反向传播到底在做什么
- Transformer 为什么能取代 RNN/LSTM
- GPT 和 BERT 的本质区别（Decoder vs Encoder）
- LoRA、量化、RLHF 等工业实践背后的原理
- 从 GPT 到 ChatGPT 经历了哪些关键步骤
