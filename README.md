# LLM Learning — 从零理解大模型

> 面向后端开发者的大模型学习路径，用代码理解原理。

## 我的学习进度

当前状态：第 0 阶段第 1 课进行中，第一阶段三课已完成、新增第 4 课「优化器」（三件套已搭好待填），第二阶段 6 课全部完成。**第三阶段第 1 课 LoRA + 第 2 课量化 self_write 均已全部完成（各 6/6 TODO）**；量化主课 Part 1-7 已精读。**第 3 课 RLHF 精读中**（Part 4 PPO 改成真两阶段结构、看懂 old/ref 两参照物；Part 5 DPO「为什么 work」推导已理解并写进注释）。下一步：填 `03_rlhf_self_write.py` TODO-1（DPO loss）。

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
│   ├── 04_optimizers.py                  # SGD/Momentum/RMSprop/Adam/AdamW、内存代价、Adafactor
│   ├── 04_optimizers_self_write.py       # ↳ 自写练习（6 个 TODO）
│   ├── reset_exercises_01.py             # 重置第 1 课练习
│   ├── reset_exercises_02.py             # 重置第 2 课练习
│   ├── reset_exercises_03.py             # 重置第 3 课练习
│   ├── reset_exercises_04.py             # 重置第 4 课练习
│   └── README.md                         # 第一阶段详细指南
│
├── phase2-transformer/              # 第二阶段：Transformer 架构
│   ├── 01_word_embeddings.py             # 词嵌入、位置编码
│   ├── 01_word_embeddings_self_write.py  # ↳ 自写练习（第 1 课）
│   ├── 02_self_attention.py              # Q/K/V、注意力分数、掩码
│   ├── 02_self_attention_self_write.py   # ↳ 自写练习（第 2 课，8 个 TODO）
│   ├── 03_multi_head_attention.py        # 多头注意力、残差连接、LayerNorm
│   ├── 03_multi_head_attention_self_write.py # ↳ 自写练习（第 3 课，9 个 TODO）
│   ├── 04_transformer_block.py           # 完整 Transformer Block
│   ├── 04_transformer_block_self_write.py # ↳ 自写练习（第 4 课，9 个 TODO）
│   ├── 05_gpt_from_scratch.py            # 从零搭建 GPT，文本生成
│   ├── 05_gpt_from_scratch_self_write.py # ↳ 自写练习（第 5 课，12 个 TODO，PyTorch）
│   ├── 06_scaling_laws.py                # Scaling Law、C≈6ND、Chinchilla compute-optimal
│   ├── reset_exercises_0X.py             # 每课配套重置脚本（01–05，二刷用）
│   └── README.md
│
├── phase3-training/                 # 第三阶段：训练与微调（训练流程已并入 PyTorch 专项）
│   ├── 01_lora.py                        # LoRA 低秩微调
│   ├── 01_lora_self_write.py             # ↳ 自写练习（6 个 TODO，✅ 已完成）
│   ├── 02_quantization.py                # 模型量化 (INT8/INT4)
│   ├── 03_rlhf.py                        # RLHF / DPO 人类偏好对齐
│   ├── 04_inference_optimization.py      # KV Cache、采样策略、投机解码
│   ├── kv_cache_numpy_demo.py            # 纯 NumPy 最小 KV Cache 对照演示
│   └── README.md
│
├── phase4-deepseek-reasoning/        # 第四阶段：DeepSeek 与推理优化
│   ├── README.md                         # DeepSeek、MLA/MoE、推理服务、GRPO/R1、测试时计算
│   ├── benchmark_questions.md            # 本地推理优化与 reasoning budget 固定题库
│   └── video_courses.md                  # 第四阶段视频课程材料清单
│
├── phase5-agent-architecture/        # 第五阶段：Agent 架构与 Agent 基础设施
│   ├── README.md                         # 10 课：Agent Loop/Tool/规划/记忆/RAG/Runtime/沙箱/并发缓存/观测/Multi-Agent
│   ├── 01_minimal_agent_loop.py          # 最小 agent loop（observe→think→act + 终止条件）
│   ├── 02_tool_calling.py                # tool 注册表、dispatcher、参数校验、错误回灌
│   └── 03_agent_runtime.py               # run/step、checkpoint/resume、trace 与成本报告
│
├── papers/                          # 论文库
│   ├── core-transformers/                # Transformer / GPT / BERT / InstructGPT 主线论文
│   ├── attention-extensions/             # 位置编码、Self-Attention、线性注意力延伸论文
│   ├── efficient-transformers/           # 高效 Transformer 与长上下文论文
│   ├── kv-cache/                         # KV Cache 推理优化（MQA/GQA/PagedAttention/FlashAttention）
│   ├── distributed-training/             # 分布式训练与显存切分（ZeRO / FSDP）
│   ├── scaling-laws/                     # 规模定律与 compute-optimal 训练论文（配第二阶段第 6 课）
│   ├── vision-transformers/              # Vision Transformer 论文
│   ├── deepseek/                         # DeepSeek MoE / Coder / Math / V2 / V3 / R1
│   ├── frontier-llms/                    # Llama 3、Qwen2.5 等现代开源 LLM 技术报告
│   ├── efficient-training/               # QLoRA、DPO、PPO、RLHF/RLAIF 等训练 / 微调 / 强化学习对齐论文
│   ├── retrieval-augmented/              # 检索增强 RAG（REALM / RAG / RETRO）
│   ├── foundations/                      # 跨学科思想源头论文（如 Anderson《More Is Different》/ 涌现）
│   ├── frontier-ai-2024-2025.md          # 现代前沿论文清单，含训练 / 视频材料
│   ├── notes/                            # 论文精读笔记（含 scaling_laws_kaplan_2020.md / chinchilla_compute_optimal_2022.md）
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
第 0 阶段 (4课, 按需)        第一阶段 (4课)            第二阶段 (6课)             第三阶段 (5课)             第四阶段 (8课)
矩阵运算补强            →  NumPy/梯度/网络/优化器  →  Attention/Transformer/GPT/Scaling Law  →  LoRA/量化/RLHF/推理优化  →  DeepSeek/推理优化
   形状与反向                  基础数学                   核心架构                                    工业实践
   🚧 第 1 课进行中            ✅ 已完成三课，新增第4课优化器  ✅ 6 课全部完成（GPT/Scaling Law）        ✅ LoRA self_write  🚧主课    后续进阶

                                          ┈┈┈→  第五阶段 (10课)：Agent 架构与 Agent 基础设施
                                                 Agent Loop/Tool/规划/记忆/RAG + Runtime/沙箱/并发缓存/观测/Multi-Agent
                                                 应用范式 + Agent infra 均衡 · 课程已搭好待学
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
| 第 4 课 | `04_optimizers.py` | `04_optimizers_self_write.py` | SGD/Momentum/RMSprop/Adam/AdamW、优化器内存代价、Adafactor |

> 详见 [phase1-foundations/README.md](phase1-foundations/README.md)

### 第二阶段：Transformer 架构（6 课全部完成）

| 课程 | 主课文件 | 自写练习 | 核心内容 |
|------|----------|----------|----------|
| 第 1 课 | `01_word_embeddings.py` | `01_word_embeddings_self_write.py` | 词嵌入、位置编码 |
| 第 2 课 | `02_self_attention.py` | `02_self_attention_self_write.py` | Q/K/V、注意力分数、因果掩码 |
| 第 3 课 | `03_multi_head_attention.py` | `03_multi_head_attention_self_write.py` | 多头注意力、残差连接、LayerNorm |
| 第 4 课 | `04_transformer_block.py` | `04_transformer_block_self_write.py` | 完整 Transformer Block、FFN |
| 第 5 课 | `05_gpt_from_scratch.py` | `05_gpt_from_scratch_self_write.py` | 完整 GPT 模型、文本生成（12 个 TODO，PyTorch） |
| 第 6 课 | `06_scaling_laws.py` | — | Scaling Law、`C≈6ND`、Chinchilla compute-optimal、Chinchilla 之后 |

> 详见 [phase2-transformer/README.md](phase2-transformer/README.md)

### 第三阶段：训练与微调

> 原「训练流程」课已删除（并入 `pytorch-essentials/` 第 4/5/6/8 课），其余课已重命名为 01-04。

| 课程 | 文件 | 核心内容 |
|------|------|----------|
| 第 1 课 | `01_lora.py` | LoRA 低秩微调（self_write ✅，主课待精读） |
| 第 2 课 | `02_quantization.py` | 模型量化 (INT8/INT4) |
| 第 3 课 | `03_rlhf.py` | RLHF / DPO 人类偏好对齐 |
| 第 4 课 | `04_inference_optimization.py` | KV Cache、采样策略、投机解码 |
| 第 5 课（待补） | 分布式训练专题 | DP/TP/PP、ZeRO、FSDP |

> 详见 [phase3-training/README.md](phase3-training/README.md)

### 第四阶段：DeepSeek 与推理优化

| 课程 | 主题 | 核心内容 |
|------|------|----------|
| 第 1 课 | DeepSeek 总览 | MoE、MLA、GRPO、MTP、蒸馏的整体地图 |
| 第 2 课 | MoE 与激活参数 | routed experts、shared experts、负载均衡、active params |
| 第 3 课 | MLA 与 KV Cache 压缩 | MHA/GQA/MLA 对比、长上下文 KV Cache 估算 |
| 第 4 课 | Attention Kernel 与长上下文 | FlashAttention、稀疏注意力、DeepSeek-V4 DSA |
| 第 5 课 | Serving Runtime | vLLM、SGLang、TensorRT-LLM、PagedAttention、continuous batching |
| 第 6 课 | 解码加速 | Speculative Decoding、draft model、DeepSeek-V3 MTP |
| 第 7 课 | 推理能力后训练 | DeepSeekMath、GRPO、R1-Zero、cold start、RL reasoning |
| 第 8 课 | 蒸馏与测试时计算 | reasoning distillation、self-consistency、budgeted thinking、评测 |

> 详见 [phase4-deepseek-reasoning/README.md](phase4-deepseek-reasoning/README.md)

### 第五阶段：Agent 架构与 Agent 基础设施（课程已搭建，待学习）

应用范式与基础设施均衡，配套零依赖可运行脚本（`FakeLLM` 跑通骨架，无需 GPU/网络/API key）。

| 课程 | 主题 | 核心内容 | 配套代码 |
|------|------|----------|----------|
| 第 1 课 | Agent 总览与 Agent Loop | ReAct、agent loop、终止条件 | `01_minimal_agent_loop.py` |
| 第 2 课 | Tool Use / Function Calling | tool schema、parse/dispatch、错误回灌 | `02_tool_calling.py` |
| 第 3 课 | 规划与反思 | Plan-and-Execute、Reflexion、self-critique | — |
| 第 4 课 | 记忆系统 | short/long-term memory、向量检索、压缩 | — |
| 第 5 课 | RAG 与上下文工程 | chunking、retrieval、context assembly、context rot | — |
| 第 6 课 | Agent Runtime 与状态管理 | run/step、checkpoint/resume、durable execution | `03_agent_runtime.py` |
| 第 7 课 | 工具沙箱与安全 | sandbox、权限边界、超时、prompt injection、人审 | — |
| 第 8 课 | 并发、调度与上下文复用 | continuous batching、prefix/KV cache 复用、限流 | — |
| 第 9 课 | 可观测性、评测与成本 | tracing、span、token/latency/cost、agent eval | — |
| 第 10 课 | Multi-Agent 编排与框架对比 | orchestrator-worker、handoff、LangGraph/AutoGen 等 | — |

> 详见 [phase5-agent-architecture/README.md](phase5-agent-architecture/README.md)

### 核心论文

配合第二阶段一起阅读：

| 论文 | 年份 | 一句话概括 |
|------|------|-----------|
| Attention Is All You Need | 2017 | Transformer 的开山之作 |
| GPT-1（*Improving Language Understanding by Generative Pre-Training*） | 2018 | 生成式预训练 + 判别式微调，Decoder 迁移学习 |
| BERT | 2018 | 双向编码器，预训练+微调范式 |
| GPT-2 | 2019 | 纯 Decoder 语言模型，无监督多任务 |
| GPT-3 | 2020 | 175B 参数，In-context Learning |
| Scaling Laws for Neural Language Models | 2020 | GPT-3 的理论背景：loss 随 N/D/C 呈幂律下降，外推可计算 |
| Chinchilla（Training Compute-Optimal LLMs） | 2022 | 修正 Kaplan：N 和 D 应按 1:20 同步扩，70B + 1.4T 打败 280B + 300B |
| An Empirical Model of Large-Batch Training | 2018 | critical batch size 与 gradient noise scale 的源头：batch 多大才不浪费 compute，Kaplan §5.3 引用 |
| More Is Different（Anderson, *Science*） | 1972 | 「涌现 / emergence」思想源头：还原论 ≠ 构成论，规模累积引发质变——scaling laws 量变之外的互补线 |
| InstructGPT | 2022 | RLHF 落地，让模型遵循人类指令 |
| PPO（Proximal Policy Optimization） | 2017 | RLHF 三段式里 RL 那一步的默认算法：clip 裁剪策略更新，稳又好实现 |
| Deep RL from Human Preferences（Christiano） | 2017 | RLHF 思想源头：只用人类两两比较就能训出奖励模型，InstructGPT reward model 的前身 |
| DPO（Direct Preference Optimization） | 2023 | 把 RLHF 简化成一个偏好分类损失，无需奖励模型和 PPO，当下最流行对齐法之一 |
| Constitutional AI（RLAIF） | 2022 | 用一套「宪法」原则让模型自我批判，并用 AI 反馈替代人类标注做 RL 对齐 |
| GRPO（在 DeepSeekMath 内） | 2024 | PPO 变体，去掉 value model，用组内相对得分当基线；DeepSeek-R1 推理型 RL 核心 |
| ZeRO（Rajbhandari et al.） | 2020 | 分布式训练显存切分：把优化器状态/梯度/参数沿 N 卡切成 1/N，训得起万亿参数；FSDP 的鼻祖 |
| DeepSeek / 现代开源 LLM 论文清单 | 2023-2025 | MoE、MLA、代码/数学数据、推理型 RL、QLoRA、DPO |
| 检索增强 RAG 三件套（REALM / RAG / RETRO） | 2020-2021 | 知识不必全压进参数，可在预训练/生成时从外部知识库检索——RAG 系统的源头 |
| Harness engineering（OpenAI） | 2026 | Agent 为先：人设计环境与反馈回路，Codex 产出代码与工程资产 |
| Harness Engineering Is Cybernetics（George Zhang，X 原文） | 2026 | 将 Harness 置于控制论史：反馈回路在架构层闭合时的工程含义 |

> 详见 [papers/README.md](papers/README.md) 和 [papers/frontier-ai-2024-2025.md](papers/frontier-ai-2024-2025.md)（Harness 章节附 Anthropic *Emotion Concepts* 长文延伸阅读）

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
- DeepSeek 系列中的 MoE、MLA、GRPO、R1 推理后训练和现代推理服务优化
