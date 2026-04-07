# LLM Learning — 从零理解大模型

> 面向后端开发者的大模型学习路径，用代码理解原理。

## 项目结构

```
llm-learning/
├── phase1-foundations/       # 第一阶段：深度学习基础
│   ├── 01_numpy_basics.py        # 张量运算、矩阵乘法、Softmax
│   ├── 02_gradient_descent.py    # 损失函数、梯度、参数更新
│   └── 03_neural_network.py      # 前向/反向传播、激活函数
│
├── phase2-transformer/       # 第二阶段：Transformer 架构
│   ├── 01_word_embeddings.py     # 词嵌入、位置编码
│   ├── 02_self_attention.py      # Q/K/V、注意力分数、掩码
│   ├── 03_multi_head_attention.py# 多头注意力、残差连接、LayerNorm
│   ├── 04_transformer_block.py   # 完整 Transformer Block
│   └── 05_gpt_from_scratch.py    # 从零搭建 GPT，文本生成
│
├── phase3-training/          # 第三阶段：训练与微调
│   ├── 01_training_pipeline.py   # DataLoader、AMP、梯度累积、Checkpoint
│   ├── 02_lora.py                # LoRA 低秩微调
│   ├── 03_quantization.py        # 模型量化 (INT8/INT4)
│   ├── 04_rlhf.py                # RLHF / DPO 人类偏好对齐
│   └── 05_inference_optimization.py # KV Cache、采样策略、投机解码
│
├── papers/                   # 核心论文
│   ├── Attention_Is_All_You_Need_2017.pdf
│   ├── BERT_2018.pdf
│   ├── GPT2_*.pdf
│   ├── GPT3_*.pdf
│   └── InstructGPT_*.pdf
│
├── attention_paper_prerequisites.md  # 读论文的前置知识清单
└── requirements.txt
```

## 学习路线

```
第一阶段 (3课)          第二阶段 (5课)           第三阶段 (5课)
NumPy/梯度/神经网络  →  Attention/Transformer/GPT  →  LoRA/量化/RLHF/推理优化
     基础数学              核心架构                   工业实践
```

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 从第一课开始
python3 phase1-foundations/01_numpy_basics.py
```

## 每课的学习方式

1. 先运行一遍，看输出
2. 逐段阅读代码，理解原理
3. 动手修改参数，观察变化
4. 完成课后练习

## 依赖

- Python 3.8+
- NumPy + Matplotlib（第一阶段）
- PyTorch（第二、三阶段）
