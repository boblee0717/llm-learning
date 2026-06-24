# 第三阶段：训练与微调

> 从训练流程到 LoRA、量化、RLHF —— 掌握大模型的实用技术

## 前置要求

完成第二阶段的全部课程，理解：
- Transformer 完整架构
- 自注意力与多头注意力
- GPT 的训练与文本生成

## 环境准备

```bash
# 在项目根目录激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 课程结构

按顺序学习，每课约 60-120 分钟：

| 课程 | 文件 | 核心内容 | 关键概念 |
|------|------|----------|----------|
| 第 1 课 | `01_training_pipeline.py` | DataLoader、训练循环、验证、模型保存 | 工业级训练流程的完整实现 |
| 第 2 课 | `02_lora.py` | 低秩分解、LoRA 层、参数冻结 | 用 0.1% 的参数微调大模型 |
| 第 3 课 | `03_quantization.py` | FP32→FP16→INT8→INT4 量化 | 让大模型跑在消费级显卡上 |
| 第 4 课 | `04_rlhf.py` | 奖励模型、PPO、DPO | 让模型变得"有用且安全" |
| 第 5 课 | `05_inference_optimization.py` | KV Cache、采样策略、投机解码 | 让推理速度快 10 倍 |
| 第 5 课·附 | `kv_cache_numpy_demo.py` | 纯 NumPy 最小 KV Cache 对照演示 | 剥掉框架看懂「缓存复用」 |
| 第 6 课（待补） | 分布式训练专题 | 数据/张量/流水并行、ZeRO、FSDP | 单卡放不下时怎么沿多卡切分训练 |

## 每课详细大纲

### 第 1 课：完整训练流程

- 数据集准备与 DataLoader
- 训练循环的五步曲：forward → loss → backward → step → zero_grad
- 验证集评估与早停（Early Stopping）
- 混合精度训练（AMP）：用 FP16 加速训练
- 梯度累积：用小显存模拟大 batch
- 模型保存与加载（checkpoint）
- **与 LLM 的关系**：这就是预训练 GPT 时的完整流程

### 第 2 课：LoRA 微调

- 全参数微调的问题：175B 参数的 GPT-3 你存都存不下
- 低秩分解的数学原理：W + ΔW ≈ W + BA
- LoRA 的核心思想：冻结原始权重，只训练小矩阵 A 和 B
- 从零实现 LoRA 层
- rank 的选择对效果的影响
- **与 LLM 的关系**：几乎所有开源模型微调都在用 LoRA

### 第 3 课：模型量化

- 浮点数回顾：FP32、FP16、BF16 的区别
- 量化的原理：把浮点数映射到整数
- 对称量化 vs 非对称量化
- 逐张量 vs 逐通道量化
- 量化对模型精度的影响
- **与 LLM 的关系**：4-bit 量化让 70B 模型跑在单张 24GB 显卡上

### 第 4 课：RLHF 人类偏好对齐

- 为什么预训练后的模型不好用？——"能力"vs"对齐"
- SFT（监督微调）：教模型学会对话格式
- 奖励模型（Reward Model）：学习人类偏好
- PPO 强化学习：用奖励信号优化生成策略
- DPO：不需要奖励模型的更简洁方法
- **与 LLM 的关系**：ChatGPT = GPT + SFT + RLHF

### 第 5 课：推理优化

- 自回归生成的瓶颈：每次只生成一个 token
- KV Cache：避免重复计算，加速 10 倍
- 采样策略对比：贪心、Temperature、Top-K、Top-P
- Beam Search vs Sampling
- 投机解码（Speculative Decoding）：用小模型加速大模型
- **与 LLM 的关系**：ChatGPT 能秒回你消息，靠的就是这些优化
- **附加演示** `kv_cache_numpy_demo.py`：纯 NumPy 手写的 KV Cache 最小对照版（无缓存整段重算 vs 有缓存逐 token，验证结果一致 + 投影次数 O(n²) vs O(n) + 显存估算），配合 `papers/kv-cache/` 的 MQA / GQA / PagedAttention / FlashAttention 一起看

### 第 6 课：分布式训练专题（待补）

> 前面 1-5 课都是**单卡训练技巧**（AMP、梯度累积、LoRA、量化）。这一课补上「多卡 / 集群训练」这块空白，把第一阶段第 4 课算过的「每参数 ~16 字节显存账」从单卡推到集群。

- 三种并行：数据并行（DP）、张量并行（TP）、流水并行（PP）各切什么、各自的通信代价
- **ZeRO（Zero Redundancy Optimizer）**：数据并行下每张卡都冗余存一份优化器状态/梯度/参数，ZeRO-1/2/3 依次把这三部分沿 N 张卡切成 1/N
- FSDP（PyTorch 原生的 ZeRO 等价实现）
- **配套论文**: [ZeRO (Rajbhandari et al., 2020)](../papers/distributed-training/ZeRO_Memory_Optimizations_Toward_Training_Trillion_Parameter_Models_2020.pdf)，见 [papers/README.md 第 12 节](../papers/README.md)
- **与 LLM 的关系**：万亿参数模型、7B/70B 全参训练为什么必须多卡，靠的就是这些切分策略
- **读法提示**：ZeRO 是系统/分布式工程，不适合像 Attention 那样从零手搓，定位成「概念精读 + 看图」即可

## 学习方式

1. **先理解概念**：每课开头有详细的原理讲解
2. **跑代码看效果**：观察量化前后精度变化、LoRA 微调效果等
3. **对比实验**：改参数（rank、量化位数、学习率），观察影响
4. **读源码**：课后去看 HuggingFace PEFT、bitsandbytes 的实现

## 完成后你将理解

- 工业界如何训练和微调大模型
- 为什么 LoRA 能用极少参数达到接近全参数微调的效果
- 为什么量化后模型变小了但效果没差太多
- ChatGPT 从 GPT 到"能聊天"经历了哪些步骤
- 推理时的各种加速技巧背后的原理

## 推荐配套资源

- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft) - LoRA 等参数高效微调的官方实现
- [Andrej Karpathy - Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) - 完整的训练实战
- [李宏毅 - RLHF](https://www.youtube.com/watch?v=73kEe5bsLiQ) - 中文讲解 RLHF
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - 图解 GPT-2 的生成过程
- [QLoRA 论文](https://arxiv.org/abs/2305.14314) - 量化 + LoRA 的结合

## 下一步

完成第三阶段后，你已具备理解和使用大模型的完整知识体系。
可以开始：
- 用 HuggingFace 微调开源模型（Llama、Qwen 等）
- 部署自己的 LLM 服务
- 深入研究某个方向（多模态、Agent、长上下文等）
