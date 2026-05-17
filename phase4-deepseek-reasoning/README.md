# 第四阶段：DeepSeek 与推理优化

> 接在主线课程之后，用 DeepSeek 系列把「模型为什么更会推理」和「服务为什么更快更省」两条线合起来学。

截至 2026-05-17，本阶段把 DeepSeek 的论文主线、官方开源模型、推理系统论文和服务框架放在同一张地图里。这里的「推理优化」同时覆盖两层含义：

- **Inference optimization**：降低延迟、吞吐成本和 KV Cache 显存，让模型服务跑得更快更稳。
- **Reasoning optimization**：通过数据、后训练、RL、蒸馏、测试时计算，让模型在数学、代码、复杂任务上更会思考。

## 前置要求

完成第三阶段后再进入本阶段，至少理解：

- GPT 自回归生成流程、KV Cache、采样策略、投机解码
- LoRA / QLoRA / 量化的基本思想
- SFT / RLHF / DPO 的差别
- Transformer Block、Multi-Head Attention、FFN、LayerNorm

如果只学完第二阶段，也可以先看第 1-3 课，但第 6-8 课会用到第三阶段的训练与对齐知识。

## 快速索引

| 入口 | 用途 |
|------|------|
| [课程结构](#course-structure) | 8 课主线，一课一个技术问题 |
| [学习节奏](#rhythm) | 和前面阶段一致的「读论文 -> 跑代码/实验 -> 复盘」节奏 |
| [本地推理实验室](#local-lab) | 本阶段最重要的实战训练路线 |
| [Benchmark 题库](benchmark_questions.md) | 数学、代码、逻辑、长上下文、服务压测题目 |
| [视频与课程材料](video_courses.md) | DeepLearning.AI、Stanford CS336、Hugging Face、中文 DeepSeek/R1 视频 |
| [材料清单](#materials) | 必读论文、官方文档、实现框架 |
| [实践项目](#projects) | 学完后能做的 4 个小项目 |
| [读法建议](#reading-guide) | DeepSeek 论文怎么抓主线 |

<a id="course-structure"></a>

## 课程结构

按顺序学习，每课约 90-150 分钟。第 1-5 课偏 inference，第 6-8 课偏 reasoning。

| 课程 | 主题 | 核心问题 | 关键概念 | 实战训练 |
|------|------|----------|----------|----------|
| 第 1 课 | DeepSeek 总览 | DeepSeek 为什么能把能力、成本、开源结合起来？ | MoE、MLA、GRPO、MTP、distillation | 选定本地模型与基准题库 |
| 第 2 课 | MoE 与激活参数 | 为什么总参数很大但每 token 只激活一小部分？ | routed experts、shared experts、load balancing、activated params | 估算 active params 与权重显存 |
| 第 3 课 | MLA 与 KV Cache 压缩 | 长上下文推理为什么卡在 KV Cache？DeepSeek-V2 怎么压？ | MHA/GQA/MLA、latent KV、KV cache formula | 做 KV Cache 显存计算器 |
| 第 4 课 | Attention Kernel 与长上下文 | 为什么 FlashAttention 快？长上下文还需要哪些结构优化？ | IO-aware attention、tiling、sparse attention、DSA | 做 1K/4K/16K/32K 长上下文实验 |
| 第 5 课 | Serving Runtime | vLLM/SGLang/TensorRT-LLM 到底在优化什么？ | PagedAttention、continuous batching、prefix cache、RadixAttention | 本地服务压测 TTFT/TPOT/QPS |
| 第 6 课 | 解码加速 | 单请求生成为什么慢？怎样一次多拿几个 token？ | speculative decoding、draft model、verification、MTP | 测 speculative decoding 或不同解码参数 |
| 第 7 课 | 推理能力后训练 | R1 的「会思考」来自哪里？ | cold start、RLVR、GRPO、self-verification、reflection | 对比 instruct 模型与 R1 distill |
| 第 8 课 | 蒸馏、评测与测试时计算 | 怎么把大模型推理能力迁移到小模型？怎么评估成本收益？ | reasoning distillation、self-consistency、budgeted thinking、math/code eval | 做 reasoning budget 评测报告 |

## 视频课程总览

详细清单见 [video_courses.md](video_courses.md)。本阶段视频/课程材料按下面优先级使用：

| 优先级 | 材料 | 对应主题 |
|--------|------|----------|
| 1 | [DeepLearning.AI - Efficiently Serving LLMs](https://www.deeplearning.ai/courses/efficiently-serving-llms) | KV Cache、batching、continuous batching、量化、LoRA serving |
| 2 | [DeepLearning.AI - Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth) | 量化原理与误差分析 |
| 3 | [DeepLearning.AI - Fine-tuning & RL for LLMs](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training) | SFT、RLHF、PPO、GRPO、eval、post-training |
| 4 | [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/index.html) | MoE、GPU、kernels、inference、evaluation、alignment |
| 5 | [Hugging Face GRPO Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) | GRPO reasoning post-training 实操 |
| 6 | [李宏毅 ML 2025 第七讲笔记：DeepSeek-R1 等推理模型如何深度思考](https://alanhou.org/blog/hungyi-lee-ml2025-lecture7/) | test-time compute、distillation、reasoning model 直觉 |
| 7 | [DeepSeek-R1 论文精读 + GRPO 原理&代码讲解](https://www.bilibili.com/video/BV1wU98YcEGy/) | R1 论文精读和 GRPO 代码理解 |

每课具体看哪些视频，按 [video_courses.md](video_courses.md#每课视频任务) 的「每课视频任务」执行。

<a id="rhythm"></a>

## 学习节奏

每课沿用前面阶段的架构：

| 步骤 | 做什么 | 产出 |
|------|--------|------|
| 1 | 先读本课「一句话目标」和关键概念 | 明确这节课解决哪个瓶颈 |
| 2 | 读论文 Abstract / Introduction / Figure / Table | 建立论文级主线 |
| 3 | 对照第三阶段代码或开源框架文档 | 把概念落到实现位置 |
| 4 | 做一个小实验或画一张结构图 | 不只会背名词 |
| 5 | 写 3 条复盘 | 记录「瓶颈 -> 技术 -> 代价」 |

复盘固定回答：

1. 这个技术优化的是 **latency、throughput、memory、quality** 中哪一个？
2. 它依赖的是 **模型结构、训练方法、解码算法、服务调度、硬件 kernel** 中哪一层？
3. 它牺牲了什么：实现复杂度、显存、精度、稳定性、通用性，还是输出可控性？

<a id="local-lab"></a>

## 本地推理实验室

这是本阶段最重要的成长训练。不要只追求「本地跑起来」，而是建立一个可重复实验流程：同一组题、同一个模型、固定采样参数，逐步更换推理框架和优化手段，记录指标变化。

### 第一轮推荐模型

| 场景 | 推荐模型 | 目的 |
|------|----------|------|
| 普通指令基线 | `Qwen2.5-7B-Instruct` 或同级 instruct 模型 | 建立非 reasoning 基线 |
| 推理模型基线 | `DeepSeek-R1-Distill-Qwen-7B` | 观察 thinking token 对质量和延迟的影响 |
| 代码方向 | `Qwen2.5-Coder-7B` 或 DeepSeek-Coder 系列 | 单独测试代码题 |
| 显存更充足 | 14B / 32B distill 模型 | 观察模型规模带来的质量与成本变化 |

第一轮不要急着上 70B。7B/14B 足够让你看清大多数推理优化现象。

### 实验指标

| 指标 | 含义 | 记录方式 |
|------|------|----------|
| TTFT | Time To First Token，首 token 延迟 | 客户端计时或框架日志 |
| TPOT | Time Per Output Token，平均每 token 延迟 | 总 decode 时间 / 输出 token 数 |
| tokens/s | 生成速度 | 输出 token 数 / decode 时间 |
| peak VRAM | 峰值显存 | `nvidia-smi`、框架日志或监控工具 |
| prompt tokens | 输入 token 数 | tokenizer 或服务返回 usage |
| output tokens | 输出 token 数 | tokenizer 或服务返回 usage |
| correctness | 题目是否答对 | 用 [benchmark_questions.md](benchmark_questions.md) 评分 |
| failure type | 错因 | 算错、代码错、理解错、超时、OOM、格式错 |

### 实验记录模板

| 日期 | 模型 | 框架 | 量化 | prompt 长度 | 并发 | TTFT | tokens/s | 显存 | 正确率 | 备注 |
|------|------|------|------|-------------|------|------|----------|------|--------|------|
|  |  |  |  |  |  |  |  |  |  |  |

### 四个必做训练

| 训练 | 对应课程 | 目标 |
|------|----------|------|
| 本地推理基线 | 第 1 课 | 跑通 instruct 与 reasoning 模型，记录第一版指标 |
| 量化对比 | 第 2-3 课 | 比较 FP16/BF16、INT8、INT4/GGUF Q4 的显存、速度、质量 |
| KV Cache / 长上下文实验 | 第 3-5 课 | 比较 1K/4K/16K/32K prompt 的 prefill、decode、显存变化 |
| Reasoning Budget 评测 | 第 7-8 课 | 比较普通模型、R1 distill、低/高输出预算、多采样的成本收益 |

第一轮目标不是得到漂亮分数，而是形成工程直觉：一个优化到底省了多少显存、快了多少、损失了多少质量。

## 每课详细大纲

### 第 1 课：DeepSeek 技术路线总览

**一句话目标**：先把 DeepSeek 系列放进一条演化线，不急着啃细节。

**主线**

- DeepSeekMoE：专家细分 + 共享专家，解决稀疏模型的专家冗余。
- DeepSeek-V2：MLA + DeepSeekMoE，把 KV Cache 压缩和 MoE 结合起来。
- DeepSeek-V3：更大规模 MoE，加上 FP8、MTP、无辅助损失负载均衡等工程技巧。
- DeepSeekMath：数学数据继续预训练 + GRPO，R1 的前置技术土壤。
- DeepSeek-R1：冷启动 SFT + 大规模 RL + 蒸馏，形成开源推理模型路线。
- DeepSeek-R1-0528 / V3.1 / V3.2 / V4 Preview：官方后续版本，作为前沿观察材料，不作为第一轮必读。

**要做**

- 读 [papers/frontier-ai-2024-2025.md](../papers/frontier-ai-2024-2025.md) 的 DeepSeek 1-6 项。
- 画一条时间线：`MoE -> V2(MLA) -> V3(MTP/FP8) -> Math(GRPO) -> R1(RL reasoning) -> V4(1M context/DSA)`。
- 复盘：DeepSeek 的关键词不是单个模型，而是「稀疏架构 + KV 压缩 + 推理型后训练 + 开源蒸馏」。

**实战训练**

- 选定第一轮本地模型：一个普通 instruct 模型 + 一个 R1 distill reasoning 模型。
- 从 [benchmark_questions.md](benchmark_questions.md) 里抽取 10 道题做 smoke test，确认模型能稳定输出。
- 建立第一版实验表，记录模型、框架、量化格式、显存和 tokens/s。

### 第 2 课：MoE 与激活参数

**一句话目标**：理解「671B total / 37B active」这种说法，不再把总参数等同于每 token 计算量。

**核心内容**

- Dense FFN vs MoE FFN：每层 FFN 从「一个大 MLP」变成「多个专家 MLP」。
- Router / gate：每个 token 选择少数专家。
- routed experts 与 shared experts：前者负责差异化知识，后者负责通用知识。
- 负载均衡：避免所有 token 都挤到少数专家。
- 工程代价：专家并行、通信、路由不均、批内 token 分布不稳定。

**材料**

- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 本地 PDF：[DeepSeekMoE](../papers/deepseek/DeepSeekMoE_Towards_Ultimate_Expert_Specialization_2024.pdf)、[DeepSeek-V3](../papers/deepseek/DeepSeek-V3_Technical_Report_2024.pdf)

**动手任务**

- 画出普通 FFN 和 MoE FFN 的 forward 流程。
- 用一句话解释：为什么 MoE 可以「参数多、计算不按总参数线性增长」。
- 估算 7B、14B、32B 模型在 FP16、INT8、INT4 下的权重显存。
- 如果使用 MoE 模型，区分 total params、active params、实际权重加载显存、每 token 计算量。

### 第 3 课：MLA 与 KV Cache 压缩

**一句话目标**：把第三阶段第 5 课的 KV Cache 推到现代长上下文服务的真实瓶颈。

**核心内容**

- KV Cache 显存估算：`layers × batch × seq_len × heads × head_dim × 2(K,V) × dtype_bytes`。
- MHA / MQA / GQA / MLA 的差别。
- MLA 的直觉：不直接缓存每个 head 的完整 K/V，而是缓存低维 latent，再按需恢复。
- DeepSeek-V2 把 MLA 和 MoE 结合，主打「高性能 + 低 KV Cache」。

**材料**

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- 本地 PDF：[DeepSeek-V2](../papers/deepseek/DeepSeek-V2_Strong_Economical_Efficient_MoE_2024.pdf)
- 复习代码：[phase3-training/05_inference_optimization.py](../phase3-training/05_inference_optimization.py)

**动手任务**

- 用 7B / 32B / 70B 三个假设模型，估算 4K、32K、128K 上下文下 KV Cache 量级。
- 写一张对比表：MHA、MQA、GQA、MLA 分别牺牲什么、换来什么。
- 做一个 `kv_cache_calculator` 小脚本或表格，输入层数、KV heads、head_dim、batch、seq_len、dtype，输出显存。
- 对比同一个模型在 1K / 4K / 16K prompt 下的峰值显存，验证公式和实测差异。

### 第 4 课：Attention Kernel 与长上下文

**一句话目标**：理解为什么「理论复杂度」之外，还要关心 GPU 内存读写。

**核心内容**

- 标准 attention 慢在哪里：不是只有 FLOPs，还有 HBM 与 SRAM 之间的数据搬运。
- FlashAttention：用 tiling 减少 HBM 读写，是 exact attention，不是近似 attention。
- 长上下文进一步需要：稀疏注意力、滑窗、分块、分层 KV、prefix/context cache。
- DeepSeek-V4 Preview 的前沿观察：官方提到 token-wise compression + DSA，并把 1M context 作为官方服务默认能力。

**材料**

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [DeepSeek-V4 Preview 官方发布](https://api-docs.deepseek.com/news/news260424)
- 可选回顾：[Sparse Transformer](../papers/efficient-transformers/Generating_Long_Sequences_with_Sparse_Transformers_2019.pdf)

**动手任务**

- 画出标准 attention 与 FlashAttention 的「读写矩阵」差异。
- 复盘：FlashAttention 主要优化的是训练、prefill、decode 中哪一段？为什么 decode 仍然常被 KV Cache 和逐 token 串行限制？
- 用同一个模型测试 1K / 4K / 16K / 32K prompt，分别记录 prefill 时间、decode tokens/s、峰值显存。
- 如果框架支持 FlashAttention 开关，做一次开/关对比；如果不支持，只写清当前框架默认使用了什么 attention 后端。

### 第 5 课：Serving Runtime：vLLM / SGLang / TensorRT-LLM

**一句话目标**：从单次 forward 走到真实在线服务：请求是动态来的，长度不一样，GPU 不能干等。

**核心内容**

- TTFT、TPOT、QPS、throughput、batch size、并发数。
- prefill 与 decode：吞吐瓶颈不同。
- PagedAttention：像操作系统分页一样管理 KV Cache，减少碎片和浪费。
- Continuous batching / in-flight batching：每个 decode step 都可以插入新请求或移除完成请求。
- Prefix cache / context cache：多轮对话、Agent、RAG 中复用相同前缀的 KV。
- SGLang RadixAttention：用 radix tree 管理可复用前缀。
- TensorRT-LLM：更靠近 NVIDIA GPU 生产部署栈，关注 kernel、quantization、paged KV、in-flight batching。

**材料**

- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM 官网](https://www.vllm.ai/)
- [SGLang 文档](https://docs.sglang.io/)
- [TensorRT-LLM 文档](https://docs.nvidia.com/tensorrt-llm/index.html)
- [DeepSeek API Context Caching](https://api-docs.deepseek.com/guides/kv_cache)

**动手任务**

- 用一个表格比较 vLLM、SGLang、TensorRT-LLM、llama.cpp：

| 框架 | 适合场景 | 核心优化 | 第一轮学习重点 |
|------|----------|----------|----------------|
| vLLM | 通用高吞吐 API 服务 | PagedAttention、continuous batching | KV 管理与调度 |
| SGLang | Agent / 结构化生成 / 前缀复用多 | RadixAttention、prefix cache | 多轮和程序化调用 |
| TensorRT-LLM | NVIDIA 生产部署 | kernel、paged KV、quantization、IFB | 极致性能栈 |
| llama.cpp | 本地/边缘/消费级硬件 | GGUF、量化、CPU/GPU 混合 | 模型压缩与本地跑通 |

- 至少选择一个服务框架做压测：本地优先 `llama.cpp` / Ollama / LM Studio，有 NVIDIA GPU 再上 vLLM 或 SGLang。
- 固定同一组 prompt，测试并发 `1 / 2 / 4 / 8` 下的 TTFT、TPOT、tokens/s、失败率。
- 对比短 prompt + 长输出、长 prompt + 短输出两类负载，区分 prefill-heavy 和 decode-heavy。

### 第 6 课：解码加速与 Multi-Token Prediction

**一句话目标**：自回归 decode 的串行性很硬，所以要么让一次 forward 验证多个 token，要么让模型学会多预测几个 token。

**核心内容**

- Speculative decoding：小模型先草拟，大模型并行验证，输出分布可保持不变。
- draft model 的质量和速度决定加速上限。
- DeepSeek-V3 的 MTP：训练时引入多 token 预测目标，既可能增强表示，也为推理加速提供结构基础。
- 解码加速要和服务框架结合看：单请求 latency 与批量 throughput 不总是同一个目标。

**材料**

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 复习代码：[phase3-training/05_inference_optimization.py](../phase3-training/05_inference_optimization.py)

**动手任务**

- 解释为什么 speculative decoding 能做到「加速但不改输出分布」。
- 思考：为什么 draft model 太弱或太慢都会让加速失败？
- 如果框架支持 speculative decoding，固定目标模型，换不同 draft 配置记录接受率、tokens/s、质量变化。
- 如果暂时不支持，就用第三阶段 `05_inference_optimization.py` 的投机解码演示复盘，并补一张「draft -> verify -> accept/reject」流程图。

### 第 7 课：DeepSeekMath / R1 与推理能力后训练

**一句话目标**：理解 R1 不是单纯「prompt 让它多想」，而是后训练目标改变了模型的行为分布。

**核心内容**

- DeepSeekMath：数学数据继续预训练，配合 SFT 和 GRPO。
- GRPO：PPO 的变体，用 group scores 做相对基线，减少 critic 负担。
- R1-Zero：直接在 base model 上 RL，出现自我验证、反思、长 CoT，但也有重复、可读性差、语言混杂。
- R1：加入 cold-start data、多阶段训练和蒸馏，让推理过程更可读、更稳定。
- RLVR：可验证奖励在数学、代码题上的价值。

**材料**

- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
- 本地 PDF：[DeepSeekMath](../papers/deepseek/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_2024.pdf)、[DeepSeek-R1](../papers/deepseek/DeepSeek-R1_Incentivizing_Reasoning_Capability_2025.pdf)

**动手任务**

- 画出 R1 pipeline：`base -> R1-Zero(RL) -> cold start SFT -> reasoning RL -> rejection sampling/SFT -> final RL -> distill`。
- 写一段 200 字复盘：为什么「没有 SFT 的 RL」能激发推理，但不一定适合直接给用户用？
- 用同一批数学/代码/逻辑题，对比普通 instruct 模型和 R1 distill 模型。
- 记录每道题的输出 token 数、是否答对、是否出现过度推理、是否格式混乱。

### 第 8 课：蒸馏、评测与测试时计算

**一句话目标**：最终要能回答一个工程问题：多花 token 思考，值不值？

**核心内容**

- Reasoning distillation：用大 reasoning model 生成高质量思维轨迹，小模型做 SFT。
- Self-consistency：多采样、多解路径投票，换质量但增加成本。
- Verifier / reward model / unit test：用外部可验证信号筛掉坏答案。
- Budgeted thinking：根据任务难度控制 thinking token，不是每个问题都开最大预算。
- 评测维度：正确率、平均输出 token、TTFT、TPOT、总成本、失败样例类型。
- 风险：过长思维链、幻觉细节、不可控工具调用、benchmark overfitting。

**材料**

- [Open-R1](https://github.com/huggingface/open-r1)
- [DeepSeek-R1-0528 官方发布](https://api-docs.deepseek.com/news/news250528)
- [DeepSeek-V3.1 官方发布](https://api-docs.deepseek.com/news/news250821)
- [DeepSeek-V3.2 官方发布](https://api-docs.deepseek.com/news/news251201)
- [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)

**动手任务**

- 选 20 道数学/代码/逻辑题，比较「普通模型」「reasoning 模型」「reasoning 模型低/高 token 预算」。
- 记录：正确率、平均生成 token、平均响应时间、失败类型。
- 复盘：哪些题值得开 thinking mode，哪些题不值得？
- 从 [benchmark_questions.md](benchmark_questions.md) 选一组固定题，做最终报告。
- 至少比较 3 种配置：普通模型、R1 distill 单次生成、R1 distill 多采样 self-consistency。
- 把结果写成一页结论：质量提升多少，成本增加多少，最典型的失败样例是什么。

<a id="materials"></a>

## 必要学习材料

### A. DeepSeek 主线论文

| 优先级 | 材料 | 读法 |
|--------|------|------|
| 必读 | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | 抓 shared experts / routed experts |
| 必读 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 重点读 MLA 与 KV Cache 压缩 |
| 必读 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 抓 MoE、MLA、MTP、FP8、负载均衡 |
| 必读 | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 看数学数据与 GRPO |
| 必读 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 看 R1-Zero、cold start、多阶段 RL、蒸馏 |
| 观察 | [DeepSeek-R1-0528](https://api-docs.deepseek.com/news/news250528) | 看 R1 后续产品化改进 |
| 观察 | [DeepSeek-V3.1](https://api-docs.deepseek.com/news/news250821) / [V3.2](https://api-docs.deepseek.com/news/news251201) | 看 thinking/non-thinking、tool-use、agent 方向 |
| 观察 | [DeepSeek-V4 Preview](https://api-docs.deepseek.com/news/news260424) | 看 1M context、DSA、V4-Pro/V4-Flash |

### B. 推理系统与解码优化

| 优先级 | 材料 | 读法 |
|--------|------|------|
| 必读 | [FlashAttention](https://arxiv.org/abs/2205.14135) | 理解 IO-aware exact attention |
| 必读 | [PagedAttention](https://arxiv.org/abs/2309.06180) | 理解 KV Cache 分页管理 |
| 必读 | [Speculative Decoding](https://arxiv.org/abs/2211.17192) | 理解 draft + verify |
| 必读 | [vLLM](https://www.vllm.ai/) | 看 PagedAttention、continuous batching、OpenAI-compatible server |
| 必读 | [SGLang](https://docs.sglang.io/) | 看 RadixAttention、prefix cache、Agent/structured output 场景 |
| 选读 | [TensorRT-LLM](https://docs.nvidia.com/tensorrt-llm/index.html) | 有 NVIDIA 部署需求时再深入 |

### C. 推理能力与后训练

| 优先级 | 材料 | 读法 |
|--------|------|------|
| 必读 | [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1) | 看 model summary、distill models、usage recommendations |
| 必读 | [Open-R1](https://github.com/huggingface/open-r1) | 看复现路线和数据/训练 pipeline |
| 选读 | [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/) | 从闭源 reasoning 模型角度看能力与安全评测 |
| 选读 | [DeepSeek API Thinking Mode](https://api-docs.deepseek.com/guides/reasoning_model) | 看产品层如何暴露 thinking 模式 |

<a id="projects"></a>

## 实践项目

### 项目 1：KV Cache 显存计算器

输入模型层数、head 数、head_dim、batch、seq_len、dtype，输出 KV Cache 显存，并比较 MHA / GQA / MLA 的数量级差异。

**目标**：看到长上下文服务的真实瓶颈。

**验收标准**

- 能输出 1K / 4K / 16K / 32K / 128K 上下文的 KV Cache 显存。
- 能解释为什么 batch、seq_len、KV heads 会线性放大显存。
- 能把公式估算值和一次本地实测结果放在同一张表里。

### 项目 2：vLLM / SGLang 推理基准

用同一个开源小模型，分别测试不同并发、不同 prompt 长度、不同输出长度下的 TTFT / TPOT / throughput。

**目标**：理解 prefill-heavy 与 decode-heavy 的差别。

**验收标准**

- 至少测试并发 `1 / 2 / 4 / 8`。
- 至少测试短 prompt 长输出、长 prompt 短输出两类任务。
- 能说清吞吐变高时，单请求延迟是否也变好，以及为什么。

### 项目 3：Speculative Decoding 小实验

用一个小 draft model 和一个 target model，测接受率、速度提升和输出一致性。

**目标**：理解 draft 质量、验证成本和加速比的关系。

**验收标准**

- 至少记录 draft 长度、接受率、tokens/s、输出质量。
- 能解释为什么 draft model 太弱、太慢或分布差太多都会拖累加速。
- 如果本地框架暂不支持，就用第三阶段脚本做模拟实验并写出限制。

### 项目 4：Reasoning Budget 评测

准备一小组数学/代码题，对比普通模式、thinking 模式、不同 token budget、多采样 self-consistency。

**目标**：学会用「质量-延迟-成本」三角形评估 reasoning 模型。

**验收标准**

- 使用 [benchmark_questions.md](benchmark_questions.md) 的最小正式评测组合。
- 至少比较普通 instruct、R1 distill 单次、R1 distill 多采样三种配置。
- 输出一张表：正确率、平均输出 token、平均耗时、典型失败样例。

## 阶段性训练安排

如果你想把本阶段当成 4 周实战小课，可以按这个节奏：

| 周次 | 训练重点 | 产出 |
|------|----------|------|
| Week 1 | 本地模型基线 + 题库 smoke test | 第一版实验记录表 |
| Week 2 | 量化 + KV Cache + 长上下文 | 显存/速度对比表 |
| Week 3 | vLLM/SGLang/llama.cpp 服务压测 | TTFT/TPOT/QPS 报告 |
| Week 4 | Reasoning budget + self-consistency | 质量-延迟-成本总结 |

每周只改一类变量。比如 Week 2 只改量化和上下文长度，不同时换模型、换框架、换题目。这样数据才有解释力。

<a id="reading-guide"></a>

## 读法建议

第一轮不要逐字读 DeepSeek 系列论文。建议按这条线：

1. **先读 V2**：因为 MLA 是推理优化主线的核心入口。
2. **再读 MoE + V3**：MoE 解释参数/成本，V3 把架构和训练工程接起来。
3. **然后读 Math + R1**：从 GRPO 过渡到推理型后训练。
4. **最后读 serving 论文**：FlashAttention、PagedAttention、Speculative Decoding，把模型论文落到部署系统。
5. **前沿版本只做观察**：R1-0528、V3.1、V3.2、V4 Preview 先看官方发布页和模型卡，等技术报告稳定后再精读。

一个好用的判断标准：每读到一个新名词，都把它归到下面五层之一。

| 层级 | 例子 | 主要优化 |
|------|------|----------|
| 模型结构 | MoE、MLA、MTP、DSA | 计算量、KV Cache、能力上限 |
| 训练/后训练 | SFT、GRPO、RLVR、distillation | 推理能力、输出稳定性 |
| 解码算法 | sampling、beam、speculative decoding、self-consistency | 延迟、质量、成本 |
| 服务调度 | PagedAttention、continuous batching、prefix cache | 吞吐、显存利用率 |
| 硬件 kernel | FlashAttention、FP8、TensorRT-LLM | 实际 wall-clock 性能 |

## 完成后你将理解

- DeepSeek-V2/V3 为什么强调 MLA、MoE、MTP，而不是只堆参数。
- R1 为什么能把 RL 用到 reasoning 上，以及 R1-Zero 和 R1 的差别。
- KV Cache 为什么是长上下文推理的核心成本之一。
- vLLM/SGLang/TensorRT-LLM 分别在服务链路的哪一层发力。
- 「更会推理」和「推理更快」是两件不同但会互相牵制的事。
- 面对一个新模型或新框架时，能用 latency / throughput / memory / quality / cost 五个维度判断它到底值不值得学。
