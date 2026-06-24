# 必读论文

按阅读顺序排列，建议配合第二阶段的代码课程一起学习。

## 目录分类

- `core-transformers/`: Transformer、GPT、BERT、InstructGPT 等主线论文
- `attention-extensions/`: 位置编码、Self-Attention 表达能力、线性注意力等延伸论文
- `efficient-transformers/`: 高效 Transformer 与长上下文 benchmark / survey
- `kv-cache/`: KV Cache 推理优化（MQA / GQA / PagedAttention / FlashAttention）
- `distributed-training/`: 分布式训练与显存切分（ZeRO / FSDP 等多卡训练）
- `scaling-laws/`: 规模定律与 compute-optimal 训练论文
- `vision-transformers/`: Vision Transformer 方向论文
- `deepseek/`: DeepSeek MoE、Coder、Math、V2/V3/R1 系列论文
- `frontier-llms/`: Llama 3、Qwen2.5 等现代开源 LLM 技术报告
- `efficient-training/`: QLoRA、DPO 等训练 / 微调 / 对齐论文
- `agents/`: 工具调用型 LLM agent、上下文工程 / 记忆等方向论文
- `retrieval-augmented/`: 检索增强（RAG / REALM / RETRO）—— 推理时外接知识库的方向
- `foundations/`: 跨学科「思想源头」论文（如涌现 / emergence）
- `notes/`: 论文精读笔记

## 现代前沿补充

如果你已经读完 GPT-3 / InstructGPT / Chinchilla，建议接着看：

- [frontier-ai-2024-2025.md](frontier-ai-2024-2025.md): DeepSeek 与新一代开源 LLM 的 10 篇进阶论文，附训练 / 视频材料

## 阅读顺序

### 1. Attention Is All You Need (2017)
- **文件**: [Attention_Is_All_You_Need_2017.pdf](core-transformers/Attention_Is_All_You_Need_2017.pdf)
- **配套笔记**: [attention_is_all_you_need_reading_3.4_3.5.md](notes/attention_is_all_you_need_reading_3.4_3.5.md)
- **配套笔记**: [attention_is_all_you_need_reading_3.1_3.2.2.md](notes/attention_is_all_you_need_reading_3.1_3.2.2.md)
- **配套精读**: [notes_attention_qkv.md](notes/notes_attention_qkv.md)
- **作者**: Vaswani et al. (Google)
- **重点章节**: Section 3 (Model Architecture), Figure 1, Figure 2
- **配合课程**: 第 2-4 课
- **一句话**: 提出 Transformer 架构，用纯注意力机制取代 RNN

#### 位置编码延伸阅读（未读）

- **Learning to Encode Position for Transformer with Continuous Dynamical Model** (2020)
  - **状态**: 未读
  - **文件**: [Learning_to_Encode_Position_for_Transformer_with_Continuous_Dynamical_Model_2020.pdf](attention-extensions/Learning_to_Encode_Position_for_Transformer_with_Continuous_Dynamical_Model_2020.pdf)
  - **来源**: [arxiv.org/abs/2003.09229](https://arxiv.org/abs/2003.09229)
  - **作者**: Xuanqing Liu, Hsiang-Fu Yu, Inderjit Dhillon, Cho-Jui Hsieh
  - **配合课程**: 第 1 课（位置编码）延伸阅读
  - **一句话**: 用连续动力系统 / Neural ODE 的视角学习可外推的位置编码，对比正弦位置编码、可学习位置 embedding 与 RoPE 等方法的取舍。

#### Self-Attention 延伸阅读（未读）

- **On the Relationship between Self-Attention and Convolutional Layers** (2020)
  - **状态**: 未读
  - **文件**: [On_the_Relationship_between_Self_Attention_and_Convolutional_Layers_2020.pdf](attention-extensions/On_the_Relationship_between_Self_Attention_and_Convolutional_Layers_2020.pdf)
  - **来源**: [arxiv.org/abs/1911.03584](https://arxiv.org/abs/1911.03584)
  - **作者**: Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi
  - **配合课程**: 第 2-3 课（Self-Attention / Multi-Head Attention）延伸阅读
  - **一句话**: 从表达能力和视觉实验角度解释多头自注意力与卷积层的关系，说明 self-attention 在足够 head 下可以模拟卷积，并常会学到类似像素网格的注意力模式。

- **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** (2020)
  - **状态**: 未读
  - **文件**: [Transformers_are_RNNs_Fast_Autoregressive_Transformers_with_Linear_Attention_2020.pdf](attention-extensions/Transformers_are_RNNs_Fast_Autoregressive_Transformers_with_Linear_Attention_2020.pdf)
  - **来源**: [arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236)
  - **作者**: Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, Francois Fleuret
  - **配合课程**: 第 2-3 课（Self-Attention / Multi-Head Attention）线性注意力延伸阅读
  - **一句话**: 将自注意力写成 kernel feature map 的线性点积形式，用矩阵乘法结合律把复杂度从 O(N^2) 降到 O(N)，并展示自回归 Transformer 与 RNN 的联系。

#### Efficient Transformer / Long Context 延伸阅读（未读）

- **Generating Long Sequences with Sparse Transformers** (2019)
  - **状态**: 未读
  - **文件**: [Generating_Long_Sequences_with_Sparse_Transformers_2019.pdf](efficient-transformers/Generating_Long_Sequences_with_Sparse_Transformers_2019.pdf)
  - **来源**: [arxiv.org/abs/1904.10509](https://arxiv.org/abs/1904.10509)、[OpenAI blog](https://openai.com/index/sparse-transformer)
  - **作者**: Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever
  - **配合课程**: 第 2-3 课（Sparse Attention / Long Context）延伸阅读；也可配合 GPT-3 `2.1 Model and Architectures` 理解 `locally banded sparse attention`
  - **一句话**: GPT-3 §2.1 引用的 Sparse Transformer 原始论文，用稀疏注意力模式把标准 attention 的二次复杂度降下来，使模型能处理更长序列。

- **Long Range Arena: A Benchmark for Efficient Transformers** (2020)
  - **状态**: 未读
  - **文件**: [Long_Range_Arena_A_Benchmark_for_Efficient_Transformers_2020.pdf](efficient-transformers/Long_Range_Arena_A_Benchmark_for_Efficient_Transformers_2020.pdf)
  - **来源**: [arxiv.org/abs/2011.04006](https://arxiv.org/abs/2011.04006)
  - **作者**: Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler
  - **配合课程**: 第 2-3 课（Self-Attention / Multi-Head Attention）高效长序列 Transformer 延伸阅读
  - **一句话**: 提出 Long Range Arena (LRA) benchmark，用 1K-16K token 的多模态长序列任务系统比较 Reformer、Linformer、Performer、Longformer 等高效 Transformer。

- **Efficient Transformers: A Survey** (2022 edition)
  - **状态**: 未读
  - **文件**: [Efficient_Transformers_A_Survey_2022.pdf](efficient-transformers/Efficient_Transformers_A_Survey_2022.pdf)
  - **来源**: [arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)
  - **作者**: Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler
  - **配合课程**: 第 2-3 课（Self-Attention / Multi-Head Attention）高效 Transformer 综述
  - **一句话**: 系统梳理 Reformer、Linformer、Performer、Longformer 等 X-former 家族，按计算和内存效率改进路线组织高效 Transformer 的整体地图。

#### Vision Transformer 延伸阅读（未读）

- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (2020)
  - **状态**: 未读
  - **文件**: [An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_2020.pdf](vision-transformers/An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_2020.pdf)
  - **来源**: [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
  - **作者**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
  - **配合课程**: 第 2-3 课（Self-Attention / Multi-Head Attention）视觉方向延伸阅读
  - **一句话**: Vision Transformer (ViT) 把图像切成 patch 序列，直接用纯 Transformer 做图像分类，展示大规模预训练后可与卷积网络竞争。

### 2. GPT-1 (2018)
- **文件**: [GPT1_2018_improving_language_understanding.pdf](core-transformers/GPT1_2018_improving_language_understanding.pdf)
- **来源**: [OpenAI 官方 PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- **作者**: Radford et al. (OpenAI)
- **重点章节**: Section 3 (Framework), Section 4 (Experiments)
- **配合课程**: 第 5 课（与从零搭建 GPT 对照阅读）
- **一句话**: 生成式预训练 + 判别式微调，用 Transformer Decoder 学通用表示并迁移到下游任务

### 3. BERT (2018)
- **文件**: [BERT_2018.pdf](core-transformers/BERT_2018.pdf)
- **配套笔记**: [bert_reading_model_architecture.md](notes/bert_reading_model_architecture.md)
- **作者**: Devlin et al. (Google)
- **重点章节**: Section 3 (Pre-training BERT), Figure 1
- **配合课程**: 第 3 课（对照 encoder 结构、多头注意力、残差与 LayerNorm）
- **一句话**: 双向 Transformer Encoder，开创预训练+微调范式

### 4. GPT-2 (2019)
- **文件**: [GPT2_Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf](core-transformers/GPT2_Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf)
- **配套笔记**: [notes_gpt2_input_and_model.md](notes/notes_gpt2_input_and_model.md)
- **作者**: Radford et al. (OpenAI)
- **重点章节**: Section 2 (Approach), Table 1
- **配合课程**: 第 5 课
- **一句话**: 纯 Decoder 的语言模型，证明无监督预训练的强大

### 5. GPT-3 (2020)
- **文件**: [GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf](core-transformers/GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf)
- **作者**: Brown et al. (OpenAI)
- **重点章节**: Section 1 (Introduction), Figure 1.1, Section 2.1 (Model and Architectures), Section 3
- **一句话**: 175B 参数，展示 In-context Learning 和 Scaling Law

#### Scaling Law 阅读路径（已升级为第二阶段第 6 课正文）

> 配套课程：[`phase2-transformer/06_scaling_laws.py`](../phase2-transformer/06_scaling_laws.py) 与第二阶段 README 的 [第 6 课大纲](../phase2-transformer/README.md#lesson-6)。读这两篇论文等于读完第 6 课的「② 读论文」环节。

- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
  - **状态**: 配第二阶段第 6 课（已下载，待精读）
  - **文件**: [Scaling_Laws_for_Neural_Language_Models_2020.pdf](scaling-laws/Scaling_Laws_for_Neural_Language_Models_2020.pdf)
  - **来源**: [OpenAI 论文页](https://openai.com/research/scaling-laws-for-neural-language-models)、[arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
  - **作者**: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
  - **配套笔记**: [scaling_laws_kaplan_2020.md](notes/scaling_laws_kaplan_2020.md)
  - **建议读法**: 先读 Abstract / Introduction，再看 §3.1 三条幂律曲线（Figure 1）。重点理解 `N` 不含 embedding、`C` 以 PF-days 为单位、loss 以 nats/token 为单位。
  - **一句话**: GPT-3 背后的直接理论背景，说明语言模型 loss 会随参数量、数据量和训练 compute 呈近似幂律下降；论文最后给出「优先扩 N」的 compute-optimal 结论（后被 Chinchilla 修正）。

- **Training Compute-Optimal Large Language Models** (Chinchilla, 2022)
  - **状态**: 配第二阶段第 6 课（已下载，待精读）
  - **文件**: [Training_Compute_Optimal_Large_Language_Models_Chinchilla_2022.pdf](scaling-laws/Training_Compute_Optimal_Large_Language_Models_Chinchilla_2022.pdf)
  - **来源**: [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)、[DeepMind 官方解读](https://deepmind.google/discover/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)
  - **作者**: Jordan Hoffmann et al. (DeepMind)
  - **配套笔记**: [chinchilla_compute_optimal_2022.md](notes/chinchilla_compute_optimal_2022.md)（含「Chinchilla 之后」：Llama 3 over-training、DeepSeek-V3、推理成本视角）
  - **建议读法**: 重点看 Abstract、§1、§3 三种 approach 与 Table 3、§4.1 Chinchilla vs Gopher。
  - **一句话**: 对 Kaplan scaling 路线的关键修正：固定训练 compute 时，N 和 D 应该按 ≈ 1:20 同步扩；Chinchilla 70B + 1.4T tokens 用相同 compute 击败 Gopher 280B + 300B tokens。

- **An Empirical Model of Large-Batch Training** (McCandlish et al., 2018)
  - **状态**: 配第二阶段第 6 课 + 第三阶段第 1 课（已下载，待精读）
  - **文件**: [An_Empirical_Model_of_Large_Batch_Training_2018.pdf](scaling-laws/An_Empirical_Model_of_Large_Batch_Training_2018.pdf)
  - **来源**: [arxiv.org/abs/1812.06162](https://arxiv.org/abs/1812.06162)、[OpenAI blog](https://openai.com/index/how-ai-training-scales/)
  - **作者**: Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team
  - **配套笔记**: [large_batch_training_mccandlish_2018.md](notes/large_batch_training_mccandlish_2018.md)
  - **建议读法**: 抓两个概念即可——critical batch size `B_crit`（步数 vs 样本数双曲线的拐点）与 gradient noise scale `B_noise`（可测、能预测 `B_crit`）。配合第三阶段第 1 课 Part 5 梯度累积一起看。
  - **一句话**: Kaplan scaling laws §5.3 critical batch size 的源头论文。同一作者班底先把「batch size 该多大」这条轴量化清楚：batch 增大能省步数但收益递减，拐点 `B_crit ≈ B_noise`，且 `B_noise` 随训练推进 / 任务难度增大——正是 GPT-3 把 batch 从 0.5M warmup 到 3.2M tokens 的理论依据。

#### Scaling Law 的思想源头（跨学科）

> 不是深度学习论文，而是「涌现 / emergence」概念的源头。读完 Kaplan 的「平滑量变」后看它，理解「规模累积可能引发质变」这条互补线索。

- **More Is Different** (Anderson, 1972)
  - **状态**: 选读（思想背景，已下载）
  - **文件**: [More_Is_Different_Anderson_1972.pdf](foundations/More_Is_Different_Anderson_1972.pdf)
  - **来源**: *Science*, Vol. 177, No. 4047, pp. 393–396（1972 年 8 月 4 日）；副标题 *Broken symmetry and the nature of the hierarchical structure of science*
  - **作者**: Philip W. Anderson（贝尔实验室 / 剑桥卡文迪许，1977 年诺贝尔物理学奖得主）
  - **配套笔记**: [more_is_different_anderson_1972.md](notes/more_is_different_anderson_1972.md)
  - **建议读法**: 正文不到 4 页，抓两点即可——① 还原论（接受）≠ 构成论（反对）；② 每上升一个复杂度层级会涌现新规律，「整体非常不同于部分之和」。读完对照笔记里「量变（Kaplan）↔ 质变（Anderson）」一节。
  - **一句话**: 「涌现（emergence）」的思想源头。Anderson 反对极端还原论：即便知道全部微观定律，也无法据此重建宇宙——大量粒子聚集会在每个复杂度层级涌现出全新性质。这正是 Kaplan 等人借用的精神：LLM 的 loss 是平滑量变，但规模累积到一定程度可能涌现出全新能力。

### 6. InstructGPT (2022)
- **文件**: [InstructGPT_Training_LMs_to_Follow_Instructions_2022.pdf](core-transformers/InstructGPT_Training_LMs_to_Follow_Instructions_2022.pdf)
- **作者**: Ouyang et al. (OpenAI)
- **重点章节**: Section 3 (Methods), Figure 2
- **一句话**: RLHF 的落地实践，让模型学会遵循人类指令

### 7. 现代前沿论文：DeepSeek 与新一代开源 LLM (2023-2025)
- **清单**: [frontier-ai-2024-2025.md](frontier-ai-2024-2025.md)
- **包含**: DeepSeekMoE、DeepSeek-Coder、DeepSeekMath、DeepSeek-V2、DeepSeek-V3、DeepSeek-R1、Llama 3、Qwen2.5、QLoRA、DPO
- **配合课程**: 第三阶段训练与微调，尤其是 LoRA、量化、RLHF/DPO、推理优化
- **一句话**: 从“读懂 Transformer”过渡到“读懂现代开源大模型是怎么训练、对齐、压缩和高效推理的”。

### 8. Harness engineering: leveraging Codex in an agent-first world (OpenAI, 2026)
- **链接**: [openai.com/index/harness-engineering](https://openai.com/index/harness-engineering/)
- **作者**: Ryan Lopopolo（OpenAI）
- **一句话**: 在「以 agent 为先」的团队里，人负责设计环境、表达意图与搭建反馈回路，让 Codex 类 agent 可靠地产出代码与配套资产（测试、CI、文档、可观测性等），而不是以手写代码为主业。

### 9. Harness Engineering Is Cybernetics（George Zhang, 2026）
- **链接（X 原文长帖）**: [x.com/odysseus0z/status/2030416758138634583](https://x.com/odysseus0z/status/2030416758138634583)
- **作者**: George Zhang（OpenClaw 维护者）
- **一句话**: 把 Harness engineering 放进控制论视角：从瓦特调速器到 Kubernetes 控制器，再到在架构层用 LLM 闭合反馈回路——工程师从「拧阀门」转向「设计调速器与约束」。

#### Harness 延伸阅读（非论文，机制可解释性长文）

> 不是必读论文，但与 Harness #7/#8 的「反馈如何塑造模型行为」主题强相关。李宏毅 Harness Engineering 视频中曾引用，建议读完 #7/#8 后选读。

- **Emotion Concepts and their Function in a Large Language Model**（Anthropic，*Transformer Circuits Thread*，2024）
  - **链接**: [transformer-circuits.pub/2024/emotions](https://transformer-circuits.pub/2024/emotions/index.html)
  - **配合笔记**: [harness-engineering-li-hongyi.md](../harness-engineering-li-hongyi.md)
  - **一句话**: 在电路/特征视角下讨论 LLM 中的「情绪」概念如何形成与起作用，并与交互方式（含对 Agent 的责备方式）对模型行为的影响相关联——可与「Harness 与反馈设计」对照阅读。

### 10. Decision-Aware Memory Cards: CICL（2026）
- **状态**: 未读
- **文件**: [Decision-Aware_Memory_Cards_CICL_2026.pdf](agents/Decision-Aware_Memory_Cards_CICL_2026.pdf)
- **来源**: [arxiv.org/abs/2606.08151](https://arxiv.org/abs/2606.08151)、[代码 GitHub](https://github.com/stephen-guan-researcher/CICL)
- **作者**: Xinyu Guan, Qianyang Zhao, Yuming Deng
- **配合课程**: agent 架构 / 上下文工程方向，可与 Harness #7/#8 对照阅读
- **一句话**: 工具调用型 agent 真正需要的不是更长上下文，而是「行动时刻的决策相关证据」；提出 Counterfactual-Inspired Context Layer (CICL)，构建实例上下文图、按「对下一步动作的预期影响」而非语义相似度给候选证据打分排序，再压缩成带类型的 memory cards——在 SWE-bench Verified 上把检索 hit@1 从 0.58 提到 0.78，并在压缩模式下每条 query 平均省下约 45 个 token。

### 11. KV Cache 推理优化（2019-2023）

> 主题：自回归推理时，把历史 token 的 Key / Value 缓存下来避免重复计算，是 LLM 推理提速的核心。这条线索从「砍注意力头共享 KV」一路压到「低维潜向量」，再到「系统层显存管理」。
>
> 建议读法：先理解标准 KV Cache 的动机（用显存换算力，单步注意力从 O(n²) 降到 O(n)），再按 MQA → GQA → PagedAttention → FlashAttention 顺序读，最后回到 [`deepseek/DeepSeek-V2`](deepseek/DeepSeek-V2_Strong_Economical_Efficient_MoE_2024.pdf) 的 MLA，串成「KV Cache 怎么越压越小」的主线。
>
> 配合课程：第三阶段「推理优化」；与 [`attention-extensions/Transformers are RNNs`](attention-extensions/Transformers_are_RNNs_Fast_Autoregressive_Transformers_with_Linear_Attention_2020.pdf)（线性注意力 = 用固定大小状态替代无限增长的 KV Cache）对照阅读。

- **Fast Transformer Decoding: One Write-Head is All You Need**（MQA, 2019）
  - **状态**: 未读
  - **文件**: [Fast_Transformer_Decoding_MQA_2019.pdf](kv-cache/Fast_Transformer_Decoding_MQA_2019.pdf)
  - **来源**: [arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)
  - **作者**: Noam Shazeer (Google)
  - **一句话**: 提出 Multi-Query Attention (MQA)，所有注意力头共享同一份 K、V，大幅缩小 KV Cache，是 KV Cache 压缩的鼻祖级方案。

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**（GQA, 2023）
  - **状态**: 未读
  - **文件**: [GQA_Training_Generalized_Multi_Query_Transformer_2023.pdf](kv-cache/GQA_Training_Generalized_Multi_Query_Transformer_2023.pdf)
  - **来源**: [arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)
  - **作者**: Joshua Ainslie, James Lee-Thorp, Santiago Ontañón, et al. (Google)
  - **一句话**: MHA 与 MQA 的折中，把注意力头分组、每组共享一份 K、V，在质量和 KV Cache 大小间取平衡；Llama 2/3 等主流模型采用。

- **Efficient Memory Management for Large Language Model Serving with PagedAttention**（vLLM, 2023）
  - **状态**: 未读
  - **文件**: [PagedAttention_vLLM_Efficient_Memory_Management_2023.pdf](kv-cache/PagedAttention_vLLM_Efficient_Memory_Management_2023.pdf)
  - **来源**: [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
  - **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. (UC Berkeley)
  - **一句话**: 从系统视角优化 KV Cache：像操作系统分页一样分块管理显存，消除碎片、支持共享，大幅提升推理吞吐；vLLM 的核心论文，做推理服务必读。

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**（2022）
  - **状态**: 未读
  - **文件**: [FlashAttention_Fast_and_Memory_Efficient_Exact_Attention_2022.pdf](kv-cache/FlashAttention_Fast_and_Memory_Efficient_Exact_Attention_2022.pdf)
  - **来源**: [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
  - **作者**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford)
  - **一句话**: 严格说是优化注意力的 IO / 访存（用 tiling 在 SRAM 里算 attention，不落地巨大的注意力矩阵），但与 KV Cache、推理访存瓶颈强相关，常与上面三篇一起读。

### 12. 检索增强生成 / RAG（2020-2021）

> 主题：除了把知识「压」进模型参数，还能在预训练 / 生成时**外接一个知识库**，按需检索证据再生成。这条线解决「知识更新、可溯源、长尾事实」等参数化模型的痛点，是后来一切 RAG 系统的源头。
>
> 建议读法：按 REALM → RAG → RETRO 顺序，正好是「检索进预训练 → 检索进生成 → 检索 scale 到万亿 token」。先抓三个共性问题——检索什么（chunk / passage）、怎么检索（dense retriever / 近邻）、检索到的内容如何喂给模型（拼接 / cross-attention）。
>
> 配合课程：第三 / 第四阶段延伸阅读（推理服务 + 知识增强方向），与 DeepSeek 的「稀疏架构 + KV 压缩」并列，作为「参数化知识 vs 非参数化知识」的互补视角；也对应第五阶段第 5 课「RAG 与上下文工程」。
>
> 配套精读骨架：[retrieval_augmented_rag_notes.md](notes/retrieval_augmented_rag_notes.md)（三篇合一的待填笔记，带「检索什么 / 怎么检索 / 怎么喂给模型」三个共性问题与对照表）。

- **REALM: Retrieval-Augmented Language Model Pre-Training**（Guu et al., 2020）
  - **状态**: 未读
  - **文件**: [REALM_Retrieval_Augmented_Language_Model_PreTraining_2020.pdf](retrieval-augmented/REALM_Retrieval_Augmented_Language_Model_PreTraining_2020.pdf)
  - **来源**: [arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)
  - **作者**: Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang (Google)
  - **一句话**: 第一个把「检索器」端到端做进**预训练**的工作——用掩码语言建模的信号反向学习一个 neural knowledge retriever，让模型在预训练阶段就学会从知识库里取证据，再用证据预测被掩码的内容。

- **RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**（Lewis et al., 2020）
  - **状态**: 未读
  - **文件**: [RAG_Retrieval_Augmented_Generation_for_Knowledge_Intensive_NLP_2020.pdf](retrieval-augmented/RAG_Retrieval_Augmented_Generation_for_Knowledge_Intensive_NLP_2020.pdf)
  - **来源**: [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
  - **作者**: Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. (Facebook AI Research / UCL)
  - **一句话**: 「RAG」这个名字的出处。把 DPR 稠密检索器和 BART seq2seq 生成器组合，**生成时**取回 Top-K 文档当条件来生成答案，在开放域问答等知识密集型任务上刷新 SOTA，是后续所有 RAG 系统的范式之作。

- **RETRO: Improving Language Models by Retrieving from Trillions of Tokens**（Borgeaud et al., 2021）
  - **状态**: 未读
  - **文件**: [RETRO_Improving_Language_Models_by_Retrieving_from_Trillions_of_Tokens_2021.pdf](retrieval-augmented/RETRO_Improving_Language_Models_by_Retrieving_from_Trillions_of_Tokens_2021.pdf)
  - **来源**: [arxiv.org/abs/2112.04426](https://arxiv.org/abs/2112.04426)
  - **作者**: Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, et al. (DeepMind)
  - **一句话**: 把检索增强 scale 到万亿 token 级别的知识库，用 chunked cross-attention 把检索到的近邻 chunk 融进解码，让 7.5B 的 RETRO 在部分指标上逼近 25× 大的 GPT-3 / Jurassic-1——证明「检索」可以部分替代「堆参数」。

### 13. 分布式训练 / 显存切分（2020）

> 主题：单卡放不下大模型时，如何把训练状态沿多张卡切开。这条线索把第一阶段第 4 课算过的「每参数 ~16 字节显存账」从**单卡视角**推到**集群视角**——既然优化器状态（m+v）最吃显存，那就沿 N 张数据并行卡把它切成 1/N。
>
> 建议读法：定位成**概念精读 + 看图**，不必抠通信实现细节（all-gather / reduce-scatter 等）。重点看三段切分（ZeRO-1 优化器状态 → ZeRO-2 +梯度 → ZeRO-3 +参数）的示意图与显存对比表。读完回头看 `llm-interview-questions.md` 里的 ZeRO-1/2/3 对照表，把「背结论」变成「懂原理」。
>
> 配合课程：第三阶段「分布式训练」专题（待补）；与第一阶段 [第 4 课优化器显存账](../phase1-foundations/04_optimizers.py) 和 [pytorch-essentials 显存估算器](../pytorch-essentials/07_debug_profile_memory.py) 对照阅读。

- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**（Rajbhandari et al., 2020）
  - **状态**: 未读
  - **文件**: [ZeRO_Memory_Optimizations_Toward_Training_Trillion_Parameter_Models_2020.pdf](distributed-training/ZeRO_Memory_Optimizations_Toward_Training_Trillion_Parameter_Models_2020.pdf)
  - **来源**: [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)（发表于 SC20）
  - **作者**: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (Microsoft)
  - **建议读法**: 抓三点即可——① 训练显存都花在哪（参数 / 梯度 / 优化器状态 / 激活）；② ZeRO-1/2/3 分别切掉哪部分冗余、单卡省多少（÷N）；③ 它与传统数据并行（DP）/ 模型并行（TP/PP）的关系。PyTorch 的 FSDP 本质就是 ZeRO 的等价实现。
  - **一句话**: 微软 DeepSpeed 里 ZeRO 的奠基论文。核心思想是把数据并行下每张卡都冗余存一份的「优化器状态 / 梯度 / 参数」沿 N 张卡切分，让单卡只存 1/N，从而把万亿参数级模型训得起来——是理解现代大模型「怎么真的训出来」绕不开的入口。

## 阅读技巧

- **不要试图一次读完** —— 每篇花 1-2 小时，分多次读
- **先读 Abstract + Introduction + Conclusion** —— 建立全局印象
- **重点看图和表** —— 一图胜千言
- **跳过不懂的数学** —— 先建立直觉，后面再补
- **和代码对照** —— 论文里的公式对应代码里的哪一行？
