# 必读论文

按阅读顺序排列，建议配合第二阶段的代码课程一起学习。

## 阅读顺序

### 1. Attention Is All You Need (2017)
- **文件**: `Attention_Is_All_You_Need_2017.pdf`
- **作者**: Vaswani et al. (Google)
- **重点章节**: Section 3 (Model Architecture), Figure 1, Figure 2
- **配合课程**: 第 2-4 课
- **一句话**: 提出 Transformer 架构，用纯注意力机制取代 RNN

#### 位置编码延伸阅读（未读）

- **Learning to Encode Position for Transformer with Continuous Dynamical Model** (2020)
  - **状态**: 未读
  - **文件**: `Learning_to_Encode_Position_for_Transformer_with_Continuous_Dynamical_Model_2020.pdf`
  - **链接**: [arxiv.org/abs/2003.09229](https://arxiv.org/abs/2003.09229)
  - **作者**: Xuanqing Liu, Hsiang-Fu Yu, Inderjit Dhillon, Cho-Jui Hsieh
  - **配合课程**: 第 1 课（位置编码）延伸阅读
  - **一句话**: 用连续动力系统 / Neural ODE 的视角学习可外推的位置编码，对比正弦位置编码、可学习位置 embedding 与 RoPE 等方法的取舍。

#### Self-Attention 延伸阅读（未读）

- **On the Relationship between Self-Attention and Convolutional Layers** (2020)
  - **状态**: 未读
  - **文件**: `On_the_Relationship_between_Self_Attention_and_Convolutional_Layers_2020.pdf`
  - **链接**: [arxiv.org/abs/1911.03584](https://arxiv.org/abs/1911.03584)
  - **作者**: Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi
  - **配合课程**: 第 2-3 课（Self-Attention / Multi-Head Attention）延伸阅读
  - **一句话**: 从表达能力和视觉实验角度解释多头自注意力与卷积层的关系，说明 self-attention 在足够 head 下可以模拟卷积，并常会学到类似像素网格的注意力模式。

- **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** (2020)
  - **状态**: 未读
  - **文件**: `Transformers_are_RNNs_Fast_Autoregressive_Transformers_with_Linear_Attention_2020.pdf`
  - **链接**: [arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236)
  - **作者**: Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, Francois Fleuret
  - **配合课程**: 第 2-3 课（Self-Attention / Multi-Head Attention）线性注意力延伸阅读
  - **一句话**: 将自注意力写成 kernel feature map 的线性点积形式，用矩阵乘法结合律把复杂度从 O(N^2) 降到 O(N)，并展示自回归 Transformer 与 RNN 的联系。

#### Vision Transformer 延伸阅读（未读）

- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (2020)
  - **状态**: 未读
  - **文件**: `An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_2020.pdf`
  - **链接**: [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
  - **作者**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
  - **配合课程**: 第 2-3 课（Self-Attention / Multi-Head Attention）视觉方向延伸阅读
  - **一句话**: Vision Transformer (ViT) 把图像切成 patch 序列，直接用纯 Transformer 做图像分类，展示大规模预训练后可与卷积网络竞争。

### 2. GPT-1 (2018)
- **文件**: `GPT1_2018_improving_language_understanding.pdf`
- **官方 PDF**: [https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- **作者**: Radford et al. (OpenAI)
- **重点章节**: Section 3 (Framework), Section 4 (Experiments)
- **配合课程**: 第 5 课（与从零搭建 GPT 对照阅读）
- **一句话**: 生成式预训练 + 判别式微调，用 Transformer Decoder 学通用表示并迁移到下游任务

### 3. BERT (2018)
- **文件**: `BERT_2018.pdf`
- **作者**: Devlin et al. (Google)
- **重点章节**: Section 3 (Pre-training BERT), Figure 1
- **一句话**: 双向 Transformer Encoder，开创预训练+微调范式

### 4. GPT-2 (2019)
- **文件**: `GPT2_Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf`
- **作者**: Radford et al. (OpenAI)
- **重点章节**: Section 2 (Approach), Table 1
- **配合课程**: 第 5 课
- **一句话**: 纯 Decoder 的语言模型，证明无监督预训练的强大

### 5. GPT-3 (2020)
- **文件**: `GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf`
- **作者**: Brown et al. (OpenAI)
- **重点章节**: Section 1 (Introduction), Figure 1.1, Section 3
- **一句话**: 175B 参数，展示 In-context Learning 和 Scaling Law

### 6. InstructGPT (2022)
- **文件**: `InstructGPT_Training_LMs_to_Follow_Instructions_2022.pdf`
- **作者**: Ouyang et al. (OpenAI)
- **重点章节**: Section 3 (Methods), Figure 2
- **一句话**: RLHF 的落地实践，让模型学会遵循人类指令

### 7. Harness engineering: leveraging Codex in an agent-first world (OpenAI, 2026)
- **链接**: [openai.com/index/harness-engineering](https://openai.com/index/harness-engineering/)
- **作者**: Ryan Lopopolo（OpenAI）
- **一句话**: 在「以 agent 为先」的团队里，人负责设计环境、表达意图与搭建反馈回路，让 Codex 类 agent 可靠地产出代码与配套资产（测试、CI、文档、可观测性等），而不是以手写代码为主业。

### 8. Harness Engineering Is Cybernetics（George Zhang, 2026）
- **链接（X 原文长帖）**: [x.com/odysseus0z/status/2030416758138634583](https://x.com/odysseus0z/status/2030416758138634583)
- **作者**: George Zhang（OpenClaw 维护者）
- **一句话**: 把 Harness engineering 放进控制论视角：从瓦特调速器到 Kubernetes 控制器，再到在架构层用 LLM 闭合反馈回路——工程师从「拧阀门」转向「设计调速器与约束」。

#### Harness 延伸阅读（非论文，机制可解释性长文）

> 不是必读论文，但与 Harness #7/#8 的「反馈如何塑造模型行为」主题强相关。李宏毅 Harness Engineering 视频中曾引用，建议读完 #7/#8 后选读。

- **Emotion Concepts and their Function in a Large Language Model**（Anthropic，*Transformer Circuits Thread*，2024）
  - **链接**: [transformer-circuits.pub/2024/emotions](https://transformer-circuits.pub/2024/emotions/index.html)
  - **配合笔记**: [harness-engineering-li-hongyi.md](../harness-engineering-li-hongyi.md)
  - **一句话**: 在电路/特征视角下讨论 LLM 中的「情绪」概念如何形成与起作用，并与交互方式（含对 Agent 的责备方式）对模型行为的影响相关联——可与「Harness 与反馈设计」对照阅读。

## 阅读技巧

- **不要试图一次读完** —— 每篇花 1-2 小时，分多次读
- **先读 Abstract + Introduction + Conclusion** —— 建立全局印象
- **重点看图和表** —— 一图胜千言
- **跳过不懂的数学** —— 先建立直觉，后面再补
- **和代码对照** —— 论文里的公式对应代码里的哪一行？
