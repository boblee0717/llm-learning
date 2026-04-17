# 必读论文

按阅读顺序排列，建议配合第二阶段的代码课程一起学习。

## 阅读顺序

### 1. Attention Is All You Need (2017)
- **文件**: `Attention_Is_All_You_Need_2017.pdf`
- **作者**: Vaswani et al. (Google)
- **重点章节**: Section 3 (Model Architecture), Figure 1, Figure 2
- **配合课程**: 第 2-4 课
- **一句话**: 提出 Transformer 架构，用纯注意力机制取代 RNN

### 2. BERT (2018)
- **文件**: `BERT_2018.pdf`
- **作者**: Devlin et al. (Google)
- **重点章节**: Section 3 (Pre-training BERT), Figure 1
- **一句话**: 双向 Transformer Encoder，开创预训练+微调范式

### 3. GPT-2 (2019)
- **文件**: `GPT2_Language_Models_are_Unsupervised_Multitask_Learners_2019.pdf`
- **作者**: Radford et al. (OpenAI)
- **重点章节**: Section 2 (Approach), Table 1
- **配合课程**: 第 5 课
- **一句话**: 纯 Decoder 的语言模型，证明无监督预训练的强大

### 4. GPT-3 (2020)
- **文件**: `GPT3_Language_Models_are_Few_Shot_Learners_2020.pdf`
- **作者**: Brown et al. (OpenAI)
- **重点章节**: Section 1 (Introduction), Figure 1.1, Section 3
- **一句话**: 175B 参数，展示 In-context Learning 和 Scaling Law

### 5. InstructGPT (2022)
- **文件**: `InstructGPT_Training_LMs_to_Follow_Instructions_2022.pdf`
- **作者**: Ouyang et al. (OpenAI)
- **重点章节**: Section 3 (Methods), Figure 2
- **一句话**: RLHF 的落地实践，让模型学会遵循人类指令

### 6. Harness engineering: leveraging Codex in an agent-first world (OpenAI, 2026)
- **链接**: [openai.com/index/harness-engineering](https://openai.com/index/harness-engineering/)
- **作者**: Ryan Lopopolo（OpenAI）
- **一句话**: 在「以 agent 为先」的团队里，人负责设计环境、表达意图与搭建反馈回路，让 Codex 类 agent 可靠地产出代码与配套资产（测试、CI、文档、可观测性等），而不是以手写代码为主业。

### 7. Harness Engineering Is Cybernetics（George Zhang, 2026）
- **链接（中文转载，原文讨论首发于 X [@odysseus0z](https://x.com/odysseus0z)）**: [微信公众号 · 海外独角兽](https://mp.weixin.qq.com/s/SVUybMZb6uh5OCR3ceoBVA)
- **作者**: George Zhang（OpenClaw 维护者）
- **一句话**: 把 Harness engineering 放进控制论视角：从瓦特调速器到 Kubernetes 控制器，再到在架构层用 LLM 闭合反馈回路——工程师从「拧阀门」转向「设计调速器与约束」。

## 阅读技巧

- **不要试图一次读完** —— 每篇花 1-2 小时，分多次读
- **先读 Abstract + Introduction + Conclusion** —— 建立全局印象
- **重点看图和表** —— 一图胜千言
- **跳过不懂的数学** —— 先建立直觉，后面再补
- **和代码对照** —— 论文里的公式对应代码里的哪一行？
