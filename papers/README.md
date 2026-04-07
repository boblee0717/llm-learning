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

## 阅读技巧

- **不要试图一次读完** —— 每篇花 1-2 小时，分多次读
- **先读 Abstract + Introduction + Conclusion** —— 建立全局印象
- **重点看图和表** —— 一图胜千言
- **跳过不懂的数学** —— 先建立直觉，后面再补
- **和代码对照** —— 论文里的公式对应代码里的哪一行？
