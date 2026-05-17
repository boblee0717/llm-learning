# 现代前沿论文：DeepSeek 与新一代开源 LLM

> 这份清单接在 Transformer / GPT-3 / InstructGPT / Chinchilla 之后读。目标不是追热点，而是把 2024-2025 年大模型的几个关键变化接到本项目现有主线：MoE 架构、MLA/KV Cache 压缩、代码与数学数据、后训练、推理型 RL、LoRA/QLoRA 与 DPO。

## 推荐阅读顺序

| 顺序 | 论文 | 主题 | 配合课程 |
|------|------|------|----------|
| 1 | DeepSeekMoE | MoE 专家拆分与共享专家 | phase2 第 4-5 课 |
| 2 | DeepSeek-V2 | MLA + DeepSeekMoE 的系统化模型 | phase2 第 5 课、phase3 第 5 课 |
| 3 | DeepSeek-V3 | 大规模 MoE、FP8、MTP、训练工程 | phase3 第 1/5 课 |
| 4 | DeepSeek-Coder | 代码模型的数据与训练路线 | phase3 第 1 课 |
| 5 | DeepSeekMath | 数学继续预训练与 GRPO | phase3 第 4 课 |
| 6 | DeepSeek-R1 | 推理能力与强化学习后训练 | phase3 第 4 课 |
| 7 | Llama 3 | 现代开源 dense LLM 全流程报告 | phase3 全阶段 |
| 8 | Qwen2.5 | 强开源基座、长上下文与多尺寸家族 | phase3 全阶段 |
| 9 | QLoRA | 4-bit 量化微调 | phase3 第 2-3 课 |
| 10 | DPO | 偏好对齐的简化目标 | phase3 第 4 课 |

## 1. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (2024)

- **状态**: 未读
- **文件**: [DeepSeekMoE_Towards_Ultimate_Expert_Specialization_2024.pdf](deepseek/DeepSeekMoE_Towards_Ultimate_Expert_Specialization_2024.pdf)
- **来源**: [arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)、[GitHub](https://github.com/deepseek-ai/DeepSeek-MoE)
- **建议读法**: 先看 Abstract / Introduction，再看 shared experts 与 routed experts 的设计动机。
- **一句话**: DeepSeek 系列 MoE 路线的底层积木：用更细粒度专家和共享专家减少冗余，让“总参数很大、每 token 激活很少”变得更有效。

## 2. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2024)

- **状态**: 未读
- **文件**: [DeepSeek-V2_Strong_Economical_Efficient_MoE_2024.pdf](deepseek/DeepSeek-V2_Strong_Economical_Efficient_MoE_2024.pdf)
- **来源**: [arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)、[GitHub](https://github.com/deepseek-ai/DeepSeek-V2)
- **建议读法**: 重点读 MLA、DeepSeekMoE 与推理效率相关章节。
- **一句话**: 把 MoE 与 Multi-head Latent Attention 组合起来，解释 DeepSeek 为什么能在训练和推理成本上做得更“经济”。

## 3. DeepSeek-V3 Technical Report (2024)

- **状态**: 未读
- **文件**: [DeepSeek-V3_Technical_Report_2024.pdf](deepseek/DeepSeek-V3_Technical_Report_2024.pdf)
- **来源**: [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)、[GitHub](https://github.com/deepseek-ai/DeepSeek-V3)、[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- **建议读法**: 先读模型架构总览，再看 FP8 训练、Multi-Token Prediction、训练稳定性与成本描述。
- **一句话**: DeepSeek-V3 是现代大规模 MoE 工程报告，适合把 phase3 的训练 pipeline、混合精度、推理优化串起来读。

## 4. DeepSeek-Coder: When the Large Language Model Meets Programming (2024)

- **状态**: 未读
- **文件**: [DeepSeek-Coder_When_the_Large_Language_Model_Meets_Programming_2024.pdf](deepseek/DeepSeek-Coder_When_the_Large_Language_Model_Meets_Programming_2024.pdf)
- **来源**: [arxiv.org/abs/2401.14196](https://arxiv.org/abs/2401.14196)、[GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)
- **建议读法**: 重点看数据构建、预训练 token 配比、代码 benchmark。
- **一句话**: 代码模型不只是“多喂代码”，还涉及数据清洗、repo 级上下文、训练混合比例和评测设计。

## 5. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (2024)

- **状态**: 未读
- **文件**: [DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_2024.pdf](deepseek/DeepSeekMath_Pushing_the_Limits_of_Mathematical_Reasoning_2024.pdf)
- **来源**: [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)、[Hugging Face](https://huggingface.co/deepseek-ai/deepseek-math-7b-base)
- **建议读法**: 先看 math 数据构建，再看 RL / GRPO 相关章节。
- **一句话**: 这是理解 DeepSeek-R1 前的一块垫脚石：数学能力来自继续预训练、SFT 与强化学习组合，而不是单一技巧。

## 6. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025)

- **状态**: 未读
- **文件**: [DeepSeek-R1_Incentivizing_Reasoning_Capability_2025.pdf](deepseek/DeepSeek-R1_Incentivizing_Reasoning_Capability_2025.pdf)
- **来源**: [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)、[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1)、[Open-R1 复现项目](https://github.com/huggingface/open-r1)
- **建议读法**: 重点看 R1-Zero、cold-start data、多阶段训练和蒸馏模型。
- **一句话**: 把 phase3 的 RLHF/DPO 扩展到“推理型后训练”：模型不只学会偏好，还学会在数学、代码、逻辑任务上花更多测试时计算。

## 7. The Llama 3 Herd of Models (2024)

- **状态**: 未读
- **文件**: [Llama_3_Herd_of_Models_2024.pdf](frontier-llms/Llama_3_Herd_of_Models_2024.pdf)
- **来源**: [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)、[Meta AI 论文页](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- **建议读法**: 不必一次读完，先看 pre-training / post-training / safety / multimodal 的目录结构。
- **一句话**: 一份现代开源大模型“全流程工程说明书”，可以对照 DeepSeek-V3 看 dense 模型与 MoE 模型路线的差别。

## 8. Qwen2.5 Technical Report (2024)

- **状态**: 未读
- **文件**: [Qwen2.5_Technical_Report_2024.pdf](frontier-llms/Qwen2.5_Technical_Report_2024.pdf)
- **来源**: [arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115)、[Hugging Face 论文页](https://huggingface.co/papers/2412.15115)
- **建议读法**: 关注模型家族、长上下文、指令模型、代码/数学能力与评测。
- **一句话**: Qwen2.5 是理解中文和多语言开源模型生态的重要参照，可以和 DeepSeek-Coder、DeepSeekMath 对照读。

## 9. QLoRA: Efficient Finetuning of Quantized LLMs (2023)

- **状态**: 未读
- **文件**: [QLoRA_Efficient_Finetuning_of_Quantized_LLMs_2023.pdf](efficient-training/QLoRA_Efficient_Finetuning_of_Quantized_LLMs_2023.pdf)
- **来源**: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)、[GitHub](https://github.com/artidoro/qlora)、[Hugging Face PEFT LoRA 文档](https://huggingface.co/docs/peft/developer_guides/lora)
- **建议读法**: 先读 NF4、double quantization、paged optimizers，再回到 `phase3-training/02_lora.py` 和 `03_quantization.py`。
- **一句话**: 把 LoRA 和 4-bit 量化连起来，是个人显卡微调大模型时最值得掌握的论文之一。

## 10. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2023)

- **状态**: 未读
- **文件**: [Direct_Preference_Optimization_2023.pdf](efficient-training/Direct_Preference_Optimization_2023.pdf)
- **来源**: [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)、[Hugging Face TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- **建议读法**: 先理解它想绕开 PPO/RLHF 的哪几个复杂环节，再对照 `phase3-training/04_rlhf.py`。
- **一句话**: DPO 是偏好对齐路线里非常实用的一步：用偏好对直接优化语言模型，工程上比完整 RLHF 更轻。

## 配套训练 / 视频材料

- **DeepSeek-R1 复现练习**: [Hugging Face Open-R1](https://github.com/huggingface/open-r1)；适合读完 DeepSeekMath 和 DeepSeek-R1 后看数据生成、SFT、GRPO/RL 的复现路线。
- **后训练总览视频课**: [Fine-tuning & RL for LLMs: Intro to Post-training](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training/)；适合把 SFT、偏好优化、RL 和生产反馈闭环串起来。
- **RLHF 视频课**: [Reinforcement Learning From Human Feedback](https://www.deeplearning.ai/courses/reinforcement-learning-from-human-feedback)；适合配合 InstructGPT、DPO、DeepSeek-R1 阅读。
- **量化视频课**: [Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth/)；适合配合 QLoRA 和 `phase3-training/03_quantization.py`。
- **LoRA/QLoRA 实操文档**: [Hugging Face PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora)；适合从本项目的 `phase3-training/02_lora.py` 过渡到真实模型微调。

## 一个轻量学习节奏

1. 先读 DeepSeekMoE / V2 / V3，只抓住 MoE、MLA、FP8、MTP 四个关键词。
2. 再读 DeepSeek-Coder / DeepSeekMath / R1，观察“数据 -> SFT -> RL -> 蒸馏”的后训练链条。
3. 最后读 Llama 3 / Qwen2.5 / QLoRA / DPO，把开源模型工程、个人微调、偏好对齐补齐。

