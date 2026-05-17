# 第四阶段视频与课程材料

> 本页只收与 DeepSeek、推理优化、推理服务、reasoning 后训练强相关的视频/课程。学习时优先看「主线必看」，再按自己卡住的点补「专题补充」。

## 使用原则

- 不要把视频当背景音。每看一节，必须产出 3 条笔记：瓶颈是什么、技术做了什么、代价是什么。
- 优先看有代码或实验的课程。第 4 阶段的目标是能测、能部署、能解释数据。
- 中文材料用于建立直觉，英文课程用于补工程细节和一手术语。

## 主线必看

| 顺序 | 材料 | 类型 | 对应课程 | 重点 |
|------|------|------|----------|------|
| 1 | [DeepLearning.AI - Efficiently Serving LLMs](https://www.deeplearning.ai/courses/efficiently-serving-llms) | 视频短课 + 代码 | 第 3-6 课 | KV Cache、batching、continuous batching、quantization、LoRA serving |
| 2 | [DeepLearning.AI - Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth) | 视频短课 + 代码 | 第 2-4 课 | 对称/非对称量化、per-channel/per-group、量化误差 |
| 3 | [DeepLearning.AI - Fine-tuning & RL for LLMs: Intro to Post-training](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training) | 系统课程 | 第 7-8 课 | SFT、RLHF、PPO、GRPO、eval、post-training 到生产 |
| 4 | [Stanford CS336: Language Modeling from Scratch (2025)](https://cs336.stanford.edu/spring2025/index.html) | 大学课程 | 全阶段 | MoE、GPU、kernels、inference、evaluation、SFT/RLHF |
| 5 | [Stanford CS336 2025 视频清单（Class Central 汇总）](https://www.classcentral.com/course/youtube-stanford-cs336-language-modeling-from-scratch-i-2025-512656) | 视频合集入口 | 全阶段 | Lecture 4 MoE、5 GPU、6 Kernels、10 Inference、12 Evaluation、15-17 Alignment/RL |
| 6 | [Hugging Face Cookbook - Post training an LLM for reasoning with GRPO in TRL](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) | Notebook 教程 | 第 7-8 课 | 用 TRL 做 GRPO reasoning post-training |
| 7 | [Hugging Face - LLM inference optimization](https://huggingface.co/docs/transformers/v4.45.2/llm_optims) | 官方教程 | 第 3-6 课 | KV Cache、torch.compile、推理加速、TGI |
| 8 | [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit1/introduction) | 视频/交互课程 | 第 8 课 | Thought-Action-Observation、工具调用、Agent reasoning |

## DeepSeek / R1 / GRPO 视频

| 材料 | 语言 | 建议位置 | 重点 |
|------|------|----------|------|
| [李宏毅机器学习 2025 第七讲笔记：DeepSeek-R1 等推理模型如何进行深度思考](https://alanhou.org/blog/hungyi-lee-ml2025-lecture7/) | 中文 | 第 7 课前 | test-time compute、majority vote、distillation、reasoning model 直觉 |
| [李小羊学AI：DeepSeek-R1 论文精读 + GRPO 原理&代码讲解](https://www.bilibili.com/video/BV1wU98YcEGy/) | 中文 | 第 7 课 | R1 论文、GRPO、代码层理解 |
| [freeCodeCamp / Class Central: DeepSeek R1 Theory Tutorial - Architecture, GRPO, KL Divergence](https://www.classcentral.com/course/deepseek-r1-theory-tutorial-architecture-grpo-kl-divergence-435630) | 英文 | 第 7 课 | R1 架构、GRPO、KL divergence、distillation |
| [Hugging Face LLM Course 中文文档：理解 DeepSeek R1 论文](https://hugging-face.cn/learn/llm-course/chapter12/3) | 中文 | 第 1 / 7 课 | R1 论文导读和核心概念复习 |

## 推理服务 / 系统优化课程

| 材料 | 类型 | 建议位置 | 重点 |
|------|------|----------|------|
| [DeepLearning.AI - Efficiently Serving LLMs](https://www.deeplearning.ai/courses/efficiently-serving-llms) | 视频短课 | 第 3-6 课主课 | KV Cache、continuous batching、quantization、LoRAX |
| [vLLM 官网与教程入口](https://www.vllm.ai/) | 官方文档/教程 | 第 5 课 | PagedAttention、continuous batching、OpenAI-compatible API |
| [SGLang Documentation](https://docs.sglang.io/) | 官方文档 | 第 5 课 | RadixAttention、prefix cache、structured output、agent serving |
| [TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/index.html) | 官方文档 | 第 5 课选读 | NVIDIA 部署、paged KV、in-flight batching、quantization |
| [Hugging Face - LLM inference optimization](https://huggingface.co/docs/transformers/v4.45.2/llm_optims) | 官方教程 | 第 3-6 课 | Transformers 里的推理优化和 TGI |

## 大学课程精选讲次

Stanford CS336 不需要整门啃完。第 4 阶段优先看这些讲：

| 讲次 | 主题 | 对应课程 | 看什么 |
|------|------|----------|--------|
| Lecture 4 | Mixture of experts | 第 2 课 | MoE 为什么能扩参数但控制计算 |
| Lecture 5 | GPUs | 第 4-5 课 | GPU memory hierarchy、吞吐瓶颈 |
| Lecture 6 | Kernels, Triton | 第 4 课 | kernel 为什么影响 wall-clock |
| Lecture 10 | Inference | 第 3-6 课 | KV Cache、serving、decode 瓶颈 |
| Lecture 12 | Evaluation | 第 8 课 | 怎么设计评测而不是只看感觉 |
| Lecture 15 | Alignment - SFT/RLHF | 第 7-8 课 | SFT/RLHF 总览 |
| Lecture 16-17 | Alignment - RL | 第 7-8 课 | RL 后训练、reward、policy optimization |

课程主页：[Stanford CS336 Spring 2025](https://cs336.stanford.edu/spring2025/index.html)  
视频入口：[Class Central 汇总页](https://www.classcentral.com/course/youtube-stanford-cs336-language-modeling-from-scratch-i-2025-512656)

## DeepLearning.AI 课程顺序

如果你想集中看课程，建议顺序如下：

1. [Efficiently Serving LLMs](https://www.deeplearning.ai/courses/efficiently-serving-llms)：先建立 inference serving 直觉。
2. [Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth)：再理解本地模型为什么能压小、哪里会损失。
3. [Fine-tuning & RL for LLMs](https://www.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training)：最后接到 R1 / GRPO / reasoning 后训练。
4. [Generative AI with LLMs](https://www.deeplearning.ai/courses/generative-ai-with-llms)：如果第三阶段训练流程还不稳，用它补完整 LLM 生命周期。

## 中文补充材料

| 材料 | 适合什么时候看 | 注意 |
|------|----------------|------|
| [李宏毅 ML 2025 第七讲笔记](https://alanhou.org/blog/hungyi-lee-ml2025-lecture7/) | 刚开始理解 reasoning model | 用来建直觉，不替代论文 |
| [DeepSeek-R1 论文精读 + GRPO 原理&代码讲解](https://www.bilibili.com/video/BV1wU98YcEGy/) | 读 R1 论文前后 | 跟着画 R1 pipeline |
| [Hugging Face LLM Course 中文：DeepSeek R1](https://hugging-face.cn/learn/llm-course/chapter12/3) | 复习 R1 概念 | 适合快速回看 |

## 每课视频任务

| 课程 | 必看 | 选看 |
|------|------|------|
| 第 1 课 DeepSeek 总览 | 李宏毅 ML 2025 第七讲笔记、HF DeepSeek R1 导读 | freeCodeCamp DeepSeek R1 |
| 第 2 课 MoE | CS336 Lecture 4 | DeepSeekMoE 论文导读类视频 |
| 第 3 课 MLA / KV Cache | Efficiently Serving LLMs 的 Text Generation / KV Cache 相关课 | HF LLM inference optimization |
| 第 4 课 Attention Kernel | CS336 Lecture 5-6 | FlashAttention 论文讲解视频 |
| 第 5 课 Serving Runtime | Efficiently Serving LLMs 的 Batching / Continuous Batching | vLLM / SGLang 官方教程 |
| 第 6 课 解码加速 | Efficiently Serving LLMs、CS336 Lecture 10 | Speculative Decoding 论文讲解视频 |
| 第 7 课 R1 / GRPO | Fine-tuning & RL for LLMs、HF GRPO Cookbook、B站 R1 精读 | RLHF 课程 |
| 第 8 课 评测与测试时计算 | CS336 Lecture 12、Fine-tuning & RL 的 Evaluation 模块 | Hugging Face Agents Course |
