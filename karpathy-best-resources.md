# Andrej Karpathy 值得优先阅读/观看的内容

> 面向当前项目（`llm-learning`）的精选清单：少而精、可直接执行。

## 快速开始（先看这 3 个）

如果你时间有限，先按这个顺序：

1. [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)  
   - 类型：视频（通识全景）
   - 为什么先看：一次建立 LLM 全栈心智模型（数据、tokenizer、模型、训练、对齐、推理）。
   - 建议耗时：3.5 小时（可分 3 次看）

2. [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)  
   - 类型：视频系列（技术主线）
   - 为什么必看：从反向传播到 GPT，完整“从零实现”路径，最适合打牢代码直觉。
   - 建议耗时：15-20 小时（分阶段推进）

3. [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
   - 类型：视频（实战）
   - 为什么关键：把 Transformer/GPT 从概念落实成可运行代码，直接对齐 `phase2-transformer` 学习目标。
   - 建议耗时：2-4 小时（含跟敲）

## 推荐文章（优先读）

### 1) [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- 类型：文章
- 适合人群：入门到进阶
- 为什么值得读：几乎是“训练调参排障手册”，对后续 `phase3-training` 很实用。
- 前置知识：基础梯度下降、过拟合/欠拟合概念
- 建议耗时：45-60 分钟

### 2) [Software 2.0](https://karpathy.github.io/2017/11/14/software-2.0/)
- 类型：文章
- 适合人群：工程背景转 AI 的开发者
- 为什么值得读：解释“规则编程 -> 数据驱动编程”的范式迁移，帮助建立长期认知框架。
- 前置知识：基本软件工程经验
- 建议耗时：20-30 分钟

### 3) [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- 类型：文章
- 适合人群：刚学序列建模的同学
- 为什么值得读：虽然是 RNN 时代文章，但非常有助于理解“语言建模”问题本质。
- 前置知识：基础神经网络
- 建议耗时：30-45 分钟

## 推荐视频（按学习阶段）

### 阶段 A：建立直觉（先全局后细节）

#### 1) [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)
- 类型：长视频
- 适合人群：所有人
- 为什么值得看：覆盖从预训练到 RLHF 的关键链路，减少“只会写代码但不懂全局”的问题。
- 建议耗时：3.5 小时

#### 2) [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- 类型：视频
- 适合人群：入门者
- 为什么值得看：快速建立“LLM 是什么、怎么训练、怎么用”的最低认知闭环。
- 建议耗时：1 小时左右

### 阶段 B：从零实现（对齐本项目第二阶段）

#### 3) [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- 类型：系列课程
- 适合人群：希望真正“写出来”的学习者
- 为什么值得看：微观到宏观，覆盖 micrograd、makemore、GPT 构建全过程。
- 建议耗时：15-20 小时

#### 4) [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- 类型：实战视频
- 适合人群：正在学 Transformer 的同学
- 为什么值得看：和 `phase2-transformer/` 的学习目标高度一致，能建立实现细节直觉。
- 建议耗时：2-4 小时

#### 5) [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- 类型：实战视频
- 适合人群：想理解 tokenizer 影响的人
- 为什么值得看：补齐训练前最容易被忽略但非常关键的一环。
- 建议耗时：1-2 小时

### 阶段 C：训练工程化（对齐第三阶段）

#### 6) [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- 类型：实战长视频
- 适合人群：准备做训练/复现实验的人
- 为什么值得看：从模型搭建到训练提速与复现实验，工程细节密度很高。
- 建议耗时：4-6 小时（含实践）

## 与本项目的映射关系

- `phase1-foundations/`：先看 `Zero to Hero` 前半（micrograd + makemore 前几讲）
- `phase2-transformer/`：重点看 `Let's build GPT` + `GPT Tokenizer`
- `phase3-training/`：重点看 `A Recipe for Training Neural Networks` + `Let's reproduce GPT-2`

## 建议学习顺序（7 天版）

1. Day 1：`Deep Dive into LLMs`（建立全局）
2. Day 2：`Software 2.0` + `Intro to LLMs`（框架 + 入门闭环）
3. Day 3-4：`Zero to Hero` 前半（反向传播、语言建模）
4. Day 5：`Let's build GPT`
5. Day 6：`GPT Tokenizer` + 回看 `phase2-transformer` 代码
6. Day 7：`A Recipe for Training Neural Networks` + 制定 `phase3` 实验计划

## 维护说明

- 新增资源时优先官方来源（`karpathy.ai`、本人 YouTube、本人博客）。
- 优先补充“高信噪比、能直接指导实践”的内容，避免泛泛转载。
