# AGENTS.md — 给 AI Agent 的项目说明

> 本文件帮助 AI agent 快速理解本仓库的性质与约定。开始动手前请先读完。

## 这是什么项目

**面向后端开发者的大模型（LLM）学习仓库，用从零实现的代码理解原理。**
作者按阶段循序渐进地手写 NumPy / PyTorch 代码来吃透 Transformer、GPT、训练微调、DeepSeek 等知识点。

这是一个**个人学习项目**，不是要交付的产品。仓库里的代码、笔记、进度记录都是为了「学懂」，而不是为了「跑通业务」。

## 最重要的一条铁律 ⚠️

**绝对不要替作者完成 `*_self_write.py` 里的 TODO（除非用户明确要求）。**

每课都有配套的自写练习文件（`0X_topic_self_write.py`），里面是留白的 `TODO-1 ~ TODO-N`，**作者要亲手填**。这是学习的核心环节。Agent 的角色是：
- 解释原理、给思路、给提示、帮忙 debug
- 在用户明确说「帮我写/补全」时才动手
- 用户卡住时，引导而不是直接给答案

如果不确定用户是想自己写还是想让你写，**先问**。

## 目录结构

```
llm-learning/
├── phase0-math/            # 第 0 阶段：矩阵运算补强（按需复习）
├── phase1-foundations/     # 第一阶段：深度学习基础（NumPy/梯度/神经网络）
├── phase2-transformer/     # 第二阶段：Transformer/Attention/GPT
├── phase3-training/        # 第三阶段：训练、LoRA、量化、RLHF、推理优化
├── phase4-deepseek-reasoning/  # 第四阶段：DeepSeek、MoE/MLA、推理服务（多为 .md 资料）
├── papers/                 # 论文 PDF 库 + notes/ 精读笔记
├── README.md               # 项目总览与课程地图
├── learning-progress.md    # 个人学习进度（带日期的推进记录）
└── requirements.txt
```

## 每课的「三件套」约定

阶段 0–2 的每课通常包含三个文件：

| 文件 | 作用 | Agent 能否改 |
|------|------|--------------|
| `0X_topic.py` | **主课文件**：完整实现 + 讲解注释 + 打印输出，供阅读学习 | 可以改进/纠错/加注释 |
| `0X_topic_self_write.py` | **自写练习**：留 TODO 待作者填，内置 `require_*` 校验 | ⚠️ 默认不要替填 TODO |
| `reset_exercises_0X.py` | **重置脚本**：把自写练习的 TODO 恢复成待填状态，供二刷 | 可以维护 |

## 代码风格约定（写新课/改代码时遵守）

- 入口处加 `sys.stdout.reconfigure(encoding="utf-8")`——作者在 **Windows / PowerShell** 环境，否则中文输出乱码。
- 用 `section(title)` 函数打印分隔标题（`"=" * 60`）。
- 校验体系：自定义 `ValidationError` + `require_not_none` / `require_shape` / `require_close` 等 `require_*` 辅助函数，让作者填完一个 TODO 跑一次就能即时纠错。
- 注释和讲解全部用**中文**，注重「为什么这么做」而不是逐行翻译代码。
- 主课文件优先用 **NumPy** 手写实现（phase0–2），phase3 起用 PyTorch。

## 运行方式

```bash
# README 里用 python3；Windows PowerShell 下通常是 python
python phase2-transformer/04_transformer_block.py
python phase2-transformer/reset_exercises_01.py   # 重置某课练习
```

依赖见 `requirements.txt`（NumPy、Matplotlib、PyTorch、tiktoken）。

## 完成一课后要做的事

作者学完/改完一个知识点后，通常需要同步更新两个地方，agent 协助时请记得提醒或一并更新：

1. **`learning-progress.md`**：在「进展记录」加一条带日期（`YYYY-MM-DD`）的记录，写清完成了什么、关键领悟。
2. **根目录 `README.md`**：更新「当前状态」摘要和对应课程表里的进度标记。

## 沟通约定

- **始终用简体中文回复。**
- 这是学习场景，多解释原理、多给「为什么」，把 agent 当成耐心的助教而非代写工具。
