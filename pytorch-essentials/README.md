# PyTorch 专项训练 — 从「NumPy 手写原理」到「PyTorch 工程惯用法」

> phase2 → phase3 之间的桥梁课。把你在 phase0–2 用 NumPy 手写吃透的原理，
> 翻译成 PyTorch 的工程化表达，为 phase3（训练 / LoRA / 量化 / RLHF / 推理）打地基。

## 这个专项解决什么问题

你已经能用 NumPy 手写反向传播、Attention、LayerNorm、优化器，也在 phase2 第 5 课
第一次用 `nn.Module` 搭出迷你 GPT。但有一段一直没系统化：**PyTorch 本身的机制**。

- phase3 第 1 课一上来就要 `DataLoader` + `AMP` + 梯度累积 + checkpoint，默认你已是 PyTorch 熟练工。
- 这个专项就是补上中间这一层：autograd、tensor 存储、Module 参数注册、数据管道、工程化训练循环、调试与显存。

**学习哲学**：同一个东西，先回顾你 NumPy 手写版，再看 PyTorch 怎么做，再**对拍验证数值一致**。
这就是「理论结合实践」的最强形态 —— 你不是在背 API，而是在确认「PyTorch 帮我做的，正是我手写过的那件事」。

## 课程结构（8 课）

每课沿用全仓库的「三件套」约定：

| 文件 | 作用 | 你能不能改 |
|------|------|-----------|
| `0X_topic.py` | **主课**：原理讲解 + 可运行演示 + 打印输出 | 可读可改 |
| `0X_topic_self_write.py` | **自写练习**：留白 TODO，内置 `require_*` 即时校验 | ⚠️ TODO 亲手填 |
| `reset_exercises_0X.py` | **重置脚本**：把自写练习恢复成待填状态，供二刷 | 维护用 |

| 课 | 主题 | 理论锚点（对照你已学） | 实践核心 |
|----|------|----------------------|----------|
| 第 1 课 | `01_tensor_basics.py` | phase0 形状 / 广播 | view vs reshape、stride / storage、contiguous、in-place、dtype / device、↔ numpy |
| 第 2 课 ⭐ | `02_autograd.py` | phase1 手写反向传播 | requires_grad / 计算图 / backward / detach / no_grad；**autograd 梯度与 NumPy 手写梯度对拍** |
| 第 3 课 | `03_nn_module.py` | phase2 各组件 | Parameter vs buffer、`state_dict`、`register_buffer`、`train()/eval()`、参数统计 |
| 第 4 课 | `04_loss_and_optim.py` | phase1 第 4 课优化器 | `CrossEntropyLoss` 与 logits、AdamW、param groups / weight decay、warmup+cosine scheduler |
| 第 5 课 | `05_data_pipeline.py` | phase2 第 5 课 `get_batch` | `Dataset` / `DataLoader` / `collate_fn`、train/val 切分、对比手写采样 |
| 第 6 课 ⭐ | `06_training_loop.py` | phase3 第 1 课预习 | 五步曲、early stopping、checkpoint 断点续训、AMP、梯度累积 / 裁剪、复现性 |
| 第 7 课 | `07_debug_profile_memory.py` | phase1 第 4 课显存账 | 经典报错、nan 检测、显存估算（每参数 16 字节）、`C≈6ND` FLOPs、profiler |
| 第 8 课 🎓 | `08_capstone_train.py` | 前 1~7 课全部 | **毕业项目**：把全部零件拼成一条连续训练流，训练一个迷你 token 语言模型（含 `nn.Embedding`、AdamW+param groups+warmup/cosine、累积/裁剪、eval、early stop、checkpoint 续训、显存/算力体检、自回归生成验收） |

> ⭐ = 性价比最高的两课。时间紧最少做 2、3、6 三课也能打通到 phase3。
> 🎓 = 第 8 课是综合验收：把前 7 课零件在一个文件里端到端组装，对标 nanoGPT 的最小训练骨架。

## 学习方式（每课 60–120 分钟）

1. 运行主课脚本 `python pytorch-essentials/0X_topic.py`，看输出建立整体印象。
2. 逐段读主课代码，重点看注释里的「为什么」与「对照你 NumPy 手写版」。
3. 打开 `0X_topic_self_write.py`，按 TODO 从前往后填。
4. 每填完一个 TODO 运行一次，靠 `require_*` 校验即时纠错（没填的返回 `None`，提示「未实现」是正常的）。
5. 全部通过后做 5 分钟复盘，写下「本课 3 个关键结论」。

二刷：`python pytorch-essentials/reset_exercises_0X.py` 把练习恢复成待填状态。

## 环境

```bash
python -m pip install -r requirements.txt   # torch / numpy / matplotlib
```

> 全部脚本 CPU 即可跑通（已在 `torch 2.12.0+cpu` 验证）。涉及 GPU 的部分（AMP、device 搬运）
> 代码会自动检测 `torch.cuda.is_available()` 优雅降级，注释里说明 GPU 上的差异。
> Windows / PowerShell 下中文输出已统一加 `sys.stdout.reconfigure(encoding="utf-8")`。

## 学完后你将能

- 看懂任意 PyTorch 模型代码的数据流，知道每个 tensor 操作背后的存储 / 广播含义。
- 解释 autograd 如何自动求出你以前手写的那些梯度，并能 debug 梯度相关问题。
- 独立写出工程化训练脚本：DataLoader + 优化器 + 调度器 + AMP + checkpoint 断点续训。
- 估算模型显存与训练算力（接得上 phase3 / phase4 的工业实践）。
