# 我的学习进度

> 记录个人推进轨迹、阶段状态与关键领悟。根目录 `README.md` 只保留项目导览和当前状态摘要。

## 当前状态

- 第 0 阶段：第 1 课进行中
- 第一阶段：三课已完成
- 第二阶段：第 3 课进行中（Attention 论文精读 §3.1 / §3.2.2 已推进完成，GPT-3 §2.1 精读笔记已新增，代码实验继续）
- 第三阶段：待学习

## 进展记录

- 2026-04-14：完成第一阶段第 3 课（神经网络前向/反向传播与 XOR 分类）自写练习，并已重置练习文件准备二刷。
- 2026-04-18：完成第二阶段第 1 课（词嵌入与位置编码）。跑通 `01_word_embeddings.py`、吃透正弦位置编码公式与 `div_term` 的 `exp(log)` 写法，看懂位置编码热力图/曲线图，完成自写练习 `01_word_embeddings_self_write.py`，并在 [phase2-transformer/README.md](phase2-transformer/README.md) 沉淀了"相加不是无损拼接，而是低成本注入顺序信息"和"放大镜 + 望远镜（小维度高频看近处、大维度低频看远处）"等 6 点关键领悟。可用 `reset_exercises_01.py` 重置练习准备二刷。
- 2026-04-18：补充 GPT-1 论文 PDF（`papers/core-transformers/GPT1_2018_improving_language_understanding.pdf`），并整理 [papers/notes/notes_gpt2_input_and_model.md](papers/notes/notes_gpt2_input_and_model.md) 笔记，配合第 1 课对照理解 decoder-only 的输入表示。
- 2026-04-25：完成第二阶段第 2 课（自注意力机制）。跑通 `02_self_attention.py` 并按 8 个 TODO 手写 `02_self_attention_self_write.py`：数值稳定 softmax -> Q/K/V 投影 -> scores -> scale -> softmax -> 加权 -> `self_attention` 函数封装 -> 因果掩码 `np.triu` -> `causal_self_attention`（GPT 用），全部校验通过。
- 2026-04-25：开始第二阶段第 3 课（多头注意力 / 残差连接 / LayerNorm）。在 [phase2-transformer/README.md](phase2-transformer/README.md) 补充了李宏毅 Multi-Head Attention 中文视频，按"看视频 -> 读论文 -> 跑 `03_multi_head_attention.py` -> 改 `n_heads`/去残差实验"的节奏推进。
- 2026-04-26：开始第 0 阶段第 1 课（向量、形状、axis、广播）。把 `01_vectors_and_axes.py` 的 axis 注释改成"聚合方向法"，给广播规则补了 5 条 Q&A（为什么从右对齐 / 为什么补 1 / 为什么不静默通过 / 不真复制内存 / 任意维度都成立），修正错误示范并加 `(3,4)` 与 `reshape(2,1,4)` 两个正确示范，给 `X+bias` 例子加 seed 和 diff 验证证明 bias 在 batch/seq 上共享。
- 2026-04-27：第 0 阶段第 1 课自写练习推进。在 `01_vectors_and_axes_self_write.py` 新增 TODO-0（动手验证 `(3,)` / `(1,3)` / `(3,1)` 的差异，练习 reshape 与 `None`/`np.newaxis` 两种写法）；在 TODO-4 后加 4a/4b/4c（3D 张量上的 axis、`keepdims` 与多轴聚合 `axis=(0,1)`、`mean`/`max`/`argmax` 共用 axis 规则）；在 TODO-7 `can_broadcast` 用例里补 6 个 5 维 case（含 attention scores + mask 的真实形状）。
- 2026-05-01：推进第二阶段第 3 课论文精读，完善 [papers/notes/attention_is_all_you_need_reading_3.1_3.2.2.md](papers/notes/attention_is_all_you_need_reading_3.1_3.2.2.md)：补清 FFN/MLP/position-wise 的术语关系、两层 FFN 中 `W_1/b_1` + ReLU + `W_2/b_2` 的结构、残差连接与残差相加的区别、encoder/decoder `N=6` 是实验超参数和对称设计而非硬约束、cross-attention 的 Q/K/V 来源与 mask 差异、causal mask 中 `✓/✗` 与 `0/1` 约定的区别、multi-head 中 concat/拼接、head 随机初始化打破对称性，以及 `W^O` 是可学习输出投影并负责混合多个 head。
- 2026-05-01：新增 [papers/notes/gpt3_reading_2.1_model_and_architectures.md](papers/notes/gpt3_reading_2.1_model_and_architectures.md)，精读 GPT-3 `2.1 Model and Architectures`：梳理 GPT-3 与 GPT-2 架构关系、dense / locally banded sparse attention 交替、8 个模型尺寸与 Table 2.1 读表注意、`d_model / d_ff / d_head` 的含义、Scaling Law 中验证 loss 与"暴力出奇迹"的关系，以及学习率、batch size、300B tokens 和 Chinchilla 修正口径。
- 2026-05-01：补充 Scaling Law 与 Sparse Transformer 延伸阅读资料。在 `papers/scaling-laws/` 下载 Kaplan et al. *Scaling Laws for Neural Language Models* 与 Chinchilla *Training Compute-Optimal Large Language Models*；在 `papers/efficient-transformers/` 下载 Child et al. *Generating Long Sequences with Sparse Transformers*；同步更新 [papers/README.md](papers/README.md) 和 [phase2-transformer/README.md](phase2-transformer/README.md)，把这些资料加入后续学习计划。
