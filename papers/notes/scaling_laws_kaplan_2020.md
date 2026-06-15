# Kaplan 精读：Scaling Laws for Neural Language Models (2020)

> 论文 PDF：[`papers/scaling-laws/Scaling_Laws_for_Neural_Language_Models_2020.pdf`](../scaling-laws/Scaling_Laws_for_Neural_Language_Models_2020.pdf)
> 作者：Jared Kaplan, Sam McCandlish, Tom Henighan, Tom Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei（OpenAI, 2020 年 1 月）
> 配套课程：第二阶段第 6 课 [`phase2-transformer/06_scaling_laws.py`](../../phase2-transformer/06_scaling_laws.py)

本文聚焦 Kaplan et al. 的「Scaling Laws for Neural Language Models」——这是 GPT-3 存在的**直接理论背景**，OpenAI 敢一口气训 175B 的依据就来自这篇。

不做逐字翻译，目的有三个：

- 把三条幂律 `L(N) / L(D) / L(C)` 讲到「记不住公式也能口述形状」
- 讲清「N、D、C 各自指什么」「为什么 loss 用 nats per token」这种容易卡的小坑
- 跟 Chinchilla 论文形成对照——Kaplan 哪些结论后来被修正了、哪些至今仍成立

---

## 阅读导航：先记住这 5 个结论

如果只想抓主线，可以先看：

- 验证 loss 会随**模型参数量 N**、**训练 token 数 D**、**训练 compute C** 三个变量分别呈**幂律下降**（power law）
- 三个变量同时变化时，loss 也可以拟合成一个**联合幂律**——这是 GPT-3 用来「外推损失」的依据
- 在 Kaplan 的口径下，**N 应当比 D 增长更快**：给定 compute 翻 10 倍，`N ↑ ~5.5×`、`D ↑ ~1.8×`
- 验证 loss 对架构细节（层数 / 宽度 / 头数的比例、`d_model / d_ff` 比值等）**不敏感**，只要 `N` 不变就行
- batch size 不是越大越好——存在一个 **critical batch size**（与 gradient noise scale 相关），超过后 compute 利用率掉头下降

后面三条结论 Chinchilla 修正了「N 比 D 增长更快」——那是 2022 年的事，请先把 Kaplan 的视角吃透。

---

## §1 Introduction：论文在干什么

这篇论文做的事其实很朴素：

1. 训练**一大堆**不同尺寸的 Transformer 语言模型（最小 768 参数，最大 1.5B）
2. 对每个尺寸用不同的 D、C 训练
3. 把所有训练曲线放在 log-log 图上看
4. 发现**几乎所有曲线都是直线**——这就是「幂律」

> 「幂律」（power law）在 log-log 图上是直线。形式是 `y = a · x^b`，两边取 log：`log y = log a + b · log x`。直线的斜率就是指数 `b`。

为什么这件事重要？

- 直线意味着**可外推**。在 1.5B 参数上拟合的直线，可以预测 175B 时的 loss 长什么样
- 这是「**敢花几亿美金训一个超大模型**」的关键——你不是闭眼赌博，你能算出 loss 会落在哪个区间
- GPT-3 论文里那张「8 个尺寸的验证 loss 接近一条直线」的 Figure 3.1，就是 Kaplan 这套理论的实际验证

---

## §3.1 三条幂律：核心公式

论文给出了三条「**单变量极限**」下的幂律。注意每一条都有适用条件——**其他两个变量不能成为瓶颈**。

### 公式 1：参数量 N 是瓶颈时（数据足够、训练足够）

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}
$$

- `N` = **非 embedding 参数量**（这点很关键，下面单开一节讲）
- `α_N ≈ 0.076`
- `N_c ≈ 8.8 × 10^13`（拟合常数）

含义：模型参数量翻倍，loss 大约下降 `2^(-0.076) ≈ 5%`。听起来不多，但是是**绝对值**——loss 从 3.0 降到 2.85 就意味着 perplexity 从 20 降到 17.3。

### 公式 2：数据量 D 是瓶颈时（模型足够大、训练步数足够）

$$
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

- `D` = 训练 **token 数**（不是「文档数」「样本数」）
- `α_D ≈ 0.095`
- `D_c ≈ 5.4 × 10^13`

### 公式 3：compute C 是瓶颈时（模型 / 数据可以自由调）

$$
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

- `C` = 训练所消耗的**总 compute**（FLOPs，或等价的 PF-days）
- `α_C ≈ 0.050`
- `C_c ≈ 3.1 × 10^8` PF-days

> **PF-days**：1 PF-day = `10^15 × 86400 ≈ 8.64 × 10^19 FLOPs`。GPT-3 训练 ≈ 3640 PF-days。

### 一句话总结

三条线**斜率不同**：

| 变量 | 指数 `α` | 翻倍带来 loss 下降 |
|---|---|---|
| `N`（参数） | 0.076 | `~5%` |
| `D`（数据） | 0.095 | `~6%` |
| `C`（compute） | 0.050 | `~3%` |

数据的指数最大、compute 的指数最小，这是因为 compute 翻倍要分给「更大模型」和「更多步训练」两部分。

---

## 关键细节 1：N 为什么不算 embedding？

这是这篇论文一个让无数人卡住的坑：**Kaplan 在算 N 的时候把 embedding 排除了**。

**为什么排除**：

- embedding 矩阵 `(vocab_size, d_model)` 的大小由 vocab_size 决定，跟模型「容量」关系不大
- 把 embedding 算进去会让小模型的 N 被 embedding 撑大，破坏幂律的干净性
- 在大模型上 embedding 占比极小，加不加几乎不影响结论；但在小模型上影响很大

**实际怎么算**：

对于 GPT-style decoder（L 层、隐藏维度 d_model、FFN 倍数 4）：

$$
N \approx 12 \cdot L \cdot d_{\text{model}}^2
$$

推导：每层有

- Attention 投影 4 个矩阵（Q/K/V/O），每个 `d_model × d_model` → `4 · d_model^2`
- FFN 两个矩阵，`d_model × 4·d_model` 和 `4·d_model × d_model` → `8 · d_model^2`
- 合计每层 `12 · d_model^2`，乘 L 层

**注意 Chinchilla 不这么算**——Chinchilla 用「**包含 embedding 的总参数量**」。这是两篇论文口径差异的一个来源。第 6 课读到 Chinchilla 时会再提。

---

## 关键细节 2：loss 单位是什么？

论文里所有的 loss 都是 **nats per token**（自然对数下的交叉熵）。

- `nats` 是用自然对数 `ln` 算的 entropy 单位（`bits` 是用 `log_2`）
- **每个 token** 的 loss——所以 D 越大，total loss 越大，但 **per-token loss 越小**

换算：

- `loss = 2.0 nats/token` → `perplexity = e^2.0 ≈ 7.4`
- `loss = 3.0 nats/token` → `perplexity ≈ 20`

> **perplexity** 的直觉：模型在每个位置「平均在多少个候选 token 间犹豫」。GPT-2 训练后 perplexity ≈ 35（WikiText-103），人类语言的下限大概在 7-15 之间。

---

## §3.2-3.4 联合幂律：把 N、D、C 一起拟合

如果三个变量同时变化，Kaplan 用下面这个**联合函数**拟合：

$$
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}
$$

不用记这个公式，记结论：

- 当 N 远小于「相对 D 的最优值」时，loss 接近 `L(N)` 那条线
- 当 D 远小于「相对 N 的最优值」时，loss 接近 `L(D)` 那条线
- 中间区域是两者的加权平均

实践含义：**单独翻一边不会有收益**。如果你 N 已经够大但 D 不够，再加 N 也只能挪一点点 loss；反之亦然。

---

## §5-6 Kaplan 的「Compute-Optimal」结论

> ⚠️ 这一节就是 Chinchilla 后来推翻的部分。先把 Kaplan 当时的结论记住，再看 Chinchilla 怎么改。

Kaplan 的拟合给出了「给定 compute C，N 和 D 该各自取多少能让 loss 最低」的关系：

$$
N_{\text{opt}} \propto C^{0.73}, \quad D_{\text{opt}} \propto C^{0.27}
$$

**意味着**：compute 翻 10 倍，应该让

- 参数量 `N ↑ 10^0.73 ≈ 5.4×`
- 数据量 `D ↑ 10^0.27 ≈ 1.86×`

**直白翻译**：「**优先把 compute 花在加大模型上，token 数加得慢一点**」。

这就是 GPT-3 175B + 300B tokens 的理论依据：

- GPT-2（1.5B 参数）训练 ≈ 40B tokens
- GPT-3（175B 参数，约 117 倍）→ token 数只增到 ≈ 7.5 倍 ≈ 300B

按 Kaplan 的口径，这是「**compute-optimal**」的。

**Chinchilla 2022 推翻这个**：实验设计有问题（学习率衰减没到底），实际上 `N` 和 `D` 应该**按 1:1 同步扩**——`N ↑ √10`、`D ↑ √10`，对应 GPT-3 175B 应该配 ≈ 3.5T tokens 才合理。

---

## §5.2 验证 loss 对架构超参不敏感

这条结论 GPT-3 笔记里反复引用过：

> 在 `N` 固定的前提下，改变 `L`（层数）/ `d_model`（宽度）/ `n_heads`（头数）/ `d_ff/d_model`（FFN 倍数）等比例，验证 loss 变化很小（在 ±2% 以内）

实际含义：

- GPT-3 表里那 8 个尺寸的具体超参比例（如 `n_heads` 怎么取）**不是理论推导**，而是按 GPU 并行效率挑出来的
- 这给后来的工业实现留下了很大的自由度——只要 N 对，怎么切层数 / 宽度都行
- 但**不能极端**：`L=1` 或 `L=1000` 这种极端比例还是会偏离曲线

---

## §5.3 Critical Batch Size：batch size 不是越大越好

论文借用了另一篇工作（McCandlish et al. 2018, *An Empirical Model of Large-Batch Training*）的概念：**gradient noise scale**。

> 这篇源头论文已下载并配了精读笔记：[`large_batch_training_mccandlish_2018.md`](./large_batch_training_mccandlish_2018.md)（PDF：[`papers/scaling-laws/An_Empirical_Model_of_Large_Batch_Training_2018.pdf`](../scaling-laws/An_Empirical_Model_of_Large_Batch_Training_2018.pdf)）。想把「critical batch size 到底怎么算」搞透，读它。

**直觉**：

- 小 batch：每一步的梯度估计噪声大，需要更多步才能收敛
- 大 batch：每一步梯度估计稳定，但每一步 compute 贵，超过某个阈值后再加 batch 收益递减
- **critical batch size** = 「再加 batch 就开始浪费 compute」的拐点

**关键发现**：critical batch size 随**训练进度增加**而增大（loss 下降越多，越能受益于大 batch）。所以训练后期可以「**线性 warmup batch size**」——这是 GPT-3 的做法（详见 GPT-3 笔记 §2.1 句 5 的 batch size 表）。

GPT-3 175B 最终 batch size 高达 3.2M tokens，就是因为大模型的 critical batch size 大。

---

## §6 Conclusions：六个核心断言

论文最后给出六条结论，作为 takeaway：

1. **Performance depends strongly on scale, weakly on model shape**：性能强依赖于规模（N、D、C），弱依赖于架构形状
2. **Smooth power laws**：在我们能观测到的所有规模区间，loss 都呈幂律
3. **Universality of overfitting**：一旦 N、D 同时增加而比例失衡（比如 N ↑ 但 D 不变），就会过拟合——过拟合本身也呈幂律
4. **Universality of training**：训练曲线（loss vs step）的形状不依赖于具体的 N，只依赖于「相对于 critical batch size 的进度」
5. **Transfer improves with test performance**：在新分布上的迁移 loss 与训练分布的 loss 呈正相关
6. **Sample efficiency**：大模型更样本有效——用更少的 data point 就能达到同样的 loss

**第 6 条**最重要：「**大模型 sample efficient**」是支撑 Kaplan「应该优先扩 N」结论的核心论据。Chinchilla 实际上不否认这点，只是指出「**当你 compute 受限时，单纯优化每 sample loss 不是最优的，要考虑 D 的边际收益**」。

---

## 与代码对照：在 `06_scaling_laws.py` 里能验证什么？

第 6 课代码会展示：

```text
Part 1：打印三条幂律的公式与指数（对应本文 §3.1）
Part 2：画 L(N) / L(D) / L(C) 三条曲线（对应 Figure 1）
Part 3：实现 C ≈ 6ND（FLOPs 估算）
Part 4：把 GPT-3 / Chinchilla / Llama 3 / DeepSeek-V3 放在 N-D 平面上对比
```

Part 3 的 `C ≈ 6ND` Kaplan 原文里也有（§2.1 FLOP estimates），但 Chinchilla 用得更频繁。等读 Chinchilla 时会更清楚为什么这个公式重要。

---

## 与 GPT-3 笔记的衔接

读完 Kaplan，回头看 [`papers/notes/gpt3_reading_2.1_model_and_architectures.md`](./gpt3_reading_2.1_model_and_architectures.md) 应该有这些「啊原来如此」时刻：

- GPT-3 训 8 个尺寸 = 「**Kaplan 那张图的 OpenAI 自家复现**」
- GPT-3 最终 175B + 300B tokens = 「**按 Kaplan compute-optimal 算出来的**」
- 「验证 loss 对架构不敏感」直接来自 Kaplan §5.2
- batch size 从 0.5M → 3.2M 随 N 线性增长 = Kaplan §5.3 critical batch size 的工程兑现

---

## 这一篇论文最大的「时代局限」

读这篇时务必带着 **2020 年视角**：

- 当时最大 dense LM 是 GPT-2（1.5B），论文最大也只到 1.5B —— 拟合用的样本规模其实有限
- 论文里所有的训练 token 数都 ≪ 100B，远小于今天动辄数 T tokens 的常态
- 学习率 schedule 用的 cosine 没有衰减到底（这是 Chinchilla 后来找到的关键 bug）

→ Kaplan 的**三条幂律 / N 不含 embedding / C ≈ 6ND 框架**至今仍成立；**「N 比 D 增长更快」的具体口径**被 Chinchilla 修正。下一篇就读 Chinchilla。

---

## TODO（读完后回填）

- [ ] 把三条幂律的指数 `α_N / α_D / α_C` 记在脑子里
- [ ] 能用 `12 · L · d_model^2` 估 GPT-2 small（L=12, d_model=768）的 N
- [ ] 在 `06_scaling_laws.py` 跑一遍 Part 2，亲眼看三条曲线
- [ ] 跳到 Chinchilla 笔记，对照「Kaplan 错在哪」
