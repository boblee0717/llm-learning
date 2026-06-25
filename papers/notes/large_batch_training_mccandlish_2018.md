# 精读：An Empirical Model of Large-Batch Training (2018)

> 论文 PDF：[`papers/scaling-laws/An_Empirical_Model_of_Large_Batch_Training_2018.pdf`](../scaling-laws/An_Empirical_Model_of_Large_Batch_Training_2018.pdf)
> 来源：[arxiv.org/abs/1812.06162](https://arxiv.org/abs/1812.06162)、[OpenAI blog](https://openai.com/index/how-ai-training-scales/)
> 作者：Sam McCandlish, Jared Kaplan, Dario Amodei, and the OpenAI Dota Team（OpenAI, 2018 年 12 月）
> 配套课程：第二阶段第 6 课 [`phase2-transformer/06_scaling_laws.py`](../../phase2-transformer/06_scaling_laws.py)、PyTorch 专项第 6 课 [`pytorch-essentials/06_training_loop.py`](../../pytorch-essentials/06_training_loop.py) 的 Part 7（梯度累积 / 大 batch）

这是 Kaplan scaling laws 笔记 [`scaling_laws_kaplan_2020.md`](./scaling_laws_kaplan_2020.md) §5.3 反复引用的「源头论文」——**critical batch size（临界批量大小）**和 **gradient noise scale（梯度噪声尺度）**这两个概念都来自这里。同一作者班底（McCandlish、Kaplan、Amodei）两年后写了 scaling laws，可以把这篇当成「scaling laws 之前，OpenAI 先把 batch size 这条轴搞清楚」。

不做逐字翻译，目标三个：

- 把「为什么 batch 不是越大越好」「临界 batch size 是什么」讲到能口述
- 讲清 gradient noise scale `B_noise` 的直觉与它怎么预测临界 batch size
- 跟工程实践（梯度累积、GPT-3 的 batch warmup）接上，知道这篇论文「在代码里对应哪一行」

---

## 阅读导航：先记住这 4 个结论

如果只想抓主线：

- 增大 batch size 能**减少达到目标 loss 所需的优化步数**，但存在**收益递减**：超过某个临界点后，再加 batch 几乎不再省步数，只是白烧 compute
- 这个临界点叫 **critical batch size `B_crit`**，它由一个可测量的量——**gradient noise scale `B_noise`**——预测，二者量级一致
- 训练存在一条「**步数 vs 样本数**」的权衡曲线（time/compute tradeoff），是一条漂亮的双曲线（Pareto front）：`B_crit` 正是这条曲线的拐点
- `B_noise` **不是常数**：它随训练推进而增大（任务越难、loss 越低，能用的 batch 越大），也随任务复杂度增大——这解释了为什么 RL（Dota）能用百万级 batch，而简单任务不行

---

## §1 这篇论文到底在回答什么问题

一个非常实际的工程问题：**我该用多大的 batch size？**

两种极端都不对：

- batch 太小：每步梯度噪声大，要很多步才收敛，**并行度低**（GPU 吃不饱、墙钟时间长）
- batch 太大：每步梯度已经很准了，再加样本只是边际改善，**compute 浪费**（同样的 loss 烧了更多算力）

论文的贡献是：给出一个**可以提前预测**「最优 batch size 在哪」的量化模型，而不用把所有 batch size 都试一遍。这个量就是 gradient noise scale。

> 作者班底里有「OpenAI Dota Team」——因为这套理论最早是为了解释 OpenAI Five（Dota 2 强化学习）为什么能用到**百万级 batch size**还在收益区间。论文同时在监督学习（图像、语言）和 RL 上验证。

---

## §2 核心权衡：步数 vs 样本数（time/compute tradeoff）

定义两个「成本」：

- **训练步数 `S`**（optimization steps）≈ 你要等多久（串行时间，受步数瓶颈）
- **样本总数 `E = B · S`**（examples processed）≈ 你烧了多少 compute（每个样本一次前向反向）

论文发现，要达到同一个目标 loss，`S` 和 `E` 满足一条非常干净的关系：

$$
\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1
$$

- `S_min`：batch → ∞ 时所需的**最少步数**（再加 batch 也省不动了）
- `E_min`：batch → 0 时所需的**最少样本数**（再减 batch 也省不动了）

这条曲线是一条**双曲线**（在 `S`-`E` 平面上是 Pareto front）。两个端点：

- 想**最省时间**（步数最少）→ 用很大的 batch，代价是样本数（compute）暴涨
- 想**最省 compute**（样本最少）→ 用很小的 batch，代价是步数（时间）暴涨

**拐点就是 critical batch size**：

$$
B_{\text{crit}} = \frac{E_{\min}}{S_{\min}}
$$

在 `B = B_crit` 时，相比理论极限，你只多花了 **2× 的步数**和 **2× 的样本数**——这是「时间」和「compute」都还能接受的甜点。

> 一句话记忆：**`B_crit` 是「时间」和「compute」两条成本的平衡点。** 比它小，浪费时间；比它大，浪费算力。

---

## §3 gradient noise scale：可测量、能预测 B_crit

怎么不试遍所有 batch 就知道 `B_crit` 在哪？论文给出一个可以**直接从梯度算出来**的量：**gradient noise scale `B_noise`**。

直觉：

- 每个样本算出来的梯度都是「真实梯度 + 噪声」
- 把 batch 里的梯度平均，噪声会被抹平 `1/B`
- **`B_noise` 衡量「梯度本身的大小」相对「梯度噪声的大小」**——信噪比

形式上（简化版）：

$$
B_{\text{noise}} = \frac{\operatorname{tr}(\Sigma)}{|G|^2}
$$

- `|G|^2`：真实梯度的模长平方（信号）
- `tr(Σ)`：单样本梯度协方差的迹（噪声）

含义：

- 当 `B ≪ B_noise`：噪声主导，加 batch 几乎线性减少步数（**值得加**）
- 当 `B ≫ B_noise`：信号主导，梯度已经很准，加 batch 收益递减（**别加了**）
- 论文证明 **`B_noise ≈ B_crit`**（量级一致）——所以测 `B_noise` 就能预测最优 batch

**为什么这很实用**：`B_noise` 可以在训练中**顺便估出来**（用不同 batch 的梯度方差，或多机梯度的差异），不需要真的扫描 batch size。

---

## §4 B_noise 不是常数：随训练推进 / 任务难度变大

这是和 scaling laws 衔接最紧的一条：

- **随训练推进变大**：训练初期 loss 高、梯度大且一致，`B_noise` 小；训练后期 loss 低、要捕捉更细的信号，`B_noise` 变大 → **后期才适合用大 batch**
- **随任务复杂度变大**：MNIST 的 `B_noise` 很小，ImageNet 大一些，语言建模更大，Dota RL 最大（百万级）——**越难的任务越能吃大 batch**

工程含义直接对应 GPT-3 的做法：

- GPT-3 训练时 batch size 从 **0.5M tokens 线性 warmup 到 3.2M tokens**
- 原因正是：训练早期 `B_noise` 小（大 batch 浪费），后期 `B_noise` 变大（大 batch 才划算）
- 这就是 Kaplan scaling laws §5.3 里「critical batch size 随训练进度增大 → 可以 warmup batch size」的来源

---

## §5 与 scaling laws / Chinchilla 的关系

把三篇论文串起来看：

| 论文 | 管的「轴」 | 关键量 |
|---|---|---|
| **本篇（2018）** | batch size（怎么并行、怎么不浪费 compute） | `B_crit` / `B_noise` |
| Kaplan scaling laws（2020） | N / D / C 怎么分配 | 三条幂律、`α_N/α_D/α_C` |
| Chinchilla（2022） | 固定 C 下 N:D 怎么配 | `N:D ≈ 1:20` |

- scaling laws 关心「该多大模型、多少数据、多少 compute」——它**默认你已经把 batch size 调到 `B_crit` 附近**，否则 compute 估算 `C ≈ 6ND` 里的 compute 利用率就不对
- 本篇关心「**给定要烧的 compute，怎么在『步数』和『样本数』之间分配，让墙钟时间最短而又不浪费算力**」
- 所以读 scaling laws 之前/之后读这篇都行：它补上了「为什么 batch size 也是 scaling 的一条独立轴」

---

## 与代码对照：在课程里对应哪一行？

### 第二阶段第 6 课 `06_scaling_laws.py`

- Part 3 的 `C ≈ 6ND` 隐含假设「batch size 调在 `B_crit`」——否则 compute 利用率（MFU）不对
- 与 Kaplan 笔记 §5.3 的 critical batch size 段落直接呼应

### PyTorch 专项第 6 课 `pytorch-essentials/06_training_loop.py` Part 7（梯度累积）

```text
batch_size=16, accumulation_steps=4  ≡  等效 batch_size=64
```

- **梯度累积**就是「显存不够时，用多个小 batch 凑出一个大 batch」——本篇告诉你「大到 `B_crit` 就够了，再大没意义」
- 实践决策链：① 估 `B_noise` → ② 目标 batch ≈ `B_crit` → ③ 显存装不下就用梯度累积凑 → ④ 多卡数据并行时每卡分一份
- 这解释了为什么工业训练脚本里「等效 batch size」是个被精心调的超参，而不是越大越好

---

## TODO（读完后回填）

- [ ] 能口述「步数 vs 样本数」双曲线和 `B_crit = E_min / S_min`
- [ ] 说清 `B_noise` 的信噪比直觉（信号 `|G|²` vs 噪声 `tr(Σ)`）
- [ ] 把「`B_noise` 随训练变大」和 GPT-3「batch 0.5M→3.2M warmup」对上
- [ ] 回到 `pytorch-essentials/06_training_loop.py` Part 7，理解梯度累积凑大 batch 的上限就是 `B_crit`
