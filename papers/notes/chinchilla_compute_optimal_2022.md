# Chinchilla 精读：Training Compute-Optimal Large Language Models (2022)

> 论文 PDF：[`papers/scaling-laws/Training_Compute_Optimal_Large_Language_Models_Chinchilla_2022.pdf`](../scaling-laws/Training_Compute_Optimal_Large_Language_Models_Chinchilla_2022.pdf)
> 作者：Jordan Hoffmann et al.（DeepMind, 2022 年 3 月）
> 配套课程：第二阶段第 6 课 [`phase2-transformer/06_scaling_laws.py`](../../phase2-transformer/06_scaling_laws.py)
> 前置阅读：[`papers/notes/scaling_laws_kaplan_2020.md`](./scaling_laws_kaplan_2020.md)（Kaplan 2020）

**这篇论文是对 Kaplan 2020 的关键修正。** 它推翻了 GPT-3 时代「优先扩参数量、数据可以慢慢加」的设计哲学，给出了「**N 和 D 应该按 1:20 同步扩**」的新基准——这就是为什么 2022 年之后，开源大模型的训练 token 数从 GPT-3 时代的 300B 一下子跳到了 1T、3T、15T 这种量级。

---

## 阅读导航：先记住这 5 个结论

- **Kaplan 错了**：给定 compute C，N 和 D **大致按 1:1（斜率 0.5）同步扩**，而不是 Kaplan 的 0.73 / 0.27 偏向 N
- **经验口径 `D ≈ 20 · N`**：训一个 N 参数的模型，**建议训 20·N 个 token**
- **`C ≈ 6 · N · D`**：训练 compute 的工程估算（前向 2·N·D + 反向 4·N·D）——这个公式后来被全行业当成标准
- **三种独立 approach 给出一致结果**：（1）固定 N 扫 D；（2）固定 C 扫 N；（3）参数化拟合 `L(N, D)` ——三条路径都指向 1:20 这个口径
- **Chinchilla 70B > Gopher 280B**：用同样的 compute，把 280B 模型缩到 70B、把 token 数从 300B 拉到 1.4T，下游任务全面超越

---

## §1 Introduction：论文要解决什么问题

DeepMind 的疑问很直接：

> 「我们花了几千万美金训了 Gopher 280B，按 Kaplan 的口径它应该是 compute-optimal 的。但真的是吗？」

他们做的事：

1. 训练 **400 多个** 不同 (N, D) 组合的模型（最小 70M，最大 16B）
2. 用三种独立的方法**重新拟合** Scaling Law
3. 三种方法都指向同一个结论：**当前所有大模型都严重欠训**

**最有冲击力的对比**（Table 3 / §4.1）：

| 模型 | 参数量 N | 训练 token D | D/N 比 | Kaplan 视角 | Chinchilla 视角 |
|---|---|---|---|---|---|
| GPT-3 (2020) | 175B | 300B | 1.7 | optimal | **欠训 12 倍** |
| Gopher (2021) | 280B | 300B | 1.07 | optimal | **欠训 19 倍** |
| MT-NLG (2022) | 530B | 270B | 0.51 | optimal | **欠训 40 倍** |
| **Chinchilla (2022)** | **70B** | **1.4T** | **20** | suboptimal | **optimal** |

Chinchilla 用了**和 Gopher 完全一样的 compute**（≈ 5.76 × 10²³ FLOPs），但在几乎所有 benchmark 上吊打 Gopher。这就是论文标题的「compute-optimal」。

---

## §3 三种 Approach：为什么大家相信结论

这是论文最扎实的地方——**三个独立方法给出一致结果**。读这一节时，重点不是公式，而是「**三条独立证据链**」这个论证模式。

### Approach 1：固定 N，扫 D

**做法**：

- 训练一系列固定大小（70M, 250M, 500M, 1B, ..., 16B）的模型
- 每个模型训到不同的 token 数（中间不断 snapshot loss）
- 画出每个 N 下的 `loss vs compute` 曲线
- 找出每条曲线的「拐点」——即该 N 在多少 compute 时达到 loss 下界

**结论**：拐点处 `N ↑ √C`、`D ↑ √C`。换言之 `N : D ≈ 1 : (constant) · N`，且常数 ≈ 20。

**关键洞察**：这个 approach 跟 Kaplan 的不同在于——**每个模型都训到 loss 真正收敛**，而不是 Kaplan 用的固定步数（这是后面要讲的 Kaplan bug 的源头）。

### Approach 2：固定 C，扫 N（IsoFLOP）

**做法**：

- 选定几个 compute budget：`C = 6e18, 1e19, 3e19, 6e19, 1e20, 3e20 FLOPs`
- 对每个 C，训练多个不同 N 的模型，每个都把 D 调到 `D = C / (6·N)` 把 compute 用完
- 画 `loss vs N` 曲线，找最小值

**直觉**：固定预算下，N 太小则模型容量不够，N 太大则 D 不够（训不充分）。最低点就是给定 C 下的最优 N。

**结论**：最低点的 `N* ∝ C^0.5`，`D* ∝ C^0.5` ——再次给出 N:D = 1:1 同步扩。

> 这是 **IsoFLOP curves**（等 compute 曲线）的经典分析。论文的 Figure 3 用 9 条 IsoFLOP 曲线给了视觉化证据，强烈建议看这张图。

### Approach 3：参数化拟合 `L(N, D)`

**做法**：

直接假设一个联合形式：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

- `E` ≈ 1.69（语言本身的不可压缩 entropy 下界）
- `A`、`B`、`α`、`β` 通过对 400 个训练点拟合得到
- 拟合后求 `min L(N, D) s.t. C = 6ND` 给出最优 (N*, D*)

**拟合结果**：

| 系数 | 拟合值 | 含义 |
|---|---|---|
| `α` | 0.34 | N 的指数 |
| `β` | 0.28 | D 的指数 |
| `E` | 1.69 | irreducible entropy（"语言本身的随机性" 下界）|

由此推出：

$$
N^* \propto C^{0.46}, \quad D^* \propto C^{0.54}
$$

**注意**：α 跟 β 接近，所以最优解里 N 和 D 大致同步增长——再一次 1:1。

### 三条 approach 的一致性

| Approach | `N* ∝ C^?` | `D* ∝ C^?` |
|---|---|---|
| 1（固定 N 扫 D） | 0.50 | 0.50 |
| 2（IsoFLOP） | 0.49 | 0.51 |
| 3（参数化拟合） | 0.46 | 0.54 |
| **Kaplan 2020** | **0.73** | **0.27** |

**Chinchilla 三条线一致，且都跟 Kaplan 差很远**——这种「**多路径一致**」是这篇论文说服力最强的地方。

---

## §3.4 经验口径：`D ≈ 20 · N`

把三种 approach 的结论翻译成工程直觉：

> **训一个 N 参数的模型，应该训 ≈ 20·N 个 token。**

| 模型大小 N | Chinchilla-optimal D |
|---|---|
| 1B | 20B tokens |
| 7B | 140B tokens |
| 13B | 260B tokens |
| 70B | 1.4T tokens |
| 175B（GPT-3） | 3.5T tokens（不是 300B！）|
| 280B（Gopher） | 5.6T tokens |
| 540B（PaLM） | 10.8T tokens |

**Chinchilla 之所以取 70B + 1.4T**：他们手上的 compute budget（≈ 5.76e23 FLOPs）算出来正好对应这个组合。

---

## §2 `C ≈ 6 · N · D`：FLOPs 估算工程公式

这个公式来自 Kaplan 论文，但被 Chinchilla 用到极致，已经是行业标准。

**推导**：

- 一次前向：每个 token 经过模型走一遍，约 `2·N` FLOPs（每个参数参与一次乘加 = 2 FLOPs）
- 一次反向：约 `4·N` FLOPs（梯度计算的乘加约是前向的 2 倍）
- 一个 token 的训练 = 前向 + 反向 = `6·N` FLOPs
- 总 compute = `6 · N · D`

**实战用法**：

```text
GPT-3 训练 compute:
  N = 175e9, D = 300e9
  C ≈ 6 × 175e9 × 300e9 = 3.15e23 FLOPs

Llama 3-8B 训练 compute:
  N = 8e9, D = 15e12
  C ≈ 6 × 8e9 × 15e12 = 7.2e23 FLOPs
  → Llama 3-8B 的训练 compute 是 GPT-3 的 2.3 倍！

DeepSeek-V3 (671B total / 37B active) compute:
  按 active 参数算（MoE 推理路径）：
  C ≈ 6 × 37e9 × 14.8e12 ≈ 3.3e24 FLOPs
  （但论文用 dense-equivalent FLOPs，实际计算更复杂）
```

**估训练时长**：

```text
H100 BF16 算力：≈ 989 TFLOPs/s ≈ 1e15 FLOPs/s
GPT-3 训练 H100 等效 GPU-hours：
  3.15e23 / 1e15 / 3600 ≈ 87,500 GPU-hours
  → 1000 卡 ≈ 3.6 天（理想 100% 利用率）
  → 实际 MFU ≈ 40-50%，约 7-9 天
```

**这就是为什么 `C ≈ 6ND` 是大模型工程师的口算神器**——拿到任何模型的「参数量 + 训练 token 数」，秒算训练成本量级。

---

## §4 Chinchilla vs Gopher：实验验证

这一节是「**论文结论的实战检验**」。

**配置对比**：

| | Gopher | Chinchilla |
|---|---|---|
| 参数量 N | 280B | 70B |
| 训练 token D | 300B | 1.4T |
| Compute C | 5.76e23 FLOPs | 5.76e23 FLOPs（一样！）|
| Optimizer | Adam | Adam |
| Architecture | 同（都是 transformer decoder） | 同 |

**结果**（Table 5、6）：

- MMLU：Gopher 60.0 → Chinchilla **67.6**（+7.6）
- BIG-bench：Gopher → Chinchilla 全面提升
- LAMBADA：Gopher 74.5 → Chinchilla **77.4**
- TruthfulQA、推理类基准：Chinchilla 均胜

**关键论点**：

> **同样的 compute 下，小模型 + 多 token 比 大模型 + 少 token 更好**

这一发现直接催生了 2022 年之后的「**小模型猛吃 token**」浪潮——LLaMA 1（2023）就是按 Chinchilla 思路设计的：7B 训 1T tokens、65B 训 1.4T tokens。

---

## Kaplan 为什么错了？两个根因

理解 Chinchilla 不能不理解 Kaplan **错在哪**。论文附录给出了诊断：

### 根因 1：Learning rate schedule 没衰减到底

Kaplan 当时的实验里，cosine LR schedule 的**周期长度**对所有模型都用了相同步数。问题是：

- 小模型训到「设定步数」时已经接近收敛，所以 loss 接近真实最小值
- 大模型训到「设定步数」时 LR 还没衰减完，loss 还在大幅下降，**测得的 loss 偏高**

后果：

- 大模型的「真实下限 loss」被高估
- 拟合时小模型显得「相对划算」、大模型显得「相对不划算」
- 结论被推向「优先扩 N」——其实是因为 N 太大被 LR schedule 坑了

**Chinchilla 的修正**：每个模型用各自合适的 cosine 长度，确保都训到 loss 收敛。

### 根因 2：固定步数 vs 固定 token 数

Kaplan 训练时用「固定步数」，意味着大模型实际见的 token 数和小模型不一样（要看 batch size）。Chinchilla 改成「**每个 (N, D) 组合独立训练到 D 个 token**」，消除了这个混淆。

### 这个故事的工程教训

- **fit 出来的指数对实验设计极度敏感**——别看到「power law 拟合」就以为是物理定律
- **小模型上拟合的结论外推到大模型有风险**——尤其当训练 dynamics 在不同规模上不同时
- 这也是为什么 Chinchilla 之后又有 [Hoffmann 反推](https://arxiv.org/abs/2404.10102)、[Epoch AI 重审](https://epochai.org/blog/chinchilla-scaling-a-replication-attempt) 等后续争议——scaling law 的拟合工程门槛比想象中高

---

## Chinchilla 之后：over-training、推理成本、MoE

> ⚠️ 这一节不是论文里的，是把 Chinchilla 2022 之后的演化补全。第 6 课重点。

### 1. Llama 3 / Llama 3.1 的 over-training

Llama 3-8B 训了 **15T tokens**，按 Chinchilla 口径只需要 ≈ 160B tokens——**多训了 94 倍**！

为什么 Meta 要这么干？因为 **Chinchilla 只优化训练 compute，不优化推理 compute**：

- 训练一次：花 N 美金
- 推理：跑亿万次，每次推理 cost ∝ `N`（dense 模型）

如果模型要服务的 query 数巨大，「**训练成本翻 10 倍 + 推理成本省 50%**」远比「训练 compute-optimal」划算。

**Llama 3 论文的口径**：

> "We train Llama 3 8B for 15T tokens, going well beyond the Chinchilla-optimal point of ~200B tokens, because the inference cost of an 8B model that performs as well as a Chinchilla-optimal 70B model is dramatically lower."

这就是 **inference-aware scaling law**——把推理 cost 纳入优化目标后，最优解会偏向**更小的模型 + 更多的 token**。

### 2. DeepSeek-V3 的取舍（MoE 视角）

DeepSeek-V3 是 **671B 总参数 / 37B 激活**（MoE）。Scaling Law 在 MoE 下要重新讨论：

- **总 N** 决定模型容量上限
- **激活 N** 决定推理 cost
- 训练时 compute ≈ `6 · N_active · D`（不是总 N），因为只有激活的 expert 在算
- 训练 token 数 D = 14.8T

按 active 参数 37B 算 D/N ≈ 400，远超 Chinchilla 的 20。但 MoE 的 scaling law 不能直接套 dense 口径——这是开放的研究问题。

**直觉**：MoE 用「**多专家**」换「**容量**」，**用「过量训练**」换「**激活参数小**」。两个杠杆一起拉。

### 3. Qwen / Yi / Mistral 的实际选择

| 模型 | N | D | D/N |
|---|---|---|---|
| Llama 2-7B | 7B | 2T | 286 |
| Llama 3-8B | 8B | 15T | 1875 |
| Llama 3-70B | 70B | 15T | 214 |
| Mistral 7B | 7B | ~2T（未公开）| ~286 |
| Qwen2.5-7B | 7B | 18T | 2571 |
| DeepSeek-V3 | 37B(active) / 671B(total) | 14.8T | 400 / 22 |

**趋势很明显**：

- 小模型（7-8B）疯狂 over-train（D/N > 1000）
- 大模型（70B+）也 over-train，但倍数小一些（D/N ≈ 200）
- MoE 是新变量——总参数远大于训练 compute 隐含的 N

**核心结论**：「**Chinchilla 1:20 是训练 compute-optimal 的下限**，但工业界几乎都跑得比这个多得多。如果不考虑推理 cost，1:20 是最优；考虑推理时，跑到 1:200 甚至 1:2000 都合理」。

### 4. 测试时计算（test-time compute）改变了游戏规则

这是 2024-2025 年最新的 twist。

OpenAI o1、DeepSeek-R1 引入了「**推理时 thinking budget**」：

- 同一个模型，给它更多 reasoning token 就能解更难的题
- 「**测试时 compute**」成了第四个 scaling 变量（除 N、D、训练 C 之外）

新论文（如 [Snell et al. 2024](https://arxiv.org/abs/2408.03314)）开始研究「**训练 compute vs 测试时 compute 的最优分配**」——在某些任务上，「**小模型 + 大测试时 compute**」比「**大模型 + 小测试时 compute**」更划算。

→ 这也是 phase4 第 8 课的内容，第 6 课先有个印象。

---

## 第 6 课要回答的 3 个问题（复盘检验）

读完 Kaplan + Chinchilla 两篇之后，回答这 3 题：

**Q1**：手上有 1024 张 H100 跑 30 天（假设 MFU = 40%），按 Chinchilla 口径该训多大的模型？

```text
单卡算力：1e15 FLOPs/s
总 compute：1024 × 30 × 86400 × 1e15 × 0.4 ≈ 1.06e24 FLOPs

按 Chinchilla：N* ∝ C^0.46，D* ∝ C^0.54
查论文 Figure 3 / Table 3 对照：
  C ≈ 1e24 → N* ≈ 60-70B，D* ≈ 1.2-1.5T

→ 大致是「Chinchilla 70B」级别
```

**Q2**：为什么 GPT-3 175B 在 Chinchilla 视角下是「严重欠训」？

```text
GPT-3 N=175B，按 Chinchilla 应配 D = 20 × 175B = 3.5T tokens
但实际只训了 300B，少了 12 倍
→ 同样的 175B 模型如果训 3.5T tokens，loss 会显著更低
→ 反过来，给定 GPT-3 的 compute（3.15e23），Chinchilla 算出 N* ≈ 70B
   也就是说，把 175B 缩到 70B、token 数从 300B 拉到 1.4T，模型会更强
→ 这也解释了为什么 OpenAI 之后没再出更大的 dense 模型
```

**Q3**：Llama 3-8B 训了 15T tokens（D/N = 1875），为什么不是浪费？

```text
1. 训练成本是一次性的，推理成本是无穷次的
2. 一个能力跟 70B 持平的 8B，推理时 throughput 高 8-10 倍，cost 低 10 倍
3. 多花 5-10 倍训练 compute 把 8B 拉到接近 Chinchilla 70B 的水准 → 长期划算
4. 这就是 inference-aware scaling：Chinchilla 不是「最优」，只是「训练 compute-optimal」
```

---

## 与其他笔记的衔接

- 前置：[`scaling_laws_kaplan_2020.md`](./scaling_laws_kaplan_2020.md)（理解 Kaplan 三条幂律才能看懂 Chinchilla 修正了什么）
- 对照：[`gpt3_reading_2.1_model_and_architectures.md`](./gpt3_reading_2.1_model_and_architectures.md)（GPT-3 8 个尺寸 + 300B tokens 的「Kaplan 视角」）
- 配套代码：[`phase2-transformer/06_scaling_laws.py`](../../phase2-transformer/06_scaling_laws.py)（Part 3 实现 `compute_optimal_chinchilla(C)`，Part 4 把 GPT-3 / Chinchilla / Llama 3 放在 N-D 平面对比）
- 后续：phase4 第 1 课会再用这套口径解 DeepSeek-V3 的「671B total / 37B active」该怎么看

---

## TODO（读完后回填）

- [ ] 能口算「N=70B 按 Chinchilla 配 D=?」「N=175B 配 D=?」
- [ ] 用 `C ≈ 6ND` 估出 GPT-3、Llama 3-8B、Llama 3-70B、DeepSeek-V3 的训练 FLOPs
- [ ] 能解释 Kaplan 为什么错（LR schedule + 固定步数）
- [ ] 跑 `06_scaling_laws.py` Part 3-4，亲眼看 Chinchilla optimal 曲线和工业模型的偏离
- [ ] 思考：你团队如果要训自己的 7B 模型，token 数怎么定？（提示：考虑推理 QPS）
