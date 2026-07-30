# 第三阶段：训练与微调

> 从训练流程到 LoRA、量化、RLHF —— 掌握大模型的实用技术

## 前置要求

完成第二阶段 + **PyTorch 专项（桥梁课）**，理解：
- Transformer 完整架构、自注意力与多头注意力、GPT 的训练与文本生成
- **完整训练流程**（DataLoader / 五步曲 / AMP / 梯度累积 / checkpoint）——这部分已在 [`pytorch-essentials/`](../pytorch-essentials/) 第 4/5/6/8 课学过，phase3 不再重复

## 环境准备

```bash
# 在项目根目录激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 课程结构（分层 + 时间盒）

> ⚠️ **phase3 换挡说明**：phase0-2 是数学/Transformer 基本功，"全程 NumPy 从零手搓"是对的；但 phase3 是**工程/应用**课，性质变了。不是每课都值得"从零手搓 + 三件套"。下表按 ROI 给每课定了**学法档位 + 完成标准 + 建议天数**——到点就过，不追求每课 100%。

**三档学法：**
- 🔨 **手搓**：值得亲手写代码 / 填 self_write 留白（phase3 只有 LoRA 和量化核心两处）
- ⚡ **跑通**：跑主课看现象 + 能改参数，不做 self_write 留白
- 📖 **精读**：概念 + 看图 + 笔记，**禁止从零手搓**（强行实现性价比极低）

> 🗑️ **原「训练流程」课已删除**（DataLoader / 五步曲 / AMP / 梯度累积 / LR 调度 / checkpoint / 整合）——这些已被后补的 **PyTorch 专项第 4/5/6/8 课**逐一覆盖，留着是纯重复。要复习训练流程直接看 [`pytorch-essentials/06_training_loop.py`](../pytorch-essentials/06_training_loop.py) 和 [`08_capstone_train.py`](../pytorch-essentials/08_capstone_train.py)。删课后文件已重命名为 `01_lora.py`…`04_inference_optimization.py`，phase3 从 LoRA 干净开始。

| 课程 | 文件 | 档位 | 完成标准（done） | 建议 |
|------|------|------|------------------|------|
| 第 1 课 LoRA | `01_lora.py` | 🔨 **手搓（重点）** | 亲手写出 `W+BA` 层 + 跑出"0.1% 参数微调有效"现象 + 试不同 rank | **2 天** |
| 第 2 课 量化 | `02_quantization.py` | 🔨 手搓核心 | 亲手写对称/非对称量化映射（~20 行）+ 看 INT8/INT4 精度损失现象 | **2 天** |
| 第 3 课 RLHF | `03_rlhf.py` | 📖 精读（防陷阱） | 画出 SFT→RM→PPO 数据流 + 亲手推一遍 DPO loss。**禁止从零搓 PPO**——读懂、能改参数即可 | **2 天** |
| 第 4 课 推理优化 | `04_inference_optimization.py` | ⚡ 跑通+已有 demo | KV Cache 概念吃透（已有 `kv_cache_numpy_demo.py`）+ 采样策略跑一遍 + 投机解码看懂思想 | 1 天 |
| 第 4 课·附 | `kv_cache_numpy_demo.py` | ⚡ 已完成 | 纯 NumPy 最小 KV Cache 对照演示，配合上面第 4 课一起看 | — |
| 第 5 课 分布式（待补） | 分布式训练专题 | 📖 精读+看图 | DP/TP/PP 各切什么 + ZeRO-1/2/3 切哪三样。**不手搓，看图** | 1 天 |

**合计约 1.5-2 周**（含缓冲）。删掉重复的训练流程课后，省下的力气全投到"真正填完 LoRA/量化的 self_write + 跑实验看现象"上。

> 📌 **配套论文是 just-in-time，不是先囤一书架。** 学到第 4 课需要 KV Cache 论文才翻 `papers/kv-cache/`，学到第 5 课需要 ZeRO 才翻 `papers/distributed-training/`。当前那批"待精读"论文是 backlog，别让它造成"还有一堆没读"的焦虑。

## 整体学习规划

### 哪些课要 self_write 留白（亲手填）

不是每课都配「三件套」。phase3 的 self_write 只给真正值得手写、且**没在别处写过**的内容：

| 课 | self_write？ | TODO 设计 | 理由 |
|----|:---:|------|------|
| 第 1 课 LoRA | ✅ **要** | `LoRALinear.forward`（W·x + BA·x·α/r）、`apply_lora` 冻结逻辑、`merge_lora` 的转置对齐，约 5 个 TODO | LoRA 核心就这几十行、面试必问，亲手写一遍收益最高 |
| 第 2 课 量化 | ✅ **要** | 对称/非对称 quant+dequant、`per_channel_quantize`、STE 伪量化 `backward`，约 6 个 TODO | 量化映射公式必须亲手推一遍，scale/zero_point 容易想当然 |
| 第 3 课 RLHF | ⚠️ **只 1 个（可选）** | 仅 `dpo_loss` 一个函数 | DPO loss 值得亲手推（它替代了整个 PPO）；**PPO/RM 不手搓**，读懂即可 |
| 第 4 课 推理 | ❌ **不要** | — | top-k/top-p 你已在 **phase2 第 5 课**亲手写过、KV Cache 已有 `kv_cache_numpy_demo.py`，再写是重复 |
| 第 5 课 分布式 | ❌ **不要** | — | 系统/分布式工程，概念 + 看图，不适合手搓 |

> **phase3 用「两件套」（主课 + self_write），不做 reset 脚本** —— 砍掉这层纯设施开销，正合效率原则。需要二刷就 `git checkout` 还原留白文件即可。
>
> 已建好的 self_write（留白待填）：[`01_lora_self_write.py`](01_lora_self_write.py)（5 TODO）、[`02_quantization_self_write.py`](02_quantization_self_write.py)（6 TODO）、[`03_rlhf_self_write.py`](03_rlhf_self_write.py)（仅 `dpo_loss` 1 TODO）。每个都自带 `require_*` 即时校验，全填对会跑一段小训练/估算收尾。第 4/5 课无 self_write。

### 推进节奏（约 9 天）

| 阶段 | 任务 | 产出 |
|------|------|------|
| **Day 1-2** | 第 1 课 LoRA：精读主课 → 填 self_write → 跑「0.1% 参数微调有效」+ 试 rank=1/4/8/16 | 手写的 `LoRALinear`，能解释 α/r 缩放与 merge 为何无推理开销 |
| **Day 3-4** | 第 2 课 量化：精读主课 → 填 self_write → 看 INT8/INT4/INT2 误差与压缩比 | 手写的对称/非对称/逐通道量化，能解释 zero_point 何时有用 |
| **Day 5** | 第 3 课 RLHF：Part 1–4 读主课 + 用 90 分钟定点串读 Christiano → Ziegler → Stiennon → InstructGPT | 一张「轨迹偏好→文本偏好→摘要 RLHF→广泛指令」演进图 |
| **Day 6** | Part 5 跑通 `dpo_loss` → 填 self_write TODO-1 → 读 DPO §3–4；再用 45 分钟读 FLAN + T0，回答 instruction tuning 与 RLHF 的边界 | DPO loss 推导 + 一段不超过 100 字的边界说明 |
| **Day 7** | 第 4 课 推理：跑主课看采样策略差异 + KV Cache 加速 + 投机解码思想 | 能解释 greedy/temp/top-k/top-p 区别、KV Cache 为何 decode 不需 causal mask |
| **Day 8** | 第 5 课 分布式：读 ZeRO 论文（概念+看图）+ 画 DP/TP/PP + ZeRO-1/2/3 切分图 | 一张「优化器状态/梯度/参数沿 N 卡切 1/N」的图 |
| **Day 9** | 缓冲 / 复盘：补没填完的 TODO、更新 learning-progress；有余力再开始 Askell / Weidinger 对齐与风险选读 | phase3 收尾；风险阅读不构成关闭阻塞项 |

### 每课的「关闭」标准

一课只有同时满足才算关闭、才开下一课（呼应「先关线再开线」）：
1. 主课 `python3 0X_xxx.py` 能 `exit 0` 跑通、现象看懂；
2. 🔨 课的 self_write 全部 TODO 填完、内置校验通过；📖/⚡ 课完成对应的「画图 / 推导 / 改参数实验」；
3. 在 `learning-progress.md` 写一条进展（踩了什么坑、关键领悟）。

## 每课详细大纲

### 第 1 课：LoRA 微调

- 全参数微调的问题：175B 参数的 GPT-3 你存都存不下
- 低秩分解的数学原理：W + ΔW ≈ W + BA
- LoRA 的核心思想：冻结原始权重，只训练小矩阵 A 和 B
- 从零实现 LoRA 层
- rank 的选择对效果的影响
- **与 LLM 的关系**：几乎所有开源模型微调都在用 LoRA

### 第 2 课：模型量化

- 浮点数回顾：FP32、FP16、BF16 的区别
- 量化的原理：把浮点数映射到整数
- 对称量化 vs 非对称量化
- 逐张量 vs 逐通道量化
- 量化对模型精度的影响
- **与 LLM 的关系**：4-bit 量化让 70B 模型跑在单张 24GB 显卡上

### 第 3 课：RLHF 人类偏好对齐

- 为什么预训练后的模型不好用？——"能力"vs"对齐"
- SFT（监督微调）：教模型学会对话格式
- 奖励模型（Reward Model）：学习人类偏好
- PPO 强化学习：用奖励信号优化生成策略
- DPO：不需要奖励模型的更简洁方法
- **与 LLM 的关系**：ChatGPT = GPT + SFT + RLHF

**配套论文（仓库已有 PDF）：**

| 论文 | 文件 | 本课对应 | 建议读法 |
|------|------|---------|---------|
| Christiano et al. (2017) | [`papers/efficient-training/Deep_RL_from_Human_Preferences_2017.pdf`](../papers/efficient-training/Deep_RL_from_Human_Preferences_2017.pdf) | Part 3（偏好 → RM）思想源头 | Figure 1 + §2.2.3，定点读 10 分钟 |
| Ziegler et al. (2019) | [`papers/efficient-training/Fine_Tuning_Language_Models_from_Human_Preferences_2019.pdf`](../papers/efficient-training/Fine_Tuning_Language_Models_from_Human_Preferences_2019.pdf) | 从轨迹偏好迁移到自然语言任务 | Figure 1 + §2 + §4.3–4.4，定点读 20 分钟 |
| Stiennon et al. (2020) | [`papers/efficient-training/Learning_to_Summarize_from_Human_Feedback_2020.pdf`](../papers/efficient-training/Learning_to_Summarize_from_Human_Feedback_2020.pdf) | Part 3–4（文本 RM → PPO + KL）直接前身 | §3.1 + Figure 2 + §3.4 + §4.3 Figure 5，精读 25 分钟 |
| InstructGPT (2022) | [`papers/core-transformers/InstructGPT_Training_LMs_to_Follow_Instructions_2022.pdf`](../papers/core-transformers/InstructGPT_Training_LMs_to_Follow_Instructions_2022.pdf) | Part 2–4（SFT → RM → PPO） | Section 3 + Figure 2，定点读 20 分钟 |
| DPO (2023) | [`papers/efficient-training/Direct_Preference_Optimization_2023.pdf`](../papers/efficient-training/Direct_Preference_Optimization_2023.pdf) | Part 5 + self_write | Section 3–4（推导 + loss 公式） |

**InstructGPT 的 RLHF 引用链怎么读（总计约 90 分钟）：**

1. **Christiano 2017（10 分钟，源头）**：只看 Figure 1 + §2.2.3。回答：为什么人类只选 A/B，就能学出标量奖励？把偏好概率 `P(A≻B)=σ(r_A-r_B)` 对到主课 Part 3。
2. **Ziegler 2019（20 分钟，语言迁移）**：看 Figure 1 + §2，再读 §4.3–4.4 的模糊任务与 bug。回答：偏好学习怎样首次系统接到 LM？策略为什么会放大标注捷径和实现错误？
3. **Stiennon 2020（25 分钟，高质量桥梁）**：看 §3.1 / Figure 2 的三段式、§3.4 的 RM loss 与 `R=r-β·log(π_RL/π_SFT)`，最后看 §4.3 Figure 5 的 reward over-optimization。回答：KL 为什么不是装饰？
4. **InstructGPT 2022（20 分钟，扩展）**：回看 Figure 2 + §3。回答：相比单任务摘要，它怎样扩展为 broad class of written instructions？SFT、比较数据、PPO 各吃什么数据？
5. **画图复盘（15 分钟）**：每篇只写「新增加的一步 / 新暴露的风险」。PPO 论文只按需回查 §3 式 (7)，不另开全文阅读线。

> **不读范围**：Christiano 的 Atari / MuJoCo 实验细节、Ziegler / Stiennon 的数据集细节和附录都先跳过。目标是补齐方法演进，不是新增全文债。

**DPO 之后再处理 related work 的另外两条线：**

1. **Instruction tuning 对照（45 分钟）**：FLAN 25 分钟 + T0 主文 20 分钟，回答「多任务监督微调已经会遵循指令，RLHF 额外优化什么？」
2. **对齐目标与风险（约 70 分钟，选读）**：Askell → Gabriel → Kenton → Weidinger；Carlini / Xu / PALMS 是隐私、干预副作用与 values-targeted 微调案例。
3. **完整页码、PDF 与暂缓引用清单**：见 [`papers/alignment-reading-map.md`](../papers/alignment-reading-map.md)。这些风险材料不阻塞本课关闭，避免再次囤读。

**DPO 推荐学习顺序（代码先行，论文对照）：**

1. **跑通主课 Part 5**：`python3 03_rlhf.py`，重点看懂 `dpo_loss` 里 `chosen_rewards = β·(log π - log π_ref)` 与 `-log σ(Δ_chosen - Δ_rejected)` 的含义
2. **填 self_write**：[`03_rlhf_self_write.py`](03_rlhf_self_write.py) TODO-1（只手写 DPO loss 外壳），`require_*` 校验通过
3. **再读 DPO 论文 Section 3–4**：对照推导与 loss 公式，把论文符号逐项映射到代码里的 `policy_chosen` / `ref_chosen` 等四项 log 概率

> Part 1–4 按同样思路：先过主课代码 + InstructGPT Figure 2，RM/PPO **读懂即可，禁止从零手搓**。

### 第 4 课：推理优化

- 自回归生成的瓶颈：每次只生成一个 token
- KV Cache：区分 prefill / decode，缓存 K、V，显著减少重复计算
- 采样策略对比：贪心、Temperature、Top-K、Top-P
- Beam Search（扩展练习）vs Sampling
- 投机解码（Speculative Decoding）：用 draft 猜多个 token，再由 target 并行验收
- **与 LLM 的关系**：这些方法分别优化串行生成、采样行为和 KV Cache 显存/调度
- **附加演示** `kv_cache_numpy_demo.py`：纯 NumPy 手写的 KV Cache 最小对照版（无缓存整段重算 vs 有缓存逐 token，验证结果一致 + 投影次数 O(n²) vs O(n) + 显存估算），配合 `papers/kv-cache/` 的 MQA / GQA / PagedAttention / FlashAttention 一起看

### 第 5 课：分布式训练专题（待补）

> 前面 1-4 课都是**单卡训练技巧**（LoRA、量化、推理优化）。这一课补上「多卡 / 集群训练」这块空白，把第一阶段第 4 课算过的「每参数 ~16 字节显存账」从单卡推到集群。

- 三种并行：数据并行（DP）、张量并行（TP）、流水并行（PP）各切什么、各自的通信代价
- **ZeRO（Zero Redundancy Optimizer）**：数据并行下每张卡都冗余存一份优化器状态/梯度/参数，ZeRO-1/2/3 依次把这三部分沿 N 张卡切成 1/N
- FSDP（PyTorch 原生的 ZeRO 等价实现）
- **配套论文**: [ZeRO (Rajbhandari et al., 2020)](../papers/distributed-training/ZeRO_Memory_Optimizations_Toward_Training_Trillion_Parameter_Models_2020.pdf)，见 [papers/README.md 第 12 节](../papers/README.md)
- **与 LLM 的关系**：万亿参数模型、7B/70B 全参训练为什么必须多卡，靠的就是这些切分策略
- **读法提示**：ZeRO 是系统/分布式工程，不适合像 Attention 那样从零手搓，定位成「概念精读 + 看图」即可

## 学习方式（效率四规则）

1. **先关线，再开线**：进 phase3 前先把欠的 TODO 收尾，别再新增论文/专题。一次只开一条线，不并行挂多课。
2. **区分"建设"和"学习"**：搭脚手架/reset/README 同步交给 AI，**你本人的时间只投在"亲手填 self_write + 跑实验看现象"**。设施再完美，不填 TODO = 没学。
3. **按档位投力气**：🔨 的课（LoRA / 量化）做完整 self_write；RLHF 只手写 `dpo_loss` 一个；⚡（推理）和分布式只跑主课 + 画图/记笔记，不做留白练习。
4. **每课守时间盒**：到建议天数就过，每课定义里的 done 标准达成即算完成，不追求 100%。

**具体动作：**
- **先理解概念**：每课开头有详细的原理讲解
- **跑代码看效果**：观察量化前后精度变化、LoRA 微调效果等
- **对比实验**：改参数（rank、量化位数、学习率），观察影响
- **读源码（选学）**：有余力再去看 HuggingFace PEFT、bitsandbytes 的实现

## 完成后你将理解

- 工业界如何训练和微调大模型
- 为什么 LoRA 能用极少参数达到接近全参数微调的效果
- 为什么量化后模型变小了但效果没差太多
- ChatGPT 从 GPT 到"能聊天"经历了哪些步骤
- 推理时的各种加速技巧背后的原理

## 推荐配套资源

- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft) - LoRA 等参数高效微调的官方实现
- [Andrej Karpathy - Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) - 完整的训练实战
- [李宏毅 - RLHF](https://www.youtube.com/watch?v=73kEe5bsLiQ) - 中文讲解 RLHF
- [DPO 论文](https://arxiv.org/abs/2305.18290) - 偏好对齐的简化路线；仓库 PDF 见 `papers/efficient-training/`，配合第 3 课 Part 5
- [Hugging Face TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer) - DPO 工程实现（读完 self_write 后可选）
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - 图解 GPT-2 的生成过程
- [QLoRA 论文](https://arxiv.org/abs/2305.14314) - 量化 + LoRA 的结合

## 下一步

完成第三阶段后，你已具备理解和使用大模型的完整知识体系。
可以开始：
- 用 HuggingFace 微调开源模型（Llama、Qwen 等）
- 部署自己的 LLM 服务
- 深入研究某个方向（多模态、Agent、长上下文等）
